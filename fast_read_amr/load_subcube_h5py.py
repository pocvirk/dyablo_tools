"""
Standalone sub-cube loader for Dyablo HDF5 snapshots — bypasses pyablo.

Design
------
Two-pass streaming approach that never loads the full dataset into memory:

  Pass 1 — stream connectivity + coordinates in chunks
    • For each chunk of cells, reads the compact node-coordinate range
      referenced by that chunk (contiguous due to Morton ordering)
    • Computes cell centres as the average of 8 hex vertices
    • Applies a spatial pre-filter to keep only cells in the sub-cube
    • Records the surviving global index ranges (a handful of contiguous
      runs, one per overlapping MPI rank)

  Pass 2 — targeted field reads
    • For each requested field, reads only the surviving index ranges
      via HDF5 hyperslab selection — no full-array read

Memory at any time
    O(chunk_size)  during pass 1  (~150 MB for chunk_size=500_000)
    O(N_subcube)   for the output arrays

Works for any sub-cube size, including cases where the full snapshot is
many times larger than available RAM.

HDF5 layout (Dyablo output)
    connectivity  (N, 8)  uint64   — 8 node indices per cell (hex)
    coordinates   (M, 3)  float64  — node positions
    level         (N,)    uint32   — raw AMR level (lcell = level + log2(bx))
    <fieldname>   (N,)    float64  — one dataset per field

Usage
-----
    from load_subcube_h5py import load_subcube, metadata_from_config

    Lbox, bx = metadata_from_config('restart_star_279982.ini')
    cc, lcell, fields = load_subcube(
        'star_iter0070000.h5',
        fieldnames = ['rho', 'rho_HII'],
        lcube      = 9,
        Lbox       = Lbox,
        bx         = bx,
        extent     = [0, 0.5, 0, 0.5, 0, 0.5],
    )
    # cc, lcell, fields['rho'] are ready to pass to multiamr2cube_numba_full
"""

import numpy as np
import h5py
from configparser import ConfigParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def metadata_from_config(config_path):
    """
    Read Lbox and bx from a Dyablo .ini config file.

    Returns
        Lbox : (3,) float64  domain size [Lx, Ly, Lz]
        bx   : int           block size (same in x/y/z for cubic blocks)
    """
    cfg = ConfigParser(inline_comment_prefixes=('#', ';'))
    cfg.read(config_path)
    Lbox = np.array([
        cfg.getfloat('mesh', 'xmax') - cfg.getfloat('mesh', 'xmin'),
        cfg.getfloat('mesh', 'ymax') - cfg.getfloat('mesh', 'ymin'),
        cfg.getfloat('mesh', 'zmax') - cfg.getfloat('mesh', 'zmin'),
    ])
    bx = cfg.getint('amr', 'bx')
    return Lbox, bx


def _index_ranges(mask, offset=0):
    """
    Convert a boolean mask to a list of (start, end) contiguous True runs.
    end is exclusive.  offset shifts all indices by a global cell offset.
    """
    padded = np.empty(len(mask) + 2, dtype=np.bool_)
    padded[0]  = False
    padded[1:-1] = mask
    padded[-1] = False
    diff   = np.diff(padded.view(np.uint8))
    starts = np.where(diff ==  1)[0] + offset
    ends   = np.where(diff == 255)[0] + offset   # 0 - 1 wraps to 255 for uint8
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_ranges(ranges):
    """Merge adjacent (start, end) pairs that share a boundary."""
    if not ranges:
        return []
    merged = [list(ranges[0])]
    for s, e in ranges[1:]:
        if s == merged[-1][1]:      # adjacent — extend
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [tuple(r) for r in merged]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_subcube(
    h5path,
    fieldnames,
    lcube,
    Lbox,
    bx,
    extent     = [0, 1, 0, 1, 0, 1],
    chunk_size = 500_000,
    verbose    = True,
):
    """
    Stream-load a spatial sub-cube from a Dyablo HDF5 snapshot.

    Parameters
    ----------
    h5path      : str         path to the .h5 snapshot file
    fieldnames  : list[str]   field datasets to load (e.g. ['rho', 'rho_HII'])
    lcube       : int         target grid level for multiamr2cube (used only
                              for informational output here; not applied)
    Lbox        : array-like  domain size [Lx, Ly, Lz]  (from metadata_from_config)
    bx          : int         block size  (from metadata_from_config)
    extent      : list[float] [x0,x1,y0,y1,z0,z1] fractional sub-domain
    chunk_size  : int         cells per streaming chunk (tune for available RAM)
    verbose     : bool        print progress

    Returns
    -------
    cc     : (K, 3) float64  cell centres of surviving cells
    lcell  : (K,)   int64    adjusted cell levels (= raw level + log2(bx))
    fields : dict            {fieldname: (K,) float64}

    All arrays are ordered identically — row k of cc corresponds to
    lcell[k] and fields[f][k].  Ready to pass directly to any
    multiamr2cube variant.
    """
    Lbox = np.asarray(Lbox, dtype=np.float64)
    lbx  = int(round(np.log2(bx)))   # log2(bx), e.g. 2 for bx=4

    is_full_box = (
        extent[0] == 0 and extent[1] == 1 and
        extent[2] == 0 and extent[3] == 1 and
        extent[4] == 0 and extent[5] == 1
    )

    with h5py.File(h5path, 'r') as f:

        conn_ds  = f['connectivity']    # (N, 8) uint64
        coord_ds = f['coordinates']     # (M, 3) float64
        N        = int(conn_ds.shape[0])

        # level is small (~100 MB): read in full once
        level_raw = f['level'][:]       # (N,) uint32

        lmin_raw = int(level_raw.min())
        lmax_raw = int(level_raw.max())
        if verbose:
            print(f'Snapshot: {N:,} cells | '
                  f'raw levels {lmin_raw}-{lmax_raw} | '
                  f'lcell {lmin_raw+lbx}-{lmax_raw+lbx}')

        # spatial filter bounds in physical coordinates
        if not is_full_box:
            margin = Lbox / float(1 << lmin_raw)   # one coarsest-cell width
            x0 = extent[0] * Lbox[0] - margin[0];  x1 = extent[1] * Lbox[0] + margin[0]
            y0 = extent[2] * Lbox[1] - margin[1];  y1 = extent[3] * Lbox[1] + margin[1]
            z0 = extent[4] * Lbox[2] - margin[2];  z1 = extent[5] * Lbox[2] + margin[2]

        # ------------------------------------------------------------------
        # Pass 1: stream connectivity, compute cell centres, filter
        # ------------------------------------------------------------------
        cc_parts    = []
        level_parts = []
        all_ranges  = []

        n_chunks = (N + chunk_size - 1) // chunk_size
        if verbose:
            print(f'Pass 1: scanning {n_chunks} chunks of {chunk_size:,} cells ...')

        for chunk_idx, start in enumerate(range(0, N, chunk_size)):
            end = min(start + chunk_size, N)

            # read connectivity chunk: (C, 8) node indices
            conn_chunk = conn_ds[start:end]

            # compact coordinate read: only the node range referenced by
            # this chunk (contiguous due to Morton ordering of octants)
            node_min = int(conn_chunk.min())
            node_max = int(conn_chunk.max())
            coords_chunk = coord_ds[node_min : node_max + 1]   # (span, 3)

            # cell centres: average of 8 hex vertices
            local_ids = (conn_chunk - node_min).astype(np.intp)  # (C, 8)
            cc_chunk  = coords_chunk[local_ids].mean(axis=1)      # (C, 3)

            # spatial filter
            if is_full_box:
                mask = np.ones(end - start, dtype=np.bool_)
            else:
                mask = (
                    (cc_chunk[:, 0] >= x0) & (cc_chunk[:, 0] < x1) &
                    (cc_chunk[:, 1] >= y0) & (cc_chunk[:, 1] < y1) &
                    (cc_chunk[:, 2] >= z0) & (cc_chunk[:, 2] < z1)
                )

            n_kept = int(mask.sum())
            if n_kept > 0:
                cc_parts.append(cc_chunk[mask])
                level_parts.append(level_raw[start:end][mask])
                chunk_ranges = _index_ranges(mask, offset=start)
                all_ranges.extend(chunk_ranges)

            if verbose and (chunk_idx % max(1, n_chunks // 10) == 0 or
                            chunk_idx == n_chunks - 1):
                pct = (chunk_idx + 1) / n_chunks * 100
                print(f'  {pct:5.1f}%  chunk {chunk_idx+1}/{n_chunks}  '
                      f'kept so far: {sum(len(p) for p in cc_parts):,}')

        # merge index ranges that are adjacent across chunk boundaries
        all_ranges = _merge_ranges(all_ranges)

        if not cc_parts:
            raise ValueError('No cells found in the requested extent.')

        cc    = np.concatenate(cc_parts)
        lcell = (np.concatenate(level_parts).astype(np.int64) + lbx)

        total_kept = len(cc)
        if verbose:
            print(f'\nPass 1 done: {total_kept:,} cells kept '
                  f'({total_kept/N*100:.1f}% of total) '
                  f'in {len(all_ranges)} contiguous run(s)')

        # ------------------------------------------------------------------
        # Pass 2: read fields only for surviving index ranges
        # ------------------------------------------------------------------
        if verbose:
            print(f'Pass 2: reading {len(fieldnames)} field(s) '
                  f'across {len(all_ranges)} range(s) ...')

        fields = {}
        for fname in fieldnames:
            if fname not in f:
                raise KeyError(f'Field "{fname}" not found in {h5path}')
            ds     = f[fname]
            parts  = [ds[s:e] for s, e in all_ranges]
            fields[fname] = np.concatenate(parts).astype(np.float64)

        if verbose:
            print('Done.')

    return cc, lcell, fields
