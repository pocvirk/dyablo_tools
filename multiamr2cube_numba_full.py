"""
Fully-numba single-pass implementation of MultiAmr2Cube.

Supports projecting multiple fields simultaneously: index computation is
performed once per cell and shared across all fields, so N fields cost
only marginally more than 1 field (extra inner loop of size Nfields).

Fine-cell averaging
-------------------
When lcell > lcube, multiple fine cells map to the same output grid cell.
Each fine cell at level l occupies a volume fraction 1/zoom^3 of the output
cell, where zoom = 2^(l - lcube).  The output value is the volume-weighted
average:

    field[g] = sum_i( values[i] * vol_frac_i ) / sum_i( vol_frac_i )

where vol_frac_i = 1 / zoom_i^3.  In a complete AMR mesh the fractions sum
to 1.0 by construction (partition of space), but the code normalises by the
accumulated weight regardless to handle sub-cube boundaries gracefully.

Coarse cells (lcell < lcube) are non-overlapping in the output grid by AMR
construction, so they are written directly (no averaging needed) and their
contribution to the weight array is set to 1.

Two-pass design
---------------
  Pass 1 — sequential over fine cells (l >= lcube)
    Accumulates weighted values and per-cell weights.  Must be sequential
    because multiple fine cells map to the same output cell.

  Pass 2 — parallel over coarse cells (l < lcube)
    Fills zoom^3 sub-cubes.  Safe with prange because coarse cells are
    non-overlapping in the output grid.

  Post-process — divide field by weight (numpy, outside numba)
"""

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# JIT kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _fine_sequential(
    cc, lcell, values,
    inv_Lx, inv_Ly, inv_Lz,
    lcube, nx, ny, nz,
    xmin, ymin, zmin,
    field, weight,
):
    """
    Sequential pass over fine cells (lcell >= lcube).
    Accumulates volume-weighted values into field and total weight into weight.

    values : (Nc, Nf)
    field  : (nx*ny*nz, Nf)   — accumulated weighted sum
    weight : (nx*ny*nz,)      — accumulated volume fractions
    """
    nxy = nx * ny
    Nc  = lcell.shape[0]
    Nf  = values.shape[1]

    for p in range(Nc):
        l = lcell[p]
        if l < lcube:
            continue

        two_l = np.int64(1) << l
        i_own = np.int64(cc[p, 0] * two_l * inv_Lx)
        j_own = np.int64(cc[p, 1] * two_l * inv_Ly)
        k_own = np.int64(cc[p, 2] * two_l * inv_Lz)

        shift = l - lcube
        idxi = (i_own >> shift) - xmin
        idxj = (j_own >> shift) - ymin
        idxk = (k_own >> shift) - zmin

        if 0 <= idxi < nx and 0 <= idxj < ny and 0 <= idxk < nz:
            g = idxi + idxj * nx + idxk * nxy
            # volume fraction of this fine cell relative to the output cell
            zoom = np.int64(1) << shift          # 2^shift
            vol_frac = 1.0 / float(zoom * zoom * zoom)
            weight[g] += vol_frac
            for f in range(Nf):
                field[g, f] += values[p, f] * vol_frac


@njit(parallel=True, cache=True)
def _coarse_parallel(
    cc, lcell, values,
    inv_Lx, inv_Ly, inv_Lz,
    lcube, nx, ny, nz,
    xmin, ymin, zmin,
    field, weight,
):
    """
    Parallel pass over coarse cells (lcell < lcube).
    Each coarse cell fills zoom^3 output cells.  Non-overlapping by AMR
    construction → prange is safe.

    values : (Nc, Nf)
    field  : (nx*ny*nz, Nf)   — direct assignment (no accumulation needed)
    weight : (nx*ny*nz,)      — set to 1.0 for every written cell
    """
    nxy = nx * ny
    Nc  = lcell.shape[0]
    Nf  = values.shape[1]

    for p in prange(Nc):
        l = lcell[p]
        if l >= lcube:
            continue

        two_l = np.int64(1) << l
        i_own = np.int64(cc[p, 0] * two_l * inv_Lx)
        j_own = np.int64(cc[p, 1] * two_l * inv_Ly)
        k_own = np.int64(cc[p, 2] * two_l * inv_Lz)

        shift = lcube - l
        zoom  = np.int64(1) << shift
        idxi  = (i_own << shift) - xmin
        idxj  = (j_own << shift) - ymin
        idxk  = (k_own << shift) - zmin

        for kz in range(zoom):
            for jz in range(zoom):
                for iz in range(zoom):
                    ii = idxi + iz
                    jj = idxj + jz
                    kk = idxk + kz
                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                        g = ii + jj * nx + kk * nxy
                        weight[g] = 1.0
                        for f in range(Nf):
                            field[g, f] = values[p, f]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def multiamr2cube(cc, lcell, values, Lbox, lcube, extent=[0, 1, 0, 1, 0, 1]):
    """
    Project AMR data onto a regular Cartesian grid.

    Fine cells (lcell > lcube) contribute a volume-weighted average to each
    output grid cell.  Coarse cells (lcell < lcube) fill their sub-cube
    uniformly.

    Parameters
    ----------
    cc     : (Nc, 3) float64   cell centres
    lcell  : (Nc,)   int       cell levels
    values : (Nc,) or (Nc, Nfields) float64
        Single field or multiple fields.  Index computation is shared across
        all fields — projecting N fields costs little more than projecting 1.
    Lbox   : sequence of 3 floats   physical box size
    lcube  : int                    target grid level (grid is 2^lcube cells/side)
    extent : [x0,x1, y0,y1, z0,z1] fractional sub-box in [0,1]^3

    Returns
    -------
    (nx, ny, nz)          for single-field input
    (nx, ny, nz, Nfields) for multi-field input
    """
    single = (values.ndim == 1)
    if single:
        values = values[:, np.newaxis]   # (Nc, 1)

    ncube = 2 ** lcube

    nx = np.int64(round((extent[1] - extent[0]) * ncube))
    ny = np.int64(round((extent[3] - extent[2]) * ncube))
    nz = np.int64(round((extent[5] - extent[4]) * ncube))

    xmin = np.int64(round(extent[0] * ncube))
    ymin = np.int64(round(extent[2] * ncube))
    zmin = np.int64(round(extent[4] * ncube))

    inv_Lx = np.float64(1.0 / Lbox[0])
    inv_Ly = np.float64(1.0 / Lbox[1])
    inv_Lz = np.float64(1.0 / Lbox[2])

    Nf = values.shape[1]
    field  = np.zeros((nx * ny * nz, Nf), dtype=np.float64)
    weight = np.zeros(nx * ny * nz,       dtype=np.float64)

    lcell_i64  = lcell.astype(np.int64)
    values_f64 = np.ascontiguousarray(values, dtype=np.float64)

    has_coarse = int(lcell_i64.min()) < lcube

    # Pass 1: fine cells — sequential (accumulate weighted values)
    _fine_sequential(
        cc, lcell_i64, values_f64,
        inv_Lx, inv_Ly, inv_Lz,
        np.int64(lcube), nx, ny, nz, xmin, ymin, zmin,
        field, weight,
    )

    # Pass 2: coarse cells — parallel (direct fill, non-overlapping)
    if has_coarse:
        _coarse_parallel(
            cc, lcell_i64, values_f64,
            inv_Lx, inv_Ly, inv_Lz,
            np.int64(lcube), nx, ny, nz, xmin, ymin, zmin,
            field, weight,
        )

    # Normalise: divide accumulated weighted sum by total weight.
    # For coarse cells weight==1, so this is a no-op for them.
    # For fine cells this computes the volume-weighted average.
    covered = weight > 0
    field[covered] /= weight[covered, np.newaxis]

    # Reshape: spatial flat index g = ix + iy*nx + iz*nxy matches Fortran order.
    if single:
        return np.reshape(field[:, 0], (nx, ny, nz), order='F')
    else:
        return np.stack(
            [np.reshape(field[:, f], (nx, ny, nz), order='F') for f in range(Nf)],
            axis=-1,
        )


def cell_coords(lcube, extent=[0, 1, 0, 1, 0, 1]):
    """
    Return the cell-centre coordinates of the output grid in fractional [0, 1] box units.

    Uses identical arithmetic to multiamr2cube so the coordinates are
    guaranteed to be consistent with the projected field array.

    Parameters
    ----------
    lcube  : int    target grid level (same value passed to multiamr2cube)
    extent : [x0,x1, y0,y1, z0,z1]  fractional sub-box (same value passed to multiamr2cube)

    Returns
    -------
    x : (nx,) float64   cell-centre x coordinates in [0, 1]
    y : (ny,) float64   cell-centre y coordinates in [0, 1]
    z : (nz,) float64   cell-centre z coordinates in [0, 1]

    Example
    -------
    cube = multiamr2cube(cc, lcell, rho, Lbox, lcube=9, extent=EXTENT)
    x, y, z = cell_coords(lcube=9, extent=EXTENT)
    # cube[i, j, k] is centred at (x[i], y[j], z[k]) in [0,1]^3
    # To locate a particle at fractional position p in [0,1]^3:
    #   i = np.searchsorted(x, p[0]) - 1   (or np.digitize)
    """
    ncube = 2 ** lcube
    nx = int(round((extent[1] - extent[0]) * ncube))
    ny = int(round((extent[3] - extent[2]) * ncube))
    nz = int(round((extent[5] - extent[4]) * ncube))
    xmin = int(round(extent[0] * ncube))
    ymin = int(round(extent[2] * ncube))
    zmin = int(round(extent[4] * ncube))
    x = (xmin + np.arange(nx) + 0.5) / ncube
    y = (ymin + np.arange(ny) + 0.5) / ncube
    z = (zmin + np.arange(nz) + 0.5) / ncube
    return x, y, z


def warmup(cc, lcell, values, Lbox):
    """
    Trigger JIT compilation before any timed run.

    Pass a representative `values` array (1-D or 2-D) so that numba compiles
    the right specialisation.  If you plan to use multi-field projection, call
    warmup with a (Nc, Nfields) array; otherwise a 1-D array is sufficient.
    """
    cc_s = cc[:100].copy()
    lc_s = lcell[:100].copy()
    v_s  = values[:100].astype(np.float64).copy()
    multiamr2cube(cc_s, lc_s, v_s, Lbox, lcube=int(lcell.min()))
