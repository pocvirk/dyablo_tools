"""
Reference single-field implementation of MultiAmr2Cube.
Logic matches the notebook exactly; only single-field and I/O-free.
"""

import numpy as np


def multiamr2cube(cc, lcell, values, Lbox, lcube, extent=[0, 1, 0, 1, 0, 1]):
    """
    Project AMR field onto a regular Cartesian grid at cell level lcube.

    Args:
        cc      : (N, 3) float64  – cell centres
        lcell   : (N,)   int64   – adjusted cell levels (level + log2(bx))
        values  : (N,)   float64 – field values
        Lbox    : (3,)           – domain bounding box sizes [Lx, Ly, Lz]
        lcube   : int            – target grid level (grid size = 2**lcube per dim)
        extent  : [x0,x1,y0,y1,z0,z1] – fractional sub-domain, default full box

    Returns:
        (nx, ny, nz) float64 array
    """
    Nc = len(values)
    ncube = 2 ** lcube

    dimx = extent[1] - extent[0]
    dimy = extent[3] - extent[2]
    dimz = extent[5] - extent[4]

    nx = np.int64(round(dimx * ncube))
    ny = np.int64(round(dimy * ncube))
    nz = np.int64(round(dimz * ncube))

    xmin = np.int64(round(extent[0] * ncube))
    ymin = np.int64(round(extent[2] * ncube))
    zmin = np.int64(round(extent[4] * ncube))

    # cell indices at each cell's own level
    i = np.int64(cc[:, 0] / (Lbox[0] / 2 ** lcell))
    j = np.int64(cc[:, 1] / (Lbox[1] / 2 ** lcell))
    k = np.int64(cc[:, 2] / (Lbox[2] / 2 ** lcell))

    # project to lcube level
    idxi = np.int64(i / 2.0 ** (lcell - lcube)) - xmin
    idxj = np.int64(j / 2.0 ** (lcell - lcube)) - ymin
    idxk = np.int64(k / 2.0 ** (lcell - lcube)) - zmin

    field = np.zeros(nx * ny * nz, dtype=np.float64)
    nxy = nx * ny
    is_fine = lcell >= lcube

    # --- fine cells (level >= lcube): strict in-bounds filter, vectorized ---
    w_fine = np.where(
        is_fine &
        (idxi >= 0) & (idxi < nx) &
        (idxj >= 0) & (idxj < ny) &
        (idxk >= 0) & (idxk < nz)
    )[0]
    field[idxi[w_fine] + idxj[w_fine] * nx + idxk[w_fine] * nxy] = values[w_fine].astype(np.float64)

    # --- coarse cells (level < lcube): partial-overlap filter, Python loop ---
    # A coarse cell at level l occupies zoom^3 output cells.  Its top-left
    # corner (idxi) can be negative and still have sub-cells inside the grid.
    if not is_fine.all():
        shift    = np.maximum(np.int64(0), np.int64(lcube) - lcell.astype(np.int64))
        zoom_arr = np.int64(1) << shift
        w_coarse = np.where(
            (~is_fine) &
            (idxi + zoom_arr > 0) & (idxi < nx) &
            (idxj + zoom_arr > 0) & (idxj < ny) &
            (idxk + zoom_arr > 0) & (idxk < nz)
        )[0]
        if len(w_coarse) > 0:
            lc_c  = lcell[w_coarse]
            xi_c  = idxi[w_coarse]
            xj_c  = idxj[w_coarse]
            xk_c  = idxk[w_coarse]
            v_c   = values[w_coarse].astype(np.float64)
            order = np.argsort(lc_c)
            lc_c  = lc_c[order]
            xi_c  = xi_c[order]; xj_c = xj_c[order]; xk_c = xk_c[order]
            v_c   = v_c[order]

            l = -1
            for p in range(len(v_c)):
                if lc_c[p] != l:
                    l = lc_c[p]
                    zoom = 2 ** (lcube - l)
                    i1d = np.arange(zoom)
                    iz_f, jz_f, kz_f = [a.ravel() for a in np.meshgrid(i1d, i1d, i1d)]
                xi = int(xi_c[p]); yj = int(xj_c[p]); zk = int(xk_c[p])
                valid = (
                    (xi + iz_f >= 0) & (xi + iz_f < nx) &
                    (yj + jz_f >= 0) & (yj + jz_f < ny) &
                    (zk + kz_f >= 0) & (zk + kz_f < nz)
                )
                ii = xi + iz_f[valid]
                jj = yj + jz_f[valid]
                kk = zk + kz_f[valid]
                field[ii + jj * nx + kk * nxy] = v_c[p]

    return np.reshape(field, (nx, ny, nz), order='F')


def extract_snapshot_data(snap, config):
    """
    Extract raw numpy arrays from a Dyablo snapshot.
    Call this once; pass the result to multiamr2cube (no I/O in timing).

    Returns: cc, lcell, Lbox
        cc    : (N, 3) float64 cell centres
        lcell : (N,)   int64  adjusted cell levels
        Lbox  : (3,)   float64 domain sizes
    """
    Nc = snap.getNCells()
    bx = config.getint('amr', 'bx')
    Lbox = np.array(snap.getDomainBoundingBox()[1])
    cc = np.array(snap.getCellsCenter(np.arange(Nc)))
    lcell = np.int64(np.array(snap.getLevel(np.arange(Nc))) + np.log2(bx))
    return cc, lcell, Lbox
