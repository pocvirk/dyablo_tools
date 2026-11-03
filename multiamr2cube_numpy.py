"""
Vectorized numpy implementation of MultiAmr2Cube.

Optimization: replaces the Python loop over coarse cells with a
per-level batched numpy assignment using broadcasting.
Fine-cell path is identical to the reference.
"""

import numpy as np


def multiamr2cube(cc, lcell, values, Lbox, lcube, extent=[0, 1, 0, 1, 0, 1]):
    """Same interface as multiamr2cube_ref.multiamr2cube."""
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

    # --- coarse cells (level < lcube): partial-overlap filter, batched per level ---
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
            lc   = lcell[w_coarse]
            xi_c = idxi[w_coarse]
            xj_c = idxj[w_coarse]
            xk_c = idxk[w_coarse]
            vals = values[w_coarse].astype(np.float64)

            for l in np.unique(lc):
                mask = lc == l
                zoom = np.int64(1) << np.int64(lcube - l)
                xi = xi_c[mask]; yj = xj_c[mask]; zk = xk_c[mask]
                v  = vals[mask]

                i1d = np.arange(zoom, dtype=np.int64)
                iz_f, jz_f, kz_f = [a.ravel() for a in np.meshgrid(i1d, i1d, i1d)]

                # (M, zoom^3) valid mask — full bounds check on both sides
                valid = (
                    (xi[:, None] + iz_f[None, :] >= 0) &
                    (xi[:, None] + iz_f[None, :] <  nx) &
                    (yj[:, None] + jz_f[None, :] >= 0) &
                    (yj[:, None] + jz_f[None, :] <  ny) &
                    (zk[:, None] + kz_f[None, :] >= 0) &
                    (zk[:, None] + kz_f[None, :] <  nz)
                )
                ii      = (xi[:, None] + iz_f[None, :])[valid]
                jj      = (yj[:, None] + jz_f[None, :])[valid]
                kk      = (zk[:, None] + kz_f[None, :])[valid]
                all_val = np.repeat(v, zoom ** 3)[valid.ravel()]
                field[ii + jj * nx + kk * nxy] = all_val

    return np.reshape(field, (nx, ny, nz), order='F')
