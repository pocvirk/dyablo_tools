"""
Numba JIT implementation of MultiAmr2Cube.

Optimization: the coarse-cell fill loop is compiled with @njit(parallel=True),
replacing the Python loop. Fine-cell path stays vectorized numpy.
The first call includes compilation; subsequent calls use the cached kernel.
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _fill_coarse(field, idxi_c, idxj_c, idxk_c, lcells, values, lcube, nx, ny, nxy):
    """
    Fill coarse cells into a flat output array using 3D origin indices.

    Each coarse cell at level l fills a zoom^3 = 2^(3*(lcube-l)) sub-cube.
    Per-element full bounds check handles cells that partially overlap the grid
    on either the low or high side of any axis.
    """
    n = len(idxi_c)
    nz = len(field) // nxy
    for p in prange(n):
        l = lcells[p]
        zoom = np.int64(1) << np.int64(lcube - l)
        xi = idxi_c[p]; yj = idxj_c[p]; zk = idxk_c[p]
        val = values[p]
        for kz in range(zoom):
            for jz in range(zoom):
                for iz in range(zoom):
                    ii = xi + iz; jj = yj + jz; kk = zk + kz
                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                        field[ii + jj * nx + kk * nxy] = val


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

    # --- coarse cells (level < lcube): partial-overlap filter, numba parallel ---
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
            _fill_coarse(
                field,
                idxi[w_coarse].astype(np.int64),
                idxj[w_coarse].astype(np.int64),
                idxk[w_coarse].astype(np.int64),
                lcell[w_coarse].astype(np.int64),
                values[w_coarse].astype(np.float64),
                np.int64(lcube),
                nx, ny, nxy,
            )

    return np.reshape(field, (nx, ny, nz), order='F')


def warmup(cc, lcell, values, Lbox):
    """
    Trigger JIT compilation on small synthetic data so the first timed call
    does not include compilation overhead.
    """
    cc_s = cc[:10].copy()
    lc_s = lcell[:10].copy()
    v_s = values[:10].astype(np.float64).copy()
    multiamr2cube(cc_s, lc_s, v_s, Lbox, lcube=int(lcell.min()))
