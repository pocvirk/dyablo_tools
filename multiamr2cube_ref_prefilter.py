"""
Reference implementation + spatial pre-filter.

The pre-filter discards cells whose physical extent cannot overlap the
requested sub-cube before any index computation.  This reduces the number
of cells passed to the kernel from N (all cells) to ~f*N where f is the
volume fraction of the sub-cube.

A conservative margin of one coarsest-level cell width is added on each
face of the requested extent to guarantee that coarse cells straddling the
boundary are not wrongly discarded.

For the full-box extent [0,1,0,1,0,1] the pre-filter is skipped (no work,
no extra allocation).
"""

import numpy as np
from multiamr2cube_ref import multiamr2cube as _multiamr2cube


def prefilter(cc, lcell, values, Lbox, extent):
    """
    Spatial pre-filter: keep only cells that can contribute to extent.

    Args:
        cc, lcell, values : full snapshot arrays
        Lbox              : [Lx, Ly, Lz]
        extent            : [x0,x1,y0,y1,z0,z1] in fractional coords

    Returns:
        filtered (cc, lcell, values) — views/copies of the input
    """
    lmin = int(lcell.min())
    # one full coarsest-cell width as guard band (conservative)
    mx = Lbox[0] / (1 << lmin)
    my = Lbox[1] / (1 << lmin)
    mz = Lbox[2] / (1 << lmin)

    mask = (
        (cc[:, 0] >= extent[0] * Lbox[0] - mx) & (cc[:, 0] < extent[1] * Lbox[0] + mx) &
        (cc[:, 1] >= extent[2] * Lbox[1] - my) & (cc[:, 1] < extent[3] * Lbox[1] + my) &
        (cc[:, 2] >= extent[4] * Lbox[2] - mz) & (cc[:, 2] < extent[5] * Lbox[2] + mz)
    )
    return cc[mask], lcell[mask], values[mask]


def multiamr2cube(cc, lcell, values, Lbox, lcube, extent=[0, 1, 0, 1, 0, 1]):
    """Reference + pre-filter.  Same interface as multiamr2cube_ref."""
    is_full_box = (extent[0] == 0 and extent[1] == 1 and
                   extent[2] == 0 and extent[3] == 1 and
                   extent[4] == 0 and extent[5] == 1)
    if not is_full_box:
        cc, lcell, values = prefilter(cc, lcell, values, Lbox, extent)
    return _multiamr2cube(cc, lcell, values, Lbox, lcube, extent)
