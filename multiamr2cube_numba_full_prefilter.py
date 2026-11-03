"""
Fully-numba single-pass implementation + spatial pre-filter.

Combines the two optimisations:
  1. Pre-filter: reduce input from N cells to ~f*N before the kernel
  2. numba_full kernel: single-pass index compute + fill with no temp arrays

For the full-box extent [0,1,0,1,0,1] the pre-filter is skipped.
See multiamr2cube_ref_prefilter.py for pre-filter details.
See multiamr2cube_numba_full.py for kernel details.
"""

import numpy as np
from multiamr2cube_ref_prefilter import prefilter
from multiamr2cube_numba_full import multiamr2cube as _multiamr2cube, warmup as _warmup


def multiamr2cube(cc, lcell, values, Lbox, lcube, extent=[0, 1, 0, 1, 0, 1]):
    """numba_full + pre-filter.  Same interface as multiamr2cube_ref."""
    is_full_box = (extent[0] == 0 and extent[1] == 1 and
                   extent[2] == 0 and extent[3] == 1 and
                   extent[4] == 0 and extent[5] == 1)
    if not is_full_box:
        cc, lcell, values = prefilter(cc, lcell, values, Lbox, extent)
    return _multiamr2cube(cc, lcell, values, Lbox, lcube, extent)


def warmup(cc, lcell, values, Lbox):
    _warmup(cc, lcell, values, Lbox)
