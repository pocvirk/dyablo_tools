# fast_read_amr

Reads Dyablo AMR snapshots straight from HDF5 and projects them onto regular Cartesian grids, without pyablo.

- `load_subcube_h5py.py`: loads the cells inside a sub-cube (centres, levels, fields). It streams the file in chunks, so memory scales with the sub-cube rather than the snapshot.
- `multiamr2cube_numba_full.py`: projects the cells onto a regular `2^lcube` grid with numba, for one or several fields at once.
- `demo_peak_zoom.ipynb`: worked example. It finds the density peak, loads the sub-cube around it and projects `rho` and `rho_HII` at three resolutions.

## Usage

```python
import numpy as np
from load_subcube_h5py import load_subcube, metadata_from_config
import multiamr2cube_numba_full as m

extent = [0, 0.5, 0, 0.5, 0, 0.5]   # fractional sub-box [x0, x1, y0, y1, z0, z1]

Lbox, bx = metadata_from_config('restart_star_279982.ini')
cc, lcell, fields = load_subcube('star_iter0070000.h5', fieldnames=['rho', 'rho_HII'],
                                 lcube=9, Lbox=Lbox, bx=bx, extent=extent)

values = np.stack([fields['rho'], fields['rho_HII']], axis=-1)   # (Ncells, Nfields)
cube = m.multiamr2cube(cc, lcell, values, Lbox, lcube=9, extent=extent)
# cube.shape == (256, 256, 256, 2); pass a 1-D values array to get a single (nx, ny, nz) field
```

To run the demo, open it from this directory and set `SNAPDIR` in the first cell to the directory holding your snapshot and restart `.ini`.

Requirements: numpy, numba, h5py; matplotlib for the demo.
