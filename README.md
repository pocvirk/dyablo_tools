# dyablo_tools

Python tools for reading Dyablo AMR snapshots and projecting them onto regular Cartesian grids, optimised with numba.

## Main tools

- `load_subcube_h5py.py`: loads the cells of a sub-cube (centres, levels, fields) directly from a Dyablo HDF5 snapshot with h5py, without pyablo. It streams the file in chunks, so memory scales with the sub-cube rather than the snapshot.
- `multiamr2cube_numba_full.py`: projects AMR cells onto a regular `2^lcube` grid with numba kernels, for one or several fields at once.
- `demo_peak_zoom.ipynb`: pyablo-free example that finds the density peak, streams the sub-cube around it and projects `rho` and `rho_HII` at lcube = 8, 9, 10.

## Optimisation study

`project.md` describes the benchmark and its results. The other `multiamr2cube_*.py` files are the reference and intermediate versions. `benchmark.py`, `compare.ipynb`, `compare_subcube.ipynb` and `peak_density_subcube.ipynb` time and validate them.

## Setup

The notebooks and `benchmark.py` expect this repo to be checked out as `claude/` inside the directory that holds the snapshot (`*.h5`, `*.xmf` and the restart `.ini`):

    cd /path/to/snapshot_dir
    git clone git@github.com:pocvirk/dyablo_tools.git claude

Requirements: numpy, numba, h5py, matplotlib; pyablo for the reference and benchmark scripts.
