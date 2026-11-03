# dyablo_tools

Tools for working with Dyablo simulation outputs. Each tool lives in its own directory.

| Tool | What it does |
|---|---|
| [fast_read_amr](fast_read_amr/) | Reads a sub-cube of a Dyablo HDF5 snapshot without pyablo and projects the AMR cells onto a regular grid with numba. |
| [dashboard_dyablo_reio](dashboard_dyablo_reio/) | Reionization + galaxy-population dashboard for dyablo cosmo RT runs (x_HI/x_HII/Gamma/tau, UV LF, SFRD, SMHM, fesc), built on dashboard_rc and fof_dyablo. |
| [dyablo_fof](dyablo_fof/) | Parallel Friends-of-Friends halo finder for dyablo particle snapshots (RAMSES pFoF, Roy, Bouillot & Rasera 2014), MPI + HDF5 Fortran. |
