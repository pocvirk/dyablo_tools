# dashboard_dyablo_reio

Part of [dyablo_tools](https://github.com/pocvirk/dyablo_tools).

Standard reionization + galaxy-population diagnostics for **dyablo** cosmological RT runs, so every run of a
calibration campaign is analysed the same way with one command.

It reuses PO's ramses-cudaton dashboard ([dashboard_rc](https://github.com/pocvirk/dashboard_rc) @6d58002:
plotting functions, BPASS v2.2.1 tables, observational data in `aux/`) and adds a dyablo reader layer in front of it.
Halos come from **fof_dyablo** ([`../dyablo_fof`](../dyablo_fof/), Roy & Rasera pFoF adapted to dyablo).

## Usage

```bash
python -I run_dashboard.py <rundir> [--fof-exe /path/to/fof] [--snap ITER ...] [--stride N] [--out DIR] [--name NAME]
# or on Frontier, as a job:
sbatch run_dashboard.sl <rundir> [same options]
```

`<rundir>` is the dyablo run directory. All parameters (box, cosmology, units, c_rad, photon yield, t_life,
eta_SN, t_SN, T_bb) are read from the run's own `last.ini`, which dyablo writes there fully resolved in code units.
Snapshots are found from `[output] outputprefix`; their redshift is computed from the snapshot time with dyablo's own
cosmology (no job logs needed). Output in `<rundir>/dashboard/` (or `--out`):

| file | content |
|---|---|
| `PDFs/xHI, xHII, J21, gamma, tau_planck` | IGM history vs observations (dashboard_rc `plotallaverages` set) |
| `PDFs/UVLF_z=…, CSFRD, Halo_SFR_M_z_z=…` | UV LF (BPASS M_AB1600), cosmic SFRD, halo SFH vs mass (dashboard_rc) |
| `PDFs/smhm_fesc_z=…` | stellar mass – halo mass, ionizing escape fraction through R200 (new) |
| `averages.txt`, `averages/iterN.csv` | per-snapshot IGM averages (cached: re-runs only read new snapshots) |
| `halos_iterN.txt` | per-halo catalogue: position, Mh, R200, M*, M_AB1600, intrinsic photon rate, fesc |
| `fof/iterN/` | fof_dyablo run (fof.in, log, catalogue); reused if present |

Every pdf also gets a png copy.

## Definitions (dyablo cosmo_reio, H only, grey M1)

- x_HII = rho_HII / rho; averages are volume weighted (mass weighted also in `averages.txt`).
- Gamma_HI = sigma_N c_rad n_gamma with the **reduced** c_rad the code uses; sigma_N is the black-body
  number-averaged cross-section (as dyablo `computeSigma`). The `gamma.pdf` curve is the mean over ionized cells
  (x_HII > 0.5), which is what Lyα-forest measurements probe (fed to dashboard_rc's `plotgamma` through J21).
- tau_es: n_e = 0.82 rho x_HII / m_p (H + HeII), via dashboard_rc's `plottaues`.
- Stars: `birth_time < 0` (it is the physical birth time). dyablo removes eta_SN M* at t_SN, so the initial mass is
  restored only for stars older than t_SN (dashboard_rc divided every star by 0.8). Metallicity is not tracked:
  `--star-Z` (default 0.05 Zsun) sets the BPASS metallicity.
- Halo mass: FoF particle count x Om rho_c L³ / N (dashboard_rc convention); stars within R200 (DM-density based).
- fesc(R200) = outgoing photon flux through the shell [R200, R200 + max(2 dx, R200/4)] / (spawn_rate x mass of stars
  younger than t_life inside R200) -- the run's own emission model. Needs `fx_rad,fy_rad,fz_rad` in `write_variables`.

Recommended dyablo output settings for runs analysed with this dashboard:
```ini
[run]
output_slice_var = aexp
output_timeslice = 0.005
[output]
write_variables = rho,rho_HII,e_rad,fx_rad,fy_rad,fz_rad,src,zreion,level
write_particle_variables = particles/vx,particles/vy,particles/vz,particles/mass,particles/birth_time
```

## Install

Python ≥3.10 with `numpy scipy h5py matplotlib astropy pandas`. Build fof_dyablo once in `../dyablo_fof` (see its README for Frontier flags) and pass it with `--fof-exe` or `$FOF_DYABLO`.

## Changes to the dashboard_rc modules (`dashboard_reio_dyablo/rc/`)

Kept as close to the originals as possible; only:
cosmology settable (`set_cosmology`, was CoDa II hard-coded), `aux/` path relative to the package,
`interp2d` replaced (removed in scipy 1.14), `n_jobs` → `workers` (scipy), `np.histogram` tuple division split
(numpy 2 / astropy), CSFRD log of a dimensionless quantity, `computeallmags(initial_mass=)` option,
a negative error bar (Ouchi 2010) made positive.
