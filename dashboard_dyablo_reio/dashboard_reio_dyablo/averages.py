"""IGM averages of a dyablo run, in the form the ramses-cudaton dashboard (rc/dashboard_tools_averages.py) plots.

One row per grid snapshot, cached in <out>/averages/iterNNNNNNN.csv so that re-running on a growing run only
reads the new snapshots. Columns:
  aexp z xHII_V xHII_M nH xnH Gamma_V Gamma_ion f_ion S T
  - xHII_V / xHII_M: volume / mass weighted x_HII = rho_HII/rho
  - nH, xnH: volume averages of n_H and x_HII n_H, physical cm^-3 (dyablo cosmo_reio is H only: n_H = rho/m_p)
  - Gamma_V: volume average of Gamma_HI = sigma_N c_rad n_gamma over all cells (s^-1, physical, reduced c_rad as in the code)
  - Gamma_ion: same, over ionized cells only (x_HII > 0.5) -- what Lya-forest measurements probe
  - S: intrinsic ionizing photon rate of the box (photons/s) = spawn_rate x mass of stars younger than t_life (src field)
  - T: volume-weighted temperature (K) when the snapshot has the fields for it, else NaN
"""
import os

import numpy as np

from .run import MP, MPC

COLUMNS = ["iter", "aexp", "z", "xHII_V", "xHII_M", "nH", "xnH", "Gamma_V", "Gamma_ion", "f_ion", "S", "T"]


def snapshot_averages(run, grid_file):
    a = run.snapshot_aexp(grid_file)
    d = run.read_grid(grid_file, ["rho", "rho_HII", "e_rad", "src"])
    v = d["vol"]
    x = d["rho_HII"] / d["rho"]
    vt = v.sum()
    m = d["rho"] * v
    # rho is supercomoving density (code kg/m3 x length^-3 scaling): physical = code / a^3
    nh = d["rho"] / MP / a**3 * 1e-6                       # cm^-3, physical
    gam = run.sigma_n * run.c_rad * d["e_rad"] / a**3      # e_rad: photons per m^3 (supercomoving) -> s^-1
    ion = x > 0.5
    s = np.nan
    if "src" in d:
        mstar_young = np.sum(d["src"] * v) * (MPC / run.l_unit)**3 * run.m_unit / 1.98847e30   # Msun
        s = run.spawn_rate * mstar_young
    return dict(aexp=a, z=1 / a - 1,
                xHII_V=np.sum(x * v) / vt, xHII_M=np.sum(x * m) / m.sum(),
                nH=np.sum(nh * v) / vt, xnH=np.sum(x * nh * v) / vt,
                Gamma_V=np.sum(gam * v) / vt,
                Gamma_ion=np.sum(gam[ion] * v[ion]) / v[ion].sum() if ion.any() else 0.0,
                f_ion=v[ion].sum() / vt, S=s, T=np.nan)


def _averages_row(run, it, grid_file):
    r = snapshot_averages(run, grid_file)
    r["iter"] = it
    return [r[c] for c in COLUMNS]


def compute_averages(run, outdir, stride=1, nproc=1, verbose=True):
    """averages table (structured numpy array), reusing the per-snapshot cache.
    The pass is I/O bound (~13 s per 50M-cell snapshot on one Lustre OST): nproc > 1 reads snapshots in parallel."""
    cache = os.path.join(outdir, "averages")
    os.makedirs(cache, exist_ok=True)
    snaps = run.snapshots()[::stride]
    cached = lambda it, g: os.path.join(cache, f"iter{it:07d}.csv")  # noqa: E731
    todo = [(it, g) for it, g, _ in snaps
            if not (os.path.exists(cached(it, g)) and os.path.getmtime(cached(it, g)) > os.path.getmtime(g))]
    if todo:
        if nproc > 1:
            from multiprocessing import Pool
            with Pool(min(nproc, len(todo))) as pool:
                rows = pool.starmap(_averages_row, [(run, it, g) for it, g in todo])
        else:
            rows = [_averages_row(run, it, g) for it, g in todo]
        for (it, g), row in zip(todo, rows):
            np.savetxt(cached(it, g), [row], delimiter=",", header=",".join(COLUMNS), comments="")
            if verbose:
                print(f"  averages iter {it:7d} z={row[2]:6.2f} xHII_V={row[3]:.4e} Gamma_ion={row[8]:.3e}", flush=True)
    rows = []
    for it, g, _ in snaps:
        r = np.genfromtxt(cached(it, g), delimiter=",", names=True)
        rows.append([float(r[c]) for c in COLUMNS])
    tab = np.array(rows)
    out = np.zeros(len(tab), dtype=[(c, float) for c in COLUMNS])
    for i, c in enumerate(COLUMNS):
        out[c] = tab[:, i]
    np.savetxt(os.path.join(outdir, "averages.txt"), tab, header=" ".join(COLUMNS), fmt="%.6e")
    return out


def plot_averages(av, plotdir, simname):
    """the ramses-cudaton dashboard IGM plots, fed with dyablo averages."""
    from .rc import dashboard_tools_averages as dbt
    z = av["z"]
    # dbt.plotgamma/plotJ21 take a volume-averaged J21 and plot Gamma = J*K/xHII, K = 6.38e22*3e8*2.493e-22.
    # Feed J = Gamma_ion * xHII / K, so that the plotted curve is Gamma measured in ionized cells (x_HII > 0.5).
    K = 6.38e22 * 3.e8 * 2.493e-22
    J = av["Gamma_ion"] * av["xHII_V"] / K
    dbt.plotxhi(z, 1 - av["xHII_V"], dir=plotdir, simname=simname)
    dbt.plotxhii(z, av["xHII_V"], dir=plotdir, simname=simname)
    dbt.plotJ21(z, J, av["xHII_V"], dir=plotdir, simname=simname)
    dbt.plotgamma(z, J, av["xHII_V"], dir=plotdir, simname=simname)
    if np.isfinite(av["T"]).any():
        dbt.plotTemp(z, av["T"], dir=plotdir, simname=simname)
    # dbt.plottaues divides xnH by 0.76/0.82 to add HeII electrons to an H-only n_e. dyablo's n_H = rho/m_p counts
    # all baryons as H, so pass the hydrogen part 0.76 x n_H x_HII: n_e = 0.82 rho x_HII / m_p (H + HeII).
    dbt.plottaues(z, 0.76 * av["xnH"], dir=plotdir, simname=simname)
