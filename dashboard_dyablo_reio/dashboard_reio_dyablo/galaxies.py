"""Galaxy populations of a dyablo snapshot with the ramses-cudaton dashboard tools (rc/dashboard_tools_galaxies.py).

Halos: fof_dyablo (Roy & Rasera pFoF adapted to dyablo, github.com/pocvirk/halo_finder_dyablo), run here on the
snapshot particle file if its catalogue is not there yet. Stars are attached to halos within R200 (dbtg.associate_star_halo2),
magnitudes come from the BPASS v2.2.1 tables (dbtg.computeallmags), and the LF / CSFRD / halo SFH plots are dbtg's.
dyablo-specific additions: stellar mass - halo mass relation, and the ionizing escape fraction through the R200 sphere
(needs the fx_rad/fy_rad/fz_rad fields in the snapshot).
"""
import os
import subprocess

import numpy as np
import astropy.units as u
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from .run import MPC, MYR
from .rc import dashboard_tools_galaxies as dbtg

FOF_IN = """{root}
DY
{pathinput}
{namepart}
{nameinfo}
0 !grpsize (RAMSES legacy, set to 0 for dyablo)
{b} !percolation parameter
{nmin} !min mass (in particles)
100000000 !max mass (in particles)
.true. !star ? (snapshot may contain star particles to filter out)
.false. !metal ?
.false. !outcube
.true. !dofof
.false. !readfromcube
.false. !dotimings
.false. !invent_ids
{outdir}
"""


def run_fof(run, particle_file, workdir, fof_exe, b=0.2, nmin=20, launcher=""):
    """run fof_dyablo on one particle snapshot; returns the directory holding the *_masst_* files."""
    out = os.path.join(workdir, "outputs")
    if os.path.isdir(out) and any("masst" in f for f in os.listdir(out)):
        return out
    os.makedirs(workdir, exist_ok=True)
    # fof_dyablo reads 80-char strings: pass paths relative to the work directory via symlinks
    for src, name in [(particle_file, "particles.h5"), (run.ini_file, "run.ini")]:
        link = os.path.join(workdir, name)
        if not os.path.islink(link):
            os.symlink(src, link)
    with open(os.path.join(workdir, "fof.in"), "w") as f:
        f.write(FOF_IN.format(root="halos", pathinput=".", namepart="particles.h5", nameinfo="run.ini",
                              b=b, nmin=nmin, outdir="outputs"))
    cmd = f"{launcher} {os.path.abspath(fof_exe)}".strip()
    with open(os.path.join(workdir, "fof.log"), "w") as log:
        subprocess.run(cmd, shell=True, cwd=workdir, stdout=log, stderr=subprocess.STDOUT, check=True)
    return out


def star_arrays(run, particle_file, aexp):
    """stars as dbtg wants them + dyablo extras. Ages in Myr; masses in Msun."""
    pos, mass, bt, st = run.read_particles(particle_file)
    tnow = float(run.tphys(aexp))
    age = tnow - bt[st]
    m = mass[st]
    # dyablo removes eta_SN x M at t_SN: restore the initial mass of stars older than t_SN only
    m_init = np.where(age > run.t_sn, m / (1 - run.eta_sn), m)
    return pos[st], m, m_init, age


def fesc_r200(run, grid_file, aexp, centres, r200, ndot_int):
    """outgoing ionizing photon rate through the shell [R200, R200+w] / intrinsic rate, per halo."""
    d = run.read_grid(grid_file, ["fx_rad", "fy_rad", "fz_rad"])
    cc = run.cell_centers(grid_file)
    L = run.lbox_mpc
    tree = cKDTree(cc, boxsize=L * (1 + 1e-12))
    out = np.full(len(r200), np.nan)
    conv = MPC**3 / MYR / aexp**2 * (run.l_unit / MPC) / (run.t_unit / MYR)   # code flux x Mpc^3 / Mpc -> photons/s
    for k in np.where(ndot_int > 0)[0]:
        r = r200[k]
        w = max(2 * d["dx"][tree.query(centres[k])[1]], 0.25 * r)
        idx = np.array(tree.query_ball_point(centres[k], r + w), dtype=int)
        if idx.size == 0:
            continue
        dr = (cc[idx] - centres[k] + 0.5 * L) % L - 0.5 * L
        rr = np.sqrt(np.sum(dr**2, axis=1))
        sh = rr >= r
        i, dr, rr = idx[sh], dr[sh], rr[sh]
        fr = (d["fx_rad"][i] * dr[:, 0] + d["fy_rad"][i] * dr[:, 1] + d["fz_rad"][i] * dr[:, 2]) / rr
        out[k] = np.sum(fr * d["vol"][i]) / w * conv / ndot_int[k]
    return out


def analyse_snapshot(run, it, grid_file, particle_file, outdir, plotdir, simname, fof_exe=None, fof_dir=None,
                     fof_launcher="", b=0.2, nmin=20, minf=1e7, star_Z=0.05):
    a = run.snapshot_aexp(grid_file)
    z = np.float64(1 / a - 1)                                          # dbtg does list - z
    cuage = run.age(a) * u.Myr
    boxh = run.lbox_mpc * run.h                                       # h^-1 cMpc, as dbtg wants it
    ngrid = run.npart_1d
    dbtg.set_cosmology(run.cosmo)

    # halos
    fof_dir = fof_dir or run_fof(run, particle_file, os.path.join(outdir, "fof", f"iter{it:07d}"), fof_exe,
                                 b=b, nmin=nmin, launcher=fof_launcher)
    halos = dbtg.readfof(fof_dir, boxh, ngrid)
    boxsize = run.lbox_mpc * u.Mpc
    mbox = (run.cosmo.critical_density(0) * boxsize.to(u.cm)**3).to(u.Msun) * run.cosmo.Om0
    mpart = mbox / ngrid**3
    hmass = halos[:, 1] * mpart
    hx, hy, hz = (halos[:, 2] * boxsize, halos[:, 3] * boxsize, halos[:, 4] * boxsize)

    # stars
    spos, smass, sm_init, sage = star_arrays(run, particle_file, a)
    starZ = np.full(len(smass), star_Z)                               # Zsun units; dyablo run has no metals
    xs, ys, zs = (spos[:, 0] * u.Mpc, spos[:, 1] * u.Mpc, spos[:, 2] * u.Mpc)
    print(f"  iter {it} z={z:.2f}: {len(hmass)} halos, {len(smass)} stars ({smass.sum():.3e} Msun now, "
          f"{sm_init.sum():.3e} formed)", flush=True)

    dbtg.plotCSFRD(sage * u.Myr, sm_init, cuage, boxh, run.H0, dir=plotdir, simname=simname)
    allstarinds, hind = dbtg.associate_star_halo2(xs, ys, zs, hx, hy, hz, hmass, boxh, Minf=minf * u.Msun)
    hm = hmass[hind]
    dbtg.plotSFR_vs_z_M(hm, allstarinds, sage * u.Myr, sm_init, cuage, z, dir=plotdir, simname=simname)
    mags, _, mstar = dbtg.computeallmags(smass, sage, starZ, allstarinds, initial_mass=sm_init)
    dbtg.plot_LF(mags[mags > -99], z, boxh, run.H0, z, dir=plotdir, simname=simname)

    # escape fractions through R200
    centres = np.array([hx[hind].value, hy[hind].value, hz[hind].value]).T
    r200 = dbtg.R200(hm, run.cosmo).to(u.Mpc).value
    young = [np.asarray(ii, dtype=int)[sage[np.asarray(ii, dtype=int)] < run.t_life] for ii in allstarinds]
    ndot = np.array([run.spawn_rate * smass[y].sum() for y in young])
    fesc = np.full(len(hm), np.nan)
    if run.has_fields(grid_file, ["fx_rad", "fy_rad", "fz_rad"]):
        fesc = fesc_r200(run, grid_file, a, centres, r200, ndot)
    else:
        print("  no fx_rad/fy_rad/fz_rad in this snapshot: fesc skipped")

    cat = np.column_stack([halos[hind, 0], centres, hm.value, r200, [len(i) for i in allstarinds], mstar,
                           np.where(mags > -99, mags, np.nan), ndot, fesc])
    np.savetxt(os.path.join(outdir, f"halos_iter{it:07d}.txt"), cat,
               header=f"{simname} iter {it} z={z:.4f}: fof_dyablo b={b}, Mh>{minf:g} Msun, stars within R200\n"
                      "fof_id x y z[cMpc] Mh[Msun] R200[cMpc] nstar Mstar_now[Msun] M_AB1600 Ndot_int[1/s] fesc_R200",
               fmt="%.6e")
    plot_smhm_fesc(cat, z, run, plotdir, simname)
    return cat


def behroozi19_smhm(mh, z):
    """median M* (Msun, Chabrier) at peak halo mass mh (Msun): Behroozi+19 Appendix J, Table J1 True/All/All/Excl."""
    p = dict(zip(*np.genfromtxt(os.path.join(dbtg.AUX, "obs/behroozi2019/smhm_fit_true_all_all_exclIHL.txt"),
                                dtype=None, encoding=None)))
    p = {k: float(v) for k, v in p.items()}
    a = 1 / (1 + z)
    m1 = p["M0"] + p["Ma"] * (a - 1) - p["Mlna"] * np.log(a) + p["Mz"] * z
    eps = p["e0"] + p["ea"] * (a - 1) - p["elna"] * np.log(a) + p["ez"] * z
    alpha = p["alpha0"] + p["alphaa"] * (a - 1) - p["alphalna"] * np.log(a) + p["alphaz"] * z
    beta = p["beta0"] + p["betaa"] * (a - 1) + p["betaz"] * z
    gamma = 10**(p["gamma0"] + p["gammaa"] * (a - 1) + p["gammaz"] * z)
    x = np.log10(mh) - m1
    return 10**(m1 + eps - np.log10(10**(-alpha * x) + 10**(-beta * x)) + gamma * np.exp(-0.5 * (x / p["delta0"])**2))


def stefanon21_shmr():
    """Stefanon+21 Table 3 rows (z, logM*, logMh, dlogMh+, dlogMh-, SHMR, +, -), Salpeter IMF."""
    return np.loadtxt(os.path.join(dbtg.AUX, "obs/stefanon2021/shmr_table3.txt"))


def read17_smhm():
    """Read+17 Table 2 clean sample: M*, dM*, M200, dM200+, dM200- (Msun)."""
    t = np.genfromtxt(os.path.join(dbtg.AUX, "obs/read2017/smhm_clean_sample.txt"), usecols=(1, 2, 3, 4, 5))
    return t * np.array([1e7, 1e7, 1e10, 1e10, 1e10])


def plot_smhm_fesc(cat, z, run, plotdir, simname):
    mh, mstar, ndot, fesc = cat[:, 4], cat[:, 7], cat[:, 9], cat[:, 10]
    s = mstar > 0
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    ax[0].loglog(mh[s], mstar[s], ".", ms=3, alpha=0.5, label=simname)
    bins = np.logspace(np.log10(mh.min()), np.log10(mh.max()) + 1e-6, 12)
    c = np.sqrt(bins[1:] * bins[:-1])
    med = [np.median(mstar[s & (mh >= lo) & (mh < hi)]) if np.any(s & (mh >= lo) & (mh < hi)) else np.nan
           for lo, hi in zip(bins[:-1], bins[1:])]
    ax[0].loglog(c, med, "k-o", ms=4, label="median (halos with stars)")
    # observed relations (Chabrier IMF; Salpeter values / 1.7). Halo masses: B19 Mpeak (virial), S21 abundance matching.
    mhg = np.logspace(9, 12, 200)
    ok = mhg >= 10**10.5
    ax[0].loglog(mhg[ok], behroozi19_smhm(mhg[ok], z), "C1-", lw=1.5, label=f"Behroozi+19 z={z:.1f}")
    ax[0].loglog(mhg[~ok], behroozi19_smhm(mhg[~ok], z), "C1:", lw=1.5, label="  (extrapolated below fit range)")
    t = stefanon21_shmr()
    zs = np.unique(t[:, 0])
    zsel = zs[np.argmin(np.abs(zs - z))]
    r = t[t[:, 0] == zsel]
    mh_s, ms_s = 10**r[:, 2], 10**r[:, 1] / 1.7
    ax[0].errorbar(mh_s, ms_s, xerr=[mh_s - 10**(r[:, 2] - r[:, 4]), 10**(r[:, 2] + r[:, 3]) - mh_s],
                   yerr=[ms_s - 10**(r[:, 1] - 0.14) / 1.7, 10**(r[:, 1] + 0.14) / 1.7 - ms_s],
                   fmt="s", color="C2", ms=5, label=f"Stefanon+21 z~{zsel:.1f} (/1.7 Salp.)")
    rd = read17_smhm()
    ax[0].errorbar(rd[:, 2], rd[:, 0], xerr=[rd[:, 4], rd[:, 3]], yerr=rd[:, 1], fmt="D", mfc="none", color="k",
                   ms=5, label="Read+17 z=0 field dwarfs")
    shmr = 2 * 0.0297 / ((mhg / 10**11.5)**-1.35 + (mhg / 10**11.5)**0.4) / 1.7
    ax[0].loglog(mhg, shmr * mhg, "C2--", lw=1, label="Stefanon+21 fit z~6-10")
    ax[0].set_xlabel(r"$M_h$ [M$_\odot$]"); ax[0].set_ylabel(r"$M_\star$ [M$_\odot$]")
    ax[0].set_title(f"stellar mass - halo mass z={z:.2f}"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    f = np.isfinite(fesc)
    if f.any():
        glob_f = np.sum(fesc[f] * ndot[f]) / np.sum(ndot[f])
        ax[1].semilogx(mh[f], np.clip(fesc[f], -0.1, 1.2), ".", ms=4, alpha=0.6, label=simname)
        fm = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            k = f & (mh >= lo) & (mh < hi)
            fm.append(np.sum(fesc[k] * ndot[k]) / np.sum(ndot[k]) if k.any() else np.nan)
        ax[1].semilogx(c, fm, "k-o", ms=4, label="photon-weighted mean")
        ax[1].axhline(glob_f, color="C3", lw=1, label=f"global {glob_f:.3f}")
    else:
        ax[1].text(0.5, 0.5, "no RT flux fields in snapshot", ha="center", transform=ax[1].transAxes)
    ax[1].set_ylim(-0.1, 1.2)
    ax[1].set_xlabel(r"$M_h$ [M$_\odot$]"); ax[1].set_ylabel(r"$f_{\rm esc}(R_{200})$")
    ax[1].set_title(f"ionizing escape fraction z={z:.2f}"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    plt.figure(fig.number)
    plt.savefig(os.path.join(plotdir, f"smhm_fesc_z={np.around(z, 2)}.pdf"))
    plt.close(fig)
