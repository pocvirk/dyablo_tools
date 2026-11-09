"""Gas inflow / outflow rates through spheres around halos (fraction x R200), as a function of halo mass.

Mdot through the sphere of radius r = sum over the cells of the shell [r - w/2, r + w/2] of m_cell v_r / w, with
v_r the physical radial velocity relative to the halo (peculiar velocity minus the halo bulk velocity, plus the
Hubble flow H(a) a r). Cells with v_r < 0 make the inflow, v_r > 0 the outflow (both reported positive, Msun/yr).
The halo velocity is the mass-weighted mean velocity of the particles within R200.

Gas velocities: rho_vx/rho_vy/rho_vz from the snapshot, or, for runs that did not write them, from the checkpoint
of the same iteration (restart_<prefix>_<iter>.h5, blocks of bx^3 cells in the snapshot's cell order).
Units: supercomoving velocity v~ -> physical peculiar velocity v~ / a; mass m = rho~ V~ (invariant).
"""
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from .run import MPC, MYR, MSUN, KMS_MPC_PER_MYR

YR = MYR * 1e-6
GAS_FIELDS = ["rho", "rho_vx", "rho_vy", "rho_vz"]


def checkpoint_file(run, it):
    return os.path.join(run.outdir, f"restart_{run.prefix}_{it}.h5")


def has_gas_velocity(run, grid_file, it):
    return run.has_fields(grid_file, GAS_FIELDS) or os.path.exists(checkpoint_file(run, it))


def read_gas(run, grid_file, it):
    """rho, rho_v (code units) + level/dx/vol, from the snapshot or else from the same-iteration checkpoint."""
    if run.has_fields(grid_file, GAS_FIELDS):
        return run.read_grid(grid_file, GAS_FIELDS)
    d = run.read_grid(grid_file, ["rho"])
    with h5py.File(checkpoint_file(run, it), "r") as f:
        rho = f["fields/rho"][:, 0, :].ravel()
        if not np.array_equal(rho, d["rho"]):
            raise ValueError(f"{checkpoint_file(run, it)}: cell order differs from {grid_file}")
        for k in GAS_FIELDS[1:]:
            d[k] = f["fields/" + k][:, 0, :].ravel()
    return d


def halo_velocities(run, particle_file, centres, r200):
    """mass-weighted mean particle velocity within R200 (code velocity units), per halo."""
    with h5py.File(particle_file, "r") as f:
        pos = f["coordinates"][()] * run.l_unit / MPC % run.lbox_mpc
        m = f["mass"][()]
        v = np.stack([f["vx"][()], f["vy"][()], f["vz"][()]], axis=1)
    tree = cKDTree(pos, boxsize=run.lbox_mpc * (1 + 1e-12))
    out = np.zeros((len(r200), 3))
    for k, idx in enumerate(tree.query_ball_point(centres, r200)):
        if idx:
            out[k] = np.average(v[idx], axis=0, weights=m[idx])
    return out


def halo_flows(run, d, cc, tree, aexp, centres, r200, vhalo, fracs=(0.5, 1.0)):
    """inflow and outflow rates (Msun/yr) through r = frac x R200: arrays (nhalo, len(fracs)) each, plus a flag
    (nhalo, len(fracs)) True when the shell is less than 3 cells wide in radius (r < 3 dx)."""
    L = run.lbox_mpc
    lt = run.l_unit / run.t_unit                                 # code velocity -> m/s
    Ha = run.cosmo.H(1 / aexp - 1).value * KMS_MPC_PER_MYR / MYR   # 1/s
    vgas = np.stack([d["rho_vx"], d["rho_vy"], d["rho_vz"]], axis=1) / d["rho"][:, None]
    mass = d["rho"] * d["vol"] * (MPC / run.l_unit)**3 * run.m_unit / MSUN   # Msun (vol is in cMpc^3)
    nf = len(fracs)
    inflow, outflow = np.full((len(r200), nf), np.nan), np.full((len(r200), nf), np.nan)
    unresolved = np.zeros((len(r200), nf), dtype=bool)
    for k in range(len(r200)):
        dxc = d["dx"][tree.query(centres[k])[1]]
        for j, fr in enumerate(fracs):
            r = fr * r200[k]
            w = max(2 * dxc, 0.2 * r)
            idx = np.array(tree.query_ball_point(centres[k], r + 0.5 * w), dtype=int)
            if idx.size == 0:
                continue
            dr = (cc[idx] - centres[k] + 0.5 * L) % L - 0.5 * L
            rr = np.sqrt(np.sum(dr**2, axis=1))
            sh = rr >= r - 0.5 * w
            i, dr, rr = idx[sh], dr[sh], rr[sh]
            if i.size == 0:
                continue
            vpec = np.sum((vgas[i] - vhalo[k]) * dr, axis=1) / rr * lt / aexp        # m/s
            vr = vpec + Ha * aexp * rr * MPC                                        # + Hubble flow
            flux = mass[i] * vr / (w * aexp * MPC) * YR                             # Msun/yr
            inflow[k, j] = -np.sum(flux[vr < 0])
            outflow[k, j] = np.sum(flux[vr > 0])
            unresolved[k, j] = r < 3 * dxc
    return inflow, outflow, unresolved


def accretion_dekel09(mh, z, fb):
    """mean baryonic accretion rate onto halos (Msun/yr), Dekel et al. 2009 eq. 7 (Neistein & Dekel 2008 fit)."""
    return 6.6 * (mh / 1e12)**1.15 * (1 + z)**2.25 * fb / 0.165


def plot_flows(mh, sfr, inflow, outflow, unresolved, fracs, z, run, plotdir, simname):
    fb = run.ob / run.om
    bins = np.logspace(np.log10(mh.min()), np.log10(mh.max()) + 1e-6, 12)
    c = np.sqrt(bins[1:] * bins[:-1])
    mg = np.logspace(np.log10(mh.min()), np.log10(mh.max()), 50)
    fig, ax = plt.subplots(1, len(fracs) + 1, figsize=(5.2 * (len(fracs) + 1), 4.6), constrained_layout=True)
    for j, fr in enumerate(fracs):
        a = ax[j]
        ok = ~unresolved[:, j] & np.isfinite(inflow[:, j])
        for y, col, lab in [(inflow[:, j], "C0", "inflow"), (outflow[:, j], "C3", "outflow")]:
            s = ok & (y > 0)
            a.loglog(mh[s], y[s], ".", color=col, ms=2, alpha=0.3)
            med = [np.median(y[s & (mh >= lo) & (mh < hi)]) if np.sum(s & (mh >= lo) & (mh < hi)) > 2 else np.nan
                   for lo, hi in zip(bins[:-1], bins[1:])]
            a.loglog(c, med, "-o", color=col, ms=4, label=f"{lab} (median)")
        ss = sfr > 0
        med = [np.median(sfr[ss & (mh >= lo) & (mh < hi)]) if np.sum(ss & (mh >= lo) & (mh < hi)) > 2 else np.nan
               for lo, hi in zip(bins[:-1], bins[1:])]
        a.loglog(c, med, "k--", lw=1, label="SFR, 10 Myr (median, halos with SF)")
        a.loglog(mg, accretion_dekel09(mg, z, fb), "C2:", lw=1.5, label="Dekel+09 mean baryonic accretion")
        a.set_xlabel(r"$M_h$ [M$_\odot$]"); a.set_ylabel(r"$\dot M$ [M$_\odot$/yr]")
        a.set_title(f"gas flows through {fr:g} $R_{{200}}$, z={z:.2f}")
        a.legend(fontsize=7, loc="lower right"); a.grid(alpha=0.3, which="both")
        if (~ok).any():
            a.text(0.02, 0.97, f"{np.sum(unresolved[:, j])} halos with r < 3 dx left out", transform=a.transAxes,
                   fontsize=7, va="top")
    # outflow / inflow (mass loading of the flows) at each radius
    a = ax[-1]
    for j, fr in enumerate(fracs):
        ok = ~unresolved[:, j] & (inflow[:, j] > 0)
        ratio = outflow[:, j] / inflow[:, j]
        med = [np.median(ratio[ok & (mh >= lo) & (mh < hi)]) if np.sum(ok & (mh >= lo) & (mh < hi)) > 2 else np.nan
               for lo, hi in zip(bins[:-1], bins[1:])]
        a.loglog(c, med, "-o", color=f"C{j + 4}", ms=4, label=f"{fr:g} $R_{{200}}$")
    a.axhline(1, color="0.5", lw=1)
    a.set_xlabel(r"$M_h$ [M$_\odot$]"); a.set_ylabel("outflow / inflow (median)")
    a.set_title(f"{simname}"); a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
    plt.figure(fig.number)
    plt.savefig(os.path.join(plotdir, f"flows_z={np.around(z, 2)}.pdf"))
    plt.close(fig)
