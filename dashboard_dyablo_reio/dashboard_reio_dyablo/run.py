"""A dyablo cosmological RT run as seen by the dashboard: parameters, units, cosmology, snapshots.

Everything is read from the run's own `last.ini` (written by dyablo in the run directory, fully resolved,
values in code units), so the same code works on any run of the calibration campaign.

Conventions (dyablo cosmo_reio, supercomoving units):
  - a quantity with time exponent T and length exponent L is physical = code * a**(2T+L);
    lengths are comoving, number counts (n*V) are invariant, rates (1/t) scale as 1/a**2.
  - code time t~ and physical time t are measured from today (a=1), both negative (dyablo Cosmo.h):
        t~(a) = -1/H0 int_a^1 da / E(a),  t(a) = -1/H0 int_a^1 a^2 da / E(a),
        E(a) = sqrt(a^4 Ok + a^3 Om + a^6 OL)
  - star particles: birth_time = t(birth) < 0; DM particles: 1000 until their cell first reaches x_HII>0.5,
    then |t| at that moment (particle zreion tag).
"""
import configparser
import glob
import os
import re

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.integrate import cumulative_trapezoid

MPC = 3.0856775814913673e22    # m
MYR = 3.15576e13               # s
KMS_MPC_PER_MYR = 1.0e3 / MPC * MYR   # 1 km/s/Mpc in 1/Myr
MSUN = 1.98847e30              # kg
MP = 1.67262192e-27            # kg
C = 2.99792458e8               # m/s

_LENGTH = {"m": 1.0, "cm": 1e-2, "km": 1e3, "pc": MPC * 1e-6, "kpc": MPC * 1e-3, "Mpc": MPC}
_TIME = {"s": 1.0, "yr": MYR * 1e-6, "kyr": MYR * 1e-3, "Myr": MYR, "Gyr": MYR * 1e3}


def _strip(v):
    return v.split(";")[0].strip()


class DyabloRun:
    def __init__(self, rundir, ini=None, logs=None):
        self.rundir = os.path.abspath(rundir)
        self.ini_file = ini or os.path.join(self.rundir, "last.ini")
        cp = configparser.ConfigParser(inline_comment_prefixes=(";",), strict=False, interpolation=None)
        cp.read(self.ini_file)
        self.cp = cp
        g = lambda s, k, d=None: _strip(cp.get(s, k, fallback=str(d)))  # noqa: E731

        # code units: [units] time/length/density, e.g. "1 Myr", "1 Mpc", "1 kg / m3"
        tnum, tunit = g("units", "time", "1 s").split()
        lnum, lunit = g("units", "length", "1 m").split()
        dens = g("units", "density", "1 kg / m3")
        assert dens.replace(" ", "") == "1kg/m3", f"unsupported density unit {dens}"
        self.t_unit = float(tnum) * _TIME[tunit]          # s
        self.l_unit = float(lnum) * _LENGTH[lunit]        # m
        self.m_unit = self.l_unit**3                      # kg (density 1 kg/m3 x length^3)

        self.prefix = g("output", "outputprefix")
        self.outdir = g("output", "outputdir", self.rundir)
        if not os.path.isdir(self.outdir):
            self.outdir = self.rundir
        self.lbox = float(g("mesh", "xmax")) - float(g("mesh", "xmin", 0))   # comoving, code length
        self.lbox_mpc = self.lbox * self.l_unit / MPC
        self.bx = int(g("amr", "bx"))
        self.npart_1d = int(g("particle_grid", "nx", 0)) or None

        self.om = float(g("cosmology", "omegam"))
        self.ol = float(g("cosmology", "omegav"))
        self.ob = float(g("cosmology", "omegab", 0.0))
        h0_code = float(g("cosmology", "h0"))            # 1/code time
        self.H0 = h0_code / self.t_unit * MPC / 1e3       # km/s/Mpc
        self.h = self.H0 / 100.0
        self.cosmo = FlatLambdaCDM(name=os.path.basename(self.rundir), H0=self.H0 * u.km / u.s / u.Mpc,
                                   Tcmb0=0 * u.K, Om0=self.om, Ob0=self.ob)

        # RT / stars (code units in last.ini)
        self.c_rad = float(g("rad", "c_rad")) * self.l_unit / self.t_unit                 # m/s
        self.temp_bb = float(g("rad", "temp_black_body", 1e5))
        self.spawn_rate = float(g("rad", "spawn_rate")) / self.t_unit * MSUN                 # photons/s/Msun
        self.t_life = float(g("rad", "t_life_star")) * self.t_unit / MYR                     # Myr
        self.eta_sn = float(g("star_feedback", "eta_snii", 0.0))
        self.t_sn = float(g("star_feedback", "t_snii", 0.0)) * self.t_unit / MYR             # Myr
        self.epsilon_star = float(g("star_formation", "epsilon_star", "nan"))
        # stellar emission: [rad] source_model = constant (spawn_rate for t_life_star) or bpass (emissivity table)
        self.source_model = g("rad", "source_model", "constant")
        if self.source_model == "bpass":
            self.emissivity_file = g("rad", "emissivity_file")
            self.emissivity_Z = float(g("rad", "emissivity_z", 0.001))
            self.emissivity_tmax = float(g("rad", "emissivity_tmax")) * self.t_unit / MYR    # Myr
            self._read_emissivity()
        self.sigma_n = _sigma_n_blackbody(self.temp_bb)                                      # m2

        self._tables()
        self.logs = logs

    # ---------------------------------------------------------------- stellar emission
    def _read_emissivity(self):
        """log10 Q(age) [photons/s/Msun_init] at emissivity_Z, interpolated in log Z as ParticleUpdate_NGP_src_bpass."""
        rows = [[float(v) for v in l.split(",")] for l in open(self.emissivity_file)
                if l.strip() and not l.lstrip().startswith("#")]
        na, nz = int(rows[0][0]), int(rows[1][0])
        ages, zs = np.array(rows[2]), np.array(rows[3])
        lq = np.log10(np.maximum(np.array(rows[4:4 + nz]), 1e-30))
        o = np.argsort(zs)
        lz = np.clip(np.log10(self.emissivity_Z), np.log10(zs[o[0]]), np.log10(zs[o[-1]]))
        self._em_logage = np.log10(ages)
        self._em_logq = np.array([np.interp(lz, np.log10(zs[o]), lq[o, i]) for i in range(na)])

    def photon_rate(self, m_now, age):
        """intrinsic ionizing photon rate (photons/s) of star particles: mass now (Msun), age (Myr) -- the run's model."""
        if self.source_model != "bpass":
            return np.where(age < self.t_life, self.spawn_rate * m_now, 0.0)
        m_init = np.where(age > self.t_sn, m_now / (1 - self.eta_sn), m_now) if self.eta_sn > 0 else m_now
        q = 10**np.interp(np.log10(np.maximum(age, 1e-30)), self._em_logage, self._em_logq)
        return np.where(age < self.emissivity_tmax, m_init * q, 0.0)

    # ---------------------------------------------------------------- time
    def _tables(self):
        a = np.geomspace(1e-4, 1.0, 20001)
        E = np.sqrt(a**4 * (1 - self.om - self.ol) + a**3 * self.om + a**6 * self.ol)
        H0 = self.H0 * KMS_MPC_PER_MYR                                     # 1/Myr
        it = cumulative_trapezoid(1 / E, a, initial=0)
        ip = cumulative_trapezoid(a**2 / E, a, initial=0)
        self._a = a
        self._tcode = -(it[-1] - it) / H0 * MYR / self.t_unit             # code time units
        self._tphys = -(ip[-1] - ip) / H0                                  # Myr from today
        self._age = ip / H0                                                # Myr since a=1e-4 (~big bang)

    def aexp_from_code_time(self, t):
        return np.interp(t, self._tcode, self._a)

    def tphys(self, aexp):
        """dyablo physical time (Myr, negative, origin today)."""
        return np.interp(aexp, self._a, self._tphys)

    def age(self, aexp):
        """age of the universe (Myr)."""
        return np.interp(aexp, self._a, self._age)

    # ---------------------------------------------------------------- snapshots
    def snapshots(self):
        """sorted list of (iteration, grid .h5, particle .h5 or None)."""
        out = []
        for gfile in glob.glob(os.path.join(self.outdir, f"{self.prefix}_iter*.h5")):
            it = int(re.search(r"_iter(\d+)\.h5$", gfile).group(1))
            p = os.path.join(self.outdir, f"{self.prefix}_particles_particles_iter{it:07d}.h5")
            out.append((it, gfile, p if os.path.exists(p) else None))
        return sorted(out)

    def snapshot_aexp(self, grid_file):
        txt = open(grid_file[:-3] + ".xmf").read()
        t = float(re.search(r'Time TimeType="Single" Value="([^"]+)"', txt).group(1))
        return float(self.aexp_from_code_time(t))

    def read_grid(self, path, fields):
        """cell fields + level, dx (comoving Mpc), vol (comoving Mpc^3)."""
        with h5py.File(path, "r") as f:
            d = {k: f[k][()] for k in fields if k in f}
            lev = f["level"][()]
        d["level"] = lev
        d["dx"] = self.lbox_mpc / (self.bx * 2.0**lev)
        d["vol"] = d["dx"]**3
        return d

    def has_fields(self, path, fields):
        with h5py.File(path, "r") as f:
            return all(k in f for k in fields)

    def cell_centers(self, path):
        """cell centres (comoving Mpc, wrapped into the box) = mean of the 8 hexahedron nodes."""
        with h5py.File(path, "r") as f:
            conn = f["connectivity"][()]
            nodes = f["coordinates"][()]
        c = np.zeros((conn.shape[0], 3))
        for k in range(8):
            c += nodes[conn[:, k]]
        return (c / 8.0 * self.l_unit / MPC) % self.lbox_mpc

    def read_particles(self, path):
        """positions (comoving Mpc), mass (Msun), birth_time (raw), is_star."""
        with h5py.File(path, "r") as f:
            pos = f["coordinates"][()] * self.l_unit / MPC
            mass = f["mass"][()] * self.m_unit / MSUN
            bt = f["birth_time"][()]
        is_star = bt < 0
        return pos % self.lbox_mpc, mass, bt * self.t_unit / MYR, is_star

    def summary(self):
        return (f"{self.prefix}: box {self.lbox_mpc:.4f} cMpc ({self.lbox_mpc * self.h:.3f} cMpc/h), H0={self.H0:.3f}, "
                f"Om={self.om}, Ob={self.ob}; c_rad={self.c_rad / C:.3g}c, {self.spawn_rate:.3g} ph/s/Msun for "
                f"{self.t_life:g} Myr" + (f" [overridden: BPASS table Z={self.emissivity_Z:g} to {self.emissivity_tmax:g} Myr]"
                                         if self.source_model == "bpass" else "") + f", T_bb={self.temp_bb:g} K, eps*={self.epsilon_star}, eta_SN={self.eta_sn}")


def _sigma_n_blackbody(temp):
    """number-averaged HI photoionization cross-section (m2) for a black body, as dyablo Ionization_utils computeSigma."""
    egy = np.linspace(13.6, 500.0, 163840)                         # eV
    # Verner et al. 1996 HI fit (same as dyablo sigma_HI)
    e0, s0, ya, P, yw, y0, y1 = 0.4298, 5.475e4, 32.88, 2.963, 0.0, 0.0, 0.0
    x = egy / e0 - y0
    y = np.sqrt(x**2 + y1**2)
    sig = s0 * ((x - 1)**2 + yw**2) * y**(0.5 * P - 5.5) * (1 + np.sqrt(y / ya))**(-P) * 1e-18 * 1e-4   # m2
    kT = 8.617333262e-5 * temp
    spec = egy**3 / np.expm1(egy / kT)
    spec_n = spec / egy
    return np.trapezoid(sig * spec_n, egy) / np.trapezoid(spec_n, egy)
