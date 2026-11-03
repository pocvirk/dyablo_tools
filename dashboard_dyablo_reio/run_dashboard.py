#!/usr/bin/env python3
"""Reionization dashboard for a dyablo run.

  python -I run_dashboard.py <rundir> [--out DIR] [--name NAME] [--fof-exe PATH] [--snap ITER ...] [--stride N]

<rundir> is the dyablo run directory (holds last.ini and the snapshots). Produces in <out>/PDFs (pdf + png):
  IGM (all snapshots): xHI, xHII, J21, Gamma (ionized cells), tau_es          -- ramses-cudaton dashboard plots
  galaxies (--snap, default last snapshot with particles): UV LF, CSFRD, halo SFH vs M  -- idem, via fof_dyablo halos
                                                           + stellar mass-halo mass, fesc(R200)
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashboard_reio_dyablo.run import DyabloRun  # noqa: E402
from dashboard_reio_dyablo import averages, galaxies  # noqa: E402

_savefig = plt.savefig


def _savefig_pdf_png(fname, *args, **kw):
    """the dashboard writes pdf; also write a png next to it for quick looks."""
    _savefig(fname, *args, **kw)
    if str(fname).endswith(".pdf"):
        _savefig(str(fname)[:-4] + ".png", dpi=110)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rundir")
    p.add_argument("--out", help="output directory (default <rundir>/dashboard)")
    p.add_argument("--name", help="simulation name in legends (default: output prefix)")
    p.add_argument("--ini", help="config file (default <rundir>/last.ini)")
    p.add_argument("--stride", type=int, default=1, help="use every N-th snapshot for the averages")
    p.add_argument("--snap", type=int, nargs="*", help="iterations for the galaxy analysis (default: last)")
    p.add_argument("--no-galaxies", action="store_true")
    p.add_argument("--fof-exe", default=os.environ.get("FOF_DYABLO"), help="fof_dyablo executable (or $FOF_DYABLO)")
    p.add_argument("--fof-launcher", default="", help='e.g. "srun -n 8" (rank count must be a cube)')
    p.add_argument("--fof-b", type=float, default=0.2)
    p.add_argument("--fof-nmin", type=int, default=20, help="min halo size in particles")
    p.add_argument("--minf", type=float, default=1e7, help="min halo mass (Msun) for the galaxy analysis")
    p.add_argument("--star-Z", type=float, default=0.05, help="stellar metallicity (Zsun) for BPASS mags")
    args = p.parse_args()

    run = DyabloRun(args.rundir, ini=args.ini)
    name = args.name or run.prefix
    out = os.path.abspath(args.out or os.path.join(run.rundir, "dashboard"))
    plotdir = os.path.join(out, "PDFs") + "/"
    os.makedirs(plotdir, exist_ok=True)
    plt.savefig = _savefig_pdf_png
    print(run.summary(), flush=True)

    print("IGM averages", flush=True)
    av = averages.compute_averages(run, out, stride=args.stride)
    averages.plot_averages(av, plotdir, name)

    if not args.no_galaxies:
        snaps = [s for s in run.snapshots() if s[2]]
        if args.snap:
            snaps = [s for s in snaps if s[0] in args.snap]
        else:
            snaps = snaps[-1:]
        for it, g, part in snaps:
            print(f"galaxies iter {it}", flush=True)
            galaxies.analyse_snapshot(run, it, g, part, out, plotdir, name, fof_exe=args.fof_exe,
                                      fof_launcher=args.fof_launcher, b=args.fof_b, nmin=args.fof_nmin,
                                      minf=args.minf, star_Z=args.star_Z)
    print("plots in", plotdir)


if __name__ == "__main__":
    main()
