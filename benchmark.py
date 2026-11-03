"""
Benchmark MultiAmr2Cube variants.

Usage (from the multiamr2cube_optim/ directory):
    python claude/benchmark.py

Outputs:
    claude/results.csv   – timing and correctness table
    stdout               – progress and summary
"""

import sys
import os
import time
import json

import numpy as np
from configparser import ConfigParser

# make sure parent dir is importable (for pyablo etc.)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyablo
import multiamr2cube_ref        as ref
import multiamr2cube_numpy      as m_numpy
import multiamr2cube_numba      as m_numba
import multiamr2cube_numba_full as m_numba_full

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PATH = "./"
SNAPNAME = "star"
CONFIGNAME = "restart_star_279982.ini"
T = 70000
FIELD = "rho"

LCUBES = [8, 9]   # lcube=10 -> 1024^3 (~8 GB); add manually if RAM permits
N_REPEATS = 5     # timed repetitions (after warm-up)

VARIANTS = {
    "ref":        ref.multiamr2cube,
    "numpy":      m_numpy.multiamr2cube,
    "numba":      m_numba.multiamr2cube,
    "numba_full": m_numba_full.multiamr2cube,
}

OUTCSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")

# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------
def load_data():
    print("Loading snapshot...", flush=True)
    config = ConfigParser(inline_comment_prefixes=('#', ';'))
    config.read(PATH + CONFIGNAME)

    reader = pyablo.XdmfReader()
    snapfile = PATH + SNAPNAME + "_iter{:07d}.xmf".format(T)
    snap = reader.readSnapshot(snapfile)

    cc, lcell, Lbox = ref.extract_snapshot_data(snap, config)
    values = np.array(snap.readAllFloat(FIELD), dtype=np.float64)
    print(f"  {len(values):,} cells  |  lmin={lcell.min()}  lmax={lcell.max()}  |  Lbox={Lbox}")
    return cc, lcell, Lbox, values


# ---------------------------------------------------------------------------
# Run one variant / lcube
# ---------------------------------------------------------------------------
def run_variant(fn, name, cc, lcell, values, Lbox, lcube, ref_result):
    ncube = 2 ** lcube
    print(f"  [{name}] lcube={lcube} ({ncube}^3 grid)...", end=" ", flush=True)

    # warm-up (also triggers numba compilation if needed)
    _ = fn(cc, lcell, values, Lbox, lcube)

    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        result = fn(cc, lcell, values, Lbox, lcube)
        times.append(time.perf_counter() - t0)

    t_med = float(np.median(times))
    t_min = float(np.min(times))

    # correctness vs reference
    if ref_result is not None:
        max_diff = float(np.max(np.abs(result - ref_result)))
        rel_diff = float(max_diff / (np.max(np.abs(ref_result)) + 1e-300))
        ok = "OK" if rel_diff < 1e-10 else f"FAIL(rel={rel_diff:.2e})"
    else:
        max_diff = rel_diff = 0.0
        ok = "REF"

    print(f"median={t_med:.3f}s  min={t_min:.3f}s  correctness={ok}")
    return result, {"variant": name, "lcube": lcube, "ncube": ncube,
                    "t_median": t_med, "t_min": t_min,
                    "max_abs_diff": max_diff, "rel_diff": rel_diff,
                    "correct": ok}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cc, lcell, Lbox, values = load_data()

    # warm up numba JIT before any timed run
    print("Warming up numba JIT...", flush=True)
    m_numba.warmup(cc, lcell, values, Lbox)
    m_numba_full.warmup(cc, lcell, values, Lbox)
    print("  done")

    rows = []

    for lcube in LCUBES:
        print(f"\n=== lcube={lcube} ===")
        ref_result = None

        for name, fn in VARIANTS.items():
            result, row = run_variant(fn, name, cc, lcell, values, Lbox, lcube, ref_result)
            if name == "ref":
                ref_result = result
            rows.append(row)

    # write CSV
    import csv
    fieldnames = ["variant", "lcube", "ncube", "t_median", "t_min",
                  "max_abs_diff", "rel_diff", "correct"]
    with open(OUTCSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nResults saved to {OUTCSV}")

    # summary table
    print("\n{:<10} {:>6} {:>12} {:>12} {:>10}".format(
        "variant", "lcube", "t_median(s)", "t_min(s)", "correct"))
    print("-" * 55)
    for r in rows:
        print("{:<10} {:>6} {:>12.3f} {:>12.3f} {:>10}".format(
            r["variant"], r["lcube"], r["t_median"], r["t_min"], r["correct"]))


if __name__ == "__main__":
    main()
