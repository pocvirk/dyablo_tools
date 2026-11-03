#!/bin/bash
#SBATCH -A AST031
#SBATCH -J dash_reio
#SBATCH -o dash_reio-%j.out
# submit from this directory, or export DASHBOARD_DIR=<path to dashboard_dyablo_reio>
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
# usage: sbatch run_dashboard.sl <rundir> [run_dashboard.py options]
# Edit PYENV / FOF_DYABLO for your installation.
# run_dashboard.py next to this script (sbatch copies the script, so locate it from the submit dir or $DASHBOARD_DIR)
HERE=${DASHBOARD_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "$0")}}
PYENV=${PYENV:-/lustre/orion/ast031/proj-shared/pocvirk/CoDaIII/prod_sr/pocvirk/dyablo/28102026/analysis/pyenv}
export FOF_DYABLO=${FOF_DYABLO:-/lustre/orion/ast031/proj-shared/pocvirk/CoDaIII/prod_sr/pocvirk/dyablo/28102026/analysis/halo_finder_dyablo/fof}
module load cray-python/3.11.7 PrgEnv-gnu cray-hdf5-parallel
. $PYENV/bin/activate
python -I -W ignore $HERE/run_dashboard.py "$@"
