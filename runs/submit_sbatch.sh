#!/bin/bash
# Ensure Slurm can open the configured log path before submitting a job.

set -euo pipefail

ROOT=${JLENS_PANEL_ROOT:-/home/g91p721/jlens_panel}
cd "${ROOT}"
mkdir -p logs

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 runs/NN_job.sbatch" >&2
    exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to submit from a dirty Git worktree: ${ROOT}" >&2
    exit 3
fi

sbatch "$1"
