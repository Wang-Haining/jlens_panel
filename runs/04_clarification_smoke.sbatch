#!/bin/bash
#SBATCH --job-name=jlens_clarify
#SBATCH --partition=nextgen-gpu
#SBATCH --account=group-jasonclark
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/clarification_smoke_%j.out
#SBATCH --error=logs/clarification_smoke_%j.err

set -euo pipefail

cd /home/g91p721/jlens_panel
source runs/tempest_env.sh
mkdir -p logs results/clarification_smoke_v3

python scripts/check_disk.py --environment tempest --project-root .
python scripts/run_dirty_experiment.py \
  --tempest \
  --data data/generated/dirty_v3/test.jsonl \
  --scores results/readouts_v3/readout_scores.jsonl \
  --best-non-j results/readouts_v3/non_j_selection.json \
  --output results/clarification_smoke_v3/outcomes.jsonl \
  --project-root . \
  --limit 20
python scripts/analyze_dirty_run.py \
  --input results/clarification_smoke_v3/outcomes.jsonl \
  --output results/clarification_smoke_v3/analysis.json
python scripts/check_disk.py --environment tempest --project-root .
