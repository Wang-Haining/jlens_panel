#!/bin/bash
#SBATCH --job-name=jlens_sweep
#SBATCH --partition=nextgen-gpu
#SBATCH --account=group-jasonclark
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/position_sweep_%j.out
#SBATCH --error=logs/position_sweep_%j.err

set -euo pipefail

cd /home/g91p721/jlens_panel
source runs/tempest_env.sh
mkdir -p analysis artifacts/calibration artifacts/sweep_v3 logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python scripts/check_disk.py --environment tempest --project-root .
python scripts/calibrate_null.py \
  --config config/sprint.yaml \
  --lens checkpoints/qwen2.5-7b-instruct-jlens.pt \
  --output-dir artifacts/calibration \
  --project-root . \
  --environment tempest
python scripts/sweep_positions.py \
  --phase capture \
  --config config/sprint.yaml \
  --train data/generated/dirty_v3/train.jsonl \
  --dev data/generated/dirty_v3/dev.jsonl \
  --lens checkpoints/qwen2.5-7b-instruct-jlens.pt \
  --artifact-dir artifacts/sweep_v3 \
  --calibration-dir artifacts/calibration \
  --results analysis/sweep_results.csv \
  --heatmap analysis/sweep_heatmap.json \
  --analysis-manifest analysis/sweep_manifest.json \
  --project-root . \
  --environment tempest

# Small multinomial probe matrices are much slower when BLAS oversubscribes
# the node. The exact LBFGS estimator and its frozen optimization settings are
# unchanged; this cap only controls linear-algebra worker count.
export JLENS_PROBE_BLAS_THREADS=4
export OMP_NUM_THREADS=${JLENS_PROBE_BLAS_THREADS}
export MKL_NUM_THREADS=${JLENS_PROBE_BLAS_THREADS}
export OPENBLAS_NUM_THREADS=${JLENS_PROBE_BLAS_THREADS}

python scripts/sweep_positions.py \
  --phase analyze \
  --config config/sprint.yaml \
  --train data/generated/dirty_v3/train.jsonl \
  --dev data/generated/dirty_v3/dev.jsonl \
  --lens checkpoints/qwen2.5-7b-instruct-jlens.pt \
  --artifact-dir artifacts/sweep_v3 \
  --calibration-dir artifacts/calibration \
  --results analysis/sweep_results.csv \
  --heatmap analysis/sweep_heatmap.json \
  --analysis-manifest analysis/sweep_manifest.json \
  --project-root . \
  --environment tempest
python scripts/check_disk.py --environment tempest --project-root .
