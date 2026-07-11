#!/bin/bash
#SBATCH --job-name=jlens_readouts
#SBATCH --partition=nextgen-gpu
#SBATCH --account=group-jasonclark
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/extract_score_readouts_%j.out
#SBATCH --error=logs/extract_score_readouts_%j.err

set -euo pipefail

cd /home/g91p721/jlens_panel
source runs/tempest_env.sh
mkdir -p artifacts/dirty_v3 data/generated/dirty_v3 logs results/readouts_v3

python scripts/check_disk.py --environment tempest --project-root .
python scripts/generate_dirty_data.py \
  --output-dir data/generated/dirty_v3 \
  --seed 20260710 \
  --train-size 1000 \
  --dev-size 200 \
  --test-size 300 \
  --min-distractors 8 \
  --max-distractors 8
python scripts/extract_readouts.py \
  --environment tempest \
  --input data/generated/dirty_v3/train.jsonl \
  --input data/generated/dirty_v3/dev.jsonl \
  --input data/generated/dirty_v3/test.jsonl \
  --lens checkpoints/qwen2.5-7b-instruct-jlens.pt \
  --output-dir artifacts/dirty_v3 \
  --project-root . \
  --disk-check-every 25
python scripts/score_readouts.py \
  --artifact-dir artifacts/dirty_v3 \
  --output-dir results/readouts_v3
python scripts/check_disk.py --environment tempest --project-root .
