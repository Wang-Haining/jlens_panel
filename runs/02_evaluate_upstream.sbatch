#!/bin/bash
#SBATCH --job-name=jlens_eval
#SBATCH --partition=nextgen-gpu
#SBATCH --account=group-jasonclark
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/evaluate_upstream_%j.out
#SBATCH --error=logs/evaluate_upstream_%j.err

set -euo pipefail

cd /home/g91p721/jlens_panel
source runs/tempest_env.sh
mkdir -p logs results/upstream

python scripts/check_disk.py --environment tempest --project-root .
python scripts/evaluate_upstream.py \
  --lens checkpoints/qwen2.5-7b-instruct-jlens.pt \
  --eval vendor/jacobian-lens/data/evaluations/lens-eval-multihop.json \
  --output results/upstream/multihop.json \
  --project-root .
python scripts/evaluate_upstream.py \
  --lens checkpoints/qwen2.5-7b-instruct-jlens.pt \
  --eval vendor/jacobian-lens/data/evaluations/lens-eval-association.json \
  --output results/upstream/association.json \
  --project-root .
