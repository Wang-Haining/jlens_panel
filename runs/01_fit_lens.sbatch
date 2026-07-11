#!/bin/bash
#SBATCH --job-name=jlens_fit
#SBATCH --partition=nextgen-gpu
#SBATCH --account=group-jasonclark
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/fit_lens_%j.out
#SBATCH --error=logs/fit_lens_%j.err

set -euo pipefail

cd /home/g91p721/jlens_panel
source runs/tempest_env.sh
mkdir -p checkpoints logs

python scripts/check_disk.py --environment tempest --project-root .
python scripts/fit_lens.py \
  --environment tempest \
  --prompts data/fit_corpus/c4_slice.json \
  --output checkpoints/qwen2.5-7b-instruct-jlens.pt \
  --project-root .
