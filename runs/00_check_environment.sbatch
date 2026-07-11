#!/bin/bash
#SBATCH --job-name=jlens_env
#SBATCH --partition=nextgen-gpu
#SBATCH --account=group-jasonclark
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/check_environment_%j.out
#SBATCH --error=logs/check_environment_%j.err

set -euo pipefail

cd /home/g91p721/jlens_panel
source runs/tempest_env.sh
mkdir -p logs results

python scripts/check_disk.py --environment tempest --project-root .
python scripts/check_runtime.py --output results/runtime_check.json
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
