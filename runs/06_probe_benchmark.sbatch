#!/bin/bash
#SBATCH --job-name=jlens_probe_bench
#SBATCH --partition=gpuunsafe
#SBATCH --account=group-jasonclark
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=logs/probe_benchmark_%j.out
#SBATCH --error=logs/probe_benchmark_%j.err

set -euo pipefail

cd /home/g91p721/jlens_panel
source runs/tempest_env.sh
python scripts/benchmark_probe_solver.py
