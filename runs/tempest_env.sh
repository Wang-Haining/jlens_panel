#!/bin/bash
# Shared Tempest module/cache setup for jlens_panel Slurm jobs.

source /usr/local/Modules/5.6.1/init/bash
module purge

module use --append /etc/scl/modulefiles
module use --append /mnt/shared/modulefiles
module use --append /mnt/shared/ebmodules/eb/all
module use --append /mnt/global/modulefiles

module load GCCcore/13.3.0
module load zlib/1.3.1-GCCcore-13.3.0
module load binutils/2.42-GCCcore-13.3.0
module load bzip2/1.0.8-GCCcore-13.3.0
module load ncurses/6.5-GCCcore-13.3.0
module load libreadline/8.2-GCCcore-13.3.0
module load Tcl/8.6.14-GCCcore-13.3.0
module load SQLite/3.45.3-GCCcore-13.3.0
module load XZ/5.4.5-GCCcore-13.3.0
module load libffi/3.4.5-GCCcore-13.3.0
module load OpenSSL
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.3.0

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HOME=/home/g91p721/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export JLENS_PANEL_ROOT=/home/g91p721/jlens_panel
export JLENS_PANEL_VENV=${JLENS_PANEL_VENV:-/home/g91p721/jd/.venv}
export PYTHONPATH=${JLENS_PANEL_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

source "${JLENS_PANEL_VENV}/bin/activate"
