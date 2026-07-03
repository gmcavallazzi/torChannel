#!/bin/bash
#SBATCH --job-name=canopy_monti
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --output=slurm-canopy-%j.out

# Monti et al. (2022) theta=0 rigid canopy, lambda=0.35, Re_b=6000
# Local GB10 submission. PYTORCH_JIT=0: nvrtc arch workaround on GB10.
cd /home/giorgio/torChannel

export PYTORCH_JIT=0
export TORCHANNEL_COMPILE=1
export TORCHANNEL_POISSON_CUDAGRAPH=1

python main.py config_canopy_monti_stats.yaml
