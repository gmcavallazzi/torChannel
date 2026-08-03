#!/bin/bash
#SBATCH --job-name=re180_matched
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --output=slurm-re180-matched-%j.out

# Re_tau = 180 open-channel validation case (torChannel OSS release).
# Seeded from an equilibrated CaNS field; see examples/re180_open/config_continue.yaml.
# Local GB10 submission. PYTORCH_JIT=0: nvrtc arch workaround on GB10.
cd /home/giorgio/torChannel

export PYTORCH_JIT=0
export TORCHANNEL_COMPILE=1
export TORCHANNEL_POISSON_CUDAGRAPH=1

python main.py examples/re180_open/config_continue.yaml
