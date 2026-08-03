#!/bin/bash
#SBATCH --job-name=re550_open
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --output=slurm-re550-open-%j.out

# Re_tau = 550 open-channel validation case (torChannel OSS release).
# Seeded by interpolating the converged Re_tau=180 field, so run that first.
# ~15 h at float64 on one GB10 -- see examples/re550_open/config.yaml.
cd /home/giorgio/torChannel

export PYTORCH_JIT=0
export TORCHANNEL_COMPILE=1
export TORCHANNEL_POISSON_CUDAGRAPH=1

python main.py examples/re550_open/config.yaml
