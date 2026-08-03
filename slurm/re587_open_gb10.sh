#!/bin/bash
#SBATCH --job-name=re587_open
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --output=slurm-re587-open-%j.out

# Re_tau ~ 587 open-channel validation case (Re_b matched to MKM chan590) (torChannel OSS release).
# Seeded by interpolating the converged Re_tau=180 field, so run that first.
# ~15 h at float64 on one GB10 -- see examples/re587_open/config.yaml.
cd /home/giorgio/torChannel

export PYTORCH_JIT=0
export TORCHANNEL_COMPILE=1
export TORCHANNEL_POISSON_CUDAGRAPH=1

python main.py examples/re587_open/config.yaml
