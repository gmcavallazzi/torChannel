#!/bin/bash
#SBATCH --job-name=re180_cfl02
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --output=slurm-re180-cfl02-%j.out

# Halved-timestep control experiment: does the residual u'_rms deficit scale
# with dt? See examples/re180_closed/config_cfl02.yaml for the hypothesis.
cd /home/giorgio/torChannel
export PYTORCH_JIT=0
export TORCHANNEL_COMPILE=1
export TORCHANNEL_POISSON_CUDAGRAPH=1
python main.py examples/re180_closed/config_cfl02.yaml
