#!/bin/bash
#SBATCH --job-name=re180_fine
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --output=slurm-re180-fine-%j.out

# Re_tau = 180 closed channel refined to dx+ = 9.4, dy+ = 4.7 (resolution study
# against examples/re180_closed/ at 11.8, 5.9). See that config for the evidence.
#
# Submit with a dependency so it starts only after the 192^2 run has written its
# final field, which is what this one interpolates from:
#     sbatch --dependency=afterok:<jobid> slurm/re180_closed_fine_gb10.sh
#
# Local GB10. PYTORCH_JIT=0: nvrtc arch workaround on GB10.
cd /home/giorgio/torChannel

export PYTORCH_JIT=0
export TORCHANNEL_COMPILE=1
export TORCHANNEL_POISSON_CUDAGRAPH=1

python main.py examples/re180_closed_fine/config.yaml
