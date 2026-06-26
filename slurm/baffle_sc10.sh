#!/bin/bash
#SBATCH -J baffle_sc10
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/R-%x.%j.out

# GB10: JIT off (nvrtc arch), unbuffered output for live monitoring
export PYTORCH_JIT=0
export PYTHONUNBUFFERED=1
cd /home/giorgio/torChannel

PY=/home/giorgio/.conda/envs/cans_drl/bin/python
echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  $(date)"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Baffle-only (short developing duct, SMOOTH circular wall + Koch baffle inlet interface):
# the steady M(x) profile carries the near-inlet N-dependence. Fixed dt=5e-4.
# Early-stop on STEADY STATE (drift<2e-5).
$PY scripts/mixing_campaign.py --mode baffle --Sc 10 --dt 5e-4 \
    --Ns 0 1 2 3 4 --check 500 --snap 4000 --drift_tol 2e-5 \
    --min_steps 4000 --max_steps 120000 --outdir results/campaign
echo "done $(date)"
