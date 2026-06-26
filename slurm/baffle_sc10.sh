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

# Baffle-only (temporal box, smooth square duct, Koch baffle IC): the diffusive-limit
# test of Eq. 4. Fixed dt=4e-4 (verified div ~1e-13). Early-stop at M<0.05.
$PY scripts/mixing_campaign.py --mode baffle --Sc 10 --dt 4e-4 \
    --Ns 0 1 2 3 4 --check 1000 --snap 20000 --M_stop 0.05 \
    --min_steps 5000 --max_steps 500000 --outdir results/campaign
echo "done $(date)"
