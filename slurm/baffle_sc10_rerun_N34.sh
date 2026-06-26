#!/bin/bash
#SBATCH -J baffle_sc10_N34
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/R-%x.%j.out

# Rerun of baffle N=3,4 ONLY, after fixing the Koch-interface sign artifact
# (open-polyline nearest-segment sign -> robust point-in-polygon in scalar.py).
# N=0,1,2 were unaffected and are kept. N=3 cold-starts; N=4 warm-starts from N=3.
export PYTORCH_JIT=0
export PYTHONUNBUFFERED=1
cd /home/giorgio/torChannel

PY=/home/giorgio/.conda/envs/cans_drl/bin/python
echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  $(date)"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

$PY scripts/mixing_campaign.py --mode baffle --Sc 10 --dt 5e-4 \
    --Ns 3 4 --check 500 --snap 4000 --drift_tol 2e-5 \
    --min_steps 12000 --max_steps 120000 --outdir results/campaign
echo "done $(date)"
