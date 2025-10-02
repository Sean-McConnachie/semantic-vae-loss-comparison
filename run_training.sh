#!/bin/bash

# ======== SLURM Job Configuration ========
#SBATCH --job-name=vqgan-seg
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --output=/data/smcc417/taming-transformers/runs/output.log
#SBATCH --error=/data/smcc417/taming-transformers/runs/error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

# ======== Job Execution Steps ========

# Set default WANDB project
WANDB_PROJECT="cluster"

if [ "$2" == "testing" ]; then
  WANDB_PROJECT="testing"
fi

export WANDB_PROJECT

# Navigate to the working directory and run the script
source /home/smcc417/.bashrc
cd /data/smcc417/taming-transformers
conda activate taming
python main.py --base $1 --gpus 0, -t True --max_epochs 10