#!/bin/bash

# ======== SLURM Job Configuration ========
#SBATCH --job-name=vqgan-seg
#SBATCH --time=24:00:00                   # [REQUIRED] Wall time limit in HH:MM:SS
#SBATCH --open-mode=append
#SBATCH --output=/data/smcc417/taming-transformers/runs/output.log
#SBATCH --error=/data/smcc417/taming-transformers/runs/error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

# ======== Job Execution Steps ========

export WANDB_PROJECT=cluster

# Navigate to the working directory where your code and virtual environment are located
source /home/smcc417/.bashrc
cd /data/smcc417/taming-transformers
conda activate taming
python main.py --base $1 --gpus 0, -t True --max_epochs 10