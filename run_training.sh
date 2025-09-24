#!/bin/bash

# ======== SLURM Job Configuration ========
#SBATCH --job-name=vqgan-seg
#SBATCH --time=01:00:00                   # [REQUIRED] Wall time limit in HH:MM:SS
#SBATCH --open-mode=append
#SBATCH --output=/data/smcc417/output.log
#SBATCH --error=/data/smcc417/error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# ======== Job Execution Steps ========

# Navigate to the working directory where your code and virtual environment are located
cd /data/smcc417/taming-transformers
conda activate taming
python main.py --base configs/coco_cond_stage.yaml --gpus 0, -t True