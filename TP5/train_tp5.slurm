#!/bin/bash
#SBATCH --job-name=tp5-captioning
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=tp5_%j.out
#SBATCH --error=tp5_%j.err

cd "$SLURM_SUBMIT_DIR"

eval "$(mamba shell hook --shell bash)"
mamba activate deeplearning

python -u train.py