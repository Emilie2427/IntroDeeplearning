#!/bin/bash
#SBATCH --job-name=image_captioning
#SBATCH --output=caption_%j.out
#SBATCH --error=caption_%j.err
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Charger les modules système pour les GPU et CUDA
module purge
module load cuda  # ou module load cuda/12.1 (selon ce qui est disponible sur votre cluster)

# Lancer le script via mamba
mamba run -n deeplearning python train.py