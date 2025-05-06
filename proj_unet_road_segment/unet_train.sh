#!/bin/bash

#$ -P ec523
#$ -N unet_training
#$ -l h_rt=12:00:00
#$ -l gpu=2
#$ -l gpu_memory=48G
#$ -l gpu_c=7.0
#$ -cwd
#$ -j y
#$ -o unet_training.txt
#$ -m e
# Load necessary modules
module load miniconda

# Activate your virtual environment if needed
source /projectnb/ec523/projects/Proj_road_segment_fix/shared_envs/unetCompare/bin/activate
# Run your training script
module load miniconda
source /share/pkg.7/miniconda/4.12.0/install/etc/profile.d/conda.sh
conda activate /projectnb/ec523/projects/Proj_road_segment_fix/shared_envs/unetCompare
python train.py