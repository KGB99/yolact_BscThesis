#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=20G
#SBATCH --time=3:00:00

module load gcc/8.2.0 python_gpu/3.11.2
python3 train.py --batch_size=16 --num_workers=2 --config=medical_resnet101_config
# we take batch size 16 because we use 2 GPUS and batch_size=8*(num_GPUS)
