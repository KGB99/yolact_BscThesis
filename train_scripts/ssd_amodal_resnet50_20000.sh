#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=24:00:00

module load gcc/8.2.0 python_gpu/3.11.2
python3 train.py --batch_size=8 --num_workers=1 --config=ssd_amodal_resnet50_20000_config
# batch_size=8*(num_GPUS)
