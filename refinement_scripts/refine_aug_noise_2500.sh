#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=24:00:00

module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy
source myenv/bin/activate
python3 train.py --batch_size=8 --num_workers=1 --resume=weights/pbr_random_and_kinect_aug_hue_5_30000.pth --start_epoch=5 --start_iter=30000 --config=refinement_pbr_aug_hue_2500 --refinement_mode=True --save_interval=250
deactivate
