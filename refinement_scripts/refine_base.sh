#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=24:00:00

module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy
source myenv/bin/activate
python3 train.py --batch_size=8 --num_workers=1 --resume=pbr_random_and_kinect_refinement_base_3_20000.pth --start_epoch=3 --start_iter=20000 --config=refinement_pbr_kinect_random_real_1000 --refinement_mode=True
deactivate