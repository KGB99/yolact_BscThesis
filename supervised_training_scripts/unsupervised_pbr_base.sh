#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=24:00:00

module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy
source myenv/bin/activate
python3 train.py --batch_size=8 --num_workers=1 --config=self_supervised_pbr_base --save_interval=1000 --resume=weights/pbr_random_and_kinect_5_30000.pth --start_iter=30000
deactivate
