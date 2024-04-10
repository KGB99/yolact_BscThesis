#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=48:00:00
#SBATCH --gres=gpumem:24g

module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy
source myenv/bin/activate
python3 train.py --batch_size=8 --num_workers=1 --config=pbr_with_real_all --save_interval=3000
deactivate
