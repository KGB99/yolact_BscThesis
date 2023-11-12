#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00

module load gcc/8.2.0 python_gpu/3.11.2
python3 eval.py --trained_model=weights/medical_resnet50_499_500.pth --images=data/data_sample/images:data/data_sample/results --output_coco_json --config=medical_subset_resnet50_config
