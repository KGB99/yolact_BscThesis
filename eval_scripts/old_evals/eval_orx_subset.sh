#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00

module load gcc/8.2.0 python_gpu/3.11.2
#python3 eval.py --trained_model=weights/medical_ssd_resnet50_0_5000.pth  --config=medical_ssd_resnet50_config --output_coco_json --images=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp/test/004000/rgb:/cluster/scratch/kbirgi/results/folder_trial --score_threshold=0.8
python3 eval.py --trained_model=weights/medical_ssd_resnet50_0_5000.pth  --config=medical_ssd_resnet50_config --output_coco_json --dataset=medical_test_orx_subset --score_threshold=0.15 --top_k=15
