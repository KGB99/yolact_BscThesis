#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=4:00:00

module load gcc/8.2.0 python_gpu/3.11.2
python3 eval.py --trained_model=weights/pbr_random_and_kinect_0_1500.pth --config=train_pbr_random_and_kinect_1500 --output_coco_json --bbox_det_file=results/quick_pbr_bbox_detections.json --mask_det_file=results/quick_pbr_mask_detections.json --dataset=test_pbr_random_lighting --score_threshold=0.2 --top_k=5