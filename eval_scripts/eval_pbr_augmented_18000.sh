#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=4:00:00

module load gcc/8.2.0 python_gpu/3.11.2
source myenv/bin/activate
python3 eval.py --trained_model=weights/pbr_random_and_kinect_augmented_3_18000.pth --config=train_pbr_random_and_kinect_augmentations --output_coco_json --bbox_det_file=results/pbr_augmented_18000_bbox_detections.json --mask_det_file=results/pbr_augmented_18000_mask_detections.json --dataset=trial_quant_eval --score_threshold=0.0 --top_k=5
deactivate
