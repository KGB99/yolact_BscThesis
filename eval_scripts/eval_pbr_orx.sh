#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00

module load gcc/8.2.0 python_gpu/3.11.2
#python3 eval.py --trained_model=weights/pbr_modal_resnet50_old_3_20000.pth  --config=pbr_modal_resnet50_20000 --output_coco_json --bbox_det_file=results/pbr_orx_bbox_detections.json --mask_det_file=results/pbr_orx_mask_detections.json --dataset=pbr_modal --score_threshold=0.15 --top_k=15
python3 eval.py --trained_model=weights/pbr_modal_resnet50_old_3_20000.pth --output_coco_json --bbox_det_file=results/pbr_orx_bbox_detections.json --mask_det_file=results/pbr_orx_mask_detections.json --dataset=pbr_modal --score_threshold=0.15 --top_k=15