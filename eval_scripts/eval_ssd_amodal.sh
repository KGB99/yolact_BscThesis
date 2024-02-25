#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00

module load gcc/8.2.0 python_gpu/3.11.2
source myenv/bin/activate
python3 eval.py --trained_model=weights/medical_resnet50_0_5000.pth  --output_coco_json --bbox_det_file=results/ssd_amodal_bbox_detections.json --mask_det_file=results/ssd_amodal_mask_detections.json --dataset=medical_test_ssd --score_threshold=0.2 --top_k=5
deactivate
