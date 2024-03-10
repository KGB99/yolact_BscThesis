#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=4:00:00

module load gcc/8.2.0 python_gpu/3.11.2
source myenv/bin/activate
python3 eval.py --trained_model=weights/pre_error_pbr_with_ssd_all_0_5000.pth --config=pbr_with_ssd_all --output_coco_json --bbox_det_file=results/pbr_with_ssd_bbox_detections.json --mask_det_file=results/pbr_with_ssd_mask_detections.json --dataset=trial_quant_eval --score_threshold=0.0 --top_k=5
deactivate
