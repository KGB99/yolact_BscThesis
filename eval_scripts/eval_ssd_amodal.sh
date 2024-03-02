#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00

module load gcc/8.2.0 python_gpu/3.11.2
source myenv/bin/activate
python3 eval.py --trained_model=weights/ssd_amodal_resnet50_1_20000.pth  --config=ssd_amodal_resnet50_20000_config --output_coco_json --bbox_det_file=results/ssd_amodal_bbox_detections.json --mask_det_file=results/ssd_amodal_mask_detections.json --dataset=trial_quant_eval --score_threshold=0.0 --top_k=5
deactivate
