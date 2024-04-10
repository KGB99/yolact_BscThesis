#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=4:00:00

module load gcc/8.2.0 python_gpu/3.11.2
source myenv/bin/activate
python3 eval.py --trained_model=weights/self_supervised_pbr_real_142_35000.pth --config=ref_self_supervised_pbr_ref --output_coco_json --bbox_det_file=results/ref_unsupervised_real_35000_bbox_detections.json --mask_det_file=results/ref_unsupervised_real_35000_mask_detections.json --dataset=trial_quant_eval --score_threshold=0.0 --top_k=5
deactivate
