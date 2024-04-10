#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=4:00:00

module load gcc/8.2.0 python_gpu/3.11.2
source myenv/bin/activate
python3 eval.py --trained_model=weights/self_supervised_pbr_real_122_30000.pth --config=self_supervised_pbr_real --output_coco_json --bbox_det_file=results/unsupervised_real_30000_bbox_detections.json --mask_det_file=results/unsupervised_real_30000_mask_detections.json --dataset=trial_quant_eval --score_threshold=0.0 --top_k=5
deactivate
