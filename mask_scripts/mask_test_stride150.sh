#!/bin/bash
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=24:00:00
module load gcc/8.2.0 python_gpu/3.11.2
source myenv/bin/activate
python3 sampling_experiments.py --coco_file=/cluster/project/infk/cvg/heinj/students/kbirgi/Annotations/testAll/stride150_amodal_labels.json --images_dir=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp --results_dir=hue_no_noise --trained_model=weights/no_noise_pbr_rand_kinect_aug_7_40000.pth
deactivate