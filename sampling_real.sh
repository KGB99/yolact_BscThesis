#!/bin/bash
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=40G
#SBATCH --time=4:00:00
module load gcc/8.2.0 python_gpu/3.11.2
python3 sampling_experiments.py --images_dir=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp --trained_model=weights/modal_ssd_resnet50_0_5000.pth --coco_file=/cluster/project/infk/cvg/heinj/students/kbirgi/Annotations/testAll/subset_amodal_labels.json

