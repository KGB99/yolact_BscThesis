#!/bin/bash
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=24:00:00
#module load gcc/8.2.0 python_gpu/3.11.2 
module load gcc/8.2.0 python_gpu/3.8.5 gdal/3.1.2 geos/3.6.2
source sampling_env/bin/activate
python3 sampling_experiments.py --coco_file=/cluster/project/infk/cvg/heinj/students/kbirgi/Annotations/trainSSD/amodal_labels_25.json --images_dir=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp --results_dir=stride_25_pbr_base_30000 --trained_model=weights/pbr_random_and_kinect_5_30000.pth --create_labels=True --visualize_masks=False
deactivate
