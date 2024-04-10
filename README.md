The original docs of the yolact repo contains some informations that I might not list here, hence if anything is unclear it might be worth looking there and reading the original readme file. You can access it [here](https://github.com/dbolya/yolact).

# Recreating Base model Results from the Thesis
Note that all evaluation scripts merely provide the predictions of the yolact model. The predictions are evaluated afterwards in the [BachelorsThesis](https://github.com/KGB99/BachelorThesis) repo.
Essentially both training and evaluation is running a script on the cluster respectively, hence I will name the scripts that are linked to the presented models in the paper. There is of course a lot more scripts than just these due to experiments not presented in the paper. Further below I provide an explanation of everything in more thorough detail.
The train scripts are all in the train_scripts folder, except for Yolact-Mixed, thats in the refinement_scripts folder, and the eval scripts are all in the eval_scripts folder.

| Model name | Train script | Eval script | 
| ---------- | ------------ | ----------- |
| Yolact-Pbr | pbr_random_kinect_resnet50.sh | eval_pbr_30000.sh |
| Yolact-Real| ssd_amodal_resnet50_40000.sh | eval_ssd_amodal.sh |
| Yolact-Augmented | pbr_noise_hue_random_kinect_resnet50.sh | eval_pbr_augmented_30000.sh |
| Yolact-Mixed | refine_all_no_noise.sh | eval_real_no_noise_27000.sh | 

# Recreating Semi-Supervised model results from the Thesis
The scripts in the mask_scripts folder create the pseudo ground-truth annotations. Following that the models have to be trained with the scripts in the supervised_training_scripts folder.
sampling_stride_10_pbr_base.sh is to create masks with the yolact-pbr model.
sampling_refined_all_27000.sh is to create masks with the yolact-mixed model.
For evaluating use scripts in the eval_scripts folder, namely eval_unsupervised_base_33000.sh for the yolact-pbr model and eval_unsupervised_real_33000.sh for the yolact-mixed model.


# Training Yolact with a custom dataset
The code in this repo is adapted for using the dataset from [Hein et al.](https://arxiv.org/abs/2305.03535).
The original repo does a good explanation of explaining how to train on your own dataset, but it should be clear from what is written here as well.

### Command line training
The relevant packages and pip installs required can be found in the requirements.txt file.
In order to train yolact, we use the given train.py file and begin the training from the command line. When training from the starting weights the following command can be used:

```
python3 train.py --batch_size=8 --num_workers=1 --config=your_config_name
```

### ETH Cluster job submisison training
Due to the requirement of a gpu, I ran all of my trainings on the cluster.
To submit it as a slurm job many examples are provided in the train_scripts folder. The modules gcc/8.2.0,python_gpu/3.11.2, and eth_proxy are enough to run train.py, the eth_proxy is only used for wandb and can be excluded otherwise. Furthermore, for the image augmentation during training you need to use a virtual env with imgaug and imgaug.corruptors packages, if you do not wish to do the stronger augmentations you can skip the virtual python environment lines (namely source... and deactivate at the end).
To provide an example, here is the script for the Yolact_pbr model from my thesis:

```
#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=48:00:00

module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy
source myenv/bin/activate
python3 train.py --batch_size=8 --num_workers=1 --config=train_pbr_random_and_kinect_hue_40000
deactivate
```
## Config file and Annotations
Note that the config name in the above commands needs to be adapted to your own configs name, which you must define in data/config.py. The MEDICAL_CLASSES include the powerdrill and screwdriver.
To do this, copying the base data config and replacing the following infos is enough:

```
your_config = dataset_base.copy({
    'name' : 'your_config_name',
    'train_images' : '/location/to/your/training/images',
    'train_info' : '/location/to/your/training/json/file',
    'valid_images' : '/location/to/your/validation/images',
    'valid_info' : '/location/to/your/validation/json/file',
    'has_gt' : True,
    'class_names' : MEDICAL_CLASSES,
    'label_map' : None
})
```

Yolact reads its training data from a JSON file in the MS-COCO format. These are the "train_info" and valid_info" files you provide in the config. The format of these JSON's is as follows:

```
{
    "info": {"description" : "your datasets description"},
    "licenses": {},
    "images": [
        {
            "file_name": "image.jpg",
            "height": image height,
            "width": image width,
            "id": image id
        },
        ...
    ],
    "annotations": [
        {
            "segmentation": segmentation vertices,
            "area": area of segmentation,
            "iscrowd": 0,
            "image_id": id of corresponding image in the "images",
            "bbox": bbox vertices,
            "category_id": category id of the annotation,
            "id": annotation id
        },
        ...
    ]
}
```

The creation of these labels is done in the [BachelorsThesis](https://github.com/KGB99/BachelorThesis) repo and the format itself and the contents are explained more deeply there.
