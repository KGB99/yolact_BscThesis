from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
import json

import sys
import os
#sys.path.append(os.path.abspath('/cluster/project/infk/cvg/heinj/students/kbirgi/yolact_BscThesis'))
from yolact import Yolact
from data import cfg, set_cfg, set_dataset
import torch.backends.cudnn as cudnn


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Sampling from Yolact masks for Segment Anything")
    parser.add_argument('--config', default=None, required=False,
                        help='The config object to use.')
    parser.add_argument('--trained_model', required=True, type=str)
    parser.add_argument('--coco_file', required=True, type=str)
    parser.add_argument('--images_dir', required=True, type=str)
    parser.add_argument('--labels_dir', required=True, type=str)
    args = parser.parse_args()
    print("Setting configs...", flush=True)
    set_cfg("sample_config")    
    print("Done!", flush=True)

    with torch.no_grad():
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        #else:
        #    torch.set_default_tensor_type('torch.FloatTensor')
        #    dataset = None        

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.', flush=True)

    f = open(args.coco_file)
    coco_dict = json.load(f)
    f.close()

    len_coco_dict = len(coco_dict)
    for i,camera in enumerate(coco_dict):
        camera_dict = coco_dict[camera]
        len_camera_dict = len(camera_dict)
        for j,imageId in enumerate(camera_dict):
            print("Camera:" + str(i) + "/" + str(len_coco_dict) + \
                  " | Image:" + str(j) + "/" + str(len_camera_dict), flush=True)
            
            img_dict = camera_dict[imageId]['img']
            mask_dict = camera_dict[imageId]['mask']
            img_path = args.images_dir + img_dict['file_name']
            label_path = args.labels_dir + img_dict['file_name']

            image = cv2.imread(img_path)
            exit()
            # TODO: predict bbox and segmentation using YOLACT

            # TODO: read gt_mask

            # TODO: sample a point from mask

            # TODO: show sampled point in image



    
    print('OK!')