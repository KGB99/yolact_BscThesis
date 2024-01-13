from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse

import sys
import os
#sys.path.append(os.path.abspath('/cluster/project/infk/cvg/heinj/students/kbirgi/yolact_BscThesis'))
from yolact import Yolact
from data import cfg, set_cfg, set_dataset
import torch.backends.cudnn as cudnn


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Sampling from Yolact masks for Segment Anything")
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    args = parser.parse_args()

    if args.config is not None:
        set_cfg(args.config)

    #if args.trained_model == 'interrupt':
    #    args.trained_model = SavePath.get_interrupt('weights/')
    #elif args.trained_model == 'latest':
    #    args.trained_model = SavePath.get_latest('weights/', cfg.name)

    with torch.no_grad():
        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            dataset = None        

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        print("\n" + str(args.trained_model))
        print(type(args.trained_model))
        net.eval()
        print(' Done.')
    exit()


    net = Yolact()
    net.load_weights('/cluster/project/infk/cvg/heinj/students/kbirgi/yolact_BscThesis/weights/pbr_modal_resnet50_old_3_20000.pth')
    print('weights loaded...')
    if torch.cuda.is_available():
        net.cuda()
    
    print('OK!')