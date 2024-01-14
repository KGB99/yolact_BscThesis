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
from data import COCODetection, get_label_map, MEANS, COLORS
import torch.backends.cudnn as cudnn
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from layers.output_utils import postprocess, undo_image_transformation
from collections import defaultdict

SCORE_THRESHOLD = 0.15
TOP_K = 15

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        #enters here
        img_gpu = img / 255.0
        #cv2.imwrite('./testerOutput/checkup.png',(img_gpu*255.0).cpu().numpy())
        h, w, _ = img.shape
    

    save = cfg.rescore_bbox
    cfg.rescore_bbox = True
    t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                    crop_masks        = False,
                                    score_threshold   = SCORE_THRESHOLD)
    cfg.rescore_bbox = save

    idx = t[1].argsort(0, descending=True)[:TOP_K]
    
    if cfg.eval_mask_branch:
        # Masks are drawn on the GPU, so don't copy
        masks = t[3][idx]
    classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(TOP_K, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < SCORE_THRESHOLD:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if True and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        # i think masks[0] is a bit mask where its either 0 or 1
        #cv2.imwrite('./testerOutput/puremasks.png',((img_gpu * masks[0]) * 255).cpu().numpy())
        print(masks[1].shape)
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        #cv2.imwrite('./testerOutput/inv_alph_masks.png',((img_gpu * inv_alph_masks[0]) * 255).cpu().numpy())
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        #cv2.imwrite('./testerOutput/checkOutput.png',((masks_color_summand) * 255).cpu().numpy())
        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        
    if False:
            # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    #cv2.imwrite('./testerOutput/checkup.png',img_numpy)
    if False:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    if num_dets_to_consider == 0:
        return img_numpy

    if True:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            
            cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

        
            _class = cfg.dataset.class_names[classes[j]]
            text_str = '%s: %.2f' % (_class, score) if True else _class

            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1

            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

            text_pt = (x1, y1 - 3)
            text_color = [255, 255, 255]

            cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
            cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
            
    
    return img_numpy


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Sampling from Yolact masks for Segment Anything")
    parser.add_argument('--config', default=None, required=False,
                        help='The config object to use.')
    parser.add_argument('--trained_model', required=True, type=str)
    parser.add_argument('--coco_file', required=True, type=str)
    parser.add_argument('--images_dir', required=True, type=str)
    args = parser.parse_args()
    images_dir = args.images_dir
    print("Setting configs...",end='', flush=True)
    set_cfg("sample_config")    
    print(" Done.", flush=True)

    with torch.no_grad():
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        #else:
        #    torch.set_default_tensor_type('torch.FloatTensor')
        #    dataset = None        

        print('Loading model...', end='',flush=True)
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.', flush=True)

        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False
        print('Loading annotations...', end='',flush=True)
        f = open(args.coco_file)
        coco_dict = json.load(f)
        f.close()
        print(' Done.', flush=True)

        len_coco_dict = len(coco_dict)
        for i,camera in enumerate(coco_dict):
            camera_dict = coco_dict[camera]
            len_camera_dict = len(camera_dict)
            for j,imageId in enumerate(camera_dict):
                if j < 100:
                    continue
                if j > 105:
                    exit()

                print("Camera:" + str(i) + "/" + str(len_coco_dict) + \
                    " | Image:" + str(j) + "/" + str(len_camera_dict), flush=True)
                
                img_dict = camera_dict[imageId]['img']
                mask_dict = camera_dict[imageId]['mask']
                img_path = images_dir + "/" + img_dict['file_name']

                #image = cv2.imread(img_path)
                # TODO: predict bbox and segmentation using YOLACT
                frame = torch.from_numpy(cv2.imread(img_path)).cuda().float()
                batch = FastBaseTransform()(frame.unsqueeze(0))
                preds = net(batch)
                dets = preds[0]
                dets = dets['detection']
                pred_bbox = dets['box']
                pred_mask = dets['mask']
                pred_class = dets['class']
                pred_score = dets['score']
                pred_proto = dets['proto']
                if False:
                    image = cv2.imread(img_path)
                    height, width, colors_dimension = image.shape

                    keepers=[]
                    for i,score in enumerate(pred_score):
                        if score < SCORE_THRESHOLD:
                            continue
                        keepers.append(i)

                        if False:
                            # assuming bbox is in (x,y,w,h) in relation to total image width
                            pr_x = int(pred_bbox[i][0].item() * width)
                            pr_y = int(pred_bbox[i][1].item() * height)
                            pr_h = int(pred_bbox[i][2].item() * width)
                            pr_w = int(pred_bbox[i][3].item() * height)
                            
                            bbox_image = np.zeros_like(image)
                            #print(pr_x,pr_y,pr_h,pr_w)
                            cv2.rectangle(bbox_image, (pr_x, pr_y), (pr_h, pr_w), (0,0,255),3)
                            bbox_result = cv2.addWeighted(image, 1, bbox_image, 0.5, 0)

                        mask_image = np.zeros_like(image)
                        print(pred_mask[i].shape)
                        print(pred_proto[i].shape)
                        mask_result = cv2.addWeighted(image, 1, mask_image, 0.5, 0)
                        cv2.imwrite("./testerOutput/" + str(i) + "_" + str(j) + ".png", mask_result)
                    exit()


                    
                
                #for id in keepers:
                    #print(pred_mask[id])
                    #print(pred_bbox[id])
                    

                # TODO: visualize prediction
                img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
        
                #if save_path is None:
                #img_numpy = img_numpy[:, :, (2, 1, 0)]

                #if save_path is None:
                #    plt.imshow(img_numpy)
                #    plt.title(path)
                #    plt.show()
                #else:

                cv2.imwrite("./testerOutput/" + str(i) + "_" + str(j) + ".png", img_numpy)
                exit()
                # TODO: read gt_mask

                # TODO: sample a point from mask

                # TODO: show sampled point in image
                    


    
    print('OK!')