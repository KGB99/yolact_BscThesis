from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
import json
from skimage import measure
import scipy.ndimage as ndi

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
    #return (masks[0] * torch.ones_like(masks[0]) * 255).cpu().numpy()
    #cv2.imwrite('./testerOutput/puremasks.png',((masks[0] * torch.ones_like(masks[0])) * 255).cpu().numpy())
    #exit()
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
        #cv2.imwrite('./testerOutput/puremasks.png',((masks[0] * torch.ones_like(masks[0])) * 255).cpu().numpy())
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


    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    #cv2.imwrite('./testerOutput/checkup.png',img_numpy)
    
    if num_dets_to_consider == 0:
        return img_numpy


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

# if sum_all is set to True then only the first mask has valuable information and the rest is not important, 
# otherwise each mask part has its own array element
def get_yolact_preds_aux(dets_out, img, h, w, crop_masks=False, sum_all=True, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
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
                                    crop_masks        = crop_masks,
                                    score_threshold   = SCORE_THRESHOLD)
    cfg.rescore_bbox = save

    idx = t[1].argsort(0, descending=True)[:TOP_K]
    
    if cfg.eval_mask_branch:
        # Masks are drawn on the GPU, so don't copy
        masks = t[3][idx]
    classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
    result_masks = []
    for i in range(0, len(masks)):
        result_masks.append((masks[i] * torch.ones_like(masks[i]) * 255).cpu().numpy())
    result_masks = [mask.astype(np.uint8) for mask in result_masks]
    #if crop_bbox:
    #    result_masks = crop_bboxes(result_masks, boxes)

    # sum_all DISCONTINUED FOR NOW DONT USE!
    if sum_all:
        for i in range(1, len(masks)):
            result_masks[0] = np.bitwise_or(result_masks[0], result_masks[i])
        # this is so that we only return relevant masks which is index 0
        result_masks = [result_masks[0]]

    results = []
    for i in range(0, len(result_masks)):
        #(classes,scores,boxes,result_masks)
        curr_dict ={}
        curr_dict['class'] = classes[i]
        curr_dict['score'] = scores[i]
        curr_dict['bbox'] = boxes[i]
        curr_dict['mask'] = result_masks[i]
        results.append(curr_dict)
    return results

# returns np.ndarray in shape (1080,1280) with elements of type np.uint8
def get_yolact_preds(img_path:str, sum_all, crop_masks):
    # TODO: predict bbox and segmentation using YOLACT
    frame = torch.from_numpy(cv2.imread(img_path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)
    # shape = (1080, 1280)
    #img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
    #cv2.imwrite("./testerOutput/" + str(i) + "_" + str(j) + ".png", img_numpy)
    result_preds = get_yolact_preds_aux(preds, frame, None, None, undo_transform=False, sum_all=sum_all, crop_masks=crop_masks)
    return result_preds

def sampleYolact():
    #TODO
    return

# input: image path, output: list of masks from segment anythings everything mode
def segmentEverything(img_path:str, anything_generator):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = anything_generator.generate(image)
    # type masks = <class 'numpy.ndarray'>
    # shape masks = (1080, 1280)
    # so these are all bitmasks? that would be awesome
    
    return masks

def calculateIoU(mask1, mask2):
    intersection = float(np.sum(np.bitwise_and(mask1,mask2)))
    union = float(np.sum(np.bitwise_or(mask1,mask2)))
    return (intersection/union)

def iou_filter(yolact_blobs, sa_masks):
    result_mask = np.zeros_like(yolact_blobs[0])
    for blob in yolact_blobs:
        for sa_dict in sa_masks:
            sa_mask = sa_dict['segmentation']
            iou = calculateIoU(blob, sa_mask)
            if iou > IOU_THRESHOLD:
                result_mask = np.bitwise_or(result_mask, sa_mask)
    return result_mask

def calculate_blobs(mask):
    labels, num_labels = ndi.label(mask)
    blobs = []
    for i in range(1,num_labels + 1):
        temp_blob = (labels == i)
        blobs.append(temp_blob)
    # this removes the 1 dimensional part of the blobs, should i rm it??
    blobs = [blob.squeeze() for blob in blobs]
    return blobs

# facebooks show masks method from their notebook:
# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

IOU_THRESHOLD = 0.1
SCORE_THRESHOLD = 0.25
TOP_K = 5
SEGMENT_SAMPLE = False
SEGMENT_EVERYTHING = True
USE_YOLACT = True
TAKE_MAX_PREDS = False

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Sampling from Yolact masks for Segment Anything")
    parser.add_argument('--config', default=None, required=False,
                        help='The config object to use.')
    parser.add_argument('--trained_model', required=True, type=str)
    parser.add_argument('--coco_file', required=True, type=str)
    parser.add_argument('--images_dir', required=True, type=str)
    parser.add_argument('--sa_model', required=False, type=str, default='sam_vit_h_4b8939.pth')
    args = parser.parse_args()
    images_dir = args.images_dir

    if USE_YOLACT:
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

    if SEGMENT_EVERYTHING:
        sam_checkpoint = 'SA_models/' + args.sa_model
        model_type = 'vit_h' if args.sa_model == 'sam_vit_h_4b8939.pth' else 'vit_l'
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device = device)
        anything_generator = SamAutomaticMaskGenerator(sam)

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
            if j < 1000:
                continue
            if j > 1005:
                exit()

            print("Camera:" + str(i+1) + "/" + str(len_coco_dict) + \
                " | Image:" + str(j+1) + "/" + str(len_camera_dict), flush=True)
            
            img_dict = camera_dict[imageId]['img']
            mask_dict = camera_dict[imageId]['mask']
            img_path = images_dir + "/" + img_dict['file_name']

            # this part is to get the blobs from the yolact predictions
            with torch.no_grad():
                # this is a list of dicts containing keys (class,score,bbox,mask)
                pre_yolact_preds = get_yolact_preds(img_path, sum_all=False, crop_masks=True)
                if TAKE_MAX_PREDS:
                    yolact_preds = []
                    powerdrill_max_conf = -1
                    powerdrill_dict = {}
                    screwdriver_max_conf = -1
                    screwdriver_dict = {}
                    for pred in pre_yolact_preds:
                        # MEDICAL_CLASSES = ('powerdrill', 'screwdriver')
                        # powerdrill = 0, screwdriver = 1
                        if pred['class'] == 0:
                            if powerdrill_max_conf < pred['score']:
                                powerdrill_max_conf = pred['score']
                                powerdrill_dict = pred
                        else:
                            if screwdriver_max_conf < pred['score']:
                                screwdriver_max_conf = pred['score']
                                screwdriver_dict = pred
                    yolact_preds.append(powerdrill_dict)
                    yolact_preds.append(screwdriver_dict)
                else:
                    yolact_preds = pre_yolact_preds

                for yolact_pred in yolact_preds:
                    yolact_pred['blobs'] = calculate_blobs(yolact_pred['mask'])

            if SEGMENT_SAMPLE:
                with torch.no_grad():
                    sampleYolact()
                exit()

            if SEGMENT_EVERYTHING:
                sa_masks = segmentEverything(img_path, anything_generator)
                image = cv2.imread(img_path)
                plt.imshow(image)
                show_anns(sa_masks)
                plt.savefig('testerOutput/SegmentAnything_image.png')
                result_masks = []
                for i,yolact_pred in enumerate(yolact_preds):
                    result_mask = iou_filter(yolact_pred['blobs'], sa_masks)
                    mask_rgb = (cv2.cvtColor(result_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([255, 0, 0], dtype=np.uint8)
                    result = cv2.addWeighted(image, 1.0, mask_rgb, 0.5, 0)
                    cv2.imwrite('testerOutput/filteredImages_' + str(i) + '.png', result)
                exit()

    print('OK!')

    exit()



    #everything that follows is old code that i might need again 
    if False:
        image = cv2.imread(img_path)
        h, w, colors_dimension = image.shape

        keepers=[]
        for i,score in enumerate(pred_score):
            if score < SCORE_THRESHOLD:
                continue
            keepers.append(i)

            if False:
                # assuming bbox is in (x,y,w,h) in relation to total image width
                pr_x = int(pred_bbox[i][0].item() * w)
                pr_y = int(pred_bbox[i][1].item() * h)
                pr_h = int(pred_bbox[i][2].item() * w)
                pr_w = int(pred_bbox[i][3].item() * h)
                
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


    image = cv2.imread(img_path)
    #dets = preds[0]
    #dets = dets['detection']
    #proto_data = dets['proto']
    #pred_bbox = dets['box']
    #pred_mask = dets['mask']
    #masks = proto_data @ pred_mask.t()
    #masks = cfg.mask_proto_mask_activation(masks)
    #masks = masks.permute(2, 0, 1).contiguous()
    #print(masks.shape)
    #exit()
    #pred_class = dets['class']
    #pred_score = dets['score']
    #pred_proto = dets['proto']
    h,w,colors_dimension = frame.shape
    t = postprocess(preds, w, h, visualize_lincomb = False, crop_masks = False, score_threshold = SCORE_THRESHOLD)
    idx = t[1].argsort(0, descending=True)[:TOP_K]
    masks = t[3][idx]
    classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
    #cv2.imwrite('./testerOutput/puremasks.png',((masks[0] * torch.ones_like(masks[0])) * 255).cpu().numpy())
    bbox = boxes.squeeze()
    bbox_image = np.zeros_like(image)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255),3)
    bbox_result = cv2.addWeighted(image, 1, bbox_image, 0.5, 0)
    masks_bitmask = (masks[0] * torch.ones_like(masks[0]) * 255).cpu().numpy()
    contours = measure.find_contours(masks_bitmask, 0.5, positive_orientation='low')
    print(len(contours))
    #masks_image = np.ones_like(image)
    #masks_image = cv2.threshold(masks_bitmask, 128, 255, cv2.THRESH_BINARY)
    #masks_image = cv2.cvtColor(masks_image, cv2.COLOR_BGR2GRAY)
    #masks_image = cv2.threshold(masks_image, 128, 255, 0)
    #img2,masks_contours,hierarchy = cv2.findContours(masks_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(masks_contours))
    exit()