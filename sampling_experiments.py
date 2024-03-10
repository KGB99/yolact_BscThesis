from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
import json
from skimage import measure
import scipy.ndimage as ndi
import pickle

import time
import os
#sys.path.append(os.path.abspath('/cluster/project/infk/cvg/heinj/students/kbirgi/yolact_BscThesis'))
from yolact import Yolact
from data import cfg, set_cfg, set_dataset
from data import COCODetection, get_label_map, MEANS, COLORS
import torch.backends.cudnn as cudnn
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from layers.output_utils import postprocess, undo_image_transformation
from collections import defaultdict
import pycocotools.mask as maskUtils

from PIL import Image
from shapely.geometry import Polygon, MultiPolygon


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
                                    crop_masks        = CROP_MASKS,
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
    for i,blob in enumerate(yolact_blobs):
        for j,sa_dict in enumerate(sa_masks):
            sa_mask = sa_dict['segmentation']
            iou = calculateIoU(blob, sa_mask)
            if iou > IOU_THRESHOLD:
                #if sa_mask.dtype != np.uint8:
                    #sa_mask = sa_mask.astype(np.uint8)
                    #sa_mask_rgb = (cv2.cvtColor(sa_mask, cv2.COLOR_GRAY2BGR)) * np.array([0,0,255], dtype=np.uint8)
                    #cv2.imwrite('./testerOutput/sa_mask_' + str(i+j) + '.png', sa_mask_rgb)
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

def checkBbox(blob, bbox, h, w): 
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    bbox_image = np.zeros_like(image)
    cv2.circle(bbox_image, (x1,y1), 5, (0,0,255), -1)
    cv2.circle(bbox_image, (x1, y2), 5, (0,0,255), -1)
    cv2.circle(bbox_image, (x2, y1), 5, (0,0,255), -1)
    cv2.circle(bbox_image, (x2,y2), 5, (0,0,255), -1)
    for idx, elem in np.ndenumerate(blob):
        # idx[1] = x value (columns), idx[0] = y value (rows)
        if (idx[1] >= x1) & (idx[1] <= x2) & (idx[0] >= y1) & (idx[0] <= y2):
            cv2.circle(bbox_image, (idx[1],idx[0]), 2, (0,255,0), -1)
            if elem:
                return True
    result = cv2.addWeighted(image, 1, bbox_image, 0.5, 0)
    cv2.imwrite('testerOutput/bbox.png', result)
    return False


def crop_masks(yolact_preds, h, w):
    for pred in yolact_preds:
        keep = []
        blobs = pred['blobs']
        for blob in blobs:
            if checkBbox(blob, pred['bbox'], h, w):
                keep.append(blob)
        pred['blobs'] = keep
    return yolact_preds

# facebooks show masks method from their notebook:
# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    #ax = plt.gca()
    #ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    return img

def prep_path(results_path, img_path):
    temp_img_path = img_path
    temp_array = []
    while (os.path.split(temp_img_path)[1] != ''):
        temp_path_split = os.path.split(temp_img_path)
        temp_array.append(temp_path_split[1])
        temp_img_path = temp_path_split[0]
    temp_array.reverse()
    temp_path = results_path
    for i in range(len(temp_array) - 1):
        temp_path = temp_path + "/" + temp_array[i]
        if (not os.path.exists(temp_path)):
            os.mkdir(temp_path)
    return

def create_mask_annotation(image_path,APPROX):
    image = image_path#ski.io.imread(image_path)
    contour_list = measure.find_contours(image, positive_orientation='low')
    segmentations = []
    polygons = []
    poly = -1
    bbox = -1
    for contour in contour_list:
        for i in range(len(contour)):
            row,col = contour[i]
            contour[i] = (col-1,row-1)
        
        poly = Polygon(contour)
        if (poly.area <= 1): continue

        if APPROX:
            poly = poly.simplify(1.0, preserve_topology=False)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        #coords = np.array(poly.exterior.coords)
        #fig, ax = plt.subplots()
        #ax.plot(coords[:,0],coords[:,1])
        #plt.show()
        segmentations.append(segmentation)
        polygons.append(poly)
        
        multipoly = MultiPolygon(polygons)
        x1, y1, x2, y2 = multipoly.bounds
        bbox = (x1, y1, x2-x1, y2-y1)
    if (bbox == -1) or (poly == -1):
        return -1,-1,-1
    return segmentations, bbox, poly.area


IOU_THRESHOLD = 0.2 # threshold for how much iou the SA-masks need with yolact preds to be included in result
SCORE_THRESHOLD = 0.2 # threshold for yolact model
TOP_K = 5 # top-k for yolact model
SEGMENT_SAMPLE = False # use the sample method of segment anything, currently not implemented
SEGMENT_EVERYTHING = True # use segment everything mode of segment anything
USE_YOLACT = True # keep true unless you do not intend to load the yolact model
TAKE_MAX_PREDS = True # takes at most the highest scoring prediction per class, none if no prediction
CROP_MASKS = True # crop the masks or not
CROP_MASKS_PERSONAL = False # use personal cropping method which keeps blobs intact even if outside of bbox or yolact cropping method
SAVE_YOLACT_PREDS = True
SAVE_SA = True
SAVE_PLOTS = True # TODO: add details to plots like iou and legend
MAX_IMAGE = 0 # max amount of images per folder, set to 0 for all
USE_PRECALC_SA = False # set true if precalculated pickle files from Segment anything exist
CHOSEN_SCENES = ['001004'] # Write the camera angles you wish to process, discontinued rn
CREATE_TRAINING_LABELS = True # Write true if you want to create training labels for yolact with generated masks
ONLY_ONE_TOOL = True
VISUALIZE_GEN_MASKS = True

def create_labels():
    labels_dict = {} # for the generated labels
    print('Loading annotations...', end='',flush=True)
    f = open(args.coco_file)
    coco_dict = json.load(f) #gt labels
    f.close()
    print(' Done.', flush=True)


    len_coco_dict = len(coco_dict)
    passed = False
    for i,camera in enumerate(coco_dict):
        if ((not passed) & ((i+1) < start_cam)):
            continue

        #print("BEWARE: USING CHOSEN SCENES FROM CODE! NO OTHER CAMERA ANGLES WILL BE PROCESSED!")
        #if not (camera in CHOSEN_SCENES):
        #    continue
        if not (camera in labels_dict):
            labels_dict[camera] = {}

        camera_dict = coco_dict[camera]
        len_camera_dict = len(camera_dict)
        for j,imageId in enumerate(camera_dict):
            print(camera_dict[imageId]['img']['file_name'])
            #if ((not passed) & ((j+1) < start_img)):
             #   continue
            #passed = True

            #if ((j % stride) != 0):
            #    continue

            start_time = time.time()
            print("Camera:" + str(i+1) + "/" + str(len_coco_dict) + \
                " | Image:" + str(j+1) + "/" + str(len_camera_dict), end = '')

            img_dict = camera_dict[imageId]['img'] # keys: ['id', 'width', 'height', 'file_name']
            mask_dict = camera_dict[imageId]['mask'] # keys: ['segmentation', 'bbox', 'area', 'iscrowd', 'image_id', 'category_id', 'id'] 
            img_path = images_dir + "/" + img_dict['file_name']
            image = cv2.imread(img_path)
            h, w, _ = image.shape
            
            #prep_path(results_path, img_dict['file_name'])
            #img_results_path = results_path + '/' + img_dict['file_name']
            print(' | Processing times: ', end='')
            yolact_time_begin = time.time()
            with torch.no_grad():
                # if we only want the yolact preds and no processing
                
                #if SAVE_YOLACT_PREDS:
                #    prep_path(yolact_path, img_dict['file_name'])
                #    yolact_preds_path = yolact_path + '/' + img_dict['file_name']
                #    frame = torch.from_numpy(cv2.imread(img_path)).cuda().float()
                #    batch = FastBaseTransform()(frame.unsqueeze(0))
                #    preds = net(batch)
                #    img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
                #    #cv2.imwrite('./testerOutput/yolactPred.png', img_numpy)
                #    img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
                #    axs[0,1].imshow(img_numpy)
                #    axs[0,1].axis('off')
                #    axs[0,1].set_title('Yolact Prediction')
                
                
                # pre_yolact_preds is a list of dicts containing keys (class,score,bbox,mask)
                pre_yolact_preds = get_yolact_preds(img_path, sum_all=False, crop_masks=(CROP_MASKS & (not CROP_MASKS_PERSONAL)))
                if TAKE_MAX_PREDS:
                    yolact_preds = []
                    powerdrill_max_conf = -1
                    powerdrill_dict = None
                    screwdriver_max_conf = -1
                    screwdriver_dict = None
                    object_id = -1
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
                    if ONLY_ONE_TOOL: #incase we only want 1 prediction
                        if ((powerdrill_dict != None) & (screwdriver_dict != None)):
                            if screwdriver_max_conf > powerdrill_max_conf:
                                yolact_preds = [screwdriver_dict]
                            else:
                                yolact_preds= [powerdrill_dict]
                        else:
                            if powerdrill_dict != None:
                                yolact_preds.append(powerdrill_dict)
                            if screwdriver_dict != None:
                                yolact_preds.append(screwdriver_dict)
                    else:
                        if powerdrill_dict != None:
                            yolact_preds.append(powerdrill_dict)
                        if screwdriver_dict != None:
                            yolact_preds.append(screwdriver_dict)
                else:
                    yolact_preds = pre_yolact_preds
                
                if len(yolact_preds) == 0:
                    print('| no predictions found!', end='')
                    end_time = time.time()
                    print(' | Processing time: ' + str(int(end_time - start_time)) + 's' , flush=True)
                    continue
                yolact_time_end = time.time()
                print(' Yolact_preds=' + str(int(yolact_time_end - yolact_time_begin)) + 's, ', end='')
                
                crop_time_begin = time.time()
                for yolact_pred in yolact_preds:
                    yolact_pred['blobs'] = calculate_blobs(yolact_pred['mask'])
                if CROP_MASKS_PERSONAL:
                    #temp_image_bool = (yolact_preds[0]['mask']).astype(bool)
                    #temp_image = (cv2.cvtColor(temp_image_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,0,255], dtype=np.uint8)
                    #temp_res = cv2.addWeighted(image, 0.5, temp_image, 1, 0)
                    #cv2.imwrite('./testerOutput/noCropMasks.png', temp_res)
                    yolact_preds = crop_masks(yolact_preds, h, w)
                    #res_mask = np.zeros_like(temp_image_bool)
                    #for blob in yolact_preds[0]['blobs']:
                    #    res_mask = np.logical_or(res_mask, blob)
                    #temp_image_bool = (res_mask).astype(bool)
                    #temp_image = (cv2.cvtColor(temp_image_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,0,255], dtype=np.uint8)
                    #temp_res = cv2.addWeighted(image, 0.5, temp_image, 1, 0)
                    #cv2.imwrite('./testerOutput/personalCropMasks.png', temp_res)
                    #exit()
                crop_time_end = time.time()
                print(' Cropping=' + str(int(crop_time_end - crop_time_begin)) + 's, ', end='')
            
            if SEGMENT_EVERYTHING:
                sa_time_begin = time.time()
                if USE_PRECALC_SA:
                    try:
                        curr_sa_path = sa_preds_path + '/' + str.replace(img_dict['file_name'], '.png' , '.pkl')
                        with open(curr_sa_path, "rb") as f:
                            sa_masks = pickle.load(f) #load pickle data to sa_masks
                            sa_masks = sa_masks['all_masks']
                    except EOFError:
                        sa_masks = segmentEverything(img_path, anything_generator)
                else:
                    sa_masks = segmentEverything(img_path, anything_generator)
                sa_time_end = time.time()
                print('SA=' + str(int(sa_time_end - sa_time_begin)) + 's, ', end='')

                #plt.imshow(image)
                #show_anns(sa_masks)
                #plt.savefig('testerOutput/SegmentAnything_image.png')
                results_time_begin = time.time()
                result_masks = []
                for k,yolact_pred in enumerate(yolact_preds):
                    if len(yolact_pred['blobs']) == 0:
                        continue
                    
                    
                    result_mask = iou_filter(yolact_pred['blobs'], sa_masks)
                    result_mask_bool = result_mask.astype(bool)
                    gen_mask_bool = result_mask_bool
                    
                    gen_mask_final = (cv2.cvtColor(gen_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([255,255,255])
                    im_pil = Image.fromarray(gen_mask_bool)
                    im_pil.convert("1")
                    width, height = im_pil.size
                    bitmask_curr = Image.new("1", (width+2,height+2), 0)
                    bitmask_curr.paste(im_pil, (1,1))

                    gen_mask_dict = {}
                    try:
                        gen_mask_dict["segmentation"], gen_mask_dict["bbox"], gen_mask_dict["area"] = create_mask_annotation(np.array(bitmask_curr), False)
                        if (gen_mask_dict["segmentation"] == -1 or gen_mask_dict["bbox"] == -1 or gen_mask_dict["area"] == -1):
                            # continue cause theres nothing to learn from
                            continue
                    except Exception:
                        print("EXCEPTION OCCURED!")
                        continue
                        
                    #create new mask dict with generated mask
                    gen_mask_dict["iscrowd"] = 0
                    gen_mask_dict["image_id"] = mask_dict['image_id']
                    gen_mask_dict["category_id"] = yolact_pred['class']
                    gen_mask_dict["id"] = mask_dict['id']
                    
                    #from now on we can assume that this image exists
                    labels_dict[camera][imageId] = {}
                    labels_dict[camera][imageId]["img"] = img_dict
                    labels_dict[camera][imageId]["mask"] = gen_mask_dict
                    
                    if VISUALIZE_GEN_MASKS:
                        gt_mask = np.zeros((h, w), dtype=np.uint8)
                        # Draw each polygon on the mask
                        for polygon in mask_dict['segmentation']:
                            rle = maskUtils.frPyObjects([polygon], h, w)
                            m = maskUtils.decode(rle)
                            gt_mask = np.maximum(gt_mask, m[:,:,0])
                        gt_mask_bool = gt_mask.astype(bool)
                        
                        and_mask_bool = np.logical_and(result_mask_bool, gt_mask_bool)
                        result_mask_bool = np.logical_and(result_mask_bool, np.logical_not(and_mask_bool))
                        gt_mask_bool = np.logical_and(gt_mask_bool, np.logical_not(and_mask_bool))
                        gt_mask_rgb = (cv2.cvtColor(gt_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,255,0], dtype=np.uint8)
                        and_mask_rgb = (cv2.cvtColor(and_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([255,0,0], dtype=np.uint8)
                        result_mask_rgb = (cv2.cvtColor(result_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,0,255], dtype=np.uint8)

                        result_image = cv2.addWeighted(image, 1.0, and_mask_rgb, 0.5, 0)
                        result_image = cv2.addWeighted(result_image, 1.0, result_mask_rgb, 0.5, 0)
                        result_image = cv2.addWeighted(result_image, 1.0, gt_mask_rgb, 0.5, 0)
                        
                        cv2.imwrite(f"./testerOutput/{camera}{imageId}.png", result_image)
                        exit()
                        
                results_time_end = time.time()
                print(' Label-Gen=' + str(int(results_time_end - results_time_begin)) + 's' , end='')
            end_time = time.time()
            print(' | Total time: ' + str(int(end_time - start_time)) + 's' , flush=True)

    print('OK!')


    return labels_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sampling from Yolact masks for Segment Anything")
    parser.add_argument('--config', default=None, required=False,
                        help='The config object to use.')
    parser.add_argument('--trained_model', required=True, type=str)
    parser.add_argument('--coco_file', required=True, type=str)
    parser.add_argument('--images_dir', required=True, type=str)
    parser.add_argument('--results_path', required=False, default='/cluster/project/infk/cvg/heinj/students/kbirgi/generating_masks', type=str)
    parser.add_argument('--results_dir', required=False, default='output', type=str)
    parser.add_argument('--sa_model', required=False, type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--stride', help='10 means every 10th picture is processed, 100 means every 100th picture is processed, etc...', required=False, type=int, default=1)
    parser.add_argument('--start_img', default=0, type=int)
    parser.add_argument('--start_cam', default=0, type=int)
    parser.add_argument('--sa_preds', help='path to the precalculated pickle files of segment anything', default='/cluster/project/infk/cvg/heinj/students/kbirgi/Annotations/trainSSD/SA_processed_images/mvpsp', required=False, type=str)
    parser.add_argument('--create_labels', default=False, type=bool)
    args = parser.parse_args()
    stride = args.stride
    images_dir = args.images_dir
    results_dir = args.results_dir
    temp_results_path = args.results_path
    start_img = args.start_img
    start_cam = args.start_cam
    sa_preds_path = args.sa_preds
    
    # prepare results directory
    results_path = temp_results_path + '/' + results_dir
    if (not (os.path.exists(results_path))):
        os.mkdir(results_path)
    results_path = temp_results_path + '/' + results_dir + '/gen_masks'
    if (not (os.path.exists(results_path))):
        os.mkdir(results_path)

    # prepare yolact directory
    yolact_path = temp_results_path + '/' + results_dir + '/yolact_preds'
    if (not os.path.exists(yolact_path)):
        os.mkdir(yolact_path)
    
    sa_path = temp_results_path + '/' + results_dir + '/sa_preds'
    if (not os.path.exists(sa_path)):
        os.mkdir(sa_path)

    plots_path = temp_results_path + '/' + results_dir + '/plots'
    if (not os.path.exists(sa_path)):
        os.mkdir(sa_path)


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

    if args.create_labels:
        labels_dict = create_labels()
        with open("/cluster/project/infk/cvg/heinj/students/kbirgi/Annotations/gen_annotations/generated_labels.json", "w") as f:
            json.dumps(labels_dict, f)
            exit()

    
    print('Loading annotations...', end='',flush=True)
    f = open(args.coco_file)
    coco_dict = json.load(f)
    f.close()
    print(' Done.', flush=True)

    print('Using stride: ' + str(stride))

    # this is the dict that will be written to result_dict_path
    # following structure will be written:
    # {img_id : {pred_id : {file_name, gen_mask_image_path, iou_score, gt_mask, pred_mask}}}
    # img_id: is the id from the coco file, 
    # pred_id: is the prediction id from yolact, cause there can be multiple predictions,
    # file_name: is the file name from the coco file,
    # class: is the class of the predicted object, here its 0 or 1
    # gen_mask_image_path: is the path of the resulting image with both masks visualized
    # iou_score: is the iou score of the gt_mask and the pred_mask
    # im not sure if storing masks is smart, i thought maybe the colors might want to be changed
    # but it will take up a lot of space so for now im going to leave it out
    # gt_mask is the ground truth mask, stored in its original format (storing masks is optional)
    # pred_mask is the predicted mask, stored in ?? format (storing masks is optional)
    #results dict takes up too much space so get rid of it
    #results = {}

    len_coco_dict = len(coco_dict)
    passed = False
    for i,camera in enumerate(coco_dict):
        if ((not passed) & ((i+1) < start_cam)):
            continue

        #print("BEWARE: USING CHOSEN SCENES FROM CODE! NO OTHER CAMERA ANGLES WILL BE PROCESSED!")
        #if not (camera in CHOSEN_SCENES):
        #    continue

        camera_dict = coco_dict[camera]
        len_camera_dict = len(camera_dict)
        camera_results = {}
        for j,imageId in enumerate(camera_dict):
            print(camera_dict[imageId]['img']['file_name'])
            if ((MAX_IMAGE != 0) & (j > MAX_IMAGE)):
                print("FINISHING EARLY!")
                exit()
            if ((not passed) & ((j+1) < start_img)):
                continue
            passed = True

            if ((j % stride) != 0):
                continue

            start_time = time.time()
            print("Camera:" + str(i+1) + "/" + str(len_coco_dict) + \
                " | Image:" + str(j+1) + "/" + str(len_camera_dict), end = '')
            
            if SAVE_PLOTS:
                #fig, axs = plt.subplots(1,2, figsize=(10,6)) # figsize=(10,6)
                fig, axs = plt.subplots(2,2) #figsize=(10,5))
                #fig.subplots_adjust(bottom=0.15, hspace=0.3)  # Adjust bottom margin to create space for the description
            
            img_dict = camera_dict[imageId]['img'] # keys: ['id', 'width', 'height', 'file_name']
            mask_dict = camera_dict[imageId]['mask'] # keys: ['segmentation', 'bbox', 'area', 'iscrowd', 'image_id', 'category_id', 'id'] 
            img_path = images_dir + "/" + img_dict['file_name']
            image = cv2.imread(img_path)
            h, w, _ = image.shape
            
            prep_path(results_path, img_dict['file_name'])
            img_results_path = results_path + '/' + img_dict['file_name']
            print(' | Processing times: ', end='')
            yolact_time_begin = time.time()
            with torch.no_grad():
                # if we only want the yolact preds and no processing
                
                if SAVE_YOLACT_PREDS:
                    prep_path(yolact_path, img_dict['file_name'])
                    yolact_preds_path = yolact_path + '/' + img_dict['file_name']
                    frame = torch.from_numpy(cv2.imread(img_path)).cuda().float()
                    batch = FastBaseTransform()(frame.unsqueeze(0))
                    preds = net(batch)
                    img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
                    #cv2.imwrite('./testerOutput/yolactPred.png', img_numpy)
                    img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
                    axs[0,1].imshow(img_numpy)
                    axs[0,1].axis('off')
                    axs[0,1].set_title('Yolact Prediction')
                
                
                # pre_yolact_preds is a list of dicts containing keys (class,score,bbox,mask)
                pre_yolact_preds = get_yolact_preds(img_path, sum_all=False, crop_masks=(CROP_MASKS & (not CROP_MASKS_PERSONAL)))
                if TAKE_MAX_PREDS:
                    yolact_preds = []
                    powerdrill_max_conf = -1
                    powerdrill_dict = None
                    screwdriver_max_conf = -1
                    screwdriver_dict = None
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
                    if powerdrill_dict != None:
                        yolact_preds.append(powerdrill_dict)
                    if screwdriver_dict != None:
                        yolact_preds.append(screwdriver_dict)
                else:
                    yolact_preds = pre_yolact_preds
                
                if len(yolact_preds) == 0:
                    print('| no predictions found!', end='')
                    end_time = time.time()
                    print(' | Processing time: ' + str(int(end_time - start_time)) + 's' , flush=True)
                    plt.close('all')
                    continue
                yolact_time_end = time.time()
                print(' Yolact_preds=' + str(int(yolact_time_end - yolact_time_begin)) + 's, ', end='')
                
                crop_time_begin = time.time()
                for yolact_pred in yolact_preds:
                    yolact_pred['blobs'] = calculate_blobs(yolact_pred['mask'])
                if CROP_MASKS_PERSONAL:
                    #temp_image_bool = (yolact_preds[0]['mask']).astype(bool)
                    #temp_image = (cv2.cvtColor(temp_image_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,0,255], dtype=np.uint8)
                    #temp_res = cv2.addWeighted(image, 0.5, temp_image, 1, 0)
                    #cv2.imwrite('./testerOutput/noCropMasks.png', temp_res)
                    yolact_preds = crop_masks(yolact_preds, h, w)
                    #res_mask = np.zeros_like(temp_image_bool)
                    #for blob in yolact_preds[0]['blobs']:
                    #    res_mask = np.logical_or(res_mask, blob)
                    #temp_image_bool = (res_mask).astype(bool)
                    #temp_image = (cv2.cvtColor(temp_image_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,0,255], dtype=np.uint8)
                    #temp_res = cv2.addWeighted(image, 0.5, temp_image, 1, 0)
                    #cv2.imwrite('./testerOutput/personalCropMasks.png', temp_res)
                    #exit()
                crop_time_end = time.time()
                print(' Cropping=' + str(int(crop_time_end - crop_time_begin)) + 's, ', end='')
                
            if SEGMENT_SAMPLE:
                with torch.no_grad():
                    sampleYolact()
            camera_results[img_dict['id']] = {}
            
            if SEGMENT_EVERYTHING:
                sa_time_begin = time.time()
                if USE_PRECALC_SA:
                    try:
                        curr_sa_path = sa_preds_path + '/' + str.replace(img_dict['file_name'], '.png' , '.pkl')
                        with open(curr_sa_path, "rb") as f:
                            sa_masks = pickle.load(f) #load pickle data to sa_masks
                            sa_masks = sa_masks['all_masks']
                    except EOFError:
                        sa_masks = segmentEverything(img_path, anything_generator)
                else:
                    sa_masks = segmentEverything(img_path, anything_generator)

                if SAVE_SA:
                    if SAVE_PLOTS:
                        #prep_path(sa_path, img_dict['file_name'])
                        axs[0,0].imshow(image)
                        img = show_anns(sa_masks)
                        axs[0,0].imshow(img)
                        axs[0,0].axis('off')
                        axs[0,0].set_title('Segment-Anything Predictions')
                    else:
                        prep_path(sa_path, img_dict['file_name'])
                        plt.imshow(image)
                        img = show_anns(sa_masks)
                        plt.imshow(img)
                        plt.savefig(sa_path + '/' + img_dict['file_name'])
                sa_time_end = time.time()
                print('SA=' + str(int(sa_time_end - sa_time_begin)) + 's, ', end='')

                #plt.imshow(image)
                #show_anns(sa_masks)
                #plt.savefig('testerOutput/SegmentAnything_image.png')
                results_time_begin = time.time()
                result_masks = []
                for k,yolact_pred in enumerate(yolact_preds):
                    if len(yolact_pred['blobs']) == 0:
                        continue
                    
                    
                    result_mask = iou_filter(yolact_pred['blobs'], sa_masks)
                    result_mask_bool = result_mask.astype(bool)
                    gen_mask_bool = result_mask_bool
                    
                    gt_mask = np.zeros((h, w), dtype=np.uint8)
                    # Draw each polygon on the mask
                    for polygon in mask_dict['segmentation']:
                        rle = maskUtils.frPyObjects([polygon], h, w)
                        m = maskUtils.decode(rle)
                        gt_mask = np.maximum(gt_mask, m[:,:,0])
                    gt_mask_bool = gt_mask.astype(bool)
                    
                    and_mask_bool = np.logical_and(result_mask_bool, gt_mask_bool)
                    result_mask_bool = np.logical_and(result_mask_bool, np.logical_not(and_mask_bool))
                    gt_mask_bool = np.logical_and(gt_mask_bool, np.logical_not(and_mask_bool))
                    
                    gt_mask_rgb = (cv2.cvtColor(gt_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,255,0], dtype=np.uint8)
                    and_mask_rgb = (cv2.cvtColor(and_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([255,0,0], dtype=np.uint8)
                    result_mask_rgb = (cv2.cvtColor(result_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,0,255], dtype=np.uint8)

                    result_image = cv2.addWeighted(image, 1.0, and_mask_rgb, 0.5, 0)
                    result_image = cv2.addWeighted(result_image, 1.0, result_mask_rgb, 0.5, 0)
                    result_image = cv2.addWeighted(result_image, 1.0, gt_mask_rgb, 0.5, 0)

                    gen_mask_rgb = (cv2.cvtColor(gen_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([0,0,255], dtype=np.uint8)
                    gen_mask_image = (cv2.addWeighted(image, 1.0, gen_mask_rgb, 0.5, 0))

                    result_pred_path = img_results_path.replace('.png', '_' + str(k) + '.png')
                    if SAVE_PLOTS:
                        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        axs[1,0].imshow(result_image)
                        axs[1,0].axis('off')
                        axs[1,0].set_title('Mask overlaps')
                        gen_mask_image = cv2.cvtColor(gen_mask_image, cv2.COLOR_BGR2RGB)    
                        axs[1,1].imshow(gen_mask_image)
                        axs[1,1].axis('off')
                        axs[1,1].set_title('Generated Mask')
                
                        #plt.savefig(result_pred_path, bbox_inches='tight', dpi=300)
                        #plt.close(fig)
                    else:
                        cv2.imwrite(result_pred_path, result_image)
                    masks_iou = round(calculateIoU(gt_mask_bool, result_mask), 2)
                    camera_results[img_dict['id']][str(k)] = {
                        'file_name' : '',#img_dict['file_name'],
                        'masks_image_name' : '',#result_pred_path,
                        'class' : int(yolact_pred['class']),
                        'iou' : masks_iou
                        #leave out the masks for now
                    }
                    #fig.suptitle('IoU(Generated Mask, Ground Truth Mask) = ' + str(masks_iou), fontsize=10, x=0.5, y=0.05)
                    #plt.figtext(0.5, 0.01, 'IoU(Generated Mask, Ground Truth Mask) = ' + str(masks_iou), fontsize=10, ha='center')
                    plt.savefig(result_pred_path, bbox_inches='tight', dpi=300)
                    
                    if CREATE_TRAINING_LABELS:
                        gen_mask_final = (cv2.cvtColor(gen_mask_bool.astype(np.uint8), cv2.COLOR_GRAY2BGR)) * np.array([255,255,255])
                        cv2.imwrite('./testerOutput/generated_mask_final.png', gen_mask_final)
                        exit()
                    #camera_results[img_dict['id']][str(k)] = results[img_dict['id']][str(k)]
                plt.close('all')
                results_time_end = time.time()
                print(' Storing=' + str(int(results_time_end - results_time_begin)) + 's' , end='')
            end_time = time.time()
            print(' | Total time: ' + str(int(end_time - start_time)) + 's' , flush=True)

        f = open(temp_results_path + '/' + results_dir + '/camera_' + camera + '.json', 'w')
        #print(camera_results)
        #print(type(camera_results))
        json.dump(camera_results, f, indent=2)
        f.close()

    #f = open(temp_results_path + '/' + results_dir + '/all_results.json', 'w')
    #f.write(json.dumps(results))
    #f.close()

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
