import os
import torch
import numpy as np
import cv2
import json
import argparse
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

#from dataset import Dataset

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
        color_mask = np.concatenate([np.random.random(3), [0.9]])
        img[m] = color_mask
    ax.imshow(img)

def generate_segmask_from_masks(masks, img_size):
    segmask = np.zeros(img_size)
    id = 1
    mask_data = {}
    for mask in masks:
        y, x = np.nonzero(mask["segmentation"].astype(bool))
        segmask[y, x] = id
        mask_data[id] = mask["stability_score"]
        id += 1
    '''plt.imshow(segmask)
    plt.show()
    print(mask_data)'''
    return segmask, mask_data

def parse_arguments():     
    parser = argparse.ArgumentParser(description="SAM Auto Image Segmentation")     
    parser.add_argument('--sam_checkpoint', required=True, help='Path to the SAM checkpoint')     
    parser.add_argument('--data_dir', required=True, help='Path to the data directory containing input images and where output will be saved')     
    return parser.parse_args()

if __name__ == "__main__":
    np.random.seed(0)
    args = parse_arguments()

    data_dir = args.data_dir
    images = os.path.join(data_dir, 'images')
    #dataset = Dataset(data_dir, extension=".JPG")
    #images = dataset.get_images_list()
    #images_dirlist = dataset.get_images_dirlist()

    output_dir = os.path.join(data_dir, "object_mask")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam_checkpoint = args.sam_checkpoint
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32,
                                               points_per_batch=32, 
                                               pred_iou_thresh=0.7,
                                               stability_score_thresh=0.95,
                                               stability_score_offset=1.0,
                                               box_nms_thresh=0.7,
                                               crop_n_layers=0,
                                               crop_nms_thresh=0.7,
                                               crop_overlap_ratio=512 / 1500,
                                               crop_n_points_downscale_factor=2,
                                               point_grids=None,
                                               min_mask_region_area=100,
                                               output_mode= "binary_mask")
        
        
    for img_file in os.listdir(images):
        img_path = os.path.join(images, img_file)
        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(img_rgb)
        segmask, mask_data = generate_segmask_from_masks(masks, img_size=img_rgb.shape[:2])
        
        frame_id = os.path.splitext(img_file)[0]  # Extract frame ID from filename
        
        cv2.imwrite(os.path.join(output_dir, frame_id + ".png"), segmask)
            
