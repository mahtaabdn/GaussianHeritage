# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import cv2
from random import randint
import matplotlib.pyplot as plt

from ext.grounded_sam import grouned_sam_output, load_model_hf
from groundingdino.util.inference import load_model
from segment_anything import sam_model_registry, SamPredictor

from render import feature_to_rgb, get_eval_feature_map


@torch.no_grad()
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, groundingdino_model, sam_predictor, TEXT_PROMPT, thresh_cos, threshold=0.2):
    render_path = os.path.join(model_path, name, "ours_{}_text".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}_text".format(iteration), "gt")
    colormask_path = os.path.join(model_path, name, "ours_{}_text".format(iteration), "objects_feature16")
    pred_obj_path = os.path.join(model_path, name, "ours_{}_text".format(iteration), "test_mask")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)

    # Use Grounded-SAM on the first frame
    results0 = render(views[0], gaussians, pipeline, background)
    rendering0 = results0["render"]
    rendering_obj0 = results0["render_object"]

    image = (rendering0.permute(1,2,0) * 255).cpu().numpy().astype('uint8')
    text_mask, annotated_frame_with_mask = grouned_sam_output(groundingdino_model, sam_predictor, TEXT_PROMPT, image)
    Image.fromarray(annotated_frame_with_mask).save(os.path.join(render_path[:-8],'grounded-sam---'+TEXT_PROMPT+'.png'))
    
    # Get all pixels inside mask
    yx = torch.nonzero(text_mask!=0, as_tuple=False)
    
    # Get coordinates
    feature_2d_list = []

    if yx.shape[0] > 0:
        idx = yx.shape[0] // 2 - 1
        x = yx[idx, 1].long()
        y = yx[idx, 0].long()
        feature_2d_list.append(rendering_obj0[:, y, x])
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        pred_obj_img_path = os.path.join(pred_obj_path,str(idx))
        makedirs(pred_obj_img_path, exist_ok=True)
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]

        if len(feature_2d_list) > 0:
            feature_2d = feature_2d_list[0]
            pred_obj_mask = (get_eval_feature_map(rendering_obj, feature_2d, thresh=thresh_cos).cpu().numpy() * 255).astype(np.uint8)
        else:
            pred_obj_mask = torch.zeros_like(view.objects).cpu().numpy()
        
        rgb_mask = feature_to_rgb(rendering_obj / (rendering_obj.norm(dim=-1, keepdim=True) + 1e-9))
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_img_path, TEXT_PROMPT + ".png"))
        print(os.path.join(pred_obj_img_path, TEXT_PROMPT + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, thresh_cos : float):
    with torch.no_grad():
        dataset.eval = True
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # grounding-dino
        # Use this command for evaluate the Grounding DINO model
        # Or you can download the model by yourself
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

        # sam-hq
        sam_checkpoint = 'SAM_checkpoints/sam_vit_h_4b8939.pth'
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device='cuda')
        sam_predictor = SamPredictor(sam)

        # Text prompt
        if 'figurines' in dataset.model_path:
            positive_input = "green apple;green toy chair;old camera;porcelain hand;red apple;red toy chair;rubber duck with red hat"
        elif 'ramen' in dataset.model_path:
            positive_input = "chopsticks;egg;glass of water;pork belly;wavy noodles in bowl;yellow bowl"
        elif 'teatime' in dataset.model_path:
            positive_input = "apple;bag of cookies;coffee mug;cookies on a plate;paper napkin;plate;sheep;spoon handle;stuffed bear;tea in a glass"
        elif 'bed' in dataset.model_path:
            positive_input = "banana;black leather shoe;camera;hand;red bag;white sheet"
        elif 'sofa' in dataset.model_path:
            positive_input = "pikachu;a stack of UNO cards;a red Nintendo Switch joy-con controller;Gundam;Xbox wireless controller;grey sofa"
        elif 'lawn' in dataset.model_path:
            positive_input = "red apple;New York Yankees cap;stapler;black headphone;hand soap;green lawn"
        elif 'bench' in dataset.model_path:
            positive_input = "pebbled concrete wall;wood;Portuguese egg tart;orange cat;green grape;mini offroad car;dressing doll"
        elif 'room' in dataset.model_path:
            positive_input = "wood wall;shrilling chicken;weaving basket;rabbit;dinosaur;baseball"
        else:
            raise NotImplementedError   # You can provide your text prompt here
        
        positives = positive_input.split(";")
        print("Text prompts:    ", positives)

        for TEXT_PROMPT in positives:
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, None, groundingdino_model, sam_predictor, TEXT_PROMPT, thresh_cos)
            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, None, groundingdino_model, sam_predictor, TEXT_PROMPT, thresh_cos)


             

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--thresh", default=0.65, type=float)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.thresh)
