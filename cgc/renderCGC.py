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
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from scene.dataset_readers import CameraInfo
from scene.colmap_loader import qvec2rotmat
from utils.graphics_utils import focal2fov
from utils.camera_utils import loadCam

import matplotlib.pyplot as plt
import cmapy
import random

def feature_to_rgb(features):
    features = features / (torch.norm(features, dim=-1, keepdim=True) + 1e-9)

    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    pca_result = pca_result.reshape(H, W, 3)

    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array

def read_view_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

        content = content.strip('[')
        data = content.rsplit(',', 1)[0]

        data = list(map(float, data.split(',')))

    return data



def extract_matrix_and_vector(data):
    R_data = data
    R_data = np.array([R_data[0],R_data[4],R_data[8],R_data[1],R_data[5],R_data[9],R_data[2],R_data[6],R_data[10]])

    R = np.array(R_data).reshape(3, 3)
    tvec = np.array(data[12:15])

    return R, tvec




def render_set(source_path, gaussians, pipeline, background):
    path = os.path.join(source_path, "renders")

    file_path =  os.path.join(path, 'view.txt')
    data = read_view_file(file_path)
    R, tvec = extract_matrix_and_vector(data)
    R = R[:3, :3]
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    qvec = [qw, qx, qy, qz]

    imagecgc_path = os.path.join(path, 'request.png')
    imagecgc_name = os.path.basename(imagecgc_path).split(".")[0]
    imagecgc = Image.open(imagecgc_path) if os.path.exists(imagecgc_path) else None
    objectscgc = Nonewidth, height = imagecgc.size
    #objectscgc_data = None

    width, height = imagecgc.size

    FovY = focal2fov(775.31884604, height)
    FovX = focal2fov(778.73580592, width)
    
    R = np.transpose(qvec2rotmat(qvec))
    T = np.array(tvec)

    cam_info = CameraInfo(uid=777, R=R, T=T, FovY=FovY, FovX=FovX, image=imagecgc,
                              image_path=imagecgc_path, image_name=imagecgc_name, width=width, height=height, objects=objectscgc)
    cam = loadCam(-1, 777, cam_info, 1)

    
    resultcgc = render(cam, gaussians, pipeline, background) 
    rendering_obj = resultcgc["render_object"]
    rgb_mask = feature_to_rgb(rendering_obj)
    Image.fromarray(rgb_mask).save(os.path.join(path, 'output.png'))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_set(dataset.source_path, gaussians, pipeline, background)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args))
