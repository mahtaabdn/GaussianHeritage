# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from scipy.spatial import cKDTree

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def masked_l1_loss(network_output, gt, mask):
    mask = mask.float()[None,:,:].repeat(gt.shape[0],1,1)
    loss = torch.abs((network_output - gt)) * mask
    loss = loss.sum() / mask.sum()
    return loss

def weighted_l1_loss(network_output, gt, weight):
    loss = torch.abs((network_output - gt)) * weight
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def contrastive_2d_loss(segmask, features, id_unique_list, n_i_list, dim_features=16, lambda_val=1e-4):
    """
    Compute the contrastive clustering loss for a 2D feature map.

    :param segmask: Tensor of shape (H, W).
    :param features: Tensor of shape (H, W, D), where (H, W) is the resolution and D is the dimensionality of these features.
    :param id_unique_list: Tensor of shape (n_p).
    :n_i_list: Tensor of shape (n_p).
    :dim_features: is the dimensionality of the features (equal to D).
    :lambda_val: Weighting factor for the loss.

    :return: loss value.
    """
    n_p = id_unique_list.shape[0] # Number of ids
    
    f_mean_per_cluster = torch.zeros((n_p, dim_features)).cuda()
    phi_per_cluster = torch.zeros((n_p, 1)).cuda()
            
    for i in range(n_p):
        mask = segmask == id_unique_list[i]
        f_mean_per_cluster[i, ...] = torch.mean(features[mask, :], dim=0, keepdim=True)
        phi_per_cluster[i] = torch.norm(features[mask, :] - f_mean_per_cluster[i], dim=1, keepdim=True).sum() / (n_i_list[i] * torch.log(n_i_list[i] + 100))
            
    phi_per_cluster = torch.clip(phi_per_cluster * 10, min=0.1, max=1.0)
    phi_per_cluster = phi_per_cluster.detach()
    loss_per_cluster = torch.zeros(n_p).cuda()
            
    for i in range(n_p):
        f_mean = f_mean_per_cluster[i]
        phi = phi_per_cluster[i]
        mask = segmask == id_unique_list[i]
        f_ij = features[mask, :] # shape (ni, 16)
        num = torch.exp(torch.matmul(f_ij, f_mean) / (phi + 1e-6)) # dim (ni)
        den = torch.sum(torch.exp(torch.matmul(f_ij, f_mean_per_cluster.transpose(-1, -2)) / (phi_per_cluster.transpose(-1, -2) + 1e-6)), dim=1) # dim (n_i)
        loss_per_cluster[i] = torch.sum(torch.log(num / (den + 1e-6)))
            
    loss_obj = - lambda_val * torch.mean(loss_per_cluster)
    return loss_obj

def spatial_loss(xyz, features, k_pull=2, k_push=5, lambda_pull=0.05, lambda_push=0.15, max_points=200000, sample_size=800):
    """
    Compute the spatial-similarity regularization loss for a 3D point cloud using Top-k neighbors and Top-k distant elements
    
    :param xyz: Tensor of shape (N, D), where N is the number of points and D is the dimensionality.
    :param features: Tensor of shape (N, C), where C is the dimensionality of these features.
    :param k_pull: Number of neighbors to consider.
    :param k_push: Number of remote elements to consider.
    :param lambda_pull: Weighting factor for the loss.
    :param lambda_push: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.
    
    :return: Computed loss value.
    """
    # Conditionally downsample if points exceed max_points
    if xyz.size(0) > max_points:
        indices = torch.randperm(xyz.size(0))[:max_points]
        xyz = xyz[indices]
        features = features[indices]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(xyz.size(0))[:sample_size]
    sample_xyz = xyz[indices]
    sample_preds = features[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_xyz, xyz)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k_pull, largest=False)  # Get top-k smallest distances

    # Compute top-k farest gaussians
    _, faraway_indices_tensor = dists.topk(k_push, largest=True)  # Get top-k bigest distances 

    # Fetch neighbor features using indexing
    neighbor_preds = features[neighbor_indices_tensor]

    # Fetch remote features using indexing
    faraway_preds = features[faraway_indices_tensor]

    # Compute cosine similarity
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-10)
    
    pull_loss = cos(sample_preds.unsqueeze(1).expand(-1, k_pull, -1, -1), neighbor_preds) #more similar of they are close
    
    push_loss =  cos(sample_preds.unsqueeze(1).expand(-1, k_push, -1, -1), faraway_preds) #less similar if they are far away
    
    # Total loss
    loss = lambda_pull * torch.sigmoid(1.0 - torch.mean(pull_loss[..., 0].reshape(-1), dim=-1)) + lambda_push * torch.sigmoid(torch.mean(push_loss[..., 0].reshape(-1), dim=-1))
    
    return loss

def no_similarity_loss(xyz, features, k=5, lambda_val=0.15, max_points=200000, sample_size=800):
    # Conditionally downsample if points exceed max_points
    if xyz.size(0) > max_points:
        indices = torch.randperm(xyz.size(0))[:max_points]
        xyz = xyz[indices]
        features = features[indices]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(xyz.size(0))[:sample_size]
    sample_xyz = xyz[indices]
    sample_preds = features[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_xyz, xyz)  # Compute pairwise distances

    # Compute top-k farest gaussians
    _, faraway_indices_tensor = dists.topk(k, largest=True)  # Get top-k bigest distances 

    # Fetch remote features using indexing
    faraway_preds = features[faraway_indices_tensor]

    # Compute cosine similarity
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-10)
    
    push_loss =  cos(sample_preds.unsqueeze(1).expand(-1, k, -1, -1), faraway_preds) # dissimilar if they are far away
    
    # Total loss
    loss = lambda_val * torch.sigmoid(torch.mean(push_loss[..., 0].reshape(-1), dim=-1))
    
    return loss

def no_dissimilarity_loss(xyz, features, k=2, lambda_val=0.05, max_points=200000, sample_size=800):
    # Conditionally downsample if points exceed max_points
    if xyz.size(0) > max_points:
        indices = torch.randperm(xyz.size(0))[:max_points]
        xyz = xyz[indices]
        features = features[indices]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(xyz.size(0))[:sample_size]
    sample_xyz = xyz[indices]
    sample_preds = features[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_xyz, xyz)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    # Fetch neighbor features using indexing
    neighbor_preds = features[neighbor_indices_tensor]

    # Compute cosine similarity
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-10)
    
    pull_loss = cos(sample_preds.unsqueeze(1).expand(-1, k, -1, -1), neighbor_preds) # more similar of they are close
    
    # Total loss
    loss = lambda_val * torch.sigmoid(1.0 - torch.mean(pull_loss[..., 0].reshape(-1), dim=-1))
    
    return loss

def variance_in_feature_clusters(segmask, features, id_unique_list, dim_features=16):
    n_p = id_unique_list.shape[0] # Number of ids
        
    f_mean_per_cluster = torch.zeros((n_p, dim_features)).cuda()
            
    for i in range(n_p):
        mask = segmask == id_unique_list[i]
        f_mean_per_cluster[i, ...] = torch.mean(features[mask, :], dim=0, keepdim=True)
    
    variance_per_cluster = torch.zeros(n_p, dim_features).cuda()

    for i in range(n_p):
        f_mean = f_mean_per_cluster[i]
        mask = segmask == id_unique_list[i]
        f_ij = features[mask, :] # shape (ni, 16)
        variance_per_cluster[i] = torch.mean(f_ij * f_ij, dim=0) - (f_mean * f_mean)
    
    return torch.mean(variance_per_cluster)