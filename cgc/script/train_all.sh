#!/bin/bash


# Train on LERF-MASK
python train.py -s data/lerf_sam/figurines -r 1 -m output/lerf_sam/figurines --config_file config/gaussian_dataset/train.json
python train.py -s data/lerf_sam/ramen -r 1 -m output/lerf_sam/ramen --config_file config/gaussian_dataset/train.json
python train.py -s data/lerf_sam/teatime -r 1 -m output/lerf_sam/teatime --config_file config/gaussian_dataset/train.json


# Train on LERF-MASK: Ablation No Dissimilarity
python train.py -s data/lerf_sam/figurines -r 1 -m output/lerf_abla_no_diss/figurines --config_file config/gaussian_dataset/train.json
python train.py -s data/lerf_sam/ramen -r 1 -m output/lerf_abla_no_diss/ramen --config_file config/gaussian_dataset/train.json
python train.py -s data/lerf_sam/teatime -r 1 -m output/lerf_abla_no_diss/teatime --config_file config/gaussian_dataset/train.json


# Train on LERF-MASK: Ablation No Similarity
python train.py -s data/lerf_sam/figurines -r 1 -m output/lerf_abla_no_sim/figurines --config_file config/gaussian_dataset/train.json
python train.py -s data/lerf_sam/ramen -r 1 -m output/lerf_abla_no_sim/ramen --config_file config/gaussian_dataset/train.json
python train.py -s data/lerf_sam/teatime -r 1 -m output/lerf_abla_no_sim/teatime --config_file config/gaussian_dataset/train.json


# Train on LERF-dataset: Ablation No 3D Loss
python train.py -s data/lerf_sam/figurines -r 1 -m output/lerf_abla/figurines --config_file config/gaussian_dataset/train.json
python train.py -s data/lerf_sam/ramen -r 1 -m output/lerf_abla/ramen --config_file config/gaussian_dataset/train.json
python train.py -s data/lerf_sam/teatime -r 1 -m output/lerf_abla/teatime --config_file config/gaussian_dataset/train.json

# Train on 3D-OVS dataset
python train.py -s data/3D-OVS/lawn -m output/3D-OVS/lawn --config_file config/3DOVS_dataset/train.json
python train.py -s data/3D-OVS/bench -m output/3D-OVS/bench --config_file config/3DOVS_dataset/train.json
python train.py -s data/3D-OVS/room -m output/3D-OVS/room --config_file config/3DOVS_dataset/train.json
python train.py -s data/3D-OVS/bed -m output/3D-OVS/bed --config_file config/3DOVS_dataset/train.json
python train.py -s data/3D-OVS/sofa -m output/3D-OVS/sofa --config_file config/3DOVS_dataset/train.json
