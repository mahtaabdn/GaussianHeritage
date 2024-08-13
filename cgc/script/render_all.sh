#!/bin/bash

# Render LERF-Mask

python render_lerf_mask.py -m output/lerf_sam/figurines --skip_train --thresh 0.7
python render_lerf_mask.py -m output/lerf_sam/ramen --skip_train --thresh 0.7
python render_lerf_mask.py -m output/lerf_sam/teatime --skip_train --thresh 0.7


# Render LERF-Mask: Ablation No Similarity
python render_lerf_mask.py -m output/lerf_abla_no_sim/figurines --skip_train --thresh 0.7
python render_lerf_mask.py -m output/lerf_abla_no_sim/ramen --skip_train --thresh 0.7
python render_lerf_mask.py -m output/lerf_abla_no_sim/teatime --skip_train --thresh 0.7


# Render LERF-Mask: Ablation No Dissimilarity
python render_lerf_mask.py -m output/lerf_abla_no_diss/figurines --skip_train --thresh 0.7
python render_lerf_mask.py -m output/lerf_abla_no_diss/ramen --skip_train --thresh 0.7
python render_lerf_mask.py -m output/lerf_abla_no_diss/teatime --skip_train --thresh 0.7


# Render Ablation: No 3D Loss
python render_lerf_mask.py -m output/lerf_abla/figurines --skip_train --thresh 0.7
python render_lerf_mask.py -m output/lerf_abla/ramen --skip_train --thresh 0.7
python render_lerf_mask.py -m output/lerf_abla/teatime --skip_train --thresh 0.7

# Render 3D-OVS

python render_lerf_mask.py -m output/3D-OVS/lawn --skip_train --thresh 0.7
python render_lerf_mask.py -m output/3D-OVS/bench --skip_train --thresh 0.7
python render_lerf_mask.py -m output/3D-OVS/room --skip_train --thresh 0.7
python render_lerf_mask.py -m output/3D-OVS/bed --skip_train --thresh 0.7
python render_lerf_mask.py -m output/3D-OVS/sofa --skip_train --thresh 0.7
