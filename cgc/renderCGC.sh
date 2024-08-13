#!/bin/bash

# Activate Conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate gaussian_grouping
 
# Arguments
source_path=$1
model_path=$2
 
 
# Run the Python script
python3 renderCGC.py \
  --source_path ${source_path} \
  --model_path ${model_path}
