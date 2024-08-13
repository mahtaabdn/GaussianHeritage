#!/bin/bash

# Activate Conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate gaussian_grouping

# Arguments
source_path=$1
config_file=$2

# Hardcoded values
ip="127.0.0.1"
port=6009
debug_from=-1
use_wandb=false

# Run the Python script
python3 /app/cgc/train.py \
  --source_path ${source_path} \
  --config_file ${config_file} \
  --ip ${ip} \
  --port ${port} \
  --debug_from ${debug_from} \
  $( [[ ${use_wandb} == true ]] && echo "--use_wandb" )
