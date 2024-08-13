#!/bin/bash

# Arguments
sam_checkpoint=$1
data_dir=$2/

# Run the Python script
python3 /app/sam/auto_sam.py --sam_checkpoint ${sam_checkpoint} --data_dir ${data_dir}
