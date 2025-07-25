#!/bin/bash
# Control analysis 7 with naive correlation

source /home/alexandel91/.bashrc
conda activate encoding

python ./control_analysis_7.py \
    --config_dir ../config.ini \
    --config default \
    --input_type "miniclips" 

python ./control_analysis_7.py \
    --config_dir ../config.ini \
    --config default \
    --input_type "images"