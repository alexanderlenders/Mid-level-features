#!/bin/bash
# Standard encoding analysis with all features for miniclips

source /home/alexandel91/.bashrc
conda activate encoding

python ./control_analysis_7_cca.py \
    --config_dir ../config.ini \
    --config default \
    --input_type "miniclips" 

python ./control_analysis_7_cca.py \
    --config_dir ../config.ini \
    --config default \
    --input_type "images"