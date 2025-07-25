#!/bin/bash
# Control analysis 7 with CKA

source /home/alexandel91/.bashrc
conda activate encoding

python ./control_analysis_7_cka.py \
    --config_dir ../config.ini \
    --config default \
    --input_type "miniclips" 

python ./control_analysis_7_cka.py \
    --config_dir ../config.ini \
    --config default \
    --input_type "images"