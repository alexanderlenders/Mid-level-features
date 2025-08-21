#!/bin/bash
# Standard encoding analysis with all features for miniclips

source /home/alexandel91/.bashrc
conda activate encoding

python ./control_analysis_4.py \
    --config_dir ../config.ini \
    --config default 