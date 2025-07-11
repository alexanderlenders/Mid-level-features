#!/bin/bash
# Standard encoding analysis with all features for miniclips

source /home/alexandel91/.bashrc
conda activate encoding

python ./time_generalization.py \
    --config_dir ../config.ini \
    --config default \
    --input_type "miniclips"

python ./time_gen_stats.py \
    --config_dir ../config.ini \
    --config default \
    --input_type "miniclips"

python ./time_gen_plotting.py \
    --config_dir ../config.ini \
    --config default \
    --input_type "miniclips"

python ./time_generalization.py \
    --config_dir ../config.ini \
    --config default \
    --input_type "images"

python ./time_gen_stats.py \
    --config_dir ../config.ini \
    --config default \
    --input_type "images"

python ./time_gen_plotting.py \
    --config_dir ../config.ini \
    --config default \
    --input_type "images"

python ./time_gen_stats_diff.py \
    --config_dir ../config.ini \
    --config default

python ./time_gen_diff_plotting.py \
    --config_dir ../config.ini \
    --config default
