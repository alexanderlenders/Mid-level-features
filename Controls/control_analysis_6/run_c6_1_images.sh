#!/bin/bash
# Standard encoding analysis with all features for miniclips

source /home/alexandel91/.bashrc
conda activate encoding

python ../../EEG/Encoding/hyperparameter_optimization.py \
    --config_dir ../config.ini \
    --config control_6_1 \
    --input_type "images" 

python ../../EEG/Encoding/encoding.py \
    --config_dir ../config.ini \
    --config control_6_1 \
    --input_type "images"

python ./control_analysis_6.py \
    --config_dir ../config.ini \
    --config control_6_1 \
    --input_type "images" \
    --idea 1

python ../../EEG/Stats/encoding_bootstrapping.py \
    --config_dir ../config.ini \
    --config control_6_1 \
    --input_type "images"

python ../../EEG/Stats/encoding_significance_stats.py \
    --config_dir ./config.ini \
    --config control_6_1 \
    --input_type "images"

python ../../EEG/Plotting/plot_encoding.py \
    --config_dir ./config.ini \
    --config control_6_1 \
    --input_type "images"