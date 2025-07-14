#!/bin/bash
# Standard encoding analysis with all features for miniclips

source /home/alexandel91/.bashrc
conda activate encoding

python ../../EEG/Encoding/hyperparameter_optimization.py \
    --config_dir ../config.ini \
    --config control_6_2 \
    --input_type "miniclips" 

python ../../EEG/Encoding/encoding.py \
    --config_dir ../config.ini \
    --config control_6_2 \
    --input_type "miniclips"

# python ./control_analysis_6.py \
#     --config_dir ../config.ini \
#     --config control_6_2 \
#     --input_type "miniclips" \
#     --idea 2

# python ../../EEG/Stats/encoding_bootstrapping.py \
#     --config_dir ../config.ini \
#     --config control_6_2 \
#     --input_type "miniclips"

# python ../../EEG/Stats/encoding_significance_stats.py \
#     --config_dir ./config.ini \
#     --config control_6_2 \
#     --input_type "miniclips"

# python ../../EEG/Plotting/plot_encoding.py \
#     --config_dir ./config.ini \
#     --config control_6_2 \
#     --input_type "miniclips"