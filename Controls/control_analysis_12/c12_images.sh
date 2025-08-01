#!/bin/bash
# Control analysis 12

source /home/alexandel91/.bashrc
conda activate encoding

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# python ../../EEG/Encoding/hyperparameter_optimization.py \
#     --config_dir ../config.ini \
#     --config control_12 \
#     --input_type "images" 

# echo "Starting encoding analysis for images"
# # Fourth step: Encoding
# python ../../EEG/Encoding/encoding.py \
#     --config_dir ../config.ini \
#     --config control_12 \
#     --input_type "images"

# echo "Encoding analysis for images completed"
# echo "Starting bootstrapping for images"
# # Fifth step: Bootstrapping
# python ../../EEG/Stats/encoding_bootstrapping.py \
#     --config_dir ../config.ini \
#     --config control_12 \
#     --input_type "images"

# echo "Bootstrapping for images completed"
# echo "Starting stats for images"
# # Sixth step: Stats
# python ../../EEG/Stats/encoding_significance_stats.py \
#     --config_dir ../config.ini \
#     --config control_12 \
#     --input_type "images"

# echo "Stats for images completed"
echo "Starting plotting for images"
# Seventh step: Plotting
python ../../EEG/Plotting/plot_encoding.py \
    --config_dir ../config.ini \
    --config control_12 \
    --input_type "images" \
    --legend
echo "Plotting for images completed"