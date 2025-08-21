#!/bin/bash
# Control analysis 12

source /home/alexandel91/.bashrc
conda activate encoding

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# python ../../EEG/Encoding/hyperparameter_optimization.py \
#     --config_dir ../config.ini \
#     --config control_12 \
#     --input_type "miniclips" 

# echo "Starting encoding analysis for miniclips"
# # Fourth step: Encoding
# python ../../EEG/Encoding/encoding.py \
#     --config_dir ../config.ini \
#     --config control_12 \
#     --input_type "miniclips"

# echo "Encoding analysis for miniclips completed"
# echo "Starting bootstrapping for miniclips"
# # Fifth step: Bootstrapping
# python ../../EEG/Stats/encoding_bootstrapping.py \
#     --config_dir ../config.ini \
#     --config control_12 \
#     --input_type "miniclips"

# # echo "Bootstrapping for miniclips completed"
# echo "Starting stats for miniclips"
# # Sixth step: Stats
# python ../../EEG/Stats/encoding_significance_stats.py \
#     --config_dir ../config.ini \
#     --config control_12 \
#     --input_type "miniclips"

# echo "Stats for miniclips completed"
echo "Starting plotting for miniclips"
# Seventh step: Plotting 
python ../../EEG/Plotting/plot_encoding.py \
    --config_dir ../config.ini \
    --config control_12 \
    --input_type "miniclips"

echo "Plotting for miniclips completed"