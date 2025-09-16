#!/bin/bash
# Standard encoding analysis with all features for images

source /home/alexandel91/.bashrc
conda activate encoding

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# echo "Performing hyperparameter tuning..."
# python ../../CNN/Encoding/hyperparameter_optimization_cnn.py \
#     --config_dir ../config.ini \
#     --config control_12 \
#     --input_type "images"

# echo "Performing encoding..."
# python ../../CNN/Encoding/encoding_cnn.py \
#     --config_dir ../config.ini \
#     --config control_12 \
#     --input_type "images"

echo "Performing encoding with bootstrapping..."
python ../../CNN/Stats/encoding_bootstrapping_cnn.py \
    --config_dir ../config.ini \
    --config control_12 \
    --input_type "images" \
    --weighted

echo "Running significance stats..."
python ../../CNN/Stats/encoding_significance_stats_cnn.py \
    --config_dir ../config.ini \
    --config control_12 \
    --input_type "images" \
    --weighted

echo "Plotting results..."
python ../../CNN/Plotting/encoding_plot_cnn.py \
    --config_dir ../config.ini \
    --config control_12 \
    --input_type "images" \
    --weighted


