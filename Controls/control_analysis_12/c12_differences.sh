#!/bin/bash
# Statistics and plots for the differences between the two conditions: images and miniclips

source /home/alexandel91/.bashrc
conda activate encoding

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# First step: Bootstrapping
python ../../EEG/Stats/encoding_difference_bootstrapping.py \
    --config_dir ../config.ini \
    --config control_12
    
# Second step: Stats
python ../../EEG/Stats/encoding_difference_significance_stats.py \
    --config_dir ../config.ini \
    --config control_12

# Third step: Plotting 
python ../../EEG/Plotting/plot_encoding.py \
    --config_dir ../config.ini \
    --config control_12 \
    --input_type "difference"
