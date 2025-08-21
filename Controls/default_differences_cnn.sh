#!/bin/bash
# Statistics and plots for the differences between the two conditions: images and miniclips

source /home/alexandel91/.bashrc
conda activate encoding

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# First step: Bootstrapping
python ../CNN/Stats/encoding_difference_bootstrapping_cnn.py \
    --config_dir ./config.ini \
    --config default \
    --weighted
        
# Second step: Stats
python ../CNN/Stats/encoding_difference_significance_stats_cnn.py \
    --config_dir ./config.ini \
    --config default \
    --weighted

echo "Plotting results..."
python ../CNN/Plotting/encoding_plot_cnn.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "difference" \
    --weighted
