#!/bin/bash
# Statistics and plots for the differences between the two conditions: images and miniclips

source /home/alexandel91/.bashrc
conda activate encoding

# First step: Bootstrapping
python ../EEG/Stats/encoding_bootstrapping.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "difference"

# Second step: Stats
python ../EEG/Stats/encoding_significance_stats.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "difference"

# Third step: Plotting 
python ../EEG/Plotting/plot_encoding.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "difference"
