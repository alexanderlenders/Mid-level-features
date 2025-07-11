#!/bin/bash
# Encoding analysis with image annotations for miniclips EEG data

source /home/alexandel91/.bashrc
conda activate encoding
    
# First step: Hyperparameter optimization
python ./hyperparameter_optimization_c3.py \
    --config_dir ../config.ini \
    --config control_3 \
    --input_type "miniclips" 

# Second step: Encoding
python ./encoding_c3.py \
    --config_dir ../config.ini \
    --config control_3 \
    --input_type "miniclips"

# Third step: Bootstrapping
python ../../EEG/Stats/encoding_bootstrapping.py \
    --config_dir ../config.ini \
    --config control_3 \
    --input_type "miniclips"

# Fourth step: Stats
python ../../EEG/Stats/encoding_significance_stats.py \
    --config_dir ../config.ini \
    --config control_3 \
    --input_type "miniclips"

# Fifth step: Plotting 
python ../../EEG/Plotting/plot_encoding.py \
    --config_dir ../config.ini \
    --config control_3 \
    --input_type "miniclips"