#!/bin/bash
# Standard encoding analysis with all features for miniclips

source /home/alexandel91/.bashrc
conda activate encoding

echo "Extracting the features from the frames..."
python ../CNN/Activation_extraction_and_prep/activation_extraction_cnn_videos.py \
    --config_dir ./config.ini \
    --config default \
    --init

