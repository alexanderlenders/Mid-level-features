#!/bin/bash
# Standard decoding analysis (images)

source /home/alexandel91/.bashrc
conda activate encoding

sub=$1

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# First step: Decoding
python ../EEG/Decoding/decoding.py \
    --config_dir ./config.ini \
    --config default \
    --input_type "images" \
    --sub $sub
