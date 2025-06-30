#!/bin/bash

# This is the script for the first control analysis:
# Alex: Repeat the encoding analysis (at least for skeleton position and action identity) for the video condition using either the first frame and the last frame.

# First step: Preprocess the features of the first frame in each video
python ../EEG/Encoding/annotation_prep_videos.py \
    --config_dir ./ \
    --config control_1 
    
# Encoding for the first frame
...
