#!/bin/bash

echo "Extracting the features from the frames..."
python ../../CNN/Activation_extraction_and_prep/activation_extraction_cnn_images.py \
    --config_dir ../config.ini \
    --config control_11 \
    --init \
    --transform "vid"
