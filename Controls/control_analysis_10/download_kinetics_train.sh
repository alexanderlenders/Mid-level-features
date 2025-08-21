#!/bin/bash
# Script to download the Kinetics-400 dataset using torchvision.datasets.Kinetics.

num_workers=$1

echo "Removing old Kinetics-400 dataset directories if they exist..."
rm -rf /scratch/alexandel91/mid_level_features/kinetics_400/train

echo "Downloading Kinetics-400 dataset..."
python ./download_kinetics.py \
    --root_dir /scratch/alexandel91/mid_level_features/kinetics_400/train \
    --split train \
    --num_download_workers $num_workers