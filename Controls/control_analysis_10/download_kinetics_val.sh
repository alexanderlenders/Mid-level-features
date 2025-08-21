#!/bin/bash
# Script to download the Kinetics-400 dataset using torchvision.datasets.Kinetics.

num_workers=$1

echo "Removing old Kinetics-400 validation dataset directory if it exists..."
rm -rf /scratch/alexandel91/mid_level_features/kinetics_400/val

echo "Downloading Kinetics-400 validation dataset..."
python ./download_kinetics.py \
    --root_dir /scratch/alexandel91/mid_level_features/kinetics_400/val \
    --split val \
    --num_download_workers $num_workers
