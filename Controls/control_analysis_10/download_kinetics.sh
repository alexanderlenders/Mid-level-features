#!/bin/bash
# Script to download the Kinetics-400 dataset using torchvision.datasets.Kinetics.

num_workers=$1

python ./download_kinetics.py \
    --root_dir /scratch/alexandel91/mid_level_features/kinetics_400 \
    --split train \
    --num_download_workers $num_workers

python ./download_kinetics.py \
    --root_dir /scratch/alexandel91/mid_level_features/kinetics_400 \
    --split val \
    --num_download_workers $num_workers
