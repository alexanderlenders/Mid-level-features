#!/bin/bash
# Script to train the ResNet-18 on Kinetics-400 dataset.

num_workers=$1
gpus=$2

# Define variables 
data_dir="/scratch/alexandel91/mid_level_features/kinetics_400"
batch_size=64
max_epochs=100
lr=3e-4
seed=42
save_dir="/scratch/alexandel91/mid_level_features/results/CNN/training/ResNet18"


# Run script
python ./train.py \
    --data_dir "$data_dir" \
    --batch_size "$batch_size" \
    --num_workers "$num_workers" \
    --max_epochs "$max_epochs" \
    --lr "$lr" \
    --gpus "$gpus" \
    --seed "$seed" \
    --save_dir "$save_dir"
