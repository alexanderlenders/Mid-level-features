#!/bin/bash
# Script to train the ResNet-18 on Kinetics-400 dataset.

num_workers=$1
gpus=$2

# Define variables 
data_dir="/scratch/alexandel91/mid_level_features/kinetics_400"
batch_size=128
max_epochs=20
lr=3e-4
seed=42
save_dir="/scratch/alexandel91/mid_level_features/results/CNN/training/ResNet18"
weight_decay=0.0
export OMP_NUM_THREADS=$num_workers

# Run script
torchrun --nproc_per_node=$gpus ./train.py \
    --data_dir "$data_dir" \
    --batch_size "$batch_size" \
    --num_workers "$num_workers" \
    --max_epochs "$max_epochs" \
    --lr "$lr" \
    --gpus "$gpus" \
    --seed "$seed" \
    --weight_decay "$weight_decay" \
    --save_dir "$save_dir"
