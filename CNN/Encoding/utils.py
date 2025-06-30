"""
Utility functions for loading and processing CNN activations.
"""
import os
import numpy as np

def load_activation(input_type, img_type, layer_id, activation_dir):
    if input_type == "images":
        layer_dir = os.path.join(activation_dir, "2D_ResNet18", "pca_90_percent", "prepared")
    elif input_type == "miniclips":
        layer_dir = os.path.join(activation_dir, "3D_ResNet18", "pca_90_percent", "prepared")

    fileDir = f"{layer_id}_layer_activations_" + img_type + ".npy"
    total_dir = os.path.join(layer_dir, fileDir)

    # Load EEG data
    y = np.load(total_dir, allow_pickle=True)

    return y
