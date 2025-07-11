"""
Utility functions for loading and processing CNN activations.
"""
import os
import numpy as np

def load_activation(img_type, layer_id, activation_dir):

    fileDir = f"{layer_id}_layer_activations_" + img_type + ".npy"
    total_dir = os.path.join(activation_dir, fileDir)

    # Load EEG data
    y = np.load(total_dir, allow_pickle=True)

    return y

def load_alpha(feature, feat_dir):
    """
    Load optimal alpha hyperparameter for ridge regression.
    """
    alphaDir = os.path.join(feat_dir, "hyperparameter_tuning_resnet.pkl")

    alpha_values = np.load(alphaDir, allow_pickle=True)

    alpha = alpha_values[feature]["best_alpha_a_corr"]

    return alpha
