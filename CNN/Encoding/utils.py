"""
Utility functions for loading and processing CNN activations.
"""

import os
import numpy as np


def load_activation(img_type: str, layer_id: str, activation_dir: str):
    """
    Utility function to load CNN activations for a specific layer and image type.
    """
    fileDir = f"{layer_id}_layer_activations_" + img_type + ".npy"
    total_dir = os.path.join(activation_dir, fileDir)

    # Load EEG data
    y = np.load(total_dir, allow_pickle=True)

    return y


def load_alpha(feature, feat_dir, tp=None):
    """
    Load optimal alpha hyperparameter for ridge regression.
    """
    alphaDir = os.path.join(feat_dir, "hyperparameter_tuning_resnet.pkl")

    alpha_values = np.load(alphaDir, allow_pickle=True)

    if tp:
        alpha = alpha_values[feature]["best_alpha_corr"]
        alpha = alpha[tp]
    else:
        alpha = alpha_values[feature]["best_alpha_a_corr"]

    return alpha
