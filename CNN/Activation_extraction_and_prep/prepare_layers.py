#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREPARE LAYERS FOR ENCODING WITH DNN

This script prepares the unit activations within the layers from the deep nets,
by splitting them into training, validation and test set. It works for all kind of
different architectures.

@author: Alexander Lenders, Agnessa Karapetian
"""
import os
import numpy as np
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
)


def load_features(feature: str, featuresDir: str):
    """
    Loads preprocessed feature activations from a specified .npy file and returns
    training, validation, and test splits for a given feature.
    Input:
    ----------
    Loads a pickle file containing a dictionary of features, where each feature key
    maps to a tuple of (train, validation, test) arrays.
    Returns:
    ----------
    Returns three numpy arrays corresponding to the training, validation, and test
    splits for the specified feature.
    Parameters
    ----------
    feature : str
        The key corresponding to the desired feature in the loaded numpy dictionary.
    featuresDir : str
        Path to the .pkl file containing the features dictionary.
    """
    features = np.load(featuresDir, allow_pickle=True)
    X_prep = features[feature]

    X_train = X_prep[0]
    X_val = X_prep[1]
    X_test = X_prep[2]

    return X_train, X_val, X_test


def prepare_layers(layerDir: str):
    """
    Prepare the layer activations for the encoding with the DNN.
    The function loads the layer activations from the directory and splits them into
    training, validation and test set. The data is saved in the results directory.
    """
    # -------------------------------------------------------------------------
    # STEP 1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
    feature_names = (
        "layer1.0.relu_1",
        "layer1.1.relu_1",
        "layer2.0.relu_1",
        "layer2.1.relu_1",
        "layer3.0.relu_1",
        "layer3.1.relu_1",
        "layer4.0.relu_1",
        "layer4.1.relu_1",
    )

    resDir = os.path.join(layerDir, "prepared")
    layerDir = os.path.join(layerDir, "pca", "features_resnet_scenes_avg.pkl")

    # -------------------------------------------------------------------------
    # STEP 2 Create Layer Data
    # -------------------------------------------------------------------------

    for i, feature in enumerate(feature_names):
        X_train, X_val, X_test = load_features(feature, layerDir)

        layer_data_train = X_train
        layer_data_test = X_test
        layer_data_val = X_val

        if not os.path.exists(resDir):
            os.makedirs(resDir)

        train_dir = os.path.join(
            resDir, f"{feature}_layer_activations_training.npy"
        )
        test_dir = os.path.join(
            resDir, f"{feature}_layer_activations_test.npy"
        )
        val_dir = os.path.join(
            resDir, f"{feature}_layer_activations_validation.npy"
        )

        np.save(train_dir, layer_data_train)
        np.save(test_dir, layer_data_test)
        np.save(val_dir, layer_data_val)


# -----------------------------------------------------------------------------
# STEP 4 Run function
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # add arguments / inputs
    parser.add_argument(
        "--config_dir",
        type=str,
        help="Directory to the configuration file.",
        required=True,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration.",
        required=True,
    )
    parser.add_argument(
        "-inp",
        "--input_type",
        default="images",
        metavar="",
        type=str,
        help="miniclips or images",
    )

    args = parser.parse_args()  # to get values for the arguments

    input_type = args.input_type
    config = load_config(args.config_dir, args.config)

    if input_type == "images":
        layer_dir = config.get(args.config, "save_dir_cnn_img")
    elif input_type == "miniclips":
        layer_dir = config.get(args.config, "save_dir_cnn_video")
    
    prepare_layers(layer_dir)
