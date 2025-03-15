#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREPARE LAYERS FOR ENCODING WITH DNN 

This script prepares the unit activations within the layers from the deep nets, 
by splitting them into training, validation and test set. It works for all kind of
different architectures.

@author: AlexanderLenders
"""
import os
import numpy as np
import argparse


def load_features(feature: str, featuresDir: str):

    features = np.load(featuresDir, allow_pickle=True)
    X_prep = features[feature]

    X_train = X_prep[0]
    X_val = X_prep[1]
    X_test = X_prep[2]

    return X_train, X_val, X_test


def prepare_layers(layerDir: str, resDir: str):
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

    # -------------------------------------------------------------------------
    # STEP 2 Create Layer Data
    # -------------------------------------------------------------------------

    for i, feature in enumerate(feature_names):
        X_train, X_val, X_test = load_features(feature, layerDir)

        layer_data_train = X_train
        layer_data_test = X_test
        layer_data_val = X_val

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
        "-d",
        "--layerdir",
        default="/scratch/agnek95/Unreal/CNN_activations_redone/2D_ResNet18/extracted/",
        type=str,
        metavar="",
        help="Directory with extracted activations; images: /scratch/agnek95/Unreal/CNN_activations_redone/2D_ResNet18/extracted/"
        "videos: /scratch/agnek95/Unreal/CNN_activations_redone/3D_ResNet18/extracted/"
        
    )
    parser.add_argument(
        "-rd",
        "--resultsdir",
        default="",
        type=str,
        metavar="",
        help= "Where to save prepared activations; images: /scratch/agnek95/Unreal/CNN_activations_redone/2D_ResNet18/pca_90_percent/prepared/"
        "videos: /scratch/agnek95/Unreal/CNN_activations_redone/3D_ResNet18/pca_90_percent/prepared/"
    )

    args = parser.parse_args()  # to get values for the arguments

    layerDir = args.layerdir
    resDir = args.resultsdir

    prepare_layers(layerDir, resDir)
