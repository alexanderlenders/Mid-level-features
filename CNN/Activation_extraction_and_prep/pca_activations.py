#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REDUCE DIM OF EXTRACTED CNN ACTIVATIONS USING PCA

@author: Alexander Lenders, Agnessa Karapetian
"""
import os
import numpy as np
import scipy.io
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
)


def pca(
    features_train,
    features_val,
    features_test,
    pca_method: str = "linear",
    n_comp: int = 1000,
    diagnostics: bool = True,
    pca_save_dir: str = None,
    min_var: float = None,
):
    """
    Note: This implements a (simple) PCA using SVD. One could also
    implement a multilinear PCA for the images with RGB channels.

    Parameters
    ----------
    features_train: numpy array
        Matrix with dimensions num_videos x num_components
        In the case of RGB channels one first has to flatten the matrix
    features_test: numpy array
        Matrix with dimensions num_videos x num_components
        In the case of RGB channels one first has to flatten the matrix
    features_val: numpy array
        Matrix with dimensions num_videos x num_components
        In the case of RGB channels one first has to flatten the matrix
    pca_method: str
        Whether to apply a "linear" or "nonlinear" Kernel PCA
    n_comp: int
        Number of fitted components in the PCA
    diagnostics: bool
        Whether to print diagnostics
    pca_save_dir: str
        Where to save the diagnostics
    min_var: float
        Minimum variance to be explained by the PCA components.

    """

    # Standard Scaler (Best practice, see notes)
    scaler = StandardScaler().fit(features_train)
    scaled_train = scaler.transform(features_train)
    scaled_test = scaler.transform(features_test)
    scaled_val = scaler.transform(features_val)

    min_var_criterion = False
    max_iter = 5
    iteration = 0

    while not min_var_criterion and iteration < max_iter:
        iteration += 1
        print("=====================================================")
        print("iter: ", iteration)
        print("n_comp: ", n_comp)
        print("=====================================================")

        # Fit PCA on train_data
        if pca_method == "linear":
            pca_image = PCA(n_components=n_comp, random_state=42)
        elif pca_method == "nonlinear":
            pca_image = KernelPCA(
                n_components=n_comp,
                kernel="poly",
                degree=4,
                random_state=42,
            )

        pca_image.fit(scaled_train)

        if pca_method == "linear":
            # Get explained variance
            per_var = pca_image.explained_variance_ratio_ * 100
            labels = ["PC" + str(x) for x in range(1, len(per_var) + 1)]

            explained_variance_dict = {
                "labels": labels,
                "explained_variance": per_var,
            }

            explained_variance = np.sum(per_var)

        if explained_variance >= min_var:
            min_var_criterion = True

            # Transform data
            pca_train = pca_image.transform(scaled_train)
            pca_test = pca_image.transform(scaled_test)
            pca_val = pca_image.transform(scaled_val)

            # Compute cumulative variance
            cumulative_variance = np.cumsum(per_var)

            # Check where cumulative variance exceeds min_var
            min_var_index = np.where(cumulative_variance >= min_var * 100)[0][
                0
            ]

            print("=====================================================")
            print("Min Var Index: ", min_var_index)
            print("=====================================================")

            # Choose the number of components based on the cumulative variance
            pca_train = pca_train[:, : min_var_index + 1]
            pca_val = pca_val[:, : min_var_index + 1]
            pca_test = pca_test[:, : min_var_index + 1]

            # Choose labels and explained variance based on the min_var_index
            explained_variance_dict["labels"] = labels[: min_var_index + 1]
            explained_variance_dict["explained_variance"] = per_var[
                : min_var_index + 1
            ]

            if diagnostics:
                print(
                    "Explained Variance by Principal Components in Training Set:"
                )
                for label, variance in zip(
                    explained_variance_dict["labels"],
                    explained_variance_dict["explained_variance"],
                ):
                    print(f"{label}: {variance}%")

                print(
                    f"Total Explained Variance in Training Set: {explained_variance}%"
                )

            if pca_save_dir:
                if not os.path.exists(pca_save_dir):
                    os.makedirs(pca_save_dir)
                save_expl_var = os.path.join(
                    pca_save_dir, "explained_variance.pkl"
                )
                with open(save_expl_var, "wb") as f:
                    pickle.dump(explained_variance_dict, f)
        else:
            n_comp = n_comp + 250

    return pca_train, pca_val, pca_test


def apply_pca(feature_dir: str, character_dir: str, num_comp: int = 1000):
    """
    Apply PCA to the extracted features (extra script to reduce RAM usage)
    """
    # --------------------------------------
    # STEP 1: Get indices for train, val, test data
    # --------------------------------------
    num_videos = 1440

    save_dir = os.path.join(feature_dir, "pca")

    MIN_VAR = 0.9
    (print("Applying PCA with minimum variance of: ", MIN_VAR))

    # classify data sets
    # character Identity
    meta_data = scipy.io.loadmat(character_dir)
    char_data = pd.DataFrame((meta_data.get("meta_data")))
    rows, cols = char_data.shape
    char_meta = pd.DataFrame(np.zeros((rows, cols)))

    for col in range(cols):
        extracted_values = [item.item() for item in char_data.iloc[:, col]]
        char_meta[char_meta.columns[col]] = extracted_values

    # Get information about train and test data
    split_data = np.array(char_meta.iloc[:, 1])
    # 0 -> training data
    # 1 -> validation data
    # 2 -> test data
    index = np.arange(0, num_videos)
    split_data = np.column_stack([index, split_data])
    train_data = split_data[:, 0][split_data[:, 1] == 0]
    val_data = split_data[:, 0][split_data[:, 1] == 1]
    test_data = split_data[:, 0][split_data[:, 1] == 2]

    # --------------------------------------
    # STEP 2: APPLY PCA
    # --------------------------------------
    # If PCA is applied, then print diagnostics!
    features = [
        "layer1.0.relu_1",
        "layer1.1.relu_1",
        "layer2.0.relu_1",
        "layer2.1.relu_1",
        "layer3.0.relu_1",
        "layer3.1.relu_1",
        "layer4.0.relu_1",
        "layer4.1.relu_1",
    ]

    # Create dict with all PCA features
    pca_features = {}

    for layer in features:
        datasets = []
        pca_save_dir_layer = os.path.join(save_dir, layer)
        layer_dir = os.path.join(feature_dir, f"features_resnet_{layer}.pkl")

        if layer == "layer1.0.relu_1":

            with open(layer_dir, "rb") as file:
                layer_1_0 = pickle.load(file)

            features_train = layer_1_0[train_data, :]
            features_val = layer_1_0[val_data, :]
            features_test = layer_1_0[test_data, :]

            print("Applying PCA to layer fc")
            pca_features_train, pca_features_val, pca_features_test = pca(
                features_train,
                features_val,
                features_test,
                diagnostics=True,
                pca_save_dir=pca_save_dir_layer,
                n_comp=num_comp,
                min_var=MIN_VAR,
            )
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)

            del features_train, features_val, features_test, layer_1_0

        elif layer == "layer1.1.relu_1":

            with open(layer_dir, "rb") as file:
                layer_1_1 = pickle.load(file)

            features_train = layer_1_1[train_data, :]
            features_val = layer_1_1[val_data, :]
            features_test = layer_1_1[test_data, :]

            print("Applying PCA to layer fc")
            pca_features_train, pca_features_val, pca_features_test = pca(
                features_train,
                features_val,
                features_test,
                diagnostics=True,
                pca_save_dir=pca_save_dir_layer,
                n_comp=num_comp,
                min_var=MIN_VAR,
            )
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)

            del features_train, features_val, features_test, layer_1_1

        elif layer == "layer2.0.relu_1":

            with open(layer_dir, "rb") as file:
                layer_2_0 = pickle.load(file)

            features_train = layer_2_0[train_data, :]
            features_val = layer_2_0[val_data, :]
            features_test = layer_2_0[test_data, :]

            print("Applying PCA to layer fc")
            pca_features_train, pca_features_val, pca_features_test = pca(
                features_train,
                features_val,
                features_test,
                diagnostics=True,
                pca_save_dir=pca_save_dir_layer,
                n_comp=num_comp,
                min_var=MIN_VAR,
            )
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)

            del features_train, features_val, features_test, layer_2_0

        elif layer == "layer2.1.relu_1":

            with open(layer_dir, "rb") as file:
                layer_2_1 = pickle.load(file)

            features_train = layer_2_1[train_data, :]
            features_val = layer_2_1[val_data, :]
            features_test = layer_2_1[test_data, :]

            print("Applying PCA to layer fc")
            pca_features_train, pca_features_val, pca_features_test = pca(
                features_train,
                features_val,
                features_test,
                diagnostics=True,
                pca_save_dir=pca_save_dir_layer,
                n_comp=num_comp,
                min_var=MIN_VAR,
            )
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)

            del features_train, features_val, features_test, layer_2_1

        elif layer == "layer3.0.relu_1":

            with open(layer_dir, "rb") as file:
                layer_3_0 = pickle.load(file)

            features_train = layer_3_0[train_data, :]
            features_val = layer_3_0[val_data, :]
            features_test = layer_3_0[test_data, :]

            print("Applying PCA to layer fc")
            pca_features_train, pca_features_val, pca_features_test = pca(
                features_train,
                features_val,
                features_test,
                diagnostics=True,
                pca_save_dir=pca_save_dir_layer,
                n_comp=num_comp,
                min_var=MIN_VAR,
            )
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)

            del features_train, features_val, features_test, layer_3_0

        elif layer == "layer3.1.relu_1":

            with open(layer_dir, "rb") as file:
                layer_3_1 = pickle.load(file)

            features_train = layer_3_1[train_data, :]
            features_val = layer_3_1[val_data, :]
            features_test = layer_3_1[test_data, :]

            print("Applying PCA to layer fc")
            pca_features_train, pca_features_val, pca_features_test = pca(
                features_train,
                features_val,
                features_test,
                diagnostics=True,
                pca_save_dir=pca_save_dir_layer,
                n_comp=num_comp,
                min_var=MIN_VAR,
            )
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)

            del features_train, features_val, features_test, layer_3_1

        elif layer == "layer4.0.relu_1":

            with open(layer_dir, "rb") as file:
                layer_4_0 = pickle.load(file)

            features_train = layer_4_0[train_data, :]
            features_val = layer_4_0[val_data, :]
            features_test = layer_4_0[test_data, :]

            print("Applying PCA to layer fc")
            pca_features_train, pca_features_val, pca_features_test = pca(
                features_train,
                features_val,
                features_test,
                diagnostics=True,
                pca_save_dir=pca_save_dir_layer,
                n_comp=num_comp,
                min_var=MIN_VAR,
            )
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)

            del features_train, features_val, features_test, layer_4_0

        elif layer == "layer4.1.relu_1":

            with open(layer_dir, "rb") as file:
                layer_4_1 = pickle.load(file)

            features_train = layer_4_1[train_data, :]
            features_val = layer_4_1[val_data, :]
            features_test = layer_4_1[test_data, :]

            print("Applying PCA to layer fc")
            pca_features_train, pca_features_val, pca_features_test = pca(
                features_train,
                features_val,
                features_test,
                diagnostics=True,
                pca_save_dir=pca_save_dir_layer,
                n_comp=num_comp,
                min_var=MIN_VAR,
            )
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)

            del features_train, features_val, features_test, layer_4_1

        pca_features[layer] = datasets

    # --------------------------------------
    # STEP 3: SAVE FEATURES #
    # --------------------------------------
    features_dir = os.path.join(save_dir, "features_resnet_scenes_avg.pkl")

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    with open(features_dir, "wb") as f:
        pickle.dump(pca_features, f)


if __name__ == "__main__":
    # parser
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

    character_dir = config.get(args.config, "character_metadata_dir")

    if input_type == "images":
        feature_dir = config.get(args.config, "save_dir_cnn_img")
    elif input_type == "miniclips":
        feature_dir = config.get(args.config, "save_dir_cnn_video")

    # run
    apply_pca(
        feature_dir=feature_dir,
        character_dir=character_dir,
        num_comp=1000,
    )
