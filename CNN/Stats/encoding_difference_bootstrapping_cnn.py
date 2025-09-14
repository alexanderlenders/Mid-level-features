#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOOTSTRAPPING ENCODING LAYERS CNN (DIFFERENCE)

This script calculates Bootstrap 95%-CIs for the encoding accuracy for each
layer and each feature. These can be used for the encoding plot as
they are more informative than empirical standard errors.

In addition, this script calculates Bootstrap 95%-CIs for the layer
of the largest encoding peak for each feature.

@author: AlexanderLenders, Agnessa Karapetian
"""
import numpy as np
import os
import pickle
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
)


def bootstrapping_CI(n_perm: int, encoding_dir: str, weighted: bool, cnn_dir_img: str, cnn_dir_vid: str):
    """
    Bootstrapped 95%-CIs for the encoding accuracy for each timepoint and
    each feature.
    Calculates empirical CI.

    Input:
    ----------
    Output from the encoding analysis (multivariate linear regression), i.e.:
    Encoding results (multivariate linear regression), saved in a dictionary
    which contains for every feature correlation measure, i.e.:
        encoding_results[feature]['correlation']

    Returns:
    ----------
    Dictionary with level 1 features and level 2 with 95% CIs (values)
    for each timepoint (key), i.e. results[feature][timepoint]

    Parameters
    ----------
    n_perm : int
        Number of permutations for bootstrapping
    encoding_dir : str
        Where encoding results are saved
    weighted : bool
        If True, uses weighted regression results.
    cnn_dir_img : str
        Directory where the explained variance per PC for images is stored.
    cnn_dir_vid : str
        Directory where the explained variance per PC for videos is stored.
    """
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
    if weighted:
        workDir_img = os.path.join(encoding_dir, "images", "weighted")
        workDir_vid = os.path.join(encoding_dir, "miniclips", "weighted")
    else:
        workDir_img = os.path.join(encoding_dir, "images", "unweighted")
        workDir_vid = os.path.join(encoding_dir, "miniclips", "unweighted")

    saveDir = os.path.join(encoding_dir, "difference", "stats")

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    layers_names = (
        "layer1.0.relu_1",
        "layer1.1.relu_1",
        "layer2.0.relu_1",
        "layer2.1.relu_1",
        "layer3.0.relu_1",
        "layer3.1.relu_1",
        "layer4.0.relu_1",
        "layer4.1.relu_1",
    )
    n_layers = len(layers_names)
    feature_names = (
        "edges",
        "world_normal",
        "lighting",
        "scene_depth",
        "reflectance",
        "action",
        "skeleton",
    )

    explained_var_dir_img = os.path.join(cnn_dir_img, "pca")
    explained_var_dir_vid = os.path.join(cnn_dir_vid, "pca")

    # set random seed (for reproduction)
    np.random.seed(42)

    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------
    fileDir_img = os.path.join(workDir_img, "encoding_layers_resnet.pkl")
    fileDir_vid = os.path.join(workDir_vid, "encoding_layers_resnet.pkl")

    encoding_results_img = np.load(fileDir_img, allow_pickle=True)
    encoding_results_vid = np.load(fileDir_vid, allow_pickle=True)

    regression_features_img = dict.fromkeys(feature_names)
    regression_features_vid = dict.fromkeys(feature_names)

    for feature in feature_names:
        regression_features_img[feature] = encoding_results_img[feature][
            "correlation"
        ]
        regression_features_vid[feature] = encoding_results_vid[feature][
            "correlation"
        ]

    features_results = {}

    for feature in feature_names:
        results_img = regression_features_img[feature]
        results_vid = regression_features_vid[feature]

        # ---------------------------------------------------------------------
        # STEP 2.3 Bootstrapping
        # ---------------------------------------------------------------------
        bt_data = np.zeros((n_layers, n_perm))

        for l, layer in enumerate(layers_names):
            layer_data_img = results_img[layer]
            layer_data_vid = results_vid[layer]
            num_comp_layer_img = layer_data_img.shape[0]
            num_comp_layer_vid = layer_data_vid.shape[0]

            # Load explained variance per PC
            explained_var_dir_layer_img = os.path.join(
                explained_var_dir_img, layer, "explained_variance.pkl"
            )
            explained_var_dir_layer_vid = os.path.join(
                explained_var_dir_vid, layer, "explained_variance.pkl"
            )
            with open(explained_var_dir_layer_vid, "rb") as f:
                explained_var_vid = pickle.load(f)
            with open(explained_var_dir_layer_img, "rb") as f:
                explained_var_img = pickle.load(f)

            explained_var_img = np.array(explained_var_img["explained_variance"])
            explained_var_vid = np.array(explained_var_vid["explained_variance"])

            for perm in range(n_perm):
                idx_img = np.random.choice(
                    range(num_comp_layer_img),
                    size=num_comp_layer_img,
                    replace=True)
                idx_vid = np.random.choice(
                    range(num_comp_layer_vid),
                    size=num_comp_layer_vid,
                    replace=True)
                
                # Resample correlations and weights
                layer_data_img = layer_data_img[idx_img]
                layer_data_vid = layer_data_vid[idx_vid]
                explained_var_img = explained_var_img[idx_img]
                explained_var_vid = explained_var_vid[idx_vid]

                # Get weighted sum across units
                mean_p_layer_img = np.sum(
                    layer_data_img * explained_var_img
                ) / np.sum(explained_var_img)
                mean_p_layer_vid = np.sum(
                    layer_data_vid * explained_var_vid
                ) / np.sum(explained_var_vid)

                bt_data[l, perm] = mean_p_layer_img - mean_p_layer_vid

        # ---------------------------------------------------------------------
        # STEP 2.4 Calculate 95%-CI
        # ---------------------------------------------------------------------
        upper = int(np.ceil(n_perm * 0.975)) - 1
        lower = int(np.ceil(n_perm * 0.025)) - 1

        ci_dict = {}

        for l, layer in enumerate(layers_names):
            l_data = bt_data[l, :]
            l_data.sort()
            ci_dict["{}".format(layer)] = [l_data[lower], l_data[upper]]

        features_results[feature] = ci_dict

    # -------------------------------------------------------------------------
    # STEP 2.5 Save CI
    # -------------------------------------------------------------------------
    # Save the dictionary

    fileDir = "encoding_layers_CI95_accuracy_difference.pkl"

    savefileDir = os.path.join(saveDir, fileDir)

    with open(savefileDir, "wb") as f:
        pickle.dump(features_results, f)


# ------------------------------------------------------------------------------


def bootstrapping_CI_peak_layer(
    n_perm: int, encoding_dir: str, weighted: bool, cnn_dir_img: str, cnn_dir_vid: str
):
    """
    Bootstrapped 95%-CIs for the layer of the largest encoding peak
    for each feature.

    Input:
    ----------
    Output from the encoding analysis (multivariate linear regression), i.e.:
    Encoding results (multivariate linear regression), saved in a dictionary
    which contains for every feature correlation measure, i.e.:
        encoding_results[feature]['correlation']

    Returns:
    ----------
    Dictionary with level 1 features and level 2 with 95% CIs (values)
    for each peak (key), i.e. results[feature][peak]

    Parameters
    ----------
    n_perm : int
        Number of permutations for bootstrapping
    encoding_dir : str
        Where encoding results are saved
    weighted : bool
        If True, uses weighted regression results.
    cnn_dir_img : str
        Directory where the explained variance per PC for images is stored.
    cnn_dir_vid : str
        Directory where the explained variance per PC for videos is stored.
    """
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
    if weighted:
        workDir_img = os.path.join(encoding_dir, "images", "weighted")
        workDir_vid = os.path.join(encoding_dir, "miniclips", "weighted")
    else:
        workDir_img = os.path.join(encoding_dir, "images", "unweighted")
        workDir_vid = os.path.join(encoding_dir, "miniclips", "unweighted")

    saveDir = os.path.join(encoding_dir, "difference", "stats")

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    layers_names = (
        "layer1.0.relu_1",
        "layer1.1.relu_1",
        "layer2.0.relu_1",
        "layer2.1.relu_1",
        "layer3.0.relu_1",
        "layer3.1.relu_1",
        "layer4.0.relu_1",
        "layer4.1.relu_1",
    )
    n_layers = len(layers_names)

    feature_names = (
        "edges",
        "world_normal",
        "lighting",
        "scene_depth",
        "reflectance",
        "action",
        "skeleton",
    )

    explained_var_dir_img = os.path.join(cnn_dir_img, "pca")
    explained_var_dir_vid = os.path.join(cnn_dir_vid, "pca")

    # set random seed (for reproduction)
    np.random.seed(42)

    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------
    fileDir_img = os.path.join(workDir_img, "encoding_layers_resnet.pkl")
    fileDir_vid = os.path.join(workDir_vid, "encoding_layers_resnet.pkl")

    encoding_results_img = np.load(fileDir_img, allow_pickle=True)
    encoding_results_vid = np.load(fileDir_vid, allow_pickle=True)

    corr_img = {}
    corr_vid = {}

    for feature in feature_names:
        corr_img[feature] = encoding_results_img[feature][
            "correlation"
        ]
        corr_vid[feature] = encoding_results_vid[feature][
            "correlation"
        ]

    ci_diff_peaks_all = {}

    for i, feature in enumerate(feature_names):

        results_img = corr_img[feature]
        results_vid = corr_vid[feature]

        # ---------------------------------------------------------------------
        # STEP 2.3 Bootstrapping
        # ---------------------------------------------------------------------
        bt_diff_peaks = np.zeros((n_perm,))

        # get true data
        num_comp_layer_img = {}
        num_comp_layer_vid = {}
        for l, layer in enumerate(layers_names):
            layer_data_img = results_img[layer]
            layer_data_vid = results_vid[layer]

            num_comp_layer_img[layer] = layer_data_img.shape[0]
            num_comp_layer_vid[layer] = layer_data_vid.shape[0]

        # Find ground truth difference in peak latencies of vid. vs img
        peak_img_true = np.argmax(
            encoding_results_img[feature]["correlation_average"]
        )
        peak_vid_true = np.argmax(
            encoding_results_vid[feature]["correlation_average"]
        )
        diff_in_peak_true = np.abs(peak_img_true - peak_vid_true)

        # Permute and calculate peak latencies for bootstrap samples
        for perm in range(n_perm):
            perm_mean_vid = np.zeros((n_layers,))
            perm_mean_img = np.zeros((n_layers,))
            for l, layer in enumerate(layers_names):
                layer_data_img = results_img[layer]
                layer_data_vid = results_vid[layer]

                # Load explained variance per PC
                explained_var_dir_layer_img = os.path.join(
                    explained_var_dir_img, layer, "explained_variance.pkl"
                )
                explained_var_dir_layer_vid = os.path.join(
                    explained_var_dir_vid, layer, "explained_variance.pkl"
                )
                with open(explained_var_dir_layer_vid, "rb") as f:
                    explained_var_vid = pickle.load(f)
                with open(explained_var_dir_layer_img, "rb") as f:
                    explained_var_img = pickle.load(f)

                explained_var_img = np.array(explained_var_img["explained_variance"])
                explained_var_vid = np.array(explained_var_vid["explained_variance"])

                idx_img = np.random.choice(
                    range(num_comp_layer_img[layer]),
                    size=num_comp_layer_img[layer],
                    replace=True,
                )
                idx_vid = np.random.choice(
                    range(num_comp_layer_vid[layer]),
                    size=num_comp_layer_vid[layer], 
                    replace=True,
                )

                # Resample correlations and weights
                layer_data_img = layer_data_img[idx_img]
                layer_data_vid = layer_data_vid[idx_vid]
                explained_var_img = explained_var_img[idx_img]
                explained_var_vid = explained_var_vid[idx_vid]

                perm_mean_vid[l] = np.sum(layer_data_vid * explained_var_vid) / np.sum(
                    explained_var_vid
                )
                perm_mean_img[l] = np.sum(layer_data_img * explained_var_img) / np.sum(
                    explained_var_img
                )

            # difference in the peaks
            peak_lat_vid = np.argmax(perm_mean_vid)
            peak_lat_img = np.argmax(perm_mean_img)
            diff_in_peakl = np.abs(peak_lat_img - peak_lat_vid)
            bt_diff_peaks[perm] = diff_in_peakl

        # ---------------------------------------------------------------------
        # STEP 2.4 Calculate 95%-CI
        # ---------------------------------------------------------------------
        upper = round(n_perm * 0.975) - 1
        lower = round(n_perm * 0.025) - 1

        bt_diff_peaks.sort()
        lower_ci = bt_diff_peaks[lower]
        upper_ci = bt_diff_peaks[upper]

        if (
            lower_ci > upper_ci
        ):  # because of absolute difference calculated earlier
            upper_ci_final = lower_ci
            lower_ci_final = upper_ci
        else:
            upper_ci_final = upper_ci
            lower_ci_final = lower_ci

        ci_diff_peaks_all["{}".format(feature)] = [
            lower_ci_final,
            diff_in_peak_true,
            upper_ci_final,
        ]

    # -------------------------------------------------------------------------
    # STEP 2.5 Save CI
    # -------------------------------------------------------------------------
    # Save the dictionary

    fileDir = "encoding_difference_in_peak.pkl"

    savefileDir = os.path.join(saveDir, fileDir)

    with open(savefileDir, "wb") as f:
        pickle.dump(ci_diff_peaks_all, f)


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
        "-np",
        "--num_perm",
        default=10000,
        type=int,
        metavar="",
        help="Number of permutations",
    )
    parser.add_argument("--weighted", action="store_true")

    args = parser.parse_args()  # to get values for the arguments

    config = load_config(args.config_dir, args.config)
    encoding_dir = config.get(args.config, "save_dir_cnn")
    n_perm = args.num_perm

    if args.weighted:
        weighted = True
    else:
        weighted = False

    cnn_dir_img = config.get(args.config, "save_dir_cnn_img")
    cnn_dir_vid = config.get(args.config, "save_dir_cnn_vid")

    bootstrapping_CI(n_perm, encoding_dir, weighted, cnn_dir_img, cnn_dir_vid)
    bootstrapping_CI_peak_layer(n_perm, encoding_dir, weighted, cnn_dir_img, cnn_dir_vid)
