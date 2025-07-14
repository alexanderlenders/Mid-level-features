#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STATS ENCODING LAYERS CNN

This script implements the statistical tests for the encoding analyis predicting
the unit activity within the action DNN layers based on the single gaming-engine
features.

It conducts permutation tests by permuting the PCA components per layer,
alpha = .05, Benjamini-Hochberg correction.

@author: AlexanderLenders, Agnessa Karapetian
"""
import argparse
import os
import numpy as np
import pickle
from scipy.stats import rankdata
from statsmodels.stats.multitest import fdrcorrection
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
print(project_root)
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
)


def encoding_stats(n_perm, alpha_value, tail, total_var, weighted, encoding_dir):
    """
    Input:
    ----------
    Output from the encoding analysis (multivariate linear regression), i.e.:
    Encoding results (multivariate linear regression), saved in a dictionary
    which contains for every feature correlation measure, i.e.:
        encoding_results[feature]['correlation']

    Returns:
    ----------
    regression_features, dictionary with the following keys for each feature:
    a. Uncorrected_p_values_map
        - Contains uncorrected p-values
    b. Corrected_p_values_map
        - Contains corrected p-values
    c. Boolean_statistical_map
        - Contains boolean values, if True -> corrected p-value is lower than .05

    Parameters
    ----------
    n_perm : int
        Number of permutations for bootstrapping
    saveDir : str
        Where to save the results
    total_var : int
        Total variance explained by all PCA components
    alpha_value : int
        Significance level
    tail : str
        One-sided or two-sided test

    """
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
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

    feature_names = (
        "edges",
        "world_normal",
        "lighting",
        "scene_depth",
        "reflectance",
        "action",
        "skeleton",
    )

    features_dict = dict.fromkeys(feature_names)

    num_layers = len(layers_names)

    if weighted:
        workDir_img = os.path.join(encoding_dir, "images", "weighted")
        workDir_vid = os.path.join(encoding_dir, "miniclips", "weighted")
    else:
        workDir_img = os.path.join(encoding_dir, "images", "unweighted")
        workDir_vid = os.path.join(encoding_dir, "miniclips", "unweighted")

    saveDir = os.path.join(encoding_dir, "difference", "stats")

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    np.random.seed(42)

    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------
    fileDir_img = os.path.join(workDir_img, "encoding_layers_resnet.pkl")
    fileDir_vid = os.path.join(workDir_vid, "encoding_layers_resnet.pkl")

    encoding_results_img = np.load(fileDir_img, allow_pickle=True)
    encoding_results_vid = np.load(fileDir_vid, allow_pickle=True)

    # define matrix where to save the values
    regression_features = dict.fromkeys(feature_names)

    for feature in features_dict.keys():

        # create statistical map
        stat_map = np.zeros((n_perm, num_layers))
        for l, layer in enumerate(layers_names):
            # for l, corr_layer in enumerate(encoding_results[feature]['weighted_correlations'].values()):
            corr_img = encoding_results_img[feature]["weighted_correlations"][
                layer
            ]
            corr_vid = encoding_results_vid[feature]["weighted_correlations"][
                layer
            ]

            num_comp_layer_img = corr_img.shape[0]
            num_comp_layer_vid = corr_vid.shape[0]

            all_results = np.vstack(
                (corr_vid, corr_img))
            labels = np.array(
                [0] * num_comp_layer_vid + [1] * num_comp_layer_img)
            # 0 for video, 1 for image

            # create mean for each layer over all images
            # this is our "original data" and permutation 1 in the stat_map
            mean_orig_img = np.sum(corr_img) / total_var
            mean_orig_vid = np.sum(corr_vid) / total_var
            # Order does not matter, as we are using a two-tailed test
            mean_orig_diff = mean_orig_img - mean_orig_vid

            stat_map[0, l] = mean_orig_diff

            for permutation in range(1, n_perm):
                # Shuffle the labels
                shuffled_labels = np.random.permutation(labels)

                # Assign to new permuted groups
                group_1 = all_results[shuffled_labels == 0, :]
                group_2 = all_results[shuffled_labels == 1, :]

                mean_group_1 = np.sum(group_1) / total_var
                mean_group_2 = np.sum(group_2) / total_var

                stat = mean_group_2 - mean_group_1

                stat_map[permutation, l] = stat
        # ---------------------------------------------------------------------
        # Calculate ranks and p-values
        # ---------------------------------------------------------------------
        # get ranks (over all permutations), this gives us a distribution
        if tail == "right":
            ranks = np.apply_along_axis(rankdata, 0, stat_map)

        elif tail == "both":
            abs_values = np.absolute(stat_map)
            ranks = np.apply_along_axis(rankdata, 0, abs_values)

        # calculate p-values
        # create a matrix with nperm+1 values in every element (to account for
        # the observed test statistic)
        sub_matrix = np.full((n_perm, num_layers), (n_perm + 1))
        p_map = (sub_matrix - ranks) / n_perm
        p_values = p_map[0, :]

        # ---------------------------------------------------------------------
        # Benjamini-Hochberg correction
        # ---------------------------------------------------------------------
        rejected, p_values_corr = fdrcorrection(
            p_values, alpha=alpha_value, is_sorted=False
        )

        stats_results = {}
        stats_results["Uncorrected_p_values_map"] = p_values
        stats_results["Corrected_p_values_map"] = p_values_corr
        stats_results["Boolean_statistical_map"] = rejected

        regression_features[feature] = stats_results
    # -------------------------------------------------------------------------
    # STEP 2.6 Save hyperparameters and scores
    # -------------------------------------------------------------------------
    # Save the dictionary
    fileDir = "encoding_stats_layers_{}_difference.pkl".format(tail)
    savefileDir = os.path.join(saveDir, fileDir)

    with open(savefileDir, "wb") as f:
        pickle.dump(regression_features, f)

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
    parser.add_argument(
        "-tp",
        "--num_tp",
        default=5,
        type=int,
        metavar="",
        help="Number of timepoints",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        default=0.05,
        type=int,
        metavar="",
        help="Significance level (alpha)",
    )
    parser.add_argument(
        "-t",
        "--tail",
        default="both",
        type=str,
        metavar="",
        help="One-sided: right, two-sided: both",
    )
    parser.add_argument(
        "-tv",
        "--total_var",
        help="Total variance explained by all PCA components together",
        default=90,
    )
    parser.add_argument(
        '--weighted', 
        action='store_true'
    )

    args = parser.parse_args()  # to get values for the arguments

    config = load_config(args.config_dir, args.config)
    encoding_dir = config.get(args.config, "save_dir_cnn")
    n_perm = args.num_perm
    timepoints = args.num_tp
    alpha_value = args.alpha
    tail = args.tail
    total_var = args.total_var

    if args.weighted:
        weighted = True
    else:
        weighted = False

    # -----------------------------------------------------------------------------
    # STEP 3 Run function
    # -----------------------------------------------------------------------------
    encoding_stats(n_perm, alpha_value, tail, total_var, weighted, encoding_dir)

