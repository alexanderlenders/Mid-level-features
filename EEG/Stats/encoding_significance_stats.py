#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STATISTICS FOR ENCODING ANALYSIS

This script does the statistical analysis for the encoding analysis.
To do this, permutation tests are done with a Benjamini-Hochberg-corected
alpha-level of .05. The statistical tests are two-sided per default.
Chance-level of encoding is 0.

@author: Alexander Lenders, Agnessa Karapetian
"""
import os
import numpy as np
from scipy.stats import rankdata
from statsmodels.stats.multitest import fdrcorrection
import pickle
import argparse
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
    parse_list,
)

def permutation_test(list_sub: list, n_perm: int, tail: str, alpha: float, timepoints: int, input_type: str, workDir: str, feature_names: list, var_part: bool = False):
    """
    Input:
    ----------
    Output from the encoding analysis (multivariate linear regression), i.e.:
    Encoding results (multivariate linear regression), saved in a dictionary
    which contains for every feature correlation measure, i.e.:
        encoding_results[feature]['correlation']

    Returns
    ----------
    feature_results, dictionary with the following keys for each feature:
        a. Uncorrected_p_values_map
            - Contains uncorrected p-values
        b. Corrected_p_values_map
            - Contains corrected p-values
        c. Boolean_statistical_map
            - Contains boolean values, if True -> corrected p-value is lower than .05

    Parameters
    ----------
    list_sub : list
        List with subjects which should be included in the statistical analysis
    n_perm : int
        Number of permutations (Default: 10,000)
    tail : str
        Whether two conduct an one-sided test (right) or two-sided test (both)
        (Default: both)
    alpha: int
        Alpha significance level (Default: 0.05)
    timepoints: int
        Number of timepoints
    input_type: str
        Miniclips or images
    workDir: str
        Working directory where the results are saved
    feature_names: list
        List of feature names to be analyzed
    var_part: bool
        If True, only the last feature is analyzed (for control_6_1 and control
        6_2). If False, all features are analyzed (Default: False)
    """
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
    workDir = os.path.join(workDir, f"{input_type}")
    saveDir = os.path.join(workDir, "stats")
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    
    identifierDir = f"seq_50hz_posterior_encoding_results_averaged_frame_before_mvnn_{len(feature_names)}_features_onehot.pkl"

    n_sub = len(list_sub)

    # set random seed (for reproduction)
    np.random.seed(42)

    if var_part:
        feature_names = feature_names[-1:]

    temp_list = [
        f"{', '.join(f)}" if isinstance(f, (tuple, list)) else str(f)
        for f in feature_names  
    ]

    feature_names = temp_list
    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------
    results_unfiltered = {}
    for index, subject in enumerate(list_sub):
        fileDir = os.path.join(workDir, f"{subject}_{identifierDir}")
        encoding_results = np.load(fileDir, allow_pickle=True)
        results_unfiltered[str(subject)] = encoding_results

    feature_results = {}
    for feature in feature_names:
        results = np.zeros((n_sub, timepoints))

        for index, subject in enumerate(list_sub):
            subject_result = results_unfiltered[str(subject)][feature][
                "correlation"
            ]

            # averaged over all channels
            subject_result_averaged = np.mean(subject_result, axis=1)
            results[index, :] = subject_result_averaged

        # ---------------------------------------------------------------------
        # STEP 2.3 Permutation (Create null distribution)
        # ---------------------------------------------------------------------
        # create statistical map for all permutation
        stat_map = np.zeros((n_perm, timepoints))

        # create mean for each timepoint over all participants
        # this is our "original data" and permutation 1 in the stat_map
        mean_orig = np.mean(results, axis=0)
        stat_map[0, :] = mean_orig

        for permutation in range(1, n_perm):
            # create array with -1 and 1 (randomization)
            perm = np.expand_dims(
                np.random.choice([-1, 1], size=(n_sub,), replace=True), 1
            )

            # create randomization matrix
            rand_matrix = np.broadcast_to(perm, (n_sub, timepoints))

            # elementwise multiplication
            permutation_mat = np.multiply(results, rand_matrix)

            # calculate mean and put it in stats map
            stat_map[permutation, :] = np.mean(permutation_mat, axis=0)

        # ---------------------------------------------------------------------
        # STEP 2.4 Calculate ranks and p-values
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
        sub_matrix = np.full((n_perm, timepoints), (n_perm + 1))
        p_map = (sub_matrix - ranks) / n_perm
        p_values = p_map[0, :]

        # ---------------------------------------------------------------------
        # STEP 2.5 Benjamini-Hochberg correction
        # ---------------------------------------------------------------------
        rejected, p_values_corr = fdrcorrection(
            p_values, alpha=alpha, is_sorted=False
        )

        stats_results = {}
        stats_results["Uncorrected_p_values_map"] = p_values
        stats_results["Corrected_p_values_map"] = p_values_corr
        stats_results["Boolean_statistical_map"] = rejected

        feature_results[feature] = stats_results

    # -------------------------------------------------------------------------
    # STEP 2.6 Save results of analysis
    # -------------------------------------------------------------------------
    # Save the dictionary
    if input_type == "miniclips":
        fileDir = "encoding_stats_{}_nonstd.pkl".format(tail)
    elif input_type == "images":
        fileDir = "encoding_stats_{}_nonstd.pkl".format(tail)

    savefileDir = os.path.join(saveDir, fileDir)

    # Creating the directory if not existing
    if os.path.isdir(os.path.join(saveDir)) == False:  # if not a directory
        os.makedirs(os.path.join(saveDir))

    with open(savefileDir, "wb") as f:
        pickle.dump(feature_results, f)


# -----------------------------------------------------------------------------
# STEP 3: Run function
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
        default=70,
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
        "-i",
        "--input_type",
        default="images",
        type=str,
        metavar="",
        help="Font",
    )

    args = parser.parse_args()  # to get values for the arguments
    config = load_config(args.config_dir, args.config)
    workDir = config.get(args.config, "save_dir")
    feature_names = parse_list(config.get(args.config, "feature_names"))

    tail = args.tail
    n_perm = args.num_perm
    timepoints = args.num_tp
    alpha = args.alpha
    input_type = args.input_type

    VAR_PART = False

    if input_type == "miniclips":
        list_sub = [
            6,
            7,
            8,
            9,
            10,
            11,
            17,
            18,
            20,
            21,
            23,
            25,
            27,
            28,
            29,
            30,
            31,
            32,
            34,
            36,
        ]
    elif input_type == "images":
        list_sub = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    if args.config == "control_6_1" or args.config == "control_6_2":
        feature_names = feature_names[:-1]  # remove the full feature set

    permutation_test(list_sub, n_perm, tail, alpha, timepoints, input_type, workDir, feature_names, var_part=VAR_PART)
