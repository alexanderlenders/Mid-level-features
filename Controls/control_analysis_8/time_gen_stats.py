#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STATISTICS FOR TIME GEN ANALYSIS (PERMUTATION TESTS)

This script does the statistical analysis for the time generalization analysis.
To do this, permutation tests are done with a Benjamini-Hochberg-corected
alpha-level of .05. The statistical tests are two-sided per default.

@author: AlexanderLenders
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


def permutation_test(
    list_sub,
    workDir,
    n_perm,
    tail,
    alpha,
    timepoints,
    input_type,
    feature_names,
):
    """
    Statistical test on encoding time generalization matrix.
    """
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
    workDir = os.path.join(workDir, f"{input_type}")
    saveDir = os.path.join(workDir, "stats")

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    identifierDir = f"seq_50hz_posteriortime_gen_encoding_results_averaged_frame_before_mvnn_{len(feature_names)}_features_onehot.pkl"

    n_sub = len(list_sub)

    results = []  # list of dictionaries

    # set random seed (for reproduction)
    np.random.seed(42)

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
        results = np.zeros((n_sub, timepoints, timepoints))

        for index, subject in enumerate(list_sub):
            subject_result = results_unfiltered[str(subject)][feature][
                "correlation"
            ]

            # averaged over all channels
            subject_result_averaged = np.mean(subject_result, axis=2)
            results[index, :, :] = subject_result_averaged

        # -------------------------------------------------------------------------
        # STEP 2.3 Permutation (Create null distribution)
        # -------------------------------------------------------------------------
        # create statistical map for all permutation
        stat_map_tg = np.zeros((n_perm, timepoints, timepoints))

        # create standardized mean for each timepoint over all participants
        # this is our "original data" and permutation 1 in the stat_map
        mean_orig_tg = np.mean(results, axis=0)
        stat_map_tg[0, :, :] = mean_orig_tg

        permutation_mat = np.zeros((n_sub, timepoints, timepoints))
        for permutation in range(1, n_perm):
            # create array with -1 and 1 (randomization)
            perm = np.expand_dims(
                np.random.choice([-1, 1], size=(n_sub,), replace=True), 1
            )

            for subject in range(n_sub):
                scalar = perm[subject]
                permutation_mat[subject, :, :] = (
                    results[subject, :, :] * scalar
                )

            # calculate standardized mean and put it in stats map
            stat_map_tg[permutation, :] = np.mean(permutation_mat, axis=0)
        # -------------------------------------------------------------------------
        # STEP 2.4 Calculate ranks and p-values
        # -------------------------------------------------------------------------
        # get ranks (over all permutations), this gives us a distribution
        if tail == "right":
            ranks_tg = np.apply_along_axis(rankdata, 0, stat_map_tg)
        elif tail == "both":
            abs_values_tg = np.absolute(stat_map_tg)
            ranks_tg = np.apply_along_axis(rankdata, 0, abs_values_tg)

        # ranks_tg_2 = np.zeros((n_perm, timepoints, timepoints))
        # for tp in range(timepoints):
        #     ranks_tg_2[:, tp, :] = np.apply_along_axis(
        #         rankdata, 0, stat_map_tg[:, tp, :]
        #     )

        sub_matrix_tg = np.full((n_perm, timepoints, timepoints), (n_perm + 1))

        p_map_tg = (sub_matrix_tg - ranks_tg) / n_perm
        p_values_tg = p_map_tg[0, :]

        # -------------------------------------------------------------------------
        # STEP 2.5 Benjamini-Hochberg correction
        # -------------------------------------------------------------------------
        flattened_p_values = p_values_tg.flatten()
        rejected, p_values_corr = fdrcorrection(
            flattened_p_values, alpha=alpha
        )

        final_p_val = rejected.reshape((timepoints, timepoints))

        corrected_p_val = p_values_corr.reshape((timepoints, timepoints))

        # -------------------------------------------------------------------------
        # STEP 2.6 Save results of analysis
        # -------------------------------------------------------------------------
        stats_results = {}
        stats_results["Uncorrected_p_values_map"] = p_values_tg
        stats_results["Corrected_p_values_map"] = corrected_p_val
        stats_results["Boolean_statistical_map"] = final_p_val

        feature_results[feature] = stats_results

    # Save the dictionary
    fileDir = "time_gen_stats_{}.pkl".format(tail)

    savefileDir = os.path.join(saveDir, fileDir)

    # Creating the directory if not existing
    if os.path.isdir(os.path.join(saveDir)) == False:  # if not a directory
        os.makedirs(os.path.join(saveDir))

    with open(savefileDir, "wb") as f:
        pickle.dump(feature_results, f)


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
        type=float,
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

    n_perm = args.num_perm
    timepoints = args.num_tp
    alpha = args.alpha
    tail = args.tail
    input_type = args.input_type

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

    permutation_test(
        list_sub,
        workDir,
        n_perm,
        tail,
        alpha,
        timepoints,
        input_type,
        feature_names,
    )
