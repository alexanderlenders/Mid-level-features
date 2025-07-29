#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOOTSTRAPPING ENCODING

This script calculates Bootstrap 95%-CIs for the encoding accuracy for each
timepoint (in ms) and each feature. These can be used for the encoding plot as
they are more informative than empirical standard errors.

In addition, this script calculates Bootstrap 95%-CIs for the timepoint (in ms)
of the encoding peak for each feature.

@author: Alexander Lenders, Agnessa Karapetian
"""
import numpy as np
from scipy.stats import rankdata
import os
import pickle

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
    parse_list,
)
import argparse


def bootstrapping_CI(
    list_sub: list,
    n_perm: int,
    timepoints: int,
    input_type: str,
    workDir: str,
    feature_names: list,
):
    """
    Bootstrapped 95%-CIs for the encoding accuracy for each timepoint and
    each feature.

    Input:
    ----------
    Output from the encoding analysis, i.e.:
    Encoding results (multivariate linear regression), saved in a dictionary
    which contains for every feature correlation measure, i.e.:
        encoding_results[feature]['correlation']

    Returns:
    ----------
    Dictionary with level 1 features and level 2 with 95% CIs (values)
    for each timepoint (key), i.e. results[feature][timepoint]

    Parameters
    ----------
    list_sub : list
          List of subjects for which encoding results exist
    n_perm : int
          Number of permutations for bootstrapping
    timepoints : int
          Number of timepoints
    input_type : str
        Images or miniclips
    workDir : str
        Working directory where the results are saved
    feature_names : list
        List of feature names to be processed
    """
    # -------------------------------------------------------------------------
    # STEP 2.1 Define Variables
    # -------------------------------------------------------------------------
    workDir = os.path.join(workDir, f"{input_type}")
    saveDir = os.path.join(workDir, "stats")
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    identifierDir = f"seq_50hz_posterior_encoding_results_averaged_frame_before_mvnn_{len(feature_names)}_features_onehot.pkl"

    # set some vars
    n_sub = len(list_sub)
    time_ms = np.arange(-400, 1000, 20)

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

    # Loop over all features
    feature_results = {}
    results_dict = {}
    ci_dict_all = {}

    # define matrix where to save the values
    temp_list = [
        f"{', '.join(f)}" if isinstance(f, (tuple, list)) else str(f)
        for f in feature_names
    ]
    feature_names = temp_list

    for feature in feature_names:

        results = np.zeros((n_sub, timepoints))
        for index, subject in enumerate(list_sub):
            subject_result = results_unfiltered[str(subject)][feature][
                "correlation"
            ]

            # averaged over all channels
            subject_result_averaged = np.mean(subject_result, axis=1)
            results[index, :] = subject_result_averaged

        results_dict[feature] = results

        # ---------------------------------------------------------------------
        # STEP 2.3 Bootstrapping: accuracy
        # ---------------------------------------------------------------------
        bt_data = np.zeros((timepoints, n_perm))

        for tp in range(timepoints):
            tp_data = results[:, tp]
            for perm in range(n_perm):
                perm_tp_data = np.random.choice(
                    tp_data, size=(n_sub, 1), replace=True
                )
                mean_p_tp = np.mean(perm_tp_data, axis=0)
                bt_data[tp, perm] = mean_p_tp.item()

        # ---------------------------------------------------------------------
        # STEP 2.4 Calculate 95%-CI
        # ---------------------------------------------------------------------
        upper = int(np.ceil(n_perm * 0.975)) - 1
        lower = int(np.ceil(n_perm * 0.025)) - 1

        ci_dict = {}

        for tp in range(timepoints):
            tp_data = bt_data[tp, :]
            ranks = rankdata(tp_data)
            t_data = np.vstack((ranks, tp_data))
            ascending_ranks_idx = np.argsort(ranks, axis=0)
            ascending_ranks = t_data[:, ascending_ranks_idx]
            lower_CI = ascending_ranks[1, lower]
            upper_CI = ascending_ranks[1, upper]

            ci_dict["{}".format(tp)] = [lower_CI, upper_CI]

        feature_results[feature] = ci_dict

        # -------------------------------------------------------------------------
        # STEP 2.5 Bootstrapping: peak latency
        # -------------------------------------------------------------------------
        # Find ground truth peak latency (ms)
        encoding_mean = np.mean(results, axis=0)
        peak = time_ms[np.argmax(encoding_mean)]

        # Permute and calculate peak latencies for bootstrap samples
        bt_data_peaks = np.zeros(n_perm)

        for perm in range(n_perm):
            perm_peak_data_idx = np.random.choice(
                results.shape[0], size=(n_sub, 1), replace=True
            )
            perm_peak_data = np.squeeze(results[perm_peak_data_idx])
            perm_mean = np.mean(perm_peak_data, axis=0)
            bt_data_peaks[perm] = time_ms[np.argmax(perm_mean)]

        # ---------------------------------------------------------------------
        # STEP 2.6 Calculate 95%-CI
        # ---------------------------------------------------------------------
        upper = int(np.ceil(n_perm * 0.975)) - 1
        lower = int(np.ceil(n_perm * 0.025)) - 1

        ci_dict = {}

        ranks = rankdata(bt_data_peaks)
        t_data = np.vstack((ranks, bt_data_peaks))
        ascending_ranks_idx = np.argsort(ranks, axis=0)
        ascending_ranks = t_data[:, ascending_ranks_idx]
        lower_CI = ascending_ranks[1, lower]
        upper_CI = ascending_ranks[1, upper]

        ci_dict["{}".format(peak)] = [lower_CI, peak, upper_CI]

        ci_dict_all["{}".format(feature)] = [lower_CI, peak, upper_CI]

    # -------------------------------------------------------------------------
    # STEP 2.7 Save CI accuracy
    # -------------------------------------------------------------------------
    # Save the dictionary

    fileDir = "encoding_CI95_accuracy.pkl"

    savefileDir = os.path.join(saveDir, fileDir)

    with open(savefileDir, "wb") as f:
        pickle.dump(feature_results, f)

    # -------------------------------------------------------------------------
    # STEP 2.8 Save CI - peak latency
    # -------------------------------------------------------------------------
    # Save the dictionary
    fileDir = "encoding_CI95_peak.pkl"

    savefileDir = os.path.join(saveDir, fileDir)

    with open(savefileDir, "wb") as f:
        pickle.dump(ci_dict_all, f)

    # -------------------------------------------------------------------------
    # STEP 2.9 Bootstrapping - peak latency differences
    # -------------------------------------------------------------------------
    pairwise_p = {}

    for feature1 in range(len(feature_names)):
        feature_A = feature_names[feature1]
        num_comparisons = feature1 + 1
        results_feature_A = results_dict[feature_A]
        encoding_mean = np.mean(results_feature_A, axis=0)
        peak_A = time_ms[np.argmax(encoding_mean)]

        for feature2 in range(num_comparisons):
            feature_B = feature_names[feature2]

            if feature_A == feature_B:
                continue

            str_comparison = "{} vs. {}".format(feature_A, feature_B)

            results_feature_B = results_dict[feature_B]
            encoding_mean_B = np.mean(results_feature_B, axis=0)

            peak_B = time_ms[np.argmax(encoding_mean_B)]

            feature_diff = peak_A - peak_B

            bt_data_peaks = np.zeros(n_perm)

            for perm in range(n_perm):
                perm_peak_data_idx = np.random.choice(
                    results_feature_A.shape[0], size=(n_sub, 1), replace=True
                )

                perm_peak_data_A = results_feature_A[
                    perm_peak_data_idx
                ].reshape((n_sub, timepoints))
                perm_peak_data_B = results_feature_B[
                    perm_peak_data_idx
                ].reshape((n_sub, timepoints))

                perm_mean_A = np.mean(perm_peak_data_A, axis=0)
                perm_mean_B = np.mean(perm_peak_data_B, axis=0)

                peak_A = time_ms[np.argmax(perm_mean_A)]
                peak_B = time_ms[np.argmax(perm_mean_B)]

                feature_diff_bt = peak_A - peak_B

                bt_data_peaks[perm] = feature_diff_bt

            # -----------------------------------------------------------------
            # STEP 2.10 Compute p-Value and CI
            # -----------------------------------------------------------------
            # CI
            upper = int(np.ceil(n_perm * 0.975)) - 1
            lower = int(np.ceil(n_perm * 0.025)) - 1

            peak_data_sorted = bt_data_peaks[np.argsort(bt_data_peaks)]
            lower_CI = peak_data_sorted[lower]
            upper_CI = peak_data_sorted[upper]

            c_dict = {"ci": [lower_CI, feature_diff, upper_CI]}

            pairwise_p[str_comparison] = c_dict

    # -------------------------------------------------------------------------
    # STEP 2.11 Save CI
    # -------------------------------------------------------------------------
    # Save the dictionary

    fileDir = "encoding_stats_peak_latency_CI.pkl"
    savefileDir = os.path.join(saveDir, fileDir)

    with open(savefileDir, "wb") as f:
        pickle.dump(pairwise_p, f)


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

    if args.config == "control_6_1" or args.config == "control_6_2":
        feature_names = feature_names[:-1]  # remove the full feature set

    n_perm = args.num_perm
    timepoints = args.num_tp
    input_type = args.input_type

    # -----------------------------------------------------------------------------
    # STEP 3: Run functions
    # -----------------------------------------------------------------------------
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

    bootstrapping_CI(
        list_sub, n_perm, timepoints, input_type, workDir, feature_names
    )
