#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOOTSTRAPPING DECODING

This script calculates Bootstrap 95%-CIs for the decoding accuracy for each
timepoint (in ms). These can be used for the decoding plot as they are more
informative than (empirical) standard errors, i.e. point estimates.

In addition, this script calculates Bootstrap 95%-CIs for the timepoint (in ms)
of the decoding peak.

@author: Alexander Lenders, Agnessa Karapetian
"""
import argparse
import numpy as np
from scipy.stats import rankdata
import os
import pickle
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from EEG.Encoding.utils import load_config


# --------------------------------------------------------------------------------------
def bootstrapping_CI(
    list_sub: list, n_perm: int, timepoints: int, input_type: str, workDir: str
):
    """
    Bootstrapped 95%-CIs for the decoding accuracy for each timepoint.

    Input:
    ----------
    Output from the time generalization analysis, i.e.:
    Decoding and time generalization results, saved in a dictionary which contains:
        a. final_results_mean (70 Timepoints x 16110 Pairwise Decoding between Videos):
            - Contains the pairwise decoding results (RDM) for each timepoint
        b. mean_accuracies_over_conditions (70 Timepoints x 1)
            - Contains the pairwise decoding results for each timepoint averaged over
              conditions
        c. time_gen_matrix (70 Timepoints x 70 Timepoints)
            - Contains the time gen matrix averaged over conditions: First dimension
            refers to training time point, second to test timepoint
        d. triangle_matrix (180 x 180 x 70 Timepoints x 70 Timepoints)
            - RDM / Contains for each pairwise comparison the time gen matrix

    Returns:
    ----------
    Dictionary with 95% CIs (values) for each timepoint (key).

    Parameters
    ----------
    list_sub : list
          List of subjects for which time_gen results exist
    n_perm : int
          Number of permutations for bootstrapping
    timepoints : int
          Number of timepoints in each EEG epoch
    input_type: str
        Miniclips or images
    """
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
    # Number of subjects
    n_sub = len(list_sub)

    decoding_mat = np.zeros((len(list_sub), timepoints))

    # set random seed (for reproduction)
    np.random.seed(42)

    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------
    workDir = os.path.join(workDir, "decoding", input_type)
    saveDir = os.path.join(workDir, "stats")

    for index, subject in enumerate(list_sub):
        if subject < 10:
            fileDir = workDir + "/decoding_{}_sub-0{}_redone.npy".format(
                input_type, subject
            )
        else:
            fileDir = workDir + "/decoding_{}_sub-{}_redone.npy".format(
                input_type, subject
            )

        decoding_results = np.load(fileDir, allow_pickle=True).item()

        decoding_accuracy = decoding_results["mean_accuracies_over_conditions"]
        decoding_mat[index, :] = decoding_accuracy

    # -------------------------------------------------------------------------
    # STEP 2.3 Bootstrapping
    # -------------------------------------------------------------------------
    bt_data = np.zeros((timepoints, n_perm))

    for tp in range(timepoints):
        tp_data = decoding_mat[:, tp]
        for perm in range(n_perm):
            # sampling of the subjects with replacement
            perm_tp_data = np.random.choice(
                tp_data, size=(n_sub, 1), replace=True
            )
            mean_p_tp = np.mean(perm_tp_data, axis=0)
            bt_data[tp, perm] = mean_p_tp.item()

    # -------------------------------------------------------------------------
    # STEP 2.4 Calculate 95%-CI
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # STEP 2.5 Save CI
    # -------------------------------------------------------------------------
    # Save the dictionary

    fileDir = "CI_95_accuracy_redone.pkl"

    savefileDir = os.path.join(saveDir, fileDir)

    # Creating the directory if not existing
    if os.path.isdir(os.path.join(saveDir)) == False:  # if not a directory
        os.makedirs(os.path.join(saveDir))

    with open(savefileDir, "wb") as f:
        pickle.dump(ci_dict, f)


# ----------------------------------------------------------------------------------------
def bootstrapping_CI_ms(
    list_sub: list, n_perm: int, timepoints: int, input_type: str, workDir: str
):
    """
    Bootstrapped 95%-CIs for the timepoint (in ms) of the first 3 decoding peaks.

    Input:
    ----------
    Output from the time generalization analysis, i.e.:
    Decoding and time generalization results, saved in a dictionary which contains:
        a. final_results_mean (70 Timepoints x 16110 Pairwise Decoding between Videos):
            - Contains the pairwise decoding results (RDM) for each timepoint
        b. mean_accuracies_over_conditions (70 Timepoints x 1)
            - Contains the pairwise decoding results for each timepoint averaged over
              conditions
        c. time_gen_matrix (70 Timepoints x 70 Timepoints)
            - Contains the time gen matrix averaged over conditions: First dimension
            refers to training time point, second to test timepoint
        d. triangle_matrix (180 x 180 x 70 Timepoints x 70 Timepoints)
            - RDM / Contains for each pairwise comparison the time gen matrix

    Returns:
    ----------
    Dictionary with 95% CIs (values) for each peak (key).

    Parameters
    ----------
    list_sub : list
          List of subjects for which time_gen results exist
    n_perm : int
          Number of permutations for bootstrapping
    timepoints : int
          Number of timepoints in each EEG epoch
    input_type: str
        Miniclips or images
    """
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
    # time in ms (at the moment hardcoded)
    time_ms = list(range(-400, 1000, 20))

    n_sub = len(list_sub)

    # at the moment hardcoded, could be included as an argument
    n_peak = 3

    decoding_mean = []
    decoding_mat = np.zeros((len(list_sub), timepoints))

    # set random seed (for reproduction)
    np.random.seed(42)

    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------
    workDir = os.path.join(workDir, "decoding", input_type)
    saveDir = os.path.join(workDir, "stats")

    for index, subject in enumerate(list_sub):
        if subject < 10:
            fileDir = workDir + "/decoding_{}_sub-0{}_redone.npy".format(
                input_type, subject
            )
        else:
            fileDir = workDir + "/decoding_{}_sub-{}_redone.npy".format(
                input_type, subject
            )

        decoding_results = np.load(fileDir, allow_pickle=True).item()
        decoding_accuracy = decoding_results["mean_accuracies_over_conditions"]
        decoding_mat[index, :] = decoding_accuracy
        decoding_mean = np.mean(decoding_mat, axis=0)
        peaks_idx = np.argsort(decoding_mean, axis=0)

        decoding_ms = np.vstack((decoding_mean, time_ms))
        decoding_ms_ord = decoding_ms[:, peaks_idx]

        # determine the decoding peak
        peak = int(decoding_ms_ord[1, -1])

    # -------------------------------------------------------------------------
    # STEP 2.3 Bootstrapping
    # -------------------------------------------------------------------------
    bt_data_peaks = np.zeros(n_perm)

    for perm in range(n_perm):
        perm_peak_data_idx = np.random.choice(
            decoding_mat.shape[0], size=(n_sub, 1), replace=True
        )
        perm_peak_data = decoding_mat[perm_peak_data_idx].reshape(
            (n_sub, timepoints)
        )
        perm_mean = np.mean(perm_peak_data, axis=0)

        peaks_idx = np.argsort(perm_mean, axis=0)

        perm_ms = np.vstack((perm_mean, time_ms))
        perm_ms_ord = perm_ms[:, peaks_idx]

        bt_data_peaks[perm] = int(perm_ms_ord[1, -1])

    # -------------------------------------------------------------------------
    # STEP 2.4 Calculate 95%-CI
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # STEP 2.5 Save CI
    # -------------------------------------------------------------------------
    # Save the dictionary

    fileDir = "CI_95_peak_latency_redone.pkl"

    savefileDir = os.path.join(saveDir, fileDir)

    # Creating the directory if not existing
    if os.path.isdir(os.path.join(saveDir)) == False:  # if not a directory
        os.makedirs(os.path.join(saveDir))

    with open(savefileDir, "wb") as f:
        pickle.dump(ci_dict, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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
        help="Miniclips or images",
    )

    args = parser.parse_args()  # to get values for the arguments

    config = load_config(args.config_dir, args.config)
    workDir = config.get(args.config, "save_dir")
    n_perm = args.num_perm
    timepoints = args.num_tp
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

bootstrapping_CI(list_sub, n_perm, timepoints, input_type, workDir)
bootstrapping_CI_ms(list_sub, n_perm, timepoints, input_type, workDir)
