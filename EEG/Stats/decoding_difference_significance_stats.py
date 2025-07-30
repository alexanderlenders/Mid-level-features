#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STATISTICS FOR DECODING ANALYSIS (PERMUTATION TESTS) - DIFFERENCE: STATIC IMAGES - VIDEOS

This script implements the statistical analysis for decoding analysis, more precisely
it tests whether the DIFFERENCES between videos and images are significant for each tp.
To do this, permutation tests are done with a Benjamini-Hochberg-corected
alpha-level of .05. The statistical tests are two-sided per default.
Chance-level of pairwise decoding is 0.5.

@author: Alexander Lenders, Agnessa Karapetian

"""
import argparse
import os
import numpy as np
from scipy.stats import rankdata
import pickle
from statsmodels.stats.multitest import fdrcorrection
import sys
from pathlib import Path
import argparse

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from EEG.Encoding.utils import load_config


# -----------------------------------------------------------------------------
# STEP 2: Define Permutation Test Function
# -----------------------------------------------------------------------------
def permutation_test(
    list_sub_vid: list[int],
    list_sub_img: list[int],
    workDir_vid: str,
    workDir_img: str,
    saveDir: str,
    n_perm: int,
    tail: str,
    alpha: float,
    timepoints: int,
):
    """
    Inputs:
    ----------
    Decoding results with videos AND static images.


    Returns
    ----------
    Results of the statistical analysis (videos-images), dictionary with the following keys:
        a. Uncorrected_p_values_map
            - Contains uncorrected p-value for each element in time gen matrix
        b. Corrected_p_values_map
            - Contains corrected p-value for each element in time gen matrix
        c. Boolean_statistical_map
            - Contains boolean values for each element in time gen matrix, if
            True -> corrected p-value is lower than .05

    Parameters
    ----------
    list_sub_vid : list
        List with subjects which should be included in the statistical analysis for decoding with videos
    list_sub_img : list
        List with subjects which should be included in the statistical analysis for decoding with images
    workDir_vid : str
        Directory with the results of the decoding analysis with videos
    workDir_img : str
        Directory with the results of the decoding analysis with static images
    saveDir : str
        Directory where to save the results of the statistical analysis.
    n_perm : int
        Number of permutations (Default: 10,000)
    tail : str
        Whether two conduct an one-sided test (right) or two-sided test (both)
        (Default: both)
    alpha: int
        Alpha significance level (Default: 0.05)
    timepoints: int
        Number of timepoints in EEG epoch

    """
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
    n_sub_vid = len(list_sub_vid)
    n_sub_img = len(list_sub_img)

    decoding_mat_vid = np.zeros((len(list_sub_vid), timepoints))
    decoding_mat_img = np.zeros((len(list_sub_img), timepoints))

    # set random seed (for reproduction)
    np.random.seed(42)

    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------
    # Videos
    for index, subject in enumerate(list_sub_vid):
        if subject < 10:
            fileDir = (
                workDir_vid
                + "/decoding_miniclips_sub-0{}_redone.npy".format(subject)
            )
        else:
            fileDir = (
                workDir_vid
                + "/decoding_miniclips_sub-{}_redone.npy".format(subject)
            )

        decoding_results_vid = np.load(fileDir, allow_pickle=True).item()

        decoding_accuracy_vid = decoding_results_vid[
            "mean_accuracies_over_conditions"
        ]
        decoding_mat_vid[index, :] = decoding_accuracy_vid

    for index, subject in enumerate(list_sub_img):
        if subject < 10:
            fileDir = (
                workDir_img
                + "/decoding_images_sub-0{}_redone.npy".format(subject)
            )
        else:
            fileDir = (
                workDir_img
                + "/decoding_images_sub-{}_redone.npy".format(subject)
            )

        decoding_results_img = np.load(fileDir, allow_pickle=True).item()

        decoding_accuracy_img = decoding_results_img[
            "mean_accuracies_over_conditions"
        ]
        decoding_mat_img[index, :] = decoding_accuracy_img

    # -------------------------------------------------------------------------
    # STEP 2.3 Permutation (Create null distribution)
    # -------------------------------------------------------------------------
    # create statistical map for all permutation
    stat_map = np.zeros((n_perm, timepoints))

    # create decoding mean for each timepoint over all participants
    # this is our "original data" and permutation 1 in the stat_map
    mean_orig_vid = np.mean(decoding_mat_vid, axis=0)
    mean_orig_img = np.mean(decoding_mat_img, axis=0)

    mean_diff = mean_orig_img - mean_orig_vid

    stat_map[0, :] = mean_diff

    all_results = np.vstack((decoding_mat_vid, decoding_mat_img))
    labels = np.array([0] * n_sub_vid + [1] * n_sub_img)

    for permutation in range(1, n_perm):
        shuffled_labels = np.random.permutation(labels)

        # Assign to new permuted groups
        group_1 = all_results[shuffled_labels == 0, :]
        group_2 = all_results[shuffled_labels == 1, :]

        mean_group_1 = np.mean(group_1, axis=0)
        mean_group_2 = np.mean(group_2, axis=0)

        mean_diff = mean_orig_img - mean_orig_vid

        # calculate decoding mean and put it in stats map
        stat_map[permutation, :] = mean_diff

    # -------------------------------------------------------------------------
    # STEP 2.4 Calculate ranks and p-values
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # STEP 2.5 Benjamini-Hochberg correction
    # -------------------------------------------------------------------------
    rejected, p_values_corr = fdrcorrection(
        p_values, alpha=alpha, is_sorted=False
    )

    # -------------------------------------------------------------------------
    # STEP 2.6 Save results of analysis
    # -------------------------------------------------------------------------
    stats_results = {}
    stats_results["Uncorrected_p_values_map"] = p_values
    stats_results["Corrected_p_values_map"] = p_values_corr
    stats_results["Boolean_statistical_map"] = rejected

    # Save the dictionary
    fileDir = "diff_decoding_stats_{}_nonstd.pkl".format(tail)

    savefileDir = os.path.join(saveDir, fileDir)

    # Creating the directory if not existing
    if os.path.isdir(os.path.join(saveDir)) == False:  # if not a directory
        os.makedirs(os.path.join(saveDir))

    with open(savefileDir, "wb") as f:
        pickle.dump(stats_results, f)


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

    args = parser.parse_args()  # to get values for the arguments

    config = load_config(args.config_dir, args.config)
    workDir = config.get(args.config, "save_dir")
    timepoints = args.num_tp
    n_perm = args.num_perm
    alpha = args.alpha
    tail = args.tail

    list_sub_vid = [
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
    list_sub_img = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    workDir_vid = os.path.join(workDir, "decoding", "miniclips")
    workDir_img = os.path.join(workDir, "decoding", "images")
    saveDir = os.path.join(workDir, "decoding", "difference", "stats")

    permutation_test(
        list_sub_vid,
        list_sub_img,
        workDir_vid,
        workDir_img,
        saveDir,
        n_perm,
        tail,
        alpha,
        timepoints,
    )
