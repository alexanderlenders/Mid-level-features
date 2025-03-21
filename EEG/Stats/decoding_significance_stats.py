#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STATISTICS FOR DECODING ANALYSIS (PERMUTATION TESTS)

This script implements the statistical analysis for the pairwise decoding analysis.
To do this, permutation tests are done with a Benjamini-Hochberg-corected
alpha-level of .05. The statistical tests are two-sided per default.
Chance-level of pairwise decoding is 0.5.

@author: Alexander Lenders, Agnessa Karapetian

"""
# -----------------------------------------------------------------------------
# STEP 1: Initialize variables
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # add arguments / inputs
    parser.add_argument(
        "-ls",
        "--list_sub",
        default=[9],
        type=int,
        metavar="",
        help="list of subjects",
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
        help="Miniclips or images",
    )

    args = parser.parse_args()  # to get values for the arguments

    list_sub = args.list_sub
    n_perm = args.num_perm
    timepoints = args.num_tp
    alpha = args.alpha
    tail = args.tail
    input_type = args.input_type

# -----------------------------------------------------------------------------
# STEP 2: Define Permutation Test Function
# -----------------------------------------------------------------------------


def permutation_test(list_sub, n_perm, tail, alpha, timepoints, input_type):
    """
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

    Returns
    ----------
    Results of the statistical analysis, dictionary with the following keys:
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
        Number of timepoints in EEG epoch
    input_type: str
        Either miniclips or images

    """
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
    # Import modules
    import os
    import numpy as np
    from scipy.stats import rankdata
    import statsmodels
    from statsmodels.stats.multitest import multipletests
    import pickle

    n_sub = len(list_sub)

    # chance-level of time gen analysis (for now hardcoded)
    chance_level = 0.5
    decoding_mat = np.zeros((len(list_sub), timepoints))

    # set random seed (for reproduction)
    np.random.seed(42)

    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------
    workDir = "Z:/Unreal/Results/Decoding/{}/Redone".format(input_type)
    saveDir = "Z:/Unreal/Results/Decoding/{}/Redone/stats".format(input_type)

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

    # Account for chance level
    decoding_mat = decoding_mat - chance_level

    # -------------------------------------------------------------------------
    # STEP 2.3 Permutation (Create null distribution)
    # -------------------------------------------------------------------------
    # create statistical map for all permutation
    stat_map = np.zeros((n_perm, timepoints))

    # create decoding mean for each timepoint over all participants
    # this is our "original data" and permutation 1 in the stat_map
    mean_orig = np.mean(decoding_mat, axis=0)
    stat_map[0, :] = mean_orig

    for permutation in range(1, n_perm):
        # create array with -1 and 1 (randomization)
        perm = np.expand_dims(
            np.random.choice([-1, 1], size=(n_sub,), replace=True), 1
        )

        # create randomization matrix
        rand_matrix = np.broadcast_to(perm, (n_sub, timepoints))

        # elementwise multiplication
        permutation_mat = np.multiply(decoding_mat, rand_matrix)

        # calculate decoding mean and put it in stats map
        stat_map[permutation, :] = np.mean(permutation_mat, axis=0)

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
    """
    Please note, that we assume a positive depence between the different 
    statistical tests. 
    """
    rejected, p_values_corr = statsmodels.stats.multitest.fdrcorrection(
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
    fileDir = "decoding_stats_{}_nonstd.pkl".format(tail)

    savefileDir = os.path.join(saveDir, fileDir)

    # Creating the directory if not existing
    if os.path.isdir(os.path.join(saveDir)) == False:  # if not a directory
        os.makedirs(os.path.join(saveDir))

    with open(savefileDir, "wb") as f:
        pickle.dump(stats_results, f)


# -----------------------------------------------------------------------------
# STEP 3: Run function
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
permutation_test(list_sub, n_perm, tail, alpha, timepoints, input_type)
