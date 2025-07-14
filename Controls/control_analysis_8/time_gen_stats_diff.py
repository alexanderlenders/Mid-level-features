#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STATISTICS FOR TIME GEN ANALYSIS (PERMUTATION TESTS) - DIFFERENCES BETWEEN IMAGES AND MINICLIPS

This script does the statistical analysis for the time generalization analysis.
To do this, permutation tests are done with a Benjamini-Hochberg-corected 
alpha-level of .05. The statistical tests are two-sided per default.

@author: AlexanderLenders
"""
import os
import numpy as np
from scipy.stats import rankdata
import statsmodels
from statsmodels.stats.multitest import fdrcorrection
import scipy.io
import pickle
import argparse
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
print(project_root)
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
    parse_list,
)

def permutation_test(list_sub_vid, list_sub_img, workDir, n_perm, tail, alpha, timepoints, feature_names): 
    """
    Statistical test on encoding time generalization matrix.
    """
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
    workDir_img = os.path.join(workDir, "images")
    workDir_vid = os.path.join(workDir, "miniclips")
    saveDir = os.path.join(workDir, "difference", "stats")

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    identifierDir = f"seq_50hz_posteriortime_gen_encoding_results_averaged_frame_before_mvnn_{len(feature_names)}_features_onehot.pkl"

    n_sub_vid = len(list_sub_vid)
    n_sub_img = len(list_sub_img)
    
    # set random seed (for reproduction)
    np.random.seed(42)
    
    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # ------------------------------------------------------------------------- 
    results_videos = {}
    results_images = {}
    for subject in list_sub_vid:
        fileDir_vid = os.path.join(workDir_vid, f"{subject}_{identifierDir}")
        encoding_results_vid = np.load(fileDir_vid, allow_pickle=True)
        results_videos[str(subject)] = encoding_results_vid

    for subject in list_sub_img:
        fileDir_img = os.path.join(workDir_img, f"{subject}_{identifierDir}")
        encoding_results_img = np.load(fileDir_img, allow_pickle=True)
        results_images[str(subject)] = encoding_results_img

   
    feature_results = {}
    for feature in feature_names:
        results_vid = np.zeros((n_sub_vid, timepoints, timepoints))
        results_img = np.zeros((n_sub_img, timepoints, timepoints))

        for index, subject in enumerate(list_sub_vid):
            subject_result = results_videos[str(subject)][feature][
                "correlation"
            ]

            # averaged over all channels
            subject_result_averaged = np.mean(subject_result, axis=2)
            results_vid[index, :, :] = subject_result_averaged
        
        for index, subject in enumerate(list_sub_img):
            subject_result = results_images[str(subject)][feature][
                "correlation"
            ]

            # averaged over all channels
            subject_result_averaged = np.mean(subject_result, axis=2)
            results_img[index, :, :] = subject_result_averaged
        
        # -------------------------------------------------------------------------
        # STEP 2.3 Permutation (Create null distribution)
        # -------------------------------------------------------------------------
        # create statistical map for all permutation 
        stat_map_tg = np.zeros((n_perm, timepoints, timepoints))

        all_results = np.vstack((results_vid, results_img))
        labels = np.array([0] * n_sub_vid + [1] * n_sub_img)  # 0=video, 1=image
        
        # create standardized mean for each timepoint over all participants 
        # this is our "original data" and permutation 1 in the stat_map 
        mean_orig_tg_vid = np.mean(results_vid, axis = 0)
        mean_orig_tg_img = np.mean(results_img, axis = 0)

        stat_map_tg[0, :, :] = mean_orig_tg_img - mean_orig_tg_vid
        
        for permutation in range(1, n_perm): 
            # shuffle the labels
            shuffled_labels = np.random.permutation(labels)

            group1 = all_results[shuffled_labels == 0]  # video
            group2 = all_results[shuffled_labels == 1]  # image

            mean_group1 = np.mean(group1, axis=0)
            mean_group2 = np.mean(group2, axis=0)

            t_stat_perm = mean_group2 - mean_group1

            # calculate standardized mean and put it in stats map 
            stat_map_tg[permutation, :] = t_stat_perm
        # -------------------------------------------------------------------------
        # STEP 2.4 Calculate ranks and p-values
        # -------------------------------------------------------------------------
        if tail == 'right':
            ranks_tg = np.apply_along_axis(rankdata, 0, stat_map_tg)
        elif tail == 'both':
            ranks_tg = np.apply_along_axis(rankdata, 0, np.abs(stat_map_tg))

        # Compute p-values
        sub_matrix_tg = np.full((n_perm, timepoints, timepoints), n_perm + 1)
        p_map_tg = (sub_matrix_tg - ranks_tg) / n_perm
        p_values_tg = p_map_tg[0, :, :]  # observed stat is at index 0
            
        # -------------------------------------------------------------------------
        # STEP 2.5 Benjamini-Hochberg correction
        # -------------------------------------------------------------------------
        flattened_p_values = p_values_tg.flatten()
        rejected, p_values_corr = fdrcorrection(
            flattened_p_values, alpha = alpha)
        
        final_p_val = rejected.reshape((timepoints, timepoints))
        
        corrected_p_val = p_values_corr.reshape((timepoints, timepoints))
        
        # -------------------------------------------------------------------------
        # STEP 2.6 Save results of analysis
        # -------------------------------------------------------------------------  
        stats_results = {}
        stats_results['Uncorrected_p_values_map'] = p_values_tg
        stats_results['Corrected_p_values_map'] = corrected_p_val
        stats_results['Boolean_statistical_map'] = final_p_val

        feature_results[feature] = stats_results
    
    # Save the dictionary
    fileDir = ('time_gen_diff_stats_{}.pkl'.format(tail))  
    
    savefileDir = os.path.join(saveDir, fileDir) 
     
    # Creating the directory if not existing
    if os.path.isdir(os.path.join(saveDir)) == False: # if not a directory
        os.makedirs(os.path.join(saveDir))
    
    with open(savefileDir, 'wb') as f:
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

    n_perm = args.num_perm
    timepoints = args.num_tp
    alpha = args.alpha
    tail = args.tail
    input_type = args.input_type

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
    
    permutation_test(list_sub_vid, list_sub_img, workDir, n_perm, tail, alpha, timepoints, feature_names)