#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STATISTICS FOR ENCODING ANALYSIS 

This script does the statistical analysis for the encoding analysis.
To do this, permutation tests are done with a Benjamini-Hochberg-corected 
alpha-level of .05. The statistical tests are two-sided per default.
Chance-level of encoding is 0. 

At the moment the directories as well as using correlation instead of RMSE to
determine how well multivariate regression can predict (encode) EEG activity
per channel is hardcoded.

@author: AlexanderLenders

Acknowledgments: This script is based on a script in MATLAB by Agnessa 
Karapetian and other members of the Cichy lab.

Anaconda Environment on local machine: mne

TO-DO:
    Change function description
    Add feature names for taskonomy and resnet
"""

# -----------------------------------------------------------------------------
# STEP 1: Initialize variables
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # type of encoding - unreal, taskonomy, resnet
    
    # add arguments / inputs
    parser.add_argument('-ls', "--list_sub", default=0, type=int, 
                        metavar='', help="list of subjects")
    parser.add_argument('-ty', "--type", default = 'unreal_before_pca', type = str, 
                        metavar='', help="type of encoding")
    parser.add_argument('-np', "--num_perm", default = 10000, type = int, 
                        metavar='', help="Number of permutations")
    parser.add_argument('-tp', "--num_tp", default = 70, type = int, 
                        metavar='', help="Number of timepoints")
    parser.add_argument('-a', "--alpha", default = 0.05, type = int, 
                        metavar='', help="Signifance level (alpha)")
    parser.add_argument('-t', "--tail", 
                        default = 'both',
                        type = str, metavar='', 
                        help="One-sided: right, two-sided: both")
    parser.add_argument('-i', "--input_type", default = 'miniclips', type = str, 
                        metavar='', help="Font")
    
    
    args = parser.parse_args() # to get values for the arguments
    
    list_sub = args.list_sub      
    n_perm = args.num_perm
    timepoints = args.num_tp
    n_perm = args.num_perm
    alpha = args.alpha
    tail = args.tail
    type_analysis = args.type
    input_type = args.input_type
    
# -----------------------------------------------------------------------------
# STEP 2: Define Permutation Test Function
# -----------------------------------------------------------------------------

def permutation_test(list_sub, type_analysis, n_perm, tail, alpha,
                     timepoints, input_type): 
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
            - Contains uncorrected p-value for each element in time gen matrix
        b. Corrected_p_values_map
            - Contains corrected p-value for each element in time gen matrix 
        c. Boolean_statistical_map
            - Contains boolean values for each element in time gen matrix, if 
            True -> corrected p-value is lower than .05
    
    Parameters
    ----------
    list_sub : list 
        List with subjects which should be included in the statistical analysis
    type_analysis : str 
        To which encoding analysis the results belong to, i.e. 'unreal', 
        'taskonomy' or 'resnet'
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
    import csv
    
    # create workDir and saveDir
    if type_analysis == 'unreal_before_pca':
        identifierDir = 'seq_50hz_posterior_encoding_results_averaged_frame_before_mvnn_7features_onehot.pkl'
        feature_names = ('edges','world_normal', 'lightning','scene_depth', 'reflectance', 'skeleton','action')
        
        if input_type == 'miniclips':
            workDir = 'Z:/Unreal/Results/Encoding/'
            saveDir = 'Z:/Unreal/Results/Encoding/redone/stats'
    
        elif input_type == 'images':     
            workDir = 'Z:/Unreal/images_results/encoding/'
            saveDir = 'Z:/Unreal/images_results/encoding/redone/stats'

    elif type_analysis == 'taskonomy_before_pca': 
        workDir = '/Volumes/Elements/miniclip_data/results/encoding/taskonomy/video/encoding/100'
        saveDir = '/Volumes/Elements/miniclip_data/results/encoding/taskonomy/video/stats'
        
        identifierDir = 'seq_50hz_posterior_encoding_results_frame_avg.pkl'       
        feature_names = ('normal', 'edge_texture', 'depth_euclidean', 'reshading', 'curvature')
    
    elif type_analysis == 'resnet_3d_before_pca': 
        workDir = '/Volumes/Elements/miniclip_data/results/encoding/resnet/video/encoding/100'
        saveDir = '/Volumes/Elements/miniclip_data/results/encoding/resnet/video/stats'
        
        identifierDir = 'seq_50hz_posterior_encoding_results.pkl'
        feature_names = ('layer1', 'layer2', 'layer3', 'layer4', 'fc')
        
    n_sub = len(list_sub)
    
    # set random seed (for reproduction)
    np.random.seed(42)
    
    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------
    results_unfiltered = {}    
    for index, subject in enumerate(list_sub): 
        fileDir = (workDir + 'redone/7_features/{}_'.format(subject) + identifierDir)
        encoding_results = np.load(fileDir, allow_pickle= True)
        results_unfiltered[str(subject)] = encoding_results
        

    feature_results = {}
    for feature in feature_names: 
        
        results = np.zeros((n_sub, timepoints)) 

        for index, subject in enumerate(list_sub): 
    
            subject_result = results_unfiltered[str(subject)][feature]['correlation']
            # averaged over all channels 
            subject_result_averaged = np.mean(subject_result, axis = 1)
            results[index, :] = subject_result_averaged

        # ---------------------------------------------------------------------
        # STEP 2.3 Permutation (Create null distribution)
        # ---------------------------------------------------------------------
        # create statistical map for all permutation 
        stat_map = np.zeros((n_perm, timepoints))
        
        # create mean for each timepoint over all participants 
        # this is our "original data" and permutation 1 in the stat_map 
        mean_orig = np.mean(results, axis = 0)
        stat_map[0, :] = mean_orig
        
        for permutation in range(1, n_perm): 
            # create array with -1 and 1 (randomization)
            perm = np.expand_dims(np.random.choice([-1, 1], size=(n_sub,),
                                                   replace=True), 1)
            
            # create randomization matrix
            rand_matrix = np.broadcast_to(perm, (n_sub, timepoints))
            
            # elementwise multiplication 
            permutation_mat = np.multiply(results, rand_matrix)
            
            # calculate mean and put it in stats map 
            stat_map[permutation, :] = np.mean(permutation_mat,
                                               axis = 0)
                
        # ---------------------------------------------------------------------
        # STEP 2.4 Calculate ranks and p-values
        # ---------------------------------------------------------------------
        # get ranks (over all permutations), this gives us a distribution
        if tail == 'right':
            ranks = (np.apply_along_axis(rankdata, 0, stat_map))
            
        elif tail == 'both': 
            abs_values = np.absolute(stat_map)
            ranks = (np.apply_along_axis(rankdata, 0, abs_values))
        
        # calculate p-values
        # create a matrix with nperm+1 values in every element (to account for 
        # the observed test statistic)
        sub_matrix = np.full((n_perm, timepoints), (n_perm+1))
        p_map = (sub_matrix - ranks)/n_perm
        p_values = p_map[0, :]
    
        # ---------------------------------------------------------------------
        # STEP 2.5 Benjamini-Hochberg correction
        # ---------------------------------------------------------------------
        rejected, p_values_corr = statsmodels.stats.multitest.fdrcorrection(
            p_values, alpha = alpha, is_sorted = False)
        
        stats_results = {}
        stats_results['Uncorrected_p_values_map'] = p_values
        stats_results['Corrected_p_values_map'] = p_values_corr
        stats_results['Boolean_statistical_map'] = rejected
        
        feature_results[feature] = stats_results
            
    # -------------------------------------------------------------------------
    # STEP 2.6 Save results of analysis
    # -------------------------------------------------------------------------
    # Save the dictionary
    if input_type == 'miniclips':
        fileDir = ('encoding_{}_miniclips_stats_{}_nonstd.pkl'.format(type_analysis, tail))  
    elif input_type == 'images':
        fileDir = ('encoding_{}_images_stats_{}_nonstd.pkl'.format(type_analysis, tail))  
    
    savefileDir = os.path.join(saveDir, fileDir) 
     
    # Creating the directory if not existing
    if os.path.isdir(os.path.join(saveDir)) == False: # if not a directory
        os.makedirs(os.path.join(saveDir))
    
    with open(savefileDir, 'wb') as f:
        pickle.dump(feature_results, f)

# -----------------------------------------------------------------------------
# STEP 3: Run function
# -----------------------------------------------------------------------------
if input_type == 'miniclips':
    list_sub = [6, 7, 8, 9, 10, 11, 17, 18, 20, 21, 23, 25, 27, 28, 29, 30, 31, 32, 34, 36]
elif input_type == 'images':
    list_sub = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    
permutation_test(list_sub, type_analysis, n_perm, tail, alpha, timepoints, input_type) 
    

    