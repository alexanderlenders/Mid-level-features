#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STATS ENCODING LAYERS FROM RESNET3D

This script implements the statistical tests for the encoding analyis predicting
the unit activity within the action DNN layers based on the single gaming-engine
features. 

It conducts permutation tests by permuting the 100 unit activity components per layer, 
alpha = .05, Benjamini-Hochberg correction.
@author: AlexanderLenders, AgnessaKarapetian
"""
# -----------------------------------------------------------------------------
# STEP 1: Initialize variables
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    
    parser = argparse.ArgumentParser()
    
    # add arguments / inputs
    parser.add_argument('-np', "--num_perm", default = 10000, type = int, 
                        metavar='', help="Number of permutations")
    parser.add_argument('-tp', "--num_tp", default = 5, type = int, 
                        metavar='', help="Number of timepoints")
    parser.add_argument('-a', "--alpha", default = 0.05, type = int, 
                        metavar='', help="Significance level (alpha)")
    parser.add_argument('-t', "--tail", 
                        default = 'both',
                        type = str, metavar='', 
                        help="One-sided: right, two-sided: both")
    parser.add_argument('-sd', "--savedir", 
                        default = 'Z:/Unreal/Results/Encoding/CNN_redone/2D_ResNet18/stats/',
                        type = str, metavar='', help="Where to save results")
    parser.add_argument('-tv',"--total_var", help='Total variance explained by all PCA components together', 
            default = 90)


    args = parser.parse_args() # to get values for the arguments

    n_perm = args.num_perm
    timepoints = args.num_tp
    n_perm = args.num_perm
    alpha_value = args.alpha
    tail = args.tail
    saveDir = args.savedir
    total_var = args.total_var

# -----------------------------------------------------------------------------
# STEP 2: Define Encoding Function
# -----------------------------------------------------------------------------

def encoding_stats(n_perm, alpha_value, tail, saveDir, total_var): 
    """
    TO ADD
    Input: 
    ----------
    
    Returns:
    ----------
    
    Parameters:
    ----------

    """
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
    # Import modules
    import os
    import numpy as np
    import torch 
    import pickle
    from scipy.stats import rankdata
    import statsmodels
    from statsmodels.stats.multitest import multipletests
    
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
    
        
    feature_names = ('edges', 'world_normal', 'lightning',
                 'scene_depth', 'reflectance', 'action', 'skeleton')
    
    features_dict = dict.fromkeys(feature_names)
    
    num_layers = len(layers_names)
    
    
    workDir_img = 'Z:/Unreal/Results/Encoding/CNN_redone/2D_ResNet18/'
    workDir_vid = 'Z:/Unreal/Results/Encoding/CNN_redone/3D_ResNet18/'
    
    np.random.seed(42)
        
    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------
    fileDir_img = os.path.join(workDir_img, 'encoding_layers_resnet.pkl')   
    fileDir_vid = os.path.join(workDir_vid, 'encoding_layers_resnet.pkl')   

    encoding_results_img = np.load(fileDir_img, allow_pickle= True)
    encoding_results_vid = np.load(fileDir_vid, allow_pickle= True)

    # define matrix where to save the values 
    regression_features = dict.fromkeys(feature_names)
    
    for feature in features_dict.keys(): 
        
        # create statistical map
        stat_map = np.zeros((n_perm, num_layers))
        for l, layer in enumerate(layers_names):
        # for l, corr_layer in enumerate(encoding_results[feature]['weighted_correlations'].values()):
            corr_img = encoding_results_img[feature]['weighted_correlations'][layer]
            corr_vid = encoding_results_vid[feature]['weighted_correlations'][layer]
            
            num_comp_layer_img = corr_img.shape[0]
            num_comp_layer_vid = corr_vid.shape[0]

            # create mean for each layer over all images
            # this is our "original data" and permutation 1 in the stat_map 
            mean_orig_img = np.sum(corr_img)/total_var
            mean_orig_vid = np.sum(corr_vid)/total_var
            mean_orig_diff = mean_orig_img - mean_orig_vid

            stat_map[0, l] = mean_orig_diff
            
            for permutation in range(1, n_perm): 
                perm_img = np.expand_dims(np.random.choice([-1, 1], size = (num_comp_layer_img,), 
                                    replace = True), 1)
                perm_img = perm_img.T
                perm_corr_img = corr_img*perm_img
                
                perm_vid = np.expand_dims(np.random.choice([-1, 1], size = (num_comp_layer_vid,), 
                    replace = True), 1)
                perm_vid = perm_vid.T
                perm_corr_vid = corr_vid*perm_vid

                # create mean for each layer over all images for every permutation
                mean_img = np.sum(perm_corr_img)/total_var
                mean_vid = np.sum(perm_corr_vid)/total_var
                mean_diff = mean_img - mean_vid
                stat_map[permutation, l] = mean_diff
                
        # ---------------------------------------------------------------------
        # Calculate ranks and p-values
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
        sub_matrix = np.full((n_perm, num_layers), (n_perm+1))
        p_map = (sub_matrix - ranks)/n_perm
        p_values = p_map[0, :]
        
        # ---------------------------------------------------------------------
        # Benjamini-Hochberg correction
        # ---------------------------------------------------------------------
        rejected, p_values_corr = statsmodels.stats.multitest.fdrcorrection(
            p_values, alpha = alpha_value, is_sorted = False)
        
        stats_results = {}
        stats_results['Uncorrected_p_values_map'] = p_values
        stats_results['Corrected_p_values_map'] = p_values_corr
        stats_results['Boolean_statistical_map'] = rejected
        
        regression_features[feature] = stats_results
    # -------------------------------------------------------------------------
    # STEP 2.6 Save hyperparameters and scores
    # -------------------------------------------------------------------------    
    # Save the dictionary
    fileDir = 'encoding_stats_layers_{}_difference.pkl'.format(tail) 
    savefileDir = os.path.join(saveDir, fileDir) 
    
    with open(savefileDir, 'wb') as f:
        pickle.dump(regression_features, f)
    
# -----------------------------------------------------------------------------
# STEP 3 Run function
# -----------------------------------------------------------------------------
encoding_stats(n_perm, alpha_value, tail, saveDir, total_var)
