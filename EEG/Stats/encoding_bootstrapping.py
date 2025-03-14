#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOOTSTRAPPING ENCODING 

This script calculates Bootstrap 95%-CIs for the encoding accuracy for each
timepoint (in ms) and each feature. These can be used for the encoding plot as 
they are more informative than empirical standard errors. 

In addition, this script calculates Bootstrap 95%-CIs for the timepoint (in ms)
of the largest encoding peak for each feature. 

At the moment the directories as well as using correlation instead of RMSE to
determine how well multivariate regression can predict (encode) EEG activity
per channel is hardcoded.

This script implements an empirical bootstrap CI, not a bootstrap with the 
percentile method, see https://math.mit.edu/~dav/05.dir/class24-prep-a.pdf. 

In addition, it implements BCa (bias-corrected and accelerated bootstrap CIs.)

https://stats.stackexchange.com/questions/355781/is-it-true-that-the-percentile-bootstrap-should-never-be-used

@author: AlexanderLenders
"""
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # add arguments / inputs
    parser.add_argument('-ls', "--list_sub", default=0, type=int, 
                        metavar='', help="list of subjects")
    parser.add_argument('-ty', "--type", default = 'unreal_before_pca', type = str, 
                        metavar='', help="type of encoding")
    parser.add_argument('-np', "--num_perm", default = 10000, type = int, 
                        metavar='', help="Number of permutations")
    parser.add_argument('-tp', "--num_tp", default = 70, type = int, 
                        metavar='', help="Number of timepoints")
    parser.add_argument('-i', "--input_type", default = 'miniclips', type = str, 
                        metavar='', help="Font")
    args = parser.parse_args() # to get values for the arguments
    
    list_sub = args.list_sub      
    n_perm = args.num_perm
    timepoints = args.num_tp
    type_analysis = args.type
    input_type = args.input_type


def bootstrapping_CI(list_sub, n_perm, timepoints, type_analysis, input_type): 
    """
    Bootstrapped 95%-CIs for the encoding accuracy for each timepoint and 
    each feature. 
    Calculates empirical CI. 

    Input: 
    ----------
    Output from the encoding analysis (multivariate linear regression), i.e.: 
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
    plot_hist : bool
        Whether to plot the bootstrapping histograms
    input_type : str
        Images or miniclips
    """
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
    # Import modules
    import numpy as np
    from scipy.stats import rankdata
    import matplotlib.pyplot as plt
    import math
    import os
    import pickle
    import statsmodels
    from statsmodels.stats.multitest import multipletests
        
    if type_analysis == 'unreal_before_pca':
        if input_type == 'miniclips':
            workDir = 'Z:/Unreal/Results/Encoding/'
            saveDir = 'Z:/Unreal/Results/Encoding/redone/stats'

        elif input_type == 'images':     
            workDir = 'Z:/Unreal/images_results/encoding/'
            saveDir = 'Z:/Unreal/images_results/encoding/redone/stats'
            
        feature_names = ('edges','world_normal', 'scene_depth',
                        'lightning', 'reflectance', 'skeleton','action')
        
        identifierDir = 'seq_50hz_posterior_encoding_results_averaged_frame_before_mvnn_7features_onehot.pkl'


    elif type_analysis == 'taskonomy_before_pca': 
        if input_type == 'miniclips':
            workDir = '/Volumes/Elements/miniclip_data/results/encoding/taskonomy/video/encoding/100'
            saveDir = '/Volumes/Elements/miniclip_data/results/encoding/taskonomy/video/stats'
            identifierDir = 'seq_50hz_posterior_encoding_results_frame_avg.pkl'
        
        feature_names = ('normal', 'edge_texture', 'depth_euclidean', 'reshading', 'curvature')

    #set some vars
    n_sub = len(list_sub)
    time_ms = np.arange(-400,1000,20)
    n_peak = 1 # hardcoded

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
    
    
    #Loop over all features
    
    feature_results = {}
    results_dict = {}    
    ci_dict_all = {}

    for feature in feature_names: 
        
        results = np.zeros((n_sub, timepoints)) 

        for index, subject in enumerate(list_sub): 
    
            subject_result = results_unfiltered[str(subject)][feature]['correlation']
            # averaged over all channels 
            subject_result_averaged = np.mean(subject_result, axis = 1)
            results[index, :] = subject_result_averaged

        results_dict[feature] = results

    #     ---------------------------------------------------------------------
    #     STEP 2.3 Bootstrapping: accuracy
    #     ---------------------------------------------------------------------
        bt_data = np.zeros((timepoints, n_perm))
        
        for tp in range(timepoints): 
            tp_data = results[:, tp]
            # mean_tp = np.mean(tp_data, axis = 0)
            for perm in range(n_perm): 
                perm_tp_data = np.random.choice(tp_data, size = (n_sub, 1), 
                                                replace = True)
                mean_p_tp = np.mean(perm_tp_data, axis = 0)
                bt_data[tp, perm] = mean_p_tp

        # ---------------------------------------------------------------------
        # STEP 2.4 Calculate 95%-CI
        # ---------------------------------------------------------------------
        upper = int(np.ceil(n_perm * 0.975))
        lower = int(np.ceil(n_perm * 0.025))
        
        ci_dict = {}
        
        for tp in range(timepoints): 
            tp_data = bt_data[tp, :]
            ranks = rankdata(tp_data)
            t_data = np.vstack((ranks, tp_data))
            ascending_ranks_idx = np.argsort(ranks, axis = 0)
            ascending_ranks = t_data[:, ascending_ranks_idx]
            lower_CI = ascending_ranks[1, lower]
            upper_CI = ascending_ranks[1, upper]
            
            ci_dict['{}'.format(tp)] = [lower_CI, upper_CI]
        
        feature_results[feature] = ci_dict
        
        # -------------------------------------------------------------------------
        # STEP 2.5 Bootstrapping: peak latency
        # -------------------------------------------------------------------------  

        #Find ground truth peak latency (ms)
        encoding_mean = np.mean(results, axis = 0)     
        peak_1 = time_ms[np.argmax(encoding_mean)]    

        #Permute and calculate peak latencies for bootstrap samples
        bt_data_peaks = np.zeros((n_peak, n_perm))

        for perm in range(n_perm): 
            perm_peak_data_idx = np.random.choice(results.shape[0], size = (n_sub, 1), 
                                            replace = True)
            perm_peak_data = np.squeeze(results[perm_peak_data_idx])
            perm_mean = np.mean(perm_peak_data, axis = 0)        
            bt_data_peaks[0, perm] = time_ms[np.argmax(perm_mean)]
          

        # ---------------------------------------------------------------------
        # STEP 2.6 Calculate 95%-CI
        # ---------------------------------------------------------------------
        upper = round(n_perm * 0.975)
        lower = round(n_perm * 0.025)
        
        ci_dict = {}
            
        for peak in range(n_peak): 
            peak_data = bt_data_peaks[peak, :]
            ranks = rankdata(peak_data)
            t_data = np.vstack((ranks, peak_data))
            ascending_ranks_idx = np.argsort(ranks, axis = 0)
            ascending_ranks = t_data[:, ascending_ranks_idx]
            lower_CI = ascending_ranks[1, lower]
            upper_CI = ascending_ranks[1, upper]
                
            ci_dict['{}'.format(peak)] = [lower_CI, peak_1, upper_CI]
        
        ci_dict_all['{}'.format(feature)] = [lower_CI, peak_1, upper_CI]

 
    # -------------------------------------------------------------------------
    # STEP 2.7 Save CI accuracy
    # -------------------------------------------------------------------------  
    # Save the dictionary
    
    fileDir = 'encoding_{}_{}_CI95_accuracy.pkl'.format(type_analysis,input_type)  
    
    savefileDir = os.path.join(saveDir, fileDir) 
     
    # # Creating the directory if not existing
    # if os.path.isdir(os.path.join(saveDir)) == False: # if not a directory
    #     os.makedirs(os.path.join(saveDir))
    
    # with open(savefileDir, 'wb') as f:
    #     pickle.dump(feature_results, f)


    # -------------------------------------------------------------------------
    # STEP 2.8 Save CI - peak latency
    # -------------------------------------------------------------------------  
    # Save the dictionary

    fileDir = 'encoding_{}_{}_CI95_peak.pkl'.format(type_analysis,input_type)  

    savefileDir = os.path.join(saveDir, fileDir) 
        
    # with open(savefileDir, 'wb') as f:
    #     pickle.dump(ci_dict_all, f)

    
    
    # -------------------------------------------------------------------------
    # STEP 2.9 Bootstrapping - peak latency differences
    # -------------------------------------------------------------------------

    
    pairwise_p = {}
        
    for feature1 in range(len(feature_names)): 
        feature_A = feature_names[feature1]
        num_comparisons = feature1 + 1
        results_feature_A = results_dict[feature_A]
        encoding_mean = np.mean(results_feature_A, axis = 0) 
        peak_A = time_ms[np.argmax(encoding_mean)]    

        for feature2 in range(num_comparisons): 
            feature_B = feature_names[feature2]
            
            if feature_A == feature_B: 
                continue 
            
            str_comparison = '{} vs. {}'.format(feature_A, feature_B)
            
            results_feature_B = results_dict[feature_B]
            encoding_mean_B = np.mean(results_feature_B, axis = 0)     

            peak_B =  time_ms[np.argmax(encoding_mean_B)]
        
            feature_diff = peak_A - peak_B
            
            bt_data_peaks = np.zeros((n_peak, n_perm))
            
            for perm in range(n_perm): 
                perm_peak_data_idx = np.random.choice(results_feature_A.shape[0], size = (n_sub, 1), 
                                                replace = True)
                
                perm_peak_data_A = results_feature_A[perm_peak_data_idx].reshape((n_sub, timepoints))
                perm_peak_data_B = results_feature_B[perm_peak_data_idx].reshape((n_sub, timepoints))

                perm_mean_A = np.mean(perm_peak_data_A, axis = 0)
                perm_mean_B = np.mean(perm_peak_data_B, axis = 0)
                
                peak_A = time_ms[np.argmax(perm_mean_A)]
                peak_B = time_ms[np.argmax(perm_mean_B)]
                
                feature_diff_bt = peak_A - peak_B
                
                bt_data_peaks[0, perm] = feature_diff_bt

            # -----------------------------------------------------------------
            # STEP 2.10 Compute p-Value and CI
            # -----------------------------------------------------------------
            # CI 
            upper = int(np.ceil(n_perm * 0.975))
            lower = int(np.ceil(n_perm * 0.025))
            
            for peak in range(n_peak): 
                peak_data = bt_data_peaks[peak, :]
                peak_data_sorted = peak_data[np.argsort(peak_data)]
                lower_CI = peak_data_sorted[lower]
                upper_CI = peak_data_sorted[upper]

            c_dict = {'ci': [lower_CI, feature_diff, upper_CI]}

            pairwise_p[str_comparison] = c_dict
    
    
    # -------------------------------------------------------------------------
    # STEP 2.11 Save CI
    # -------------------------------------------------------------------------  
    # Save the dictionary
    
    fileDir = 'encoding_{}_{}_stats_peak_latency_CI_redone.pkl'.format(type_analysis,input_type)  
    savefileDir = os.path.join(saveDir, fileDir) 
     
    
    with open(savefileDir, 'wb') as f:
        pickle.dump(pairwise_p, f)
    
    
# -----------------------------------------------------------------------------
# STEP 3: Run functions
# -----------------------------------------------------------------------------
if input_type == 'miniclips':
    list_sub = [6, 7, 8, 9, 10, 11, 17, 18, 20, 21, 23, 25, 27, 28, 29, 30, 31, 32, 34, 36]
elif input_type == 'images':
    list_sub = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


bootstrapping_CI(list_sub, n_perm, timepoints, type_analysis, input_type) 

    
    
