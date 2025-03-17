#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOOTSTRAPPING ENCODING 

This script calculates Bootstrap 95%-CIs for the encoding accuracy for each
timepoint (in ms) and each feature. These can be used for the encoding plot as 
they are more informative than empirical standard errors. 

In addition, this script calculates Bootstrap 95%-CIs for the timepoint (in ms)
of the largest encoding peak for each feature. 

@author: AlexanderLenders, AgnessaKarapetian
"""
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # add arguments / inputs
    parser.add_argument('-ls_v', "--list_sub_vid", default=[6], type=int, 
                        metavar='', help="list of subjects for videos (see below)")
    parser.add_argument('-ls_i', "--list_sub_img", default=[9], type=int, 
                        metavar='', help="list of subjects for images (see below)")
    parser.add_argument('-np', "--num_perm", default = 10000, type = int, 
                        metavar='', help="Number of permutations")
    parser.add_argument('-tp', "--num_tp", default = 70, type = int, 
                        metavar='', help="Number of timepoints")

    args = parser.parse_args() # to get values for the arguments
    
    list_sub_vid = args.list_sub_vid     
    list_sub_img = args.list_sub_img 
    n_perm = args.num_perm
    timepoints = args.num_tp

def bootstrapping_CI(list_sub_vid, list_sub_img, n_perm, timepoints): 
    """
    Bootstrapped 95%-CIs for the encoding accuracy for each timepoint and 
    each feature. 

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
    import os
    import pickle
    import statsmodels
    from statsmodels.stats.multitest import multipletests
    

    workDir_img = 'Z:/Unreal/images_results/encoding/'
    workDir_vid = 'Z:/Unreal/Results/Encoding/'
    saveDir = 'Z:/Unreal/images_results/encoding/redone/stats'
        
    feature_names = ('edges','world_normal', 'scene_depth',
                    'lighting', 'reflectance', 'skeleton','action')
        

    identifierDir = 'seq_50hz_posterior_encoding_results_averaged_frame_before_mvnn_7features_onehot.pkl'


    #set some vars
    n_sub_vid = len(list_sub_vid)
    n_sub_img = len(list_sub_img)
    time_ms = np.arange(-400,1000,20)

    # set random seed (for reproduction)
    np.random.seed(42)
    
    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------   
    results_vid = {}
    results_img = {}
    for subject in list_sub_vid:
        fileDir_vid = (workDir_vid + 'redone/7_features/{}_'.format(subject) + identifierDir)
        encoding_results_vid = np.load(fileDir_vid, allow_pickle=True)
        results_vid[str(subject)] = encoding_results_vid

    for subject in list_sub_img:
        fileDir_img = (workDir_img + 'redone/7_features/{}_'.format(subject) + identifierDir)
        encoding_results_img = np.load(fileDir_img, allow_pickle=True)
        results_img[str(subject)] = encoding_results_img
    
    
    #Loop over all features
    
    feature_results = {}

    results_vid_dict = {}   
    results_img_dict = {} 
    results_diff_dict_avg = {}
    ci_diff_peaks_all= {}
    ci_dict_all = {}

    for feature in feature_names: 
        
        results_f_vid = np.zeros((n_sub_vid, timepoints)) 

        #videos
        for index, subject in enumerate(list_sub_vid): 
            subject_result = results_vid[str(subject)][feature]['correlation']
            subject_result_averaged = np.mean(subject_result, axis = 1) # averaged over all channels 
            results_f_vid[index, :] = subject_result_averaged
        results_vid_dict[feature] = results_f_vid
        
        #images
        results_f_img = np.zeros((n_sub_img, timepoints)) 
        for index, subject in enumerate(list_sub_img): 
    
            subject_result = results_img[str(subject)][feature]['correlation']
            subject_result_averaged = np.mean(subject_result, axis = 1) # averaged over all channels 
            results_f_img[index, :] = subject_result_averaged    
        results_img_dict[feature] = results_f_img

        #difference
        results_diff_avg = np.mean(results_f_img,axis=0) - np.mean(results_f_vid,axis=0) #avg over subjects then calc. diff - cant do diff otherwise
        results_diff_dict_avg[feature] = results_diff_avg

        # ---------------------------------------------------------------------
        # STEP 2.3 Bootstrapping: accuracy
        # ---------------------------------------------------------------------
        bt_data = np.zeros((timepoints, n_perm))
        
        for tp in range(timepoints): 
            tp_data_vid = results_f_vid[:, tp]
            tp_data_img = results_f_img[:, tp]

            for perm in range(n_perm): 
                perm_tp_data_vid = np.random.choice(tp_data_vid, size = (n_sub_vid, 1), 
                                                replace = True)
                perm_tp_data_img = np.random.choice(tp_data_img, size = (n_sub_img, 1), 
                                                replace = True)
                mean_p_tp_vid = np.mean(perm_tp_data_vid, axis = 0)
                mean_p_tp_img = np.mean(perm_tp_data_img, axis = 0)
                bt_data[tp, perm] = mean_p_tp_img - mean_p_tp_vid

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
        # STEP 2.5 Bootstrapping: peak latency -> 1) PEAK OF DIFFERENCE CURVE 2) DIFFERENCE IN PEAK LATENCIES (SIG/NON-SIG) 
        # -------------------------------------------------------------------------  

        #Find ground truth peak latency (ms) of the difference curve
        peak_true = time_ms[np.argmax(results_diff_avg)]    

        #Find ground truth difference in peak latencies (ms) of vid. vs img
        diff_in_peak_true = np.abs(time_ms[np.argmax(np.mean(results_f_img,axis=0))]
                                   -time_ms[np.argmax(np.mean(results_f_vid,axis=0))])

        #Permute and calculate peak latencies for bootstrap samples
        bt_data_peaks = np.zeros((n_perm,))
        bt_diff_peaks = np.zeros((n_perm,))
        for perm in range(n_perm): 
            perm_peak_data_idx_vid = np.random.choice(results_f_vid.shape[0], size = (n_sub_vid, 1),
                                                      replace = True)
            perm_peak_data_idx_img = np.random.choice(results_f_img.shape[0], size = (n_sub_img, 1), 
                                                      replace = True)
            perm_peak_data_vid = np.squeeze(results_f_vid[perm_peak_data_idx_vid])
            perm_peak_data_img = np.squeeze(results_f_img[perm_peak_data_idx_img])

            perm_mean_vid = np.mean(perm_peak_data_vid, axis = 0)    
            perm_mean_img = np.mean(perm_peak_data_img, axis = 0)
            perm_mean_diff = perm_mean_img - perm_mean_vid
            
            # 1) peak of the difference curve
            peak_diff = time_ms[np.argmax(perm_mean_diff)]
            bt_data_peaks[perm] = peak_diff

            # 2) difference in the peaks
            peak_lat_vid = time_ms[np.argmax(perm_mean_vid)]
            peak_lat_img = time_ms[np.argmax(perm_mean_img)]
            diff_in_peakl = np.abs(peak_lat_img - peak_lat_vid)
            bt_diff_peaks[perm] = diff_in_peakl


        # ---------------------------------------------------------------------
        # STEP 2.6 Calculate 95%-CI
        # ---------------------------------------------------------------------
        upper = round(n_perm * 0.975)
        lower = round(n_perm * 0.025)
        
        # 1) peak of the difference curve
        ci_dict = {}
        peak_data = bt_data_peaks
        ranks = rankdata(peak_data)
        t_data = np.vstack((ranks, peak_data))
        ascending_ranks_idx = np.argsort(ranks, axis = 0)
        ascending_ranks = t_data[:, ascending_ranks_idx]
        lower_CI = ascending_ranks[1, lower]
        upper_CI = ascending_ranks[1, upper]
        
        ci_dict_all['{}'.format(feature)] = [lower_CI, peak_true, upper_CI]

        #2) difference in the peaks

        peak_data_diff = bt_diff_peaks
        ranks_diff = rankdata(peak_data_diff)
        t_data_diff = np.vstack((ranks_diff, peak_data_diff))
        ascending_ranks_idx_diff = np.argsort(ranks_diff, axis = 0)
        ascending_ranks_diff = t_data_diff[:, ascending_ranks_idx_diff]
        lower_CI_diff = ascending_ranks_diff[1, lower]
        upper_CI_diff = ascending_ranks_diff[1, upper]
                        
        ci_diff_peaks_all['{}'.format(feature)] = [lower_CI_diff, diff_in_peak_true, upper_CI_diff]

    # -------------------------------------------------------------------------
    # STEP 2.7 Save CI accuracy
    # -------------------------------------------------------------------------  
    # Save the dictionary
    
    fileDir = 'encoding_difference_CI95_accuracy.pkl'
    
    savefileDir = os.path.join(saveDir, fileDir) 
     
    
    with open(savefileDir, 'wb') as f:
        pickle.dump(feature_results, f)


    # -------------------------------------------------------------------------
    # STEP 2.8 Save CI - peak latency
    # -------------------------------------------------------------------------  
    # Save the dictionary

    fileDir = 'encoding_difference_CI95_peak.pkl'

    savefileDir = os.path.join(saveDir, fileDir) 
        
    with open(savefileDir, 'wb') as f:
        pickle.dump(ci_dict_all, f)

    # -------------------------------------------------------------------------
    # STEP 2.9 Save CI - difference in peak latencies
    # -------------------------------------------------------------------------  
    # Save the dictionary

    fileDir = 'encoding_diff_in_peak.pkl'

    savefileDir = os.path.join(saveDir, fileDir) 
        
    with open(savefileDir, 'wb') as f:
        pickle.dump(ci_diff_peaks_all, f)

    
    # -------------------------------------------------------------------------
    # STEP 2.10 Bootstrapping - peak latency differences
    # -------------------------------------------------------------------------

    pairwise_p = {}
        
    for feature1 in range(len(feature_names)): 
        feature_A = feature_names[feature1]
        num_comparisons = feature1 + 1

        #get results for feature
        results_diff_feature_A = results_diff_dict_avg[feature_A] #avg over subjects -> vector of timepoints
        results_vid_feature_A = results_vid_dict[feature_A] #subjects x timepoints 
        results_img_feature_A = results_img_dict[feature_A] #subjects x timepoints

        #encoding diff (img-vid) mean 
        encoding_mean_diff = results_diff_feature_A  
        peaks_idx = np.argsort(encoding_mean_diff, axis = 0) 
        encoding_ms = np.vstack((encoding_mean_diff, time_ms))
        encoding_ms_ord = encoding_ms[:, peaks_idx]
        
        peak_A = encoding_ms_ord[1, -1]

        for feature2 in range(num_comparisons): 
            feature_B = feature_names[feature2]
            
            if feature_A == feature_B: 
                continue 
            
            str_comparison = '{} vs. {}'.format(feature_A, feature_B)
            
            
            #get results for feature
            results_diff_feature_B = results_diff_dict_avg[feature_B] #avg over subjects -> vector of timepoints
            results_vid_feature_B = results_vid_dict[feature_B] #subjects x timepoints 
            results_img_feature_B = results_img_dict[feature_B] #subjects x timepoints
            
            #encoding diff (img-vid) mean 
            encoding_mean_diff_B = results_diff_feature_B  
            peaks_idx_B = np.argsort(encoding_mean_diff_B, axis = 0) 
            encoding_ms_B = np.vstack((encoding_mean_diff_B, time_ms))
            encoding_ms_ord_B = encoding_ms_B[:, peaks_idx_B]
            
            peak_B = encoding_ms_ord_B[1, -1]
        
            feature_diff = peak_A - peak_B
            
            bt_data_peaks = np.zeros((n_perm,))
            
            for perm in range(n_perm): 
                #select X subjects with replacement
                perm_peak_data_idx_vid = np.random.choice(n_sub_vid, size = (n_sub_vid, 1),replace = True)
                perm_peak_data_idx_img = np.random.choice(n_sub_img, size = (n_sub_img, 1),replace = True)

                perm_peak_data_vid_A = np.squeeze(results_vid_feature_A[perm_peak_data_idx_vid]) #double check dimensions
                perm_peak_data_vid_B = np.squeeze(results_vid_feature_B[perm_peak_data_idx_vid])

                perm_peak_data_img_A = np.squeeze(results_img_feature_A[perm_peak_data_idx_img]) #double check dimensions
                perm_peak_data_img_B = np.squeeze(results_img_feature_B[perm_peak_data_idx_img])

                #avg over subjects
                perm_mean_vid_A = np.mean(perm_peak_data_vid_A, axis = 0)    
                perm_mean_vid_B = np.mean(perm_peak_data_vid_B, axis = 0) 
                perm_mean_img_A = np.mean(perm_peak_data_img_A, axis = 0)
                perm_mean_img_B = np.mean(perm_peak_data_img_B, axis = 0)

                #difference curve for each feature
                perm_mean_diff_A = perm_mean_img_A - perm_mean_vid_A
                perm_mean_diff_B = perm_mean_img_B - perm_mean_vid_B
                
                #difference of the difference
                diff_diff = perm_mean_diff_A - perm_mean_diff_B
                peak_diff_diff = time_ms[np.argmax(diff_diff)]

                #difference between the peak latencies of the differences curves of both features
                bt_data_peaks[perm] = peak_diff_diff


            # -----------------------------------------------------------------
            # STEP 2.11 Compute p-Value and CI
            # -----------------------------------------------------------------
            # CI 
            upper = int(np.ceil(n_perm * 0.975))
            lower = int(np.ceil(n_perm * 0.025))
            
            peak_data = bt_data_peaks
            ranks = rankdata(peak_data)
            t_data = np.vstack((ranks, peak_data))
            ascending_ranks_idx = np.argsort(ranks, axis = 0)
            ascending_ranks = t_data[:, ascending_ranks_idx]
            lower_CI = ascending_ranks[1, lower]
            upper_CI = ascending_ranks[1, upper]

            recenter = ascending_ranks[1, :] - feature_diff
            sum_higher = np.sum(np.abs(recenter) > np.abs(feature_diff))
            p_value = (1+sum_higher)/(n_perm+1)
                        
            c_dict = {'p_value': p_value, 'ci': [lower_CI, feature_diff, upper_CI]}
            pairwise_p[str_comparison] = c_dict
    
    
    # -------------------------------------------------------------------------
    # STEP 2.12 Save CI
    # -------------------------------------------------------------------------  
    # Save the dictionary
    
    fileDir = 'encoding_difference_stats_peak_latency_CI.pkl'
    
    # Benjamini-Hochberg Correction 
    p_values_vector = [pairwise_p[key]['p_value'] for key in pairwise_p]
    
    _, p_values_corr = statsmodels.stats.multitest.fdrcorrection(
    p_values_vector, alpha = 0.05, is_sorted = False)
    
    for i, key in enumerate(pairwise_p):
        pairwise_p[key]['p_value'] = p_values_corr[i]

    savefileDir = os.path.join(saveDir, fileDir) 
     
    with open(savefileDir, 'wb') as f:
        pickle.dump(pairwise_p, f)
    
    
# -----------------------------------------------------------------------------
# STEP 3: Run functions
# -----------------------------------------------------------------------------
list_sub_vid = [6, 7, 8, 9, 10, 11, 17, 18, 20, 21, 23, 25, 27, 28, 29, 30, 31, 32, 34, 36]
list_sub_img = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


bootstrapping_CI(list_sub_vid, list_sub_img, n_perm, timepoints) 

    
    
