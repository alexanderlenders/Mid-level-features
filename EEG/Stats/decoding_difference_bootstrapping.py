#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOOTSTRAPPING DECODING - DIFFERENCE CURVE: IMAGES minus VIDEOS

This script calculates Bootstrap 95%-CIs for the decoding accuracy of the DIFFERENCE
CURVE VIDEO MINUS IMAGES for each timepoint (in ms). These can be used for the decoding 
plot as they are more informative than (empirical) standard errors, i.e. point estimates. 

@author: AlexanderLenders, AgnessaKarapetian
"""
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # add arguments / inputs
    parser.add_argument('-ls_v', "--list_sub_vid", default=[6], type=int, 
                        metavar='', help="list of subjects: videos")
    parser.add_argument('-ls_i', "--list_sub_img", default=[9], type=int, 
                        metavar='', help="list of subjects: images")
    parser.add_argument('-d', "--workdirvid", 
                        default = 'Z:/Unreal/Results/Decoding/miniclips/Redone',
                        type = str, metavar='', help="Results of the decoding analysis with videos")
    parser.add_argument('-id', "--workdirimg", 
                        default = 'Z:/Unreal/Results/Decoding/images/Redone',
                        type = str, metavar='', help="Results of the decoding analysis with images")
    parser.add_argument('-sd', "--savedir", 
                        default = 'Z:/Unreal/Results/Decoding/miniclips/Redone/stats',
                        type = str, metavar='', help="Where to save CIs")
    parser.add_argument('-np', "--num_perm", default = 10000, type = int, 
                        metavar='', help="Number of permutations")
    parser.add_argument('-tp', "--num_tp", default = 70, type = int, 
                        metavar='', help="Number of timepoints")

    args = parser.parse_args() # to get values for the arguments
    
    list_sub_vid = args.list_sub_vid     
    list_sub_img = args.list_sub_img      
    workDir_img = args.workdirimg
    workDir_vid = args.workdirvid
    saveDir = args.savedir
    n_perm = args.num_perm
    timepoints = args.num_tp

# -----------------------------------------------------------------------------------------------------------------
def bootstrapping_CI(list_sub_vid, list_sub_img, workDir_vid, workDir_img, saveDir, n_perm, timepoints): 
    """
    Bootstrapped 95%-CIs for the decoding accuracy for each timepoint. 
    
    Input: 
    ----------
    Decoding results with videos AND images. 

    Returns:
    ----------
    Dictionary with 95% CIs (values) for each timepoint (key). 

    Parameters
    ----------
    list_sub : list
          List of subjects for decoding with videos
    list_sub_img: list 
          List of subjects for decoding with images
    workDir : str
          Directory where the results of decoding with videos are
    workDir_img : str 
          Directory where the results of decoding with images are
    n_perm : int
          Number of permutations for bootstrapping
    timepoints : int
          Number of timepoints in each EEG epoch
    saveDir : str
        Where to save the CIs
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
    
    # Number of subjects
    n_sub_vid = len(list_sub_vid)
    n_sub_img = len(list_sub_img)
    
    decoding_mat_vid = np.zeros((len(list_sub_vid), timepoints))
    decoding_mat_img = np.zeros((len(list_sub_img), timepoints))

    
    # set random seed (for reproduction)
    np.random.seed(42)

    
    # -------------------------------------------------------------------------
    # STEP 2.2 Load results (videos)
    # -------------------------------------------------------------------------   
    
    for index, subject in enumerate(list_sub_vid): 
        if subject < 10: 
            fileDir = (workDir_vid + 
                       '/decoding_miniclips_sub-0{}_redone.npy'.format(subject))
        else: 
            fileDir = (workDir_vid + 
                       '/decoding_miniclips_sub-{}_redone.npy'.format(subject))
                   
        decoding_results_vid = np.load(fileDir, allow_pickle= True).item()
        
        decoding_accuracy_vid = decoding_results_vid['mean_accuracies_over_conditions']
        decoding_mat_vid[index, :] = decoding_accuracy_vid
        
    # -------------------------------------------------------------------------
    # STEP 2.3 Load results (static images)
    # -------------------------------------------------------------------------   

    for index, subject in enumerate(list_sub_img): 
            if subject < 10: 
                fileDir = (workDir_img + 
                        '/decoding_images_sub-0{}_redone.npy'.format(subject))
            else: 
                fileDir = (workDir_img + 
                        '/decoding_images_sub-{}_redone.npy'.format(subject))
                    
            decoding_results_img = np.load(fileDir, allow_pickle= True).item()
            
            decoding_accuracy_img = decoding_results_img['mean_accuracies_over_conditions']
            decoding_mat_img[index, :] = decoding_accuracy_img
    
    # -------------------------------------------------------------------------
    # STEP 2.4 Bootstrapping
    # -------------------------------------------------------------------------  
    bt_data = np.zeros((timepoints, n_perm))
    bt_data_peaks_img = np.zeros((n_perm,))
    bt_data_peaks_vid = np.zeros((n_perm,))

    for tp in range(timepoints): 
        tp_data_vid = decoding_mat_vid[:, tp]
        tp_data_img = decoding_mat_img[:, tp]
        for perm in range(n_perm): 
            # sampling of the subjects with replacement
            perm_tp_data_vid = np.random.choice(tp_data_vid, size = (n_sub_vid, 1), 
                                            replace = True)
            perm_tp_data_img = np.random.choice(tp_data_img, size = (n_sub_img, 1), 
                                            replace = True)
            
            mean_p_tp_img = np.mean(perm_tp_data_img, axis = 0)
            mean_p_tp_vid = np.mean(perm_tp_data_vid, axis = 0)

            bt_data[tp, perm] = mean_p_tp_img - mean_p_tp_vid

    #peaks
    for perm in range(n_perm): 
        # sampling of the subjects with replacement
        perm_subjects_img = np.random.choice(range(n_sub_img), size=n_sub_img, replace = True)
        perm_subjects_vid = np.random.choice(range(n_sub_vid), size=n_sub_vid, replace = True)
        
        perm_tp_data_img = decoding_mat_img[perm_subjects_img,:]
        perm_tp_data_vid = decoding_mat_vid[perm_subjects_vid,:]
    
        mean_p_tp_img = np.mean(perm_tp_data_img, axis = 0)
        mean_p_tp_vid = np.mean(perm_tp_data_vid, axis = 0)

        bt_data_peaks_img[perm] = int(np.argmax(mean_p_tp_img))
        bt_data_peaks_vid[perm] = int(np.argmax(mean_p_tp_vid)) 

    # -------------------------------------------------------------------------
    # STEP 2.6 Calculate 95%-CI
    # -------------------------------------------------------------------------      
    upper = int(n_perm * 0.975)
    lower = int(n_perm * 0.025)
    
    ci_dict = {}

    for tp in range(timepoints): 
        tp_data = bt_data[tp, :]
        sorted_tp_data = tp_data[np.argsort(tp_data)]
        lower_CI = sorted_tp_data[lower]
        upper_CI = sorted_tp_data[upper]

        ci_dict['{}'.format(tp)] = [lower_CI, upper_CI]

    #peaks
    sorted_peak_img = bt_data_peaks_img[np.argsort(bt_data_peaks_img)]
    lower_CI_peak_img = sorted_peak_img[lower]
    upper_CI_peak_img = sorted_peak_img[upper]
    ci_dict_peaks_img = [lower_CI_peak_img, upper_CI_peak_img]

    sorted_peak_vid = bt_data_peaks_vid[np.argsort(bt_data_peaks_vid)]
    lower_CI_peak_vid = sorted_peak_vid[lower]
    upper_CI_peak_vid = sorted_peak_vid[upper]
    ci_dict_peaks_vid = [lower_CI_peak_vid, upper_CI_peak_vid]

    # -------------------------------------------------------------------------
    # STEP 2.7 Save CI
    # -------------------------------------------------------------------------  
    # Save the dictionary
    
    #ci
    fileDir_ci = ('diff_CI_95_accuracy_redone.pkl')      
    savefileDir_ci = os.path.join(saveDir, fileDir_ci) 
    with open(savefileDir_ci, 'wb') as f:
        pickle.dump(ci_dict, f)

    #peaks           
    fileDir_peak_img = ('peak_img_CI_95.pkl')      
    savefileDir_img = os.path.join(saveDir, fileDir_peak_img) 
    with open(savefileDir_img, 'wb') as f:
        pickle.dump(ci_dict_peaks_img, f)

    fileDir_peak_vid = ('peak_vid_CI_95.pkl')      
    savefileDir_vid = os.path.join(saveDir, fileDir_peak_vid) 
    with open(savefileDir_vid, 'wb') as f:
        pickle.dump(ci_dict_peaks_vid, f)
        
# -----------------------------------------------------------------------------
# STEP 3: Run function
# -----------------------------------------------------------------------------
list_sub_vid = [6, 7, 8, 9, 10, 11, 17, 18, 20, 21, 23, 25, 27, 28, 29, 30, 31, 32, 34, 36]
list_sub_img = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

bootstrapping_CI(list_sub_vid, list_sub_img, workDir_vid, workDir_img, saveDir, n_perm, timepoints) 


        
        
    
