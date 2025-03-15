#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOOTSTRAPPING ENCODING LAYERS CLUSTER

This script calculates Bootstrap 95%-CIs for the encoding accuracy for each
timepoint (in ms) and each feature. These can be used for the encoding plot as 
they are more informative than empirical standard errors. 

In addition, this script calculates Bootstrap 95%-CIs for the timepoint (in ms)
of the largest encoding peak for each feature. 

At the moment the directories as well as using correlation instead of RMSE to
determine how well multivariate regression can predict (encode) EEG activity
per channel is hardcoded.

This script implements a bootstrap with the percentile method.

In addition, it implements BCa (bias-corrected and accelerated bootstrap CIs.)

@author: AlexanderLenders, AgnessaKarapetian
"""
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # add arguments / inputs
    parser.add_argument('-np', "--num_perm", default = 10000, type = int, 
                        metavar='', help="Number of permutations")
    parser.add_argument('-l', "--num_layers", default = 8, type = int, 
                        metavar='', help="Number of layers")
    parser.add_argument('-p', "--plot", default = False, type = bool, 
                        metavar='', help="Whether to plot bootstrapping hists")
    parser.add_argument('-i', "--input_type", default = 'images', type = str, 
                    metavar='', help="Font")
    parser.add_argument('-ed',"--encoding_dir", help='Directory with encoding results', 
                    default ='Z:/Unreal/Results/Encoding/CNN_redone/')
    parser.add_argument('-tv',"--total_var", help='Total variance explained by all PCA components together', 
                    default = 90)

    args = parser.parse_args() # to get values for the arguments
       
    n_perm = args.num_perm
    n_layers = args.num_layers
    plot_hist = args.plot
    input_type = args.input_type
    encoding_dir = args.encoding_dir
    total_var = args.total_var

def bootstrapping_CI(n_perm, n_layers, plot_hist, input_type, encoding_dir, total_var): 
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
    saveDir : str
        Where to save the CIs
    input_type : str
        Miniclips or images
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

    if input_type == 'images':
        workDir = os.path.join(encoding_dir,'2D_ResNet18/')
    elif input_type == 'miniclips':
        workDir = os.path.join(encoding_dir,'3D_ResNet18/')
    
    saveDir = os.path.join(workDir,'stats/')   
        
    feature_names = ('edges', 'world_normal', 'lightning',
                 'scene_depth', 'reflectance', 'action', 'skeleton')
        
    
    # set random seed (for reproduction)
    np.random.seed(42)
    
    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------   
    fileDir = os.path.join(workDir, 'encoding_layers_resnet.pkl')   

    encoding_results = np.load(fileDir, allow_pickle= True)
    
    regression_features = dict.fromkeys(feature_names)
    
    for feature in feature_names:
        regression_features[feature] = encoding_results[feature]['weighted_correlations']
    
    features_results = {}
    
    for i, feature in enumerate(feature_names):
        results = regression_features[feature]

        # ---------------------------------------------------------------------
        # STEP 2.3 Bootstrapping
        # ---------------------------------------------------------------------
        bt_data = np.zeros((n_layers, n_perm))
        
        for l,layer in enumerate(layers_names): 
            layer_data = results[layer]
            num_comp_layer = layer_data.shape[0]
            for perm in range(n_perm): 
                perm_l_data = np.random.choice(layer_data, size = (num_comp_layer, 1), 
                                                replace = True)
                layer_weighted_sum = np.sum(perm_l_data)/total_var 
                bt_data[l, perm] = layer_weighted_sum
            
            
        # ---------------------------------------------------------------------
        # STEP 2.4 Plot histogram
        # ---------------------------------------------------------------------
        if plot_hist == True:
            n_bins = int(math.sqrt(n_perm))
            layer_plot = [0, 2, 4]
            
            for plot in layer_plot: 
                plt.hist(bt_data[plot, :], bins = n_bins)
                plt.show()
        
        # ---------------------------------------------------------------------
        # STEP 2.5 Calculate 95%-CI
        # ---------------------------------------------------------------------
        upper = int(np.ceil(n_perm * 0.975))
        lower = int(np.ceil(n_perm * 0.025))
        
        ci_dict = {}
        
        for l,layer in enumerate(layers_names): 
            layer_data = bt_data[l, :]
            # ranks = rankdata(layer_data)
            # l_data = np.vstack((ranks, layer_data))
            # ascending_ranks_idx = np.argsort(ranks, axis = 0)
            # # ascending_ranks = l_data[:, ascending_ranks_idx]

            # lower_CI = ascending_ranks[1, lower]
            # upper_CI = ascending_ranks[1, upper]
            layer_data.sort()
            ci_dict['{}'.format(layer)] = [layer_data[lower], layer_data[upper]]

        features_results[feature] = ci_dict
    # -------------------------------------------------------------------------
    # STEP 2.6 Save CI
    # -------------------------------------------------------------------------  
    # Save the dictionary
    
    fileDir = 'encoding_layers_CI95_accuracy.pkl'
    
    savefileDir = os.path.join(saveDir, fileDir) 
    
    with open(savefileDir, 'wb') as f:
        pickle.dump(features_results, f)
        
    
def bootstrapping_CI_peak_layer(n_perm, plot_hist, input_type, encoding_dir, total_var): 

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
    saveDir : str
        Where to save the CIs
    input_type : str
        Miniclips or images
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

    if input_type == 'images':
        workDir = os.path.join(encoding_dir,'2D_ResNet18/')
    elif input_type == 'miniclips':
        workDir = os.path.join(encoding_dir,'3D_ResNet18/')
    
    saveDir = os.path.join(workDir,'stats/')   
        
    feature_names = ('edges', 'world_normal', 'lightning',
                 'scene_depth', 'reflectance', 'action', 'skeleton')
        
  
    # set random seed (for reproduction)
    np.random.seed(42)
    
    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------   
    fileDir = os.path.join(workDir, 'encoding_layers_resnet.pkl')   

    encoding_results = np.load(fileDir, allow_pickle= True)
    regression_features = {}
    peak_cis = {}
    ci_dict_all = {}
    for feature in feature_names:
        peak_cis[feature] = encoding_results[feature]['correlation_average']
        
    peak_ci_dict_all = {}
    
    #true peaks
    for i, feature in enumerate(feature_names): 
        
        results = peak_cis[feature]
        peak_idx = np.argmax(results)
        peak_ci_dict_all[feature] = peak_idx
        
    ### Bootstrapping peak layers ###
    for feature in feature_names:
        regression_features[feature] = encoding_results[feature]['weighted_correlations']
    
    # features_results = {}
    for i, feature in enumerate(feature_names):
        results = regression_features[feature]

        # bt_data_cis = np.zeros((n_perm))
        layer_data_all = np.zeros((n_layers,n_perm))
        for l,layer in enumerate(layers_names): 
            layer_data = results[layer]
            num_comp_layer = layer_data.shape[0]
            for perm in range(n_perm): 
                perm_l_data = np.random.choice(layer_data, size = (num_comp_layer, 1), 
                                                replace = True)
                layer_weighted_sum = np.sum(perm_l_data)/total_var                 
                layer_data_all[l,perm]=(layer_weighted_sum) #correlation average for every layer

        bt_data_peaks = np.argmax(layer_data_all,axis=0)
        
        # ---------------------------------------------------------------------
        # STEP 2.4 Plot histogram
        # ---------------------------------------------------------------------
        if plot_hist == True:
            n_bins = int(math.sqrt(n_perm))
            plt.hist(bt_data_peaks[:], bins = n_bins)
            plt.show()
        # ---------------------------------------------------------------------
        # STEP 2.5 Calculate 95%-CI
        # ---------------------------------------------------------------------
        upper = round(n_perm * 0.975)
        lower = round(n_perm * 0.025)
         
        #sort and get upper and lower percentiles
        bt_data_peaks.sort()
        lower_CI = bt_data_peaks[lower]
        upper_CI = bt_data_peaks[upper]
        ci_dict_all['{}'.format(feature)] = [lower_CI, peak_ci_dict_all[feature], upper_CI]

    # -------------------------------------------------------------------------
    # STEP 2.6 Save CI
    # -------------------------------------------------------------------------  
    # Save the dictionary
    
    fileDir = 'encoding_layers_CI95_peak.pkl'
    
    savefileDir = os.path.join(saveDir, fileDir) 
    
    with open(savefileDir, 'wb') as f:
        pickle.dump(ci_dict_all, f)

# -----------------------------------------------------------------------------
def bootstrapping_stats_diff_btw_features(n_perm, n_layers, plot_hist, input_type, encoding_dir, total_var): 
    """
    Bootstrapped 95%-CIs for the timepoint (in ms) of the largest encoding peak
    for each feature.
    
    Input: 
    ----------
    Output from the encoding analysis (multivariate linear regression), i.e.: 
    Encoding results (multivariate linear regression), saved in a dictionary 
    which contains for every feature correlation measure, i.e.: 
        encoding_results[feature]['correlation']
    
    Returns:
    ----------
    Dictionary with level 1 features and level 2 with 95% CIs (values) 
    for each peak (key), i.e. results[feature][peak]
    
    Parameters
    ----------
    list_sub : list
          List of subjects for which encoding results exist
    n_perm : int
          Number of permutations for bootstrapping
    n_layers : int
          Number of layers 
    plot_hist : bool
        Whether to plot the bootstrapping histograms
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

    if input_type == 'images':
        workDir = os.path.join(encoding_dir,'2D_ResNet18/')
    elif input_type == 'miniclips':
        workDir = os.path.join(encoding_dir,'3D_ResNet18/')
    
    saveDir = os.path.join(workDir,'stats/')   
        
    feature_names = ('edges', 'world_normal', 'lightning',
                 'scene_depth', 'reflectance', 'action', 'skeleton')
        
    
    # set random seed (for reproduction)
    np.random.seed(42)
    
    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------   
    fileDir = os.path.join(workDir, 'encoding_layers_resnet.pkl')   

    encoding_results = np.load(fileDir, allow_pickle= True)
    
    regression_features = dict.fromkeys(feature_names)
    correlation_avg = dict.fromkeys(feature_names)  

    for feature in feature_names:
        regression_features[feature] = encoding_results[feature]['weighted_correlations']
        correlation_avg[feature] = encoding_results[feature]['correlation_average']
    features_results = {}
        
    # -------------------------------------------------------------------------
    # STEP 2.3 Bootstrapping
    # -------------------------------------------------------------------------

    pairwise_p = {}
    
    for f1 in range(len(feature_names)): 
        #peak for A
        feature_A = feature_names[f1]
        num_comparisons = f1 + 1
        corr_feature_A = correlation_avg[feature_A]
        peak_A = np.argmax(corr_feature_A)


        for f2 in range(num_comparisons): 
            feature_B = feature_names[f2]
            
            if feature_A == feature_B: 
                continue 
            
            #peak for B
            str_comparison = '{} vs. {}'.format(feature_A, feature_B)
            
            corr_feature_B = correlation_avg[feature_B]
            peak_B = np.argmax(corr_feature_B)
            feature_diff = peak_A - peak_B
            
            bt_data_peaks = np.zeros((n_perm,1))
            
            perm_sum_data_A = np.zeros((n_layers,n_perm))
            perm_sum_data_B = np.zeros((n_layers,n_perm))

            for l, layer in enumerate(layers_names):
                layer_data_A = regression_features[feature_A][layer]
                layer_data_B = regression_features[feature_B][layer]
                num_comp = layer_data_A.shape[0]

                for perm in range(n_perm): 
                    perm_peak_data_idx = np.random.choice(layer_data_A.shape[0], size = (num_comp, 1), 
                                                    replace = True)
                    
                    perm_peak_data_A = layer_data_A[perm_peak_data_idx]
                    perm_peak_data_B = layer_data_B[perm_peak_data_idx]

                    perm_sum_data_A[l,perm] = np.sum(perm_peak_data_A)/total_var
                    perm_sum_data_B[l,perm] = np.sum(perm_peak_data_B)/total_var


            peak_A_perm = np.argmax(perm_sum_data_A,axis=0)
            peak_B_perm = np.argmax(perm_sum_data_B,axis=0)
                
            feature_diff_bt = peak_A_perm - peak_B_perm


                    
            # -----------------------------------------------------------------
            # STEP 2.4 Plot histogram
            # -----------------------------------------------------------------
            if plot_hist == True:
                n_bins = int(math.sqrt(n_perm))
            

                plt.hist(bt_data_peaks, bins = n_bins)
                plt.show()
        
            # -----------------------------------------------------------------
            # STEP 2.5 Compute p-Value and CI
            # -----------------------------------------------------------------
            # CI 
            upper = int(np.ceil(n_perm * 0.975))
            lower = int(np.ceil(n_perm * 0.025))        

            feature_diff_bt_abs = np.abs(feature_diff_bt)
            feature_diff_bt_abs.sort()
            lower_CI = feature_diff_bt_abs[lower]
            upper_CI = feature_diff_bt_abs[upper]
            recenter = feature_diff_bt_abs - feature_diff

            sum_higher = np.sum(np.abs(recenter) > np.abs(feature_diff))
            p_value = (1+sum_higher)/(n_perm+1)
            
            # bonferroni
            # p_value = p_value * 28
            
            c_dict = {'p_value': p_value, 'ci': [lower_CI, feature_diff, upper_CI]}
            pairwise_p[str_comparison] = c_dict
    

    # -------------------------------------------------------------------------
    # STEP 2.6 Save CI
    # -------------------------------------------------------------------------  
    # Save the dictionary
    
    fileDir = 'encoding_layers_stats_peak_latency_diff.pkl' 
    
    # Benjamini-Hochberg Correction 
    p_values_vector = [pairwise_p[key]['p_value'] for key in pairwise_p]
    
    rejected, p_values_corr = statsmodels.stats.multitest.fdrcorrection(
    p_values_vector, alpha = 0.05, is_sorted = False)
    
    for i, key in enumerate(pairwise_p):
        pairwise_p[key]['p_value'] = p_values_corr[i]

    
    savefileDir = os.path.join(saveDir, fileDir) 
     
    # Creating the directory if not existing
    if os.path.isdir(os.path.join(saveDir)) == False: # if not a directory
        os.makedirs(os.path.join(saveDir))
    
    with open(savefileDir, 'wb') as f:
        pickle.dump(pairwise_p, f)

bootstrapping_CI(n_perm, n_layers, plot_hist, input_type, encoding_dir, total_var)
bootstrapping_CI_peak_layer(n_perm, plot_hist, input_type, encoding_dir, total_var)
bootstrapping_stats_diff_btw_features(n_perm, n_layers, plot_hist, input_type, encoding_dir, total_var)
    
