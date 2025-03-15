#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOOTSTRAPPING ENCODING LAYERS CLUSTER

This script calculates Bootstrap 95%-CIs for the encoding accuracy for each
layer and each feature. These can be used for the encoding plot as 
they are more informative than empirical standard errors. 

In addition, this script calculates Bootstrap 95%-CIs for the layer
of the largest encoding peak for each feature. 

@author: AlexanderLenders, AgnessaKarapetian
"""
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # add arguments / inputs
    parser.add_argument('-np', "--num_perm", default = 10000, type = int, 
                        metavar='', help="Number of permutations")
    parser.add_argument('-ed',"--encoding_dir", help='Directory with encoding results', 
                    default ='Z:/Unreal/Results/Encoding/CNN_redone/')
    parser.add_argument('-tv',"--total_var", help='Total variance explained by all PCA components together', 
                    default = 90)

    args = parser.parse_args() # to get values for the arguments
       
    n_perm = args.num_perm
    encoding_dir = args.encoding_dir
    total_var = args.total_var


def bootstrapping_CI(n_perm, encoding_dir, total_var): 
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
    ----------
    n_perm : int
        Number of permutations for bootstrapping
    encoding_dir : str
        Where encoding results are saved
    total_var : int
        Total variance explained by all PCA components

    """
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
    # Import modules
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pickle
    
    workDir_img = os.path.join(encoding_dir,'2D_ResNet18/')
    workDir_vid = os.path.join(encoding_dir,'3D_ResNet18/')
    
    saveDir = os.path.join(workDir_img,'stats/')   

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
    n_layers = len(layers_names)
    feature_names = ('edges', 'world_normal', 'lighting',
                 'scene_depth', 'reflectance', 'action', 'skeleton')
        
    # set random seed (for reproduction)
    np.random.seed(42)
    
    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------   
    fileDir_img = os.path.join(workDir_img, 'encoding_layers_resnet.pkl')   
    fileDir_vid = os.path.join(workDir_vid, 'encoding_layers_resnet.pkl')   

    encoding_results_img = np.load(fileDir_img, allow_pickle= True)
    encoding_results_vid = np.load(fileDir_vid, allow_pickle= True)

    regression_features_img = dict.fromkeys(feature_names)
    regression_features_vid = dict.fromkeys(feature_names)
    
    for feature in feature_names:
        regression_features_img[feature] = encoding_results_img[feature]['weighted_correlations']
        regression_features_vid[feature] = encoding_results_vid[feature]['weighted_correlations']

    features_results = {}
    
    for feature in feature_names:
        results_img = regression_features_img[feature]
        results_vid = regression_features_vid[feature]

        # ---------------------------------------------------------------------
        # STEP 2.3 Bootstrapping
        # ---------------------------------------------------------------------
        bt_data = np.zeros((n_layers, n_perm))
        
        for l,layer in enumerate(layers_names): 
            layer_data_img = results_img[layer]
            layer_data_vid = results_vid[layer]
            num_comp_layer_img = layer_data_img.shape[0]
            num_comp_layer_vid = layer_data_vid.shape[0]

            for perm in range(n_perm): 
                units_drawn_img = np.random.choice(range(num_comp_layer_img), size = num_comp_layer_img, replace = True)
                units_drawn_vid = np.random.choice(range(num_comp_layer_vid), size = num_comp_layer_vid, replace = True)
               
                perm_l_data_img = layer_data_img[units_drawn_img]
                perm_l_data_vid = layer_data_vid[units_drawn_vid]

                mean_p_layer_img = np.sum(perm_l_data_img)/total_var
                mean_p_layer_vid = np.sum(perm_l_data_vid)/total_var
                
                bt_data[l, perm] = mean_p_layer_img - mean_p_layer_vid
              
        # ---------------------------------------------------------------------
        # STEP 2.4 Calculate 95%-CI
        # ---------------------------------------------------------------------
        upper = int(np.ceil(n_perm * 0.975))
        lower = int(np.ceil(n_perm * 0.025))
        
        ci_dict = {}
        
        for l,layer in enumerate(layers_names): 
            l_data = bt_data[l, :]
            l_data.sort()
            ci_dict['{}'.format(layer)] = [l_data[lower], l_data[upper]]
        
        features_results[feature] = ci_dict
        
    # -------------------------------------------------------------------------
    # STEP 2.5 Save CI
    # -------------------------------------------------------------------------  
    # Save the dictionary
    
    fileDir = 'encoding_layers_CI95_accuracy_difference.pkl'
    
    savefileDir = os.path.join(saveDir, fileDir) 
    
    with open(savefileDir, 'wb') as f:
        pickle.dump(features_results, f)
        
#------------------------------------------------------------------------------
        
def bootstrapping_CI_peak_layer(n_perm, encoding_dir, total_var): 
    """
    Bootstrapped 95%-CIs for the layer of the largest encoding peak
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
    n_perm : int
        Number of permutations for bootstrapping
    encoding_dir : str
        Where encoding results are saved
    total_var : int
        Total variance explained by all PCA components
    """
    
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
    # Import modules
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pickle
    
    workDir_img = os.path.join(encoding_dir,'2D_ResNet18/')
    workDir_vid = os.path.join(encoding_dir,'3D_ResNet18/')
    
    saveDir = os.path.join(workDir_img,'stats/')   

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
    n_layers = len(layers_names)

    feature_names = ('edges', 'world_normal', 'lighting',
                 'scene_depth', 'reflectance', 'action', 'skeleton')
        
    # set random seed (for reproduction)
    np.random.seed(42)
            
    # -------------------------------------------------------------------------
    # STEP 2.2 Load results
    # -------------------------------------------------------------------------    
    fileDir_img = os.path.join(workDir_img, 'encoding_layers_resnet.pkl')   
    fileDir_vid = os.path.join(workDir_vid, 'encoding_layers_resnet.pkl')   

    encoding_results_img = np.load(fileDir_img, allow_pickle= True)
    encoding_results_vid = np.load(fileDir_vid, allow_pickle= True)
    
    corr_img = {}
    corr_vid = {}

    for feature in feature_names:
        corr_img[feature] = encoding_results_img[feature]['weighted_correlations']
        corr_vid[feature] = encoding_results_vid[feature]['weighted_correlations']

    ci_diff_peaks_all= {}

    for i, feature in enumerate(feature_names): 
        
        results_img = corr_img[feature]
        results_vid = corr_vid[feature]
    
        # ---------------------------------------------------------------------
        # STEP 2.3 Bootstrapping
        # ---------------------------------------------------------------------
        bt_diff_peaks = np.zeros((n_perm,))

        #get true data
        num_comp_layer_img = {}
        num_comp_layer_vid = {}
        for l,layer in enumerate(layers_names):
            layer_data_img = results_img[layer]
            layer_data_vid = results_vid[layer]
            num_comp_layer_img[layer] = layer_data_img.shape[0]
            num_comp_layer_vid[layer] = layer_data_vid.shape[0]

        #Find ground truth difference in peak latencies of vid. vs img
        peak_img_true = np.argmax(encoding_results_img[feature]['correlation_average'])
        peak_vid_true = np.argmax(encoding_results_vid[feature]['correlation_average'])
        diff_in_peak_true = np.abs(peak_img_true - peak_vid_true)

        #Permute and calculate peak latencies for bootstrap samples
        for perm in range(n_perm):
            perm_mean_vid = np.zeros((n_layers,))
            perm_mean_img = np.zeros((n_layers,))
            for l,layer in enumerate(layers_names):
                layer_data_img = results_img[layer]
                layer_data_vid = results_vid[layer]

                #permute units & get weighted sum across units
                units_drawn_vid = np.random.choice(range(num_comp_layer_vid[layer]), size = num_comp_layer_vid[layer], replace = True) 
                units_drawn_img = np.random.choice(range(num_comp_layer_img[layer]), size = num_comp_layer_img[layer], replace = True)
                
                perm_peak_data_vid = layer_data_vid[units_drawn_vid]
                perm_peak_data_img = layer_data_img[units_drawn_img]

                perm_mean_vid[l] = np.sum(perm_peak_data_vid)/total_var    
                perm_mean_img[l] = np.sum(perm_peak_data_img)/total_var


            # difference in the peaks
            peak_lat_vid = np.argmax(perm_mean_vid)
            peak_lat_img = np.argmax(perm_mean_img)
            diff_in_peakl = np.abs(peak_lat_img - peak_lat_vid)
            bt_diff_peaks[perm] = diff_in_peakl
        
        # ---------------------------------------------------------------------
        # STEP 2.4 Calculate 95%-CI
        # ---------------------------------------------------------------------
        upper = round(n_perm * 0.975)
        lower = round(n_perm * 0.025)

        bt_diff_peaks.sort()
        lower_ci = bt_diff_peaks[lower]
        upper_ci = bt_diff_peaks[upper]

        if lower_ci > upper_ci: #because of absolute difference calculated earlier
            upper_ci_final = lower_ci
            lower_ci_final = upper_ci
        else:
            upper_ci_final = upper_ci
            lower_ci_final = lower_ci
                
        ci_diff_peaks_all['{}'.format(feature)] = [lower_ci_final, diff_in_peak_true, upper_ci_final]
           
    # -------------------------------------------------------------------------
    # STEP 2.5 Save CI
    # -------------------------------------------------------------------------  
    # Save the dictionary
    
    fileDir = 'encoding_difference_in_peak.pkl'  

    savefileDir = os.path.join(saveDir, fileDir) 
        
    with open(savefileDir, 'wb') as f:
        pickle.dump(ci_diff_peaks_all, f)

bootstrapping_CI(n_perm, encoding_dir, total_var)
bootstrapping_CI_peak_layer(n_perm,encoding_dir,total_var)        

    
