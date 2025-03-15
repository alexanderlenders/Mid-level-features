#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STATS ENCODING LAYERS FROM RESNET3D

This script implements the statistical tests for the encoding analyis predicting
the unit activity within the action DNN layers based on the single gaming-engine
features. 

It conducts permutation tests by permuting the PCA components for every layer, 
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
    parser.add_argument('-a', "--alpha", default = 0.05, type = int, 
                        metavar='', help="Significance level (alpha)")
    parser.add_argument('-t', "--tail", 
                        default = 'both',
                        type = str, metavar='', 
                        help="One-sided: right, two-sided: both")
    parser.add_argument('-i', "--input_type", default = 'miniclips', type = str, 
                        metavar='', help="Font")
    parser.add_argument('-ed',"--encoding_dir", help='Directory with encoding results', 
                        default ='Z:/Unreal/Results/Encoding/CNN_redone/')
    parser.add_argument('-tv',"--total_var", help='Total variance explained by all PCA components together', 
                default = 90)

    args = parser.parse_args() # to get values for the arguments

    n_perm = args.num_perm
    alpha_value = args.alpha
    tail = args.tail
    input_type = args.input_type
    encoding_dir = args.encoding_dir
    total_var = args.total_var

    # -----------------------------------------------------------------------------
    # STEP 2: Define Encoding Function
    # -----------------------------------------------------------------------------

    def encoding_stats(n_perm, alpha_value, tail, input_type, encoding_dir, total_var): 
        
        """
        Input: 
        ----------
        Output from the encoding analysis (multivariate linear regression), i.e.: 
        Encoding results (multivariate linear regression), saved in a dictionary 
        which contains for every feature correlation measure, i.e.: 
            encoding_results[feature]['correlation']
        
        Returns:
        ----------
        regression_features, dictionary with the following keys for each feature: 
        a. Uncorrected_p_values_map 
            - Contains uncorrected p-values
        b. Corrected_p_values_map
            - Contains corrected p-values
        c. Boolean_statistical_map
            - Contains boolean values, if True -> corrected p-value is lower than .05
        
        Parameters
        ----------
        n_perm : int
            Number of permutations for bootstrapping
        encoding_dir : str
            Where encoding results are saved
        total_var : int
            Total variance explained by all PCA components
        alpha_value : int
            Significance level
        tail : str
            One-sided or two-sided test
        input_type : str
            Images or miniclips
        
        """
        # -------------------------------------------------------------------------
        # STEP 2.1 Import Modules & Define Variables
        # -------------------------------------------------------------------------
        # Import modules
        import os
        import numpy as np
        import pickle
        from scipy.stats import rankdata
        import statsmodels
        
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
            
        feature_names = ('edges', 'world_normal', 'lighting',
                    'scene_depth', 'reflectance', 'action', 'skeleton')
        
        features_dict = dict.fromkeys(feature_names)
        
        num_layers = len(layers_names)    

        if input_type == 'images':
            workDir = os.path.join(encoding_dir,'2D_ResNet18/')
        elif input_type == 'miniclips':
            workDir = os.path.join(encoding_dir,'3D_ResNet18/')
        np.random.seed(42)
            
        # -------------------------------------------------------------------------
        # STEP 2.2 Load results
        # -------------------------------------------------------------------------
        fileDir = os.path.join(workDir, 'encoding_layers_resnet.pkl')   

        encoding_results = np.load(fileDir, allow_pickle=True)

        # define matrix where to save the values 
        regression_features = dict.fromkeys(feature_names)
        
        for feature in features_dict.keys(): 
            print(feature)
            stat_map = {}
            # create statistical map
            stat_map = np.zeros((n_perm, num_layers))

            # corr = encoding_results[feature]['correlation']
                    
            for l, corr_layer in enumerate(encoding_results[feature]['weighted_correlations'].values()):
                print(l)
                print(corr_layer.shape)
                num_comp_layer = corr_layer.shape[0]

                # create mean for each timepoint over all participants 
                # this is our "original data" and permutation 1 in the stat_map 
                # mean_orig = np.mean(corr_layer, axis = 0)
                mean_orig = np.sum(corr_layer)/total_var
                stat_map[0,l] = mean_orig 
                
                for permutation in range(1, n_perm): 
                    perm = np.expand_dims(np.random.choice([-1, 1], size = (num_comp_layer,), 
                                        replace = True), 1)
                    perm = perm.T
                    
                    perm_corr_layer = corr_layer*perm
                        
                    # create mean for each timepoint over all participants 
                    # this is our "original data" and permutation 1 in the stat_map 
                    # mean_orig = np.mean(perm_corr_layer, axis = 1)
                    mean_perm = np.sum(perm_corr_layer)/total_var
                    stat_map[permutation, l] = mean_perm 
                    
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
        fileDir = 'encoding_stats_layers_{}.pkl'.format(tail) 
        saveDir = os.path.join(workDir,'stats/')
        savefileDir = os.path.join(saveDir, fileDir) 
        
        with open(savefileDir, 'wb') as f:
            pickle.dump(regression_features, f)
        
    # -----------------------------------------------------------------------------
    # STEP 3 Run function
    # -----------------------------------------------------------------------------
    encoding_stats(n_perm, alpha_value, tail, input_type, encoding_dir, total_var)
