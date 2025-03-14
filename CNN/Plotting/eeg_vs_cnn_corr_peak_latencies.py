#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORRELATION BETWEEN EEG AND DNN PEAK LATENCIES 

This script creates plots for the encoding analyses. 

@author: AgnessaKarapetian
"""

# -----------------------------------------------------------------------------
# STEP 1: Import modules & Define Variables
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Import modules 
    import numpy as np 
    import matplotlib.pyplot as plt
    import pickle
    import os
    from scipy import stats
    from scipy.stats import rankdata

    plt.rcParams['svg.fonttype'] = 'none'
    np.random.seed(40)

    #initialize variables
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

    alpha_value = 0.05    
    feature_names_sorted = ('edges', 'reflectance', 'lighting', 'world_normal',
                    'scene_depth', 'skeleton', 'action')

    features_dict = dict.fromkeys(feature_names_sorted)

    num_layers = len(layers_names)
    total_var = 90
    n_perm = 10000

    # Subjects 
    list_sub_images = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    list_sub_miniclips = [6, 7, 8, 9, 10, 11, 17, 18, 20, 21, 23, 25, 27, 28, 29, 30, 31, 32, 34, 36]

    # Paths 
    statsDir_eeg_img = 'Z:/Unreal/images_results/encoding/redone/stats/'   
    statsDir_eeg_mc = 'Z:/Unreal/Results/Encoding/redone/stats/'
    statsDir_dnn_img = 'Z:/Unreal/Results/Encoding/CNN_redone/2D_ResNet18/stats/'
    statsDir_dnn_mc = 'Z:/Unreal/Results/Encoding/CNN_redone/3D_ResNet18/stats/'
    saveDir = 'Z:/Unreal/Results/Encoding/plots_redone/'

    # Feature and filenames 
    identifierDir = 'seq_50hz_posterior_encoding_results_averaged_frame_before_mvnn_7features_onehot.pkl'

    # Filepaths for peak latencies 
    #1. EEG
    peaks_filename_eeg_img = 'encoding_unreal_before_pca_images_CI95_peak.pkl'
    peaks_filepath_eeg_img = os.path.join(statsDir_eeg_img, peaks_filename_eeg_img) 
    peaks_filename_eeg_mc = 'encoding_unreal_before_pca_miniclips_CI95_peak.pkl'
    peaks_filepath_eeg_mc = os.path.join(statsDir_eeg_mc, peaks_filename_eeg_mc) 

    #2. DNN
    peaks_filename_dnn_img = 'encoding_layers_CI95_peak.pkl'
    peaks_filepath_dnn_img = os.path.join(statsDir_dnn_img,peaks_filename_dnn_img)
    peaks_filename_dnn_mc = 'encoding_layers_CI95_peak.pkl'
    peaks_filepath_dnn_mc = os.path.join(statsDir_dnn_mc,peaks_filename_dnn_mc)

    # -----------------------------------------------------------------------------
    # STEP 2: Load EEG and DNN peak latencies
    # -----------------------------------------------------------------------------

    # Load peak latencies 
    #1. EEG
    with open(peaks_filepath_eeg_img, 'rb') as file:
        peaks_eeg_img = pickle.load(file)
        peaks_eeg_img = {key: peaks_eeg_img[key] for key in feature_names_sorted}
    peak_latencies_eeg_img = [item[1] for item in peaks_eeg_img.values()]

    with open(peaks_filepath_eeg_mc, 'rb') as file:
        peaks_eeg_mc = pickle.load(file)
        peaks_eeg_mc = {key: peaks_eeg_mc[key] for key in feature_names_sorted}
    peak_latencies_eeg_mc = [item[1] for item in peaks_eeg_mc.values()]

    #2. DNN
    with open(peaks_filepath_dnn_img, 'rb') as file:
        peaks_dnn_img = pickle.load(file)
        peaks_dnn_img = {key: peaks_dnn_img[key] for key in feature_names_sorted}
    peak_latencies_dnn_img = [item[1] for item in peaks_dnn_img.values()]

    with open(peaks_filepath_dnn_mc, 'rb') as file:
        peaks_dnn_mc = pickle.load(file)
        peaks_dnn_mc = {key: peaks_dnn_mc[key] for key in feature_names_sorted}
    peak_latencies_dnn_mc = [item[1] for item in peaks_dnn_mc.values()]

    #midlevel only
    peak_latencies_eeg_img_midlevel = peak_latencies_eeg_img[1:-1]
    peak_latencies_eeg_mc_midlevel = peak_latencies_eeg_mc[1:-1]
    peak_latencies_dnn_img_midlevel = peak_latencies_dnn_img[1:-1]
    peak_latencies_dnn_mc_midlevel = peak_latencies_dnn_mc[1:-1]

    # Load correlations for every unit 
    workDir_img_dnn = 'Z:/Unreal/Results/Encoding/CNN_redone/2D_ResNet18/'
    workDir_vid_dnn = 'Z:/Unreal/Results/Encoding/CNN_redone/3D_ResNet18/'
        
    fileDir_img_dnn = os.path.join(workDir_img_dnn, 'encoding_layers_resnet.pkl')   
    fileDir_vid_dnn = os.path.join(workDir_vid_dnn, 'encoding_layers_resnet.pkl')   

    encoding_results_img_dnn = np.load(fileDir_img_dnn, allow_pickle= True)
    encoding_results_vid_dnn = np.load(fileDir_vid_dnn, allow_pickle= True)

    # -----------------------------------------------------------------------------
    # STEP 3: Correlate EEG and DNN peak latencies
    # -----------------------------------------------------------------------------

    # Spearman's correlation
    correlation_results = {}
    corr_res_img = stats.spearmanr(peak_latencies_eeg_img,peak_latencies_dnn_img)
    correlation_results['images'] = {}
    correlation_results['images']['Spearmans r'] = corr_res_img.statistic
    correlation_results['images']['pvalue_scipy'] = corr_res_img.pvalue 

    corr_res_mc = stats.spearmanr(peak_latencies_eeg_mc,peak_latencies_dnn_mc)
    correlation_results['miniclips'] = {}
    correlation_results['miniclips']['Spearmans r'] = corr_res_mc.statistic
    correlation_results['miniclips']['pvalue_scipy'] = corr_res_mc.pvalue 

    #Spearman's correlation - midlevel features ONLY
    correlation_results_midlevel = {}
    corr_res_imgs_midlevel = stats.spearmanr(peak_latencies_eeg_img_midlevel,peak_latencies_dnn_img_midlevel)
    correlation_results_midlevel['images'] = {}
    correlation_results_midlevel['images']['Spearmans r'] = corr_res_imgs_midlevel.statistic
    correlation_results_midlevel['images']['pvalue_scipy'] = corr_res_imgs_midlevel.pvalue

    corr_res_mc_midlevel = stats.spearmanr(peak_latencies_eeg_mc_midlevel,peak_latencies_dnn_mc_midlevel)
    correlation_results_midlevel['miniclips'] = {}
    correlation_results_midlevel['miniclips']['Spearmans r'] = corr_res_mc_midlevel.statistic
    correlation_results_midlevel['miniclips']['pvalue_scipy'] = corr_res_mc_midlevel.pvalue

    #Stats
    #method 2: permute the DNN unit-correlations, calculate perm DNN peak latencies and correlate with true EEG peak latencies
    feat_peak_og_img_dnn = {}   
    feat_peak_og_vid_dnn = {}   

    peak_perm_img_dnn = {}
    peak_perm_vid_dnn = {}

    for feature in feature_names_sorted: 
        #get original peaks
        mean_orig_img_dnn = np.zeros((num_layers,1))
        mean_orig_vid_dnn = np.zeros((num_layers,1))
        num_comp_layer_img_dnn = {}
        num_comp_layer_vid_dnn = {}
        corr_img_dnn = {}
        corr_vid_dnn = {}
        peak_perm_img_dnn[feature] = {}
        peak_perm_vid_dnn[feature] = {}

        for l,layer in enumerate(layers_names):
            corr_img_dnn[layer] = encoding_results_img_dnn[feature]['weighted_correlations'][layer]
            corr_vid_dnn[layer] = encoding_results_vid_dnn[feature]['weighted_correlations'][layer]
            
            num_comp_layer_img_dnn[layer] = corr_img_dnn[layer].shape[0]
            num_comp_layer_vid_dnn[layer] = corr_vid_dnn[layer].shape[0]

            # create mean for each layer over all images
            mean_orig_img_dnn[l] = np.sum(corr_img_dnn[layer])/total_var
            mean_orig_vid_dnn[l] = np.sum(corr_vid_dnn[layer])/total_var
        
        #get peak
        feat_peak_og_img_dnn[feature] = np.argmax(mean_orig_img_dnn)
        feat_peak_og_vid_dnn[feature] = np.argmax(mean_orig_vid_dnn)

        peak_perm_img_dnn[feature][0] = {}
        peak_perm_vid_dnn[feature][0] = {}
        peak_perm_img_dnn[feature][0] = feat_peak_og_img_dnn[feature]
        peak_perm_vid_dnn[feature][0] = feat_peak_og_vid_dnn[feature]

        #create permutation samples & get their peak latencies
        for permutation in range(1, n_perm): 
            mean_perm_img_dnn = np.zeros((num_layers,))
            mean_perm_vid_dnn = np.zeros((num_layers,))

            for l, layer in enumerate(layers_names):

                perm_img_dnn = np.expand_dims(np.random.choice([-1, 1], size = (num_comp_layer_img_dnn[layer],), 
                                    replace = True), 1)
                perm_img_dnn = perm_img_dnn.T
                perm_corr_img_dnn = corr_img_dnn[layer]*perm_img_dnn
                
                perm_vid_dnn = np.expand_dims(np.random.choice([-1, 1], size = (num_comp_layer_vid_dnn[layer],), 
                    replace = True), 1)
                perm_vid_dnn = perm_vid_dnn.T
                perm_corr_vid_dnn = corr_vid_dnn[layer]*perm_vid_dnn

                # create mean for each layer over all images for every permutation
                mean_perm_img_dnn[l] = np.sum(perm_corr_img_dnn)/total_var
                mean_perm_vid_dnn[l] = np.sum(perm_corr_vid_dnn)/total_var

            peak_perm_img_dnn[feature][permutation] = {}
            peak_perm_vid_dnn[feature][permutation] = {}
            peak_perm_img_dnn[feature][permutation] = np.argmax(mean_perm_img_dnn)
            peak_perm_vid_dnn[feature][permutation] = np.argmax(mean_perm_vid_dnn)

    #reorganize perm peak latencies
    peak_latencies_dnn_img_perm = {}
    peak_latencies_dnn_vid_perm = {}
    for permutation in range(1,n_perm):
        peaks_dnn_img_perm = {key: peak_perm_img_dnn[key][permutation] for key in feature_names_sorted}
        peak_latencies_dnn_img_perm[permutation] = {}
        peak_latencies_dnn_img_perm[permutation] = [item for item in peaks_dnn_img_perm.values()]

        peaks_dnn_vid_perm = {key: peak_perm_vid_dnn[key][permutation] for key in feature_names_sorted}
        peak_latencies_dnn_vid_perm[permutation] = {}
        peak_latencies_dnn_vid_perm[permutation] = [item for item in peaks_dnn_vid_perm.values()]

    #true correlations
    correlation_true_img = corr_res_img.statistic
    correlation_true_vid = corr_res_mc.statistic
    peak_latencies_eeg_img_true = peak_latencies_eeg_img 
    peak_latencies_eeg_vid_true = peak_latencies_eeg_mc
    peak_latencies_dnn_img_true = peak_latencies_dnn_img
    peak_latencies_dnn_vid_true = peak_latencies_dnn_mc

    #calculate the correlation between permuted dnn peak latencies and true eeg peak latencies        
    stat_map_img = np.zeros((n_perm,))
    stat_map_vid = np.zeros((n_perm,))
    stat_map_img[0] = correlation_true_img
    stat_map_vid[0] = correlation_true_vid

    for permutation in range(1, n_perm):
        stat_map_img[permutation] = stats.spearmanr(peak_latencies_dnn_img_perm[permutation],peak_latencies_eeg_img_true)[0]
        stat_map_vid[permutation] = stats.spearmanr(peak_latencies_dnn_vid_perm[permutation],peak_latencies_eeg_vid_true)[0]

    #get ranks
    ranks_img = (np.apply_along_axis(rankdata, 0, stat_map_img))
    ranks_vid = (np.apply_along_axis(rankdata, 0, stat_map_vid))
            
    # 3. calculate p-values 
    sub_matrix = np.full((n_perm,), (n_perm+1))
    p_map_img = (sub_matrix - ranks_img)/n_perm
    p_value_img = p_map_img[0]
    correlation_results['images']['pvalue_calculated'] = p_value_img
    correlation_results['images']['significance_calculated'] = p_value_img < 0.05

    p_map_vid = (sub_matrix - ranks_vid)/n_perm
    p_value_vid = p_map_vid[0]
    correlation_results['miniclips']['pvalue_calculated'] = p_value_vid
    correlation_results['miniclips']['significance_calculated'] = p_value_vid < 0.05

    #stats for MIDLEVEL features
    feat_peak_og_img_dnn_midlevel = {}   
    feat_peak_og_vid_dnn_midlevel = {}   

    peak_perm_img_dnn_midlevel = {}
    peak_perm_vid_dnn_midlevel = {}

    for feature in feature_names_sorted[1:-1]: 
        #get original peaks
        mean_orig_img_dnn_midlevel = np.zeros((num_layers,1))
        mean_orig_vid_dnn_midlevel = np.zeros((num_layers,1))
        num_comp_layer_img_dnn_midlevel = {}
        num_comp_layer_vid_dnn_midlevel = {}
        corr_img_dnn_midlevel = {}
        corr_vid_dnn_midlevel = {}
        peak_perm_img_dnn_midlevel[feature] = {}
        peak_perm_vid_dnn_midlevel[feature] = {}

        for l,layer in enumerate(layers_names):
            corr_img_dnn_midlevel[layer] = encoding_results_img_dnn[feature]['weighted_correlations'][layer]
            corr_vid_dnn_midlevel[layer] = encoding_results_vid_dnn[feature]['weighted_correlations'][layer]
            
            num_comp_layer_img_dnn_midlevel[layer] = corr_img_dnn_midlevel[layer].shape[0]
            num_comp_layer_vid_dnn_midlevel[layer] = corr_vid_dnn_midlevel[layer].shape[0]

            # create mean for each layer over all images
            mean_orig_img_dnn_midlevel[l] = np.sum(corr_img_dnn_midlevel[layer])/total_var
            mean_orig_vid_dnn_midlevel[l] = np.sum(corr_vid_dnn_midlevel[layer])/total_var
        
        #get peak
        feat_peak_og_img_dnn_midlevel[feature] = np.argmax(mean_orig_img_dnn_midlevel)
        feat_peak_og_vid_dnn_midlevel[feature] = np.argmax(mean_orig_vid_dnn_midlevel)

        peak_perm_img_dnn_midlevel[feature][0] = {}
        peak_perm_vid_dnn_midlevel[feature][0] = {}
        peak_perm_img_dnn_midlevel[feature][0] = feat_peak_og_img_dnn_midlevel[feature]
        peak_perm_vid_dnn_midlevel[feature][0] = feat_peak_og_vid_dnn_midlevel[feature]

        #create permutation samples & get their peak latencies
        for permutation in range(1, n_perm): 
            mean_perm_img_dnn_midlevel = np.zeros((num_layers,))
            mean_perm_vid_dnn_midlevel = np.zeros((num_layers,))

            for l, layer in enumerate(layers_names):

                perm_img_dnn_midlevel = np.expand_dims(np.random.choice([-1, 1], size = (num_comp_layer_img_dnn_midlevel[layer],), 
                                    replace = True), 1)
                perm_img_dnn_midlevel = perm_img_dnn_midlevel.T
                perm_corr_img_dnn_midlevel = corr_img_dnn_midlevel[layer]*perm_img_dnn_midlevel
                
                perm_vid_dnn_midlevel = np.expand_dims(np.random.choice([-1, 1], size = (num_comp_layer_vid_dnn_midlevel[layer],), 
                    replace = True), 1)
                perm_vid_dnn_midlevel = perm_vid_dnn_midlevel.T
                perm_corr_vid_dnn_midlevel = corr_vid_dnn_midlevel[layer]*perm_vid_dnn_midlevel

                # create mean for each layer over all images for every permutation
                mean_perm_img_dnn_midlevel[l] = np.sum(perm_corr_img_dnn_midlevel)/total_var
                mean_perm_vid_dnn_midlevel[l] = np.sum(perm_corr_vid_dnn_midlevel)/total_var

            peak_perm_img_dnn_midlevel[feature][permutation] = {}
            peak_perm_vid_dnn_midlevel[feature][permutation] = {}
            peak_perm_img_dnn_midlevel[feature][permutation] = np.argmax(mean_perm_img_dnn_midlevel)
            peak_perm_vid_dnn_midlevel[feature][permutation] = np.argmax(mean_perm_vid_dnn_midlevel)

    #reorganize perm peak latencies
    peak_latencies_dnn_img_perm_midlevel = {}
    peak_latencies_dnn_vid_perm_midlevel = {}
    for permutation in range(1,n_perm):
        peaks_dnn_img_perm_midlevel = {key: peak_perm_img_dnn_midlevel[key][permutation] for key in feature_names_sorted[1:-1]}
        peak_latencies_dnn_img_perm_midlevel[permutation] = {}
        peak_latencies_dnn_img_perm_midlevel[permutation] = [item for item in peaks_dnn_img_perm_midlevel.values()]

        peaks_dnn_vid_perm_midlevel = {key: peak_perm_vid_dnn_midlevel[key][permutation] for key in feature_names_sorted[1:-1]}
        peak_latencies_dnn_vid_perm_midlevel[permutation] = {}
        peak_latencies_dnn_vid_perm_midlevel[permutation] = [item for item in peaks_dnn_vid_perm_midlevel.values()]

    #true correlations
    correlation_true_img_midlevel = corr_res_imgs_midlevel.statistic
    correlation_true_vid_midlevel = corr_res_mc_midlevel.statistic
    peak_latencies_eeg_img_true_midlevel = peak_latencies_eeg_img_midlevel 
    peak_latencies_eeg_vid_true_midlevel = peak_latencies_eeg_mc_midlevel
    peak_latencies_dnn_img_true_midlevel = peak_latencies_dnn_img_midlevel
    peak_latencies_dnn_vid_true_midlevel = peak_latencies_dnn_mc_midlevel

    #calculate the correlation between permuted dnn peak latencies and true eeg peak latencies        
    stat_map_img_midlevel = np.zeros((n_perm,))
    stat_map_vid_midlevel = np.zeros((n_perm,))
    stat_map_img_midlevel[0] = correlation_true_img_midlevel
    stat_map_vid_midlevel[0] = correlation_true_vid_midlevel

    for permutation in range(1, n_perm):
        stat_map_img_midlevel[permutation] = stats.spearmanr(peak_latencies_dnn_img_perm_midlevel[permutation],peak_latencies_eeg_img_true_midlevel)[0]
        stat_map_vid_midlevel[permutation] = stats.spearmanr(peak_latencies_dnn_vid_perm_midlevel[permutation],peak_latencies_eeg_vid_true_midlevel)[0]

    #remove any nans (there might be a few because the chance of getting the same peak layers for all features is non zero)
    stat_map_img_midlevel_final = stat_map_img_midlevel[~np.isnan(stat_map_img_midlevel)]
    stat_map_vid_midlevel_final  = stat_map_vid_midlevel[~np.isnan(stat_map_vid_midlevel)]

    #get ranks
    ranks_img_midlevel = (np.apply_along_axis(rankdata, 0, stat_map_img_midlevel_final))
    ranks_vid_midlevel = (np.apply_along_axis(rankdata, 0, stat_map_vid_midlevel_final))
            
    # 3. calculate p-values 
    new_num_perm_img = stat_map_img_midlevel_final.shape[0]
    sub_matrix_img = np.full((new_num_perm_img), (new_num_perm_img+1))
    p_map_img_midlevel = (sub_matrix_img - ranks_img_midlevel)/new_num_perm_img
    p_value_img_midlevel = p_map_img_midlevel[0]
    correlation_results_midlevel['images']['pvalue_calculated'] = p_value_img_midlevel
    correlation_results_midlevel['images']['significance_calculated'] = p_value_img_midlevel < 0.05

    new_num_perm_vid = stat_map_vid_midlevel_final.shape[0]
    sub_matrix_vid = np.full((new_num_perm_vid), (new_num_perm_vid+1))
    p_map_vid_midlevel = (sub_matrix_vid - ranks_vid_midlevel)/n_perm
    p_value_vid_midlevel = p_map_vid_midlevel[0]
    correlation_results_midlevel['miniclips']['pvalue_calculated'] = p_value_vid_midlevel
    correlation_results_midlevel['miniclips']['significance_calculated'] = p_value_vid_midlevel < 0.05

    #save
    with open(os.path.join(saveDir,f'results_corr_eeg_dnn_images_miniclips_method2.pkl'), 'wb') as f:
        pickle.dump(correlation_results, f)

    with open(os.path.join(saveDir,f'results_corr_eeg_dnn_images_miniclips_midlevel.pkl'), 'wb') as f:
        pickle.dump(correlation_results_midlevel, f)

    # -----------------------------------------------------------------------------
    # STEP 4: Plot EEG vs DNN peak latencies
    # -----------------------------------------------------------------------------

    # Plot 
    num_features = len(peak_latencies_eeg_img)
    colormap = plt.colormaps['Set2']
    colors = [colormap(i) for i in range(num_features)] 
    fig,ax = plt.subplots(2)
    sorted_indices = [0, 4, 3, 1, 2, 5, 6] #hardcoded - based on results from images 
    sorted_color_dict = [colors[i] for i in sorted_indices]

    peak_latencies_dnn_img_forplot = [p+1 for p in peak_latencies_dnn_img]
    peak_latencies_dnn_mc_forplot = [p+1 for p in peak_latencies_dnn_mc]
    for i,(x,y) in enumerate(zip(peak_latencies_eeg_img,peak_latencies_dnn_img_forplot)): #+1 so the first layer does not equal 0
        eeg_vs_dnn_img = ax[0].scatter(x,y,color=sorted_color_dict[i])

    for i,(x,y) in enumerate(zip(peak_latencies_eeg_mc,peak_latencies_dnn_mc_forplot)):
        eeg_vs_dnn_img = ax[1].scatter(x,y,color=sorted_color_dict[i])

    # Ticks and labels
    ax[0].set_xlabel('EEG peak latency (ms)',font='Arial',fontsize=14)
    ax[0].set_ylabel('CNN peak latency (layer)',font='Arial',fontsize=14)

    ax[1].set_xlabel('EEG peak latency (ms)',font='Arial',fontsize=14)
    ax[1].set_ylabel('CNN peak latency (layer)',font='Arial',fontsize=14)

    ax[0].set_xticks(ticks=np.arange(0,550,50),labels=np.arange(0,550,50),font='Arial',fontsize=11)
    ax[0].set_yticks(ticks=[0,1,2,3,4,5,6,7,8],labels=['', '1.0', '1.1', '2.0', '2.1', '3.0', '3.1', '4.0', '4.1'],font='Arial',fontsize=11)

    ax[1].set_xticks(ticks=np.arange(0,550,50),labels=np.arange(0,550,50),font='Arial',fontsize=11)
    ax[1].set_yticks(ticks=[0,1,2,3,4,5,6,7,8],labels=['', '1.0', '1.1', '2.0', '2.1', '3.0', '3.1', '4.0', '4.1'],font='Arial',fontsize=11)

    # Add correlation 
    for e,i in enumerate(['images','miniclips']):
        corr_r = correlation_results[i]['Spearmans r']
        corr_p = correlation_results[i]['pvalue_calculated']
        if correlation_results[i]['significance_calculated'] == True:
            ax[e].text(180,0.25, f"Spearman's $p$ = {np.round(corr_r,2)}* (p = {corr_p})",font='Arial',fontsize=14)
        else:
            ax[e].text(180,0.25, f"Spearman's $p$ = {np.round(corr_r,2)} (p = {corr_p})",font='Arial',fontsize=14)

    # Save plot as png and svg
    plt.tight_layout
    plt.subplots_adjust(hspace=0.7)
    plotDir = os.path.join(saveDir, f'plot_eeg_vs_dnn_images_miniclips_noregression.svg')
    plt.savefig(plotDir, dpi = 300, format = 'svg', transparent = True)

    plotDir = os.path.join(saveDir, f'plot_eeg_vs_dnn_images_miniclips_noregression.png')
    plt.savefig(plotDir, dpi = 300, format = 'png', transparent = True)

 

