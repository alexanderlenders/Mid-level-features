#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT FOR ENCODING OF DNN LAYERS (SINGLE FEATURES)

@author: AlexanderLenders, AgnessaKarapetian
"""
# -----------------------------------------------------------------------------
# STEP 1: Initialize variables
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()

    # add arguments / inputs
    parser.add_argument('-sp', "--save", default = True, type = bool, 
                        metavar='', help="Save plots in SaveDir?")
    parser.add_argument('-f', "--font", default = 'Arial', type = str, 
                        metavar='', help="Font")
    parser.add_argument('-i', "--input_type", default = 'miniclips', type = str, 
                        metavar='', help="images, miniclips or difference")

    
    args = parser.parse_args() # to get values for the arguments
      
    save = args.save
    font = args.font
    input_type = args.input_type #images, miniclips or difference

# -----------------------------------------------------------------------------
# STEP 2: Import modules & Define Variables
# -----------------------------------------------------------------------------
# Import modules
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import os
plt.rcParams['svg.fonttype'] = 'none'

if input_type == 'miniclips':
    workDir = 'Z:/Unreal/Results/Encoding/CNN_redone/3D_ResNet18/'
elif input_type == 'images':
    workDir = 'Z:/Unreal/Results/Encoding/CNN_redone/2D_ResNet18/'

elif input_type == 'difference':
    workDir_vid = 'Z:/Unreal/Results/Encoding/CNN_redone/3D_ResNet18/'
    workDir_img = 'Z:/Unreal/Results/Encoding/CNN_redone/2D_ResNet18/'

saveDir = 'Z:/Unreal/Results/Encoding/plots_redone/' #for all plots

if input_type == 'difference':
    stats_dir = os.path.join(workDir_img,'stats/','encoding_stats_layers_both_difference.pkl') 
    ci_dir = os.path.join(workDir_img,'stats/','encoding_layers_CI95_accuracy_difference.pkl')
    peakDir = os.path.join(workDir_img,'stats/','encoding_difference_in_peak.pkl')
else:
    stats_dir = os.path.join(workDir,'stats/','encoding_stats_layers_both.pkl') 
    ci_dir = os.path.join(workDir,'stats/','encoding_layers_CI95_accuracy.pkl')
    peakDir = os.path.join(workDir,'stats/','encoding_layers_CI95_peak.pkl')
    ci_stats_peaks = os.path.join(workDir,'stats/', 'encoding_layers_stats_peak_latency_diff.pkl')


feature_names = ('edges','world_normal', 'scene_depth',
                     'lightning', 'reflectance', 'skeleton','action')
    
    
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

num_layers = len(layers_names)

# Names in the plot
feature_names_graph = ('Edges','Normals', 'Depth',
                           'Lighting', 'Reflectance',
                           'Skeleton','Action')    
num_features = len(feature_names)

colormap = plt.colormaps['Set2']
colors = [colormap(i) for i in range(num_features)]

# -----------------------------------------------------------------------------
# STEP 3: Load results
# -----------------------------------------------------------------------------
if input_type == 'difference':
    fileDir_img = os.path.join(workDir_img, 'encoding_layers_resnet.pkl')   
    fileDir_vid = os.path.join(workDir_vid, 'encoding_layers_resnet.pkl') 

    encoding_results_img = np.load(fileDir_img, allow_pickle= True) 
    encoding_results_vid = np.load(fileDir_vid, allow_pickle= True)
else:
    fileDir = os.path.join(workDir, 'encoding_layers_resnet.pkl')   
    encoding_results = np.load(fileDir, allow_pickle= True)
        
# Create mean of all subjects for each feature
features_mean = []
for feature in feature_names: 
    if input_type == 'difference':
        f_img = encoding_results_img[feature]['correlation_average']
        f_vid = encoding_results_vid[feature]['correlation_average']
        f_diff = f_img - f_vid
        features_mean.append(f_diff)

    else:
        f = encoding_results[feature]['correlation_average']
        features_mean.append(f)

# -----------------------------------------------------------------------------
# STEP 4: Create encoding curves plot 
# -----------------------------------------------------------------------------
# load stats results 
with open(stats_dir, 'rb') as file:
    encoding_stats = pickle.load(file)
    
# load CIs
with open(ci_dir, 'rb') as file:
    confidence_95 = pickle.load(file)

### Create the plot ###
plt.close()
fig, ax = plt.subplots(1,2, figsize=(8,4.3))
# plt.figure(figsize=(7, 6))
ax[0].axhline(y=0, color = 'darkgrey', linestyle = 'solid', linewidth = 1)

layers = np.arange(num_layers)
x_tick_values = [0, 1, 2, 3, 4, 5, 6, 7]
x_tick_labels = ['1.0', '1.1', '2.0', '2.1', '3.0', '3.1', '4.0', '4.1']

# load stats results
with open(peakDir, 'rb') as file:
    peaks = pickle.load(file)
peaks_filt = {key: peaks[key] for key in feature_names}
accuracies = [item[1] for item in peaks_filt.values()]
lower_ci = [item[0] for item in peaks_filt.values()]
upper_ci = [item[2] for item in peaks_filt.values()]

# Sort the accuracies and get the sorted indices
# sorted_indices = sorted(range(len(accuracies)), key=lambda k: accuracies[k])
sorted_indices = [0, 4, 3, 1, 2, 5, 6] #hardcoded - based on results from images EEG 
sorted_feature_names_graph = [feature_names_graph[i] for i in sorted_indices]
sorted_feature_names = [feature_names[i] for i in sorted_indices]
sorted_color_dict = [colors[i] for i in sorted_indices]
sorted_features_mean = [features_mean[i] for i in sorted_indices]

for i, feature in enumerate(sorted_features_mean):
    # Get feature data 
    f_name = sorted_feature_names[i]
    accuracy = feature
    ci_feature = confidence_95[f_name]
    
    # Extract the CI 
    low_CI = np.array([ci_feature[key][0] for key in (ci_feature)])
    low_ins = low_CI - accuracy
    low_ins = np.absolute(low_ins)
    high_CI = np.array([ci_feature[key][1] for key in (ci_feature)])
    high_ins =high_CI - accuracy

    # Plot the accuracy curve for the current featureolors
    ax[0].plot(layers, accuracy, marker='o', markersize = 10, markerfacecolor=sorted_color_dict[i],
               markeredgecolor='black', color=sorted_color_dict[i],linewidth = 2, label=sorted_feature_names_graph[i])
    
    # Plot the accuracy curve for the current feature
    ax[0].errorbar(layers, accuracy, yerr=[low_ins, high_ins], capsize = 5, capthick = 2, marker='o', markersize = 10, 
                   markerfacecolor=sorted_color_dict[i], markeredgecolor='black', color=sorted_color_dict[i], linewidth = 2)
     
    # Extract the stats 
    stats_results = encoding_stats[f_name]['Boolean_statistical_map']
    if input_type != 'difference':
        starting_value = -0.10
        plot_value = starting_value + (i/70)
    else:
        starting_value = -0.18
        plot_value = starting_value + (i/100)
    test = np.full((num_layers,1), np.nan, dtype=float)
    
    # Get the indices of the significant time points
    significant_indices = np.where(stats_results)[0]
    
    # Get the significant time points and corresponding values
    test[significant_indices] = plot_value
    plot_accuracies = test
   
    ax[0].plot(layers, plot_accuracies, '*', color=sorted_color_dict[i], markersize=4)


ax[0].set_xlabel('ResNet-18 layer', fontdict={'family': font, 'size': 11})
ax[0].set_ylabel("Pearson's r", fontdict={'family': font, 'size': 11})

if input_type == 'difference':
    yticks = np.round(np.arange(-0.15,0.25,0.05),2)
else:
    yticks = np.round(np.arange(-0.1,0.55,0.05),2)

ax[0].set_yticks(ticks = yticks, labels = yticks, fontsize=9, fontname=font)
ax[0].set_xticks(ticks= x_tick_values, labels=x_tick_labels, fontsize=9, fontname=font)

if input_type == 'images':
    legend_font_props = {'family': font, 'size': 9}
    ax[0].legend(prop=legend_font_props, frameon=False,bbox_to_anchor=[0.6,0.6])


ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['bottom'].set_linewidth(2)  # Set x-axis spine linewidth
ax[0].spines['left'].set_linewidth(2)    # Set y-axis spine linewidth

ax[0].tick_params(axis='both', direction='inout')
ax[0].tick_params(axis='x', which='both', length=6, width=3)  # Adjust the length and width as needed
ax[0].tick_params(axis='y', which='both', length=6, width=3)  # Adjust the length and width as needed

# Displaying the plot
# plt.show()

# -----------------------------------------------------------------------------
# STEP 5: Peak latencies for single features
# -----------------------------------------------------------------------------
y_range = max(upper_ci) - min(lower_ci)
top = max(upper_ci)

#pairwise significance lines
if input_type != 'difference':
    with open(ci_stats_peaks, 'rb') as file:
        peak_stats = pickle.load(file)
    
    significant_combinations = []

    for feature, values in peak_stats.items():
        ci = values['ci']
        ci_lower = ci[0]
        ci_upper = ci[2]
        if ci_lower != 0 and ci_upper != 0: 
            significant_combinations.append((feature, ci))



    sig_combi_numbers = []
    for s in significant_combinations:
        split_name = s[0].split(" vs. ")
        idx_0 = sorted_feature_names.index(split_name[0])
        idx_1 = sorted_feature_names.index(split_name[1])       
        sig_combi_numbers.append(sorted([idx_0, idx_1]))
    
    sorted_sig_combi_numbers = sorted(sig_combi_numbers)

    #sig line params
    line_offset = 8
    line_height = 0.25 # Height of each significance line
    
    for i, sig_combi in enumerate(sorted_sig_combi_numbers):
        x1, x2 = sig_combi
        level = len(sorted_sig_combi_numbers) - i + 1
        bar_height = level * line_height + line_offset
        ax[1].plot([x1, x2], [bar_height, bar_height], lw=1, c='black', ls='solid')


# Rearrange the accuracies, lower CIs, and upper CIs based on the sorted indices
accuracies = [accuracies[i] for i in sorted_indices]
lower_ci = [lower_ci[i] for i in sorted_indices]
upper_ci = [upper_ci[i] for i in sorted_indices]

#STARTING PLOT FROM 0 - DO THIS
if input_type != 'difference':
    accuracies = [acc+1 for acc in accuracies] 
    lower_ci = [lci + 1 for lci in lower_ci]
    upper_ci = [uci +1 for uci in upper_ci]

    
x_pos = np.arange(len(feature_names))
bars = ax[1].bar(x_pos, accuracies, align='center', alpha=1, color = sorted_color_dict)

# Calculating the error bars as the differences between lower CI and accuracies
y_err_lower = [acc - lower for acc, lower in zip(accuracies, lower_ci)]

# Calculating the error bars as the differences between upper CI and accuracies
y_err_upper = [upper - acc for acc, upper in zip(accuracies, upper_ci)]

# Adding the CI error bars
ax[1].errorbar(x_pos, accuracies, yerr=[y_err_lower, y_err_upper], fmt='none', color='black', capsize=10, lw=2, elinewidth=2, ecolor='black', capthick=2)

if input_type == 'difference':
    #if ci doesnt contain 0 (significant difference), add stars
    for i,c in enumerate(lower_ci):
        if c!=0:
            ax[1].plot(x_pos[i], upper_ci[i]+10, '*', color = sorted_color_dict[i])

# Setting labels and title
ax[1].set_xticks(x_pos, sorted_feature_names_graph, fontsize = 9, fontname= font, rotation = 45)
if input_type != 'difference':
    ax[1].set_ylabel('ResNet-18 layer', fontdict={'family': font, 'size': 11})
else:
    ax[1].set_ylabel('# layers', fontdict={'family': font, 'size': 11})


if input_type == 'difference':
    y_ticks_bar = range(9)
    ylabel_bar = range(9)
else:
    y_ticks_bar = range(9)
    ylabel_bar =  ['', '1.0', '1.1', '2.0', '2.1', '3.0', '3.1', '4.0', '4.1'] 

ax[1].set_yticks(ticks = y_ticks_bar, 
           labels = ylabel_bar, 
           fontname=font, fontsize = 9)

# if input_type != 'difference':
#     ax[1].set_ylim(0,10)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['bottom'].set_linewidth(2)  # Set x-axis spine linewidth
ax[1].spines['left'].set_linewidth(2)    # Set y-axis spine linewidth

ax[1].tick_params(axis='both', direction='inout')
ax[1].tick_params(axis='x', which='both', length=6, width=3)  # Adjust the length and width as needed
ax[1].tick_params(axis='y', which='both', length=6, width=3)  # Adjust the length and width as needed

# Displaying the plot
# plt.show()
plt.subplots_adjust(bottom=0.2)

#save as svg and png
plotDir = os.path.join(saveDir, 'plot_encoding_layers_resnet_{}_ordered.svg'.format(input_type))
plotDir_png = os.path.join(saveDir, 'plot_encoding_layers_resnet_{}_ordered.png'.format(input_type))
plt.savefig(plotDir, dpi = 300, format = 'svg', transparent = True)
plt.savefig(plotDir_png, dpi = 300, format = 'png', transparent = True)

