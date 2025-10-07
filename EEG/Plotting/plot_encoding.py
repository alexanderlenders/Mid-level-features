#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOTS FOR ENCODING ANALYSIS

This script creates plots for the encoding analyses.

@author: Alexander Lenders, Agnessa Karapetian
"""
# -----------------------------------------------------------------------------
# STEP 1: Initialize variables & import modules
# -----------------------------------------------------------------------------
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import pdb
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
    parse_list,
)

parser = argparse.ArgumentParser()

# add arguments / inputs
parser.add_argument(
    "--config_dir",
    type=str,
    help="Directory to the configuration file.",
    required=True
)
parser.add_argument(
    "--config",
    type=str,
    help="Configuration.",
    required=True
)
parser.add_argument(
    "-f", "--font", default="Arial", type=str, metavar="", help="Font"
)
parser.add_argument(
    "-i",
    "--input_type",
    default="images",
    type=str,
    metavar="",
    help="images, miniclips or difference"
)
parser.add_argument(
    "--legend",
    action="store_true",
    help="Show legend in plots"
)

parser.add_argument(
    '-un', 
    "--upper_noise_ceiling",
    default = False,
    type = bool, 
    metavar='',
    help="plot upper noise ceiling or not"
)

args = parser.parse_args()  # to get values for the arguments

config = load_config(args.config_dir, args.config)

workDir = config.get(args.config, "save_dir")
noise_ceiling_dir = config.get(args.config, "noise_ceiling_dir")
feature_names = parse_list(config.get(args.config, "feature_names"))
plot_legend = args.legend
upper_noise_ceiling = args.upper_noise_ceiling

feature_names_graph = parse_list(
    config.get(args.config, "feature_names_graph")
)

font = args.font
input_type = args.input_type

if input_type == "miniclips":
    list_sub = [
        6,
        7,
        8,
        9,
        10,
        11,
        17,
        18,
        20,
        21,
        23,
        25,
        27,
        28,
        29,
        30,
        31,
        32,
        34,
        36,
    ]
elif input_type == "images":
    list_sub = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
elif input_type == "difference":
    list_sub_vid = [
        6,
        7,
        8,
        9,
        10,
        11,
        17,
        18,
        20,
        21,
        23,
        25,
        27,
        28,
        29,
        30,
        31,
        32,
        34,
        36,
    ]
    list_sub_img = [
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
    ]

# -----------------------------------------------------------------------------
# STEP 2: Import modules & Define Variables
# -----------------------------------------------------------------------------
plt.rcParams["svg.fonttype"] = "none"

### Paths ###
# Results #
original_workDir = workDir  # save original workDir
workDir = os.path.join(workDir, f"{input_type}")
statsDir = os.path.join(workDir, "stats")
saveDir = os.path.join(workDir, "plots")

if os.path.exists(saveDir) == False:
    os.makedirs(saveDir)

identifierDir = f"seq_50hz_posterior_encoding_results_averaged_frame_before_mvnn_{len(feature_names)}_features_onehot.pkl"

# Stats #
ci_dir = os.path.join(
    statsDir, "encoding_CI95_accuracy.pkl"
)  # CI 95% encoding curve
stats_dir = os.path.join(
    statsDir, "encoding_stats_both_nonstd.pkl"
)  # Permutation tests encoding curve
ci_stats = os.path.join(
    statsDir,
    "encoding_stats_peak_latency_CI.pkl",
)  # CI peak latency
if input_type == "images" or input_type == "miniclips":
    peakDir = os.path.join(statsDir, "encoding_CI95_peak.pkl")  # CI 95% peaks
elif input_type == "difference":
    peakDir = os.path.join(
        statsDir, "encoding_diff_in_peak.pkl"
    )  # CI 95% peaks

### Define some variables ###
num_timepoints_og = 70  # full epoch
time_ms = np.arange(-400, 1000, 20)
timepoints = np.arange(60)  # for plotting
num_features = len(feature_names)
colormap = plt.colormaps["Set2"]
colors = [colormap(i) for i in range(num_features)]

# -----------------------------------------------------------------------------
# STEP 3: Load results
# -----------------------------------------------------------------------------
### Load subject-specific results ###

if input_type == "miniclips" or input_type == "images":
    results = []  # list of dictionaries

    for subject in list_sub:
        fileDir = os.path.join(workDir, f"{subject}_{identifierDir}")
        with open(fileDir, "rb") as f:
            encoding_results = pickle.load(f)
        results.append(encoding_results)

elif input_type == "difference":
    results_vid = []
    results_img = []

    for subject in list_sub_vid:
        workDir_vid = os.path.join(original_workDir, "miniclips")
        fileDir_vid = os.path.join(workDir_vid, f"{subject}_{identifierDir}")
        with open(fileDir_vid, "rb") as f:
            encoding_results_vid = pickle.load(f)
        results_vid.append(encoding_results_vid)

    for subject in list_sub_img:
        workDir_img = os.path.join(original_workDir, "images")
        fileDir_img = os.path.join(workDir_img, f"{subject}_{identifierDir}")
        with open(fileDir_img, "rb") as f:
            encoding_results_img = pickle.load(f)
        results_img.append(encoding_results_img)

### Create mean of all subjects for each feature ###
features_mean = []

if input_type == "miniclips" or input_type == "images":
    for feature in feature_names:
        features = []

        for sub, subject in enumerate(list_sub):
            f = results[sub][feature]["correlation"]
            f_averaged = np.mean(f, axis=1)  # avg over channels
            features.append(f_averaged)

        mean_feature = np.zeros((num_timepoints_og,), dtype=float)

        for sub, subject in enumerate(list_sub):
            mean_feature = np.add(mean_feature, features[sub])

        mean_feature = np.divide(mean_feature, len(list_sub))
        features_mean.append(mean_feature)

elif input_type == "difference":

    for feature in feature_names:
        features_vid = []
        features_img = []

        # get feature encoding values for each subject
        for sub, subject in enumerate(list_sub_vid):
            f_v = results_vid[sub][feature]["correlation"]
            f_v_averaged = np.mean(f_v, axis=1)
            features_vid.append(f_v_averaged)

        for sub, subject in enumerate(list_sub_img):
            f_i = results_img[sub][feature]["correlation"]
            f_i_averaged = np.mean(f_i, axis=1)
            features_img.append(f_i_averaged)

        # average over subjects
        mean_feature_vid = np.zeros((num_timepoints_og,), dtype=float)
        mean_feature_img = np.zeros((num_timepoints_og,), dtype=float)

        for sub, subject in enumerate(list_sub_vid):
            mean_feature_vid = np.add(mean_feature_vid, features_vid[sub])
        for sub, subject in enumerate(list_sub_img):
            mean_feature_img = np.add(mean_feature_img, features_img[sub])

        mean_feature_vid = np.divide(mean_feature_vid, len(list_sub_vid))
        mean_feature_img = np.divide(mean_feature_img, len(list_sub_img))
        peak_img = time_ms[np.argmax(mean_feature_img)]
        peak_vid = time_ms[np.argmax(mean_feature_vid)]
        peak_diff = peak_img - peak_vid

        # difference
        mean_feature_diff = mean_feature_img - mean_feature_vid
        features_mean.append(mean_feature_diff)

# -----------------------------------------------------------------------------
# STEP 4: Subplot 1 -> Encoding curve
# -----------------------------------------------------------------------------
### Load CIs and stats ###
with open(ci_dir, "rb") as file:
    confidence_95 = pickle.load(file)
with open(stats_dir, "rb") as file:
    encoding_stats = pickle.load(file)

### Noise ceiling ###
# Load: noise ceiling for images
if input_type != "difference":
    if input_type == "images":
        noiseDir = os.path.join(noise_ceiling_dir, "images")
    elif input_type == "miniclips":
        noiseDir = os.path.join(noise_ceiling_dir, "miniclips")

    lower_ceilings_all = np.zeros((len(list_sub), num_timepoints_og))
    upper_ceilings_all = np.zeros((len(list_sub), num_timepoints_og))

    for s, subject in enumerate(list_sub):

        # lower
        fileDir_lower = (
            str(subject) + "_seq_50hz_posterior_lower_noise_ceiling.pkl"
        )
        path_noise_l = os.path.join(noiseDir, fileDir_lower)

        with open(path_noise_l, "rb") as file:
            lower_ceiling = pickle.load(file)

        lower_ceilings_all[s, :] = lower_ceiling

        # upper
        fileDir_upper = (
            str(subject) + "_seq_50hz_posterior_upper_noise_ceiling.pkl"
        )
        path_noise_u = os.path.join(noiseDir, fileDir_upper)

        with open(path_noise_u, "rb") as file:
            upper_ceiling = pickle.load(file)

        upper_ceilings_all[s, :] = upper_ceiling

    avg_lower_ceil = np.mean(lower_ceilings_all, axis=0)[10:]
    avg_upper_ceil = np.mean(upper_ceilings_all, axis=0)[10:]

### Create the plot ###
plt.close()
fig, ax = plt.subplots(1, 2, figsize=(8, 4.3))
ax[0].axvline(x=10, color="lightgrey", linestyle="solid", linewidth=1)
ax[0].axhline(y=0, color="lightgrey", linestyle="solid", linewidth=1)

x_tick_values_curves = np.arange(0, 65, 5)
x_tick_labels_curves = [
    -200,
    -100,
    0,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
]

# plot noise ceilings
if upper_noise_ceiling:
    if input_type != 'difference':
        ax[0].fill_between(timepoints, avg_lower_ceil, avg_upper_ceil, color='grey', alpha=0.3) 
else:
    if input_type != 'difference':
        ax[0].plot(timepoints,avg_lower_ceil,color='grey', linestyle='--', alpha=0.3)

### Load CIs and stats ###
with open(peakDir, "rb") as file:
    peaks = pickle.load(file)
    peaks = {key: peaks[key] for key in feature_names}
if input_type != "difference":
    with open(ci_stats, "rb") as file:
        peak_stats = pickle.load(file)

accuracies = [item[1] for item in peaks.values()]
lower_ci = [item[0] for item in peaks.values()]
upper_ci = [item[2] for item in peaks.values()]

### Rearrange the accuracies, lower CIs, and upper CIs based on the sorted indices ###
# sorted_indices = sorted(range(len(accuracies)), key=lambda k: accuracies[k])

if (
    args.config == "default"
    or args.config == "control_3"
    or args.config == "control_1"
    or args.config == "control_2"
    or args.config == "control_9"
    or args.config == "control_12"
):
    sorted_indices = [
        0,
        4,
        3,
        1,
        2,
        5,
        6,
    ]
    sorted_color_dict = [colors[i] for i in sorted_indices]
else:
    sorted_color_dict = colors

sorted_features_mean = features_mean
sorted_feature_names_graph = feature_names_graph
sorted_feature_names = feature_names

for i, feature in enumerate(sorted_features_mean):
    # accuracy
    f_name = sorted_feature_names[i]
    accuracy = feature
    accuracy = accuracy[10:]

    # CIs
    ci_feature = confidence_95[f_name]
    low_CI = np.array([ci_feature[key][0] for key in (ci_feature)])
    low_CI = low_CI[10:]
    high_CI = np.array([ci_feature[key][1] for key in (ci_feature)])
    high_CI = high_CI[10:]

    # Stats
    stats_results = encoding_stats[f_name]["Boolean_statistical_map"]
    stats_results = stats_results[10:]
    if input_type == "images" or input_type == "miniclips":
        starting_value = -0.15
    elif input_type == "difference":
        starting_value = -0.05
    plot_value = starting_value + (i / 50)

    test = np.full((60, 1), np.nan, dtype=float)
    significant_indices = np.where(stats_results)[0]
    test[significant_indices] = plot_value
    plot_accuracies = test

    # Plot
    ax[0].plot(
        timepoints,
        accuracy,
        color=sorted_color_dict[i],
        label=sorted_feature_names_graph[i],
        linewidth=2,
    )  # accuracy
    ax[0].fill_between(
        timepoints, low_CI, high_CI, color=sorted_color_dict[i], alpha=0.3
    )  # CIs
    ax[0].plot(
        timepoints,
        plot_accuracies,
        "*",
        color=sorted_color_dict[i],
        markersize=4,
    )

    # Plot peak ticks
    if input_type == "difference":
        peak_tick_halflength = 0.015
    else:
        peak_tick_halflength = 0.017
    if input_type != "difference":
        ax[0].plot(
            [np.argmax(accuracy), np.argmax(accuracy)],
            [
                np.max(accuracy) - peak_tick_halflength,
                np.max(accuracy) + peak_tick_halflength,
            ],
            color="k",
            alpha=0.6,
            linestyle=":",
            linewidth=2,
        )

### Set plotting parameters ###
ax[0].set_xlabel("Time (ms)", fontdict={"family": font, "size": 11})
ax[0].set_ylabel("Pearson's r", fontdict={"family": font, "size": 11})

#yticks and labels
if input_type == 'miniclips' or input_type == 'images':
    if upper_noise_ceiling:
        ax[0].set_yticks(ticks=[-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        labels = [-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 
                        fontsize=9, fontname=font)
    else:
        ax[0].set_yticks(ticks=[-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4],
                labels = [-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4], 
                fontsize=9, fontname=font)
elif input_type == 'difference':
    ax[0].set_yticks(ticks=[-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15],
                     labels = [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15],
                     fontsize=9, fontname=font) 
    
ax[0].set_xticks(
    ticks=x_tick_values_curves,
    labels=x_tick_labels_curves,
    fontsize=9,
    fontname=font,
    rotation=45,
)
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[0].spines["bottom"].set_linewidth(2)  # Set x-axis spine linewidth
ax[0].spines["left"].set_linewidth(2)  # Set y-axis spine linewidth
ax[0].tick_params(
    axis="x", which="both", length=6, width=3
)  # Adjust the length and width as needed
ax[0].tick_params(
    axis="y", which="both", length=6, width=3
)  # Adjust the length and width as needed
ax[0].set_xlim(0, 60)
# if input_type == 'difference':
# ax[0].set_ylim(-0.15,0.15)

# -----------------------------------------------------------------------------
# STEP 5: Subplot 2 -> Peak latencies
# -----------------------------------------------------------------------------
# the difference plot is the difference in peak latencies
y_range = max(upper_ci) - min(lower_ci)
top = max(upper_ci)

x_pos = np.arange(len(feature_names))

# Plot the sig. difference lines
if input_type != "difference":
    significant_combinations = []

    for feature, values in peak_stats.items():
        ci = values["ci"]
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

    # sig line params
    line_offset = 435 + y_range * 0.2  # Offset for the lines
    line_height = 20  # Height of each significance line
    text_offset = line_offset - y_range * 0.02  # Offset for the text labels

    for i, sig_combi in enumerate(sorted_sig_combi_numbers):
        x1, x2 = sig_combi
        level = len(sorted_sig_combi_numbers) - i + 1
        bar_height = level * line_height + line_offset
        ax[1].plot(
            [x1, x2], [bar_height, bar_height], lw=1, c="black", ls="solid"
        )

### Plotting parameters ###
x_pos = np.arange(len(sorted_feature_names))
bars = ax[1].bar(
    x_pos, accuracies, align="center", alpha=1, color=sorted_color_dict
)
y_err_lower = [acc - lower for acc, lower in zip(accuracies, lower_ci)]
y_err_upper = [upper - acc for acc, upper in zip(accuracies, upper_ci)]

ax[1].errorbar(
    x_pos,
    accuracies,
    yerr=[y_err_lower, y_err_upper],
    fmt="none",
    color="black",
    capsize=10,
    lw=2,
    elinewidth=2,
    ecolor="black",
    capthick=2,
)
ax[1].set_xticks(
    x_pos,
    sorted_feature_names_graph,
    fontsize=9,
    fontname=font,
    rotation=45,
)
ax[1].set_ylabel("Time (ms)", fontdict={"family": font, "size": 11})

# stat significance in peak latency
if input_type == "difference":
    # if ci doesnt contain 0 (significant difference), add stars
    for i, c in enumerate(lower_ci):
        if c != 0:
            ax[1].plot(x_pos[i], upper_ci[i] + 10, "*", color="black")

plt.subplots_adjust(bottom=0.2)

# y ticks and labels
if input_type == "miniclips" or input_type == "images":
    ax[1].set_ylim((0.0, 973.35))
    ax[1].set_yticks(
        ticks=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        labels=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        fontname=font,
        fontsize=9,
    )
elif input_type == "difference":
    ax[1].set_yticks(
        ticks=[0, 50, 100, 150, 200, 250, 300, 350, 400],
        labels=[0, 50, 100, 150, 200, 250, 300, 350, 400],
        fontname=font,
        fontsize=9,
    )

ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[1].spines["bottom"].set_linewidth(2)  # Set x-axis spine linewidth
ax[1].spines["left"].set_linewidth(2)  # Set y-axis spine linewidth

ax[1].tick_params(axis="both", direction="inout")
ax[1].tick_params(
    axis="x", which="both", length=6, width=3
)  # Adjust the length and width as needed
ax[1].tick_params(
    axis="y", which="both", length=6, width=3
)  # Adjust the length and width as needed

plt.tight_layout()

# legend
if plot_legend:
    legend_font_props = {"family": font, "size": 9}
    ax[0].legend(
        prop=legend_font_props, frameon=False, bbox_to_anchor=[0.7,0.6]
    )

ax[0].tick_params(axis="both", direction="inout")

# -----------------------------------------------------------------------------
# STEP 6: Saving the plot
# -----------------------------------------------------------------------------
plt.show()
if upper_noise_ceiling:
    plotDir = os.path.join(
        saveDir,
        f"plot_encoding_{len(feature_names)}_features_{input_type}_nonstd_bothnc.svg",
    )
    plt.savefig(plotDir, dpi=300, format="svg", transparent=True)

    plotDir = os.path.join(
        saveDir,
        f"plot_encoding_full_{len(feature_names)}_features_{input_type}_nonstd_bothnc.png",
    )
    plt.savefig(plotDir, dpi=300, format="png", transparent=True)

else:
    plotDir = os.path.join(
        saveDir,
        f"plot_encoding_{len(feature_names)}_features_{input_type}_nonstd_nouppernc.svg",
    )
    plt.savefig(plotDir, dpi=300, format="svg", transparent=True)

    plotDir = os.path.join(
        saveDir,
        f"plot_encoding_full_{len(feature_names)}_features_{input_type}_nonstd_nouppernc.png",
    )
    plt.savefig(plotDir, dpi=300, format="png", transparent=True)

