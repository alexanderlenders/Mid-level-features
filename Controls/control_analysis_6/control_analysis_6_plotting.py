#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOTS FOR VARIANCE PARTITIONING ANALYSES

This script creates plots for the variance partitioning analyses.

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
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
print(project_root)
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
    required=True,
)
parser.add_argument(
    "--config",
    type=str,
    help="Configuration.",
    required=True,
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
    help="images or miniclips",
)

args = parser.parse_args()  # to get values for the arguments

config = load_config(args.config_dir, args.config)
workDir = config.get(args.config, "save_dir")
noise_ceiling_dir = config.get(args.config, "noise_ceiling_dir")
feature_names = parse_list(config.get(args.config, "feature_names"))

full_feature_set = feature_names[-1]
full_feature_set = ', '.join(full_feature_set)  # Convert to string for printing
feature_names = feature_names[:-1]  # remove the full feature set

temp_list = [
    f"{', '.join(f)}" if isinstance(f, (tuple, list)) else str(f)
    for f in feature_names  
]
feature_names = temp_list

print(full_feature_set)

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
peakDir = os.path.join(statsDir, "encoding_CI95_peak.pkl")  # CI 95% peaks

### Define some variables ###
num_timepoints_og = 70  # full epoch
time_ms = np.arange(-400, 1000, 20)
timepoints = np.arange(60)  # for plotting
num_features = len(feature_names)
colormap = plt.colormaps["Set2"]

if args.config == "control_6_1":
    colors = [colormap(i) for i in range(num_features)]
else:
    # To have the same colour scheme
    colors = [colormap(i+1) for i in range(num_features)]

# -----------------------------------------------------------------------------
# STEP 3: Load results
# -----------------------------------------------------------------------------
### Load subject-specific results ###
results = []  # list of dictionaries

for subject in list_sub:
    fileDir = os.path.join(workDir, f"{subject}_{identifierDir}")
    with open(fileDir, "rb") as f:
        encoding_results = pickle.load(f)
    results.append(encoding_results)

### Create mean of all subjects for each feature ###
features_mean = []

for feature in feature_names:
    print(feature)
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

# -----------------------------------------------------------------------------
# STEP 4: Subplot 1 -> Encoding curve
# -----------------------------------------------------------------------------
### Load CIs and stats ###
with open(ci_dir, "rb") as file:
    confidence_95 = pickle.load(file)
with open(stats_dir, "rb") as file:
    encoding_stats = pickle.load(file)

### Create the plot ###
plt.close()
fig, ax = plt.subplots(1, 1, figsize=(5, 4.3))
ax.axvline(x=10, color="lightgrey", linestyle="solid", linewidth=1)
ax.axhline(y=0, color="lightgrey", linestyle="solid", linewidth=1)

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

sorted_feature_names_graph = feature_names_graph
sorted_feature_names = feature_names
sorted_color_dict = colors
sorted_features_mean = features_mean

# Load stats for the full model 
print(full_feature_set)
stats_results = encoding_stats[full_feature_set]["Boolean_statistical_map"]
stats_results = stats_results[10:]  # Exclude the first 10 timepoints
# significant_indices = np.where(stats_results)[0]

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

    # To get variance explained in %
    accuracy = accuracy * 100
    low_CI = low_CI * 100
    high_CI = high_CI * 100

    # Mask arrays for significant and non-significant parts
    accuracy_sig = np.ma.masked_where(~stats_results, accuracy)
    accuracy_nonsig = np.ma.masked_where(stats_results, accuracy)
    low_CI_sig = np.ma.masked_where(~stats_results, low_CI)
    high_CI_sig = np.ma.masked_where(~stats_results, high_CI)
    low_CI_nonsig = np.ma.masked_where(stats_results, low_CI)
    high_CI_nonsig = np.ma.masked_where(stats_results, high_CI)

    # Plot lines
    ax.plot(timepoints, accuracy, color='lightgrey', linewidth=2, alpha = 0.5)
    ax.plot(timepoints, accuracy_sig, color=sorted_color_dict[i], linewidth=2, label=sorted_feature_names_graph[i],
)

    # Plot shaded CI areas
    # ax.fill_between(timepoints, low_CI, high_CI, color='lightgrey', alpha=0.15)
    ax.fill_between(timepoints, low_CI_sig, high_CI_sig, color=sorted_color_dict[i], alpha=0.2)

    # Plot peak ticks
    peak_tick_halflength = 0.017
    ax.plot(
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
ax.set_xlabel("Time (ms)", fontdict={"family": font, "size": 11})
ax.set_ylabel(
    "Unique variance explained (%)", fontdict={"family": font, "size": 11}
)

# yticks and labels
if args.config == "control_6_1":
    ax.set_yticks(
        ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fontsize=9,
        fontname=font,
    )
else:
    ax.set_yticks(
        ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        fontsize=9,
        fontname=font,
    )
ax.set_ylim(bottom=0)

ax.set_xticks(
    ticks=x_tick_values_curves,
    labels=x_tick_labels_curves,
    fontsize=9,
    fontname=font,
    rotation=45,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_linewidth(2)  # Set x-axis spine linewidth
ax.spines["left"].set_linewidth(2)  # Set y-axis spine linewidth
ax.tick_params(
    axis="x", which="both", length=6, width=3
)  # Adjust the length and width as needed
ax.tick_params(
    axis="y", which="both", length=6, width=3
)  # Adjust the length and width as needed
ax.set_xlim(0, 60)
# if input_type == 'difference':
# ax.set_ylim(-0.15,0.15)

plt.tight_layout(rect=[0, 0, 1, 1])

legend_font_props = {"family": font, "size": 9}
ax.legend(
    prop=legend_font_props, frameon=False, bbox_to_anchor=[0.7, 0.6]
)

ax.tick_params(axis="both", direction="inout")

# -----------------------------------------------------------------------------
# STEP 6: Saving the plot
# -----------------------------------------------------------------------------
plt.show()
plotDir = os.path.join(
    saveDir,
    f"plot_encoding_{len(feature_names)}_features_{input_type}_nonstd.svg",
)
plt.savefig(plotDir, dpi=300, format="svg", transparent=True)
plotDir = os.path.join(
    saveDir,
    f"plot_encoding_full_{len(feature_names)}_features_{input_type}_nonstd.png",
)
plt.savefig(plotDir, dpi=300, format="png", transparent=True)