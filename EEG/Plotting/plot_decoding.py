#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLOT DECODING OF IMAGES AND VIDEOS

This script creates plots for the decoding analysis.

@author: Alexander Lenders, Agnessa Karapetian
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
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

args = parser.parse_args()  # to get values for the arguments

config = load_config(args.config_dir, args.config)
workDir = config.get(args.config, "save_dir")
font = args.font  # which font should be used

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
list_sub_img = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

workDir_vid = os.path.join(workDir, "decoding", "miniclips")
workDir_img = os.path.join(workDir, "decoding", "images")

saveDir = os.path.join(workDir, "decoding", "plots")

# -----------------------------------------------------------------------------
# STEP 1: Import modules
# -----------------------------------------------------------------------------
plt.rcParams["svg.fonttype"] = "none"

# -----------------------------------------------------------------------------
# STEP 2: Load results (Videos)
# -----------------------------------------------------------------------------
results_vid = []  # list of dictionaries
decoding_mean_vid = []

for subject in list_sub_vid:
    if subject < 10:
        fileDir = (
            workDir_vid
            + "/decoding_miniclips_sub-0{}_redone.npy".format(subject)
        )
    else:
        fileDir = workDir_vid + "/decoding_miniclips_sub-{}_redone.npy".format(
            subject
        )

    vid_decoding = np.load(fileDir, allow_pickle=True).item()
    results_vid.append(vid_decoding)
    print("appended sub", subject)
    decoding_accuracy_vid = vid_decoding["mean_accuracies_over_conditions"]
    decoding_mean_vid.append(decoding_accuracy_vid)

# -----------------------------------------------------------------------------
# STEP 3: Load results (Images)
# -----------------------------------------------------------------------------
results_img = []  # list of dictionaries
decoding_mean_img = []
timepoints = 70
plot_timepoints = 60

for subject in list_sub_img:
    if subject < 10:
        fileDir = workDir_img + "/decoding_images_sub-0{}_redone.npy".format(
            subject
        )
    else:
        fileDir = workDir_img + "/decoding_images_sub-{}_redone.npy".format(
            subject
        )

    img_decoding = np.load(fileDir, allow_pickle=True).item()
    results_img.append(img_decoding)
    print("appended sub", subject)
    decoding_accuracy_img = img_decoding["mean_accuracies_over_conditions"]
    decoding_mean_img.append(decoding_accuracy_img)

# -----------------------------------------------------------------------------
# STEP 4: Create decoding plot (Figure 3A)
# -----------------------------------------------------------------------------
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plotDir_svg = saveDir + "/plot_comparison_mean_accuracy_nonstd.svg"
plotDir_png = saveDir + "/plot_comparison_mean_accuracy_nonstd.png"
chance = 0.5
### 1. Mean curve ###
# videos
mean_decoding_vid = np.zeros((timepoints,), dtype=float)
for array in range(len(decoding_mean_vid)):
    to_add = decoding_mean_vid[array]
    mean_decoding_vid = np.add(mean_decoding_vid, to_add)

# images
mean_decoding_img = np.zeros((timepoints,), dtype=float)
for array in range(len(decoding_mean_img)):
    to_add = decoding_mean_img[array]
    mean_decoding_img = np.add(mean_decoding_img, to_add)

# mean over subjects
mean_decoding_vid = np.divide(mean_decoding_vid, (len(decoding_mean_vid)))
mean_decoding_vid = mean_decoding_vid[10:]
mean_decoding_img = np.divide(mean_decoding_img, (len(decoding_mean_img)))
mean_decoding_img = mean_decoding_img[10:]

# calculate and save difference
mean_decoding_diff = mean_decoding_img - mean_decoding_vid
fileDir_diff = "difference_decoding_all_subjects.npy"
np.save(os.path.join(workDir_img, fileDir_diff), mean_decoding_diff)

### 2. CIs ###
##videos
ci_dir = os.path.join(workDir_vid, "stats", "CI_95_accuracy_redone.pkl")

with open(ci_dir, "rb") as file:
    confidence_95_vid = pickle.load(file)

# Extract the first values from each list in the dictionary
low_CI_vid = np.array(
    [confidence_95_vid[key][0] for key in (confidence_95_vid)]
)
low_CI_vid = low_CI_vid[10:] - chance
high_CI_vid = np.array(
    [confidence_95_vid[key][1] for key in (confidence_95_vid)]
)
high_CI_vid = high_CI_vid[10:] - chance

##images
ci_dir_img = os.path.join(workDir_img, "stats", "CI_95_accuracy_redone.pkl")

with open(ci_dir_img, "rb") as file:
    img_confidence_95 = pickle.load(file)

# Extract the first values from each list in the dictionary
low_CI_img = np.array(
    [img_confidence_95[key][0] for key in (img_confidence_95)]
)
low_CI_img = low_CI_img[10:] - chance
high_CI_img = np.array(
    [img_confidence_95[key][1] for key in (img_confidence_95)]
)
high_CI_img = high_CI_img[10:] - chance

##diff
ci_dir_diff = os.path.join(
    workDir,
    "decoding",
    "difference",
    "stats",
    "diff_CI_95_accuracy_redone.pkl",
)

with open(ci_dir_diff, "rb") as file:
    confidence_95_diff = pickle.load(file)

# Extract the first values from each list in the dictionary
low_CI_diff = np.array(
    [confidence_95_diff[key][0] for key in (confidence_95_diff)]
)
low_CI_diff = low_CI_diff[10:]
high_CI_diff = np.array(
    [confidence_95_diff[key][1] for key in (confidence_95_diff)]
)
high_CI_diff = high_CI_diff[10:]

### 3. Stats ###
##vid
decoding_dir = os.path.join(
    workDir_vid, "stats", "decoding_stats_both_nonstd.pkl"
)
with open(decoding_dir, "rb") as file:
    decoding_stats_vid = pickle.load(file)
bool_decoding_vid = decoding_stats_vid["Boolean_statistical_map"]
bool_decoding_vid = bool_decoding_vid[10:]
array_vid = np.full((plot_timepoints, 1), 0.33)

##img
decoding_stats_dir_img = os.path.join(
    workDir_img, "stats", "decoding_stats_both_nonstd.pkl"
)

with open(decoding_stats_dir_img, "rb") as file:
    decoding_stats_img = pickle.load(file)
bool_decoding_img = decoding_stats_img["Boolean_statistical_map"]
bool_decoding_img = bool_decoding_img[10:]
array_img = np.full((plot_timepoints, 1), 0.32)

##diff
decoding_stats_dir_diff = os.path.join(
    workDir,
    "decoding",
    "difference",
    "stats",
    "diff_decoding_stats_both_nonstd.pkl",
)

with open(decoding_stats_dir_diff, "rb") as file:
    decoding_stats_diff = pickle.load(file)

bool_decoding_diff = decoding_stats_diff["Boolean_statistical_map"]
bool_decoding_diff = bool_decoding_diff[10:]
array_diff = np.full((plot_timepoints, 1), 0.31)

### 4. Plot ###

##vid
color_vid = "purple"
mean_decoding_vid_minus_chance = mean_decoding_vid - chance
plt.plot(
    mean_decoding_vid_minus_chance,
    color=color_vid,
    linewidth=2,
    zorder=6,
    label="Videos",
)
plt.fill_between(
    range(plot_timepoints),
    low_CI_vid,
    high_CI_vid,
    color=color_vid,
    alpha=0.2,
    zorder=5,
)
significant_time_points = np.where(bool_decoding_vid)[0]
plt.plot(
    significant_time_points,
    array_vid[significant_time_points],
    "*",
    color=color_vid,
    markersize=4,
)
peak_tick_halflength = 0.015
plt.plot(
    [
        np.argmax(mean_decoding_vid_minus_chance),
        np.argmax(mean_decoding_vid_minus_chance),
    ],
    [
        np.max(mean_decoding_vid_minus_chance) - peak_tick_halflength,
        np.max(mean_decoding_vid_minus_chance) + peak_tick_halflength,
    ],
    color="black",
    alpha=0.6,
    linestyle=":",
    linewidth=2,
)
plt.show()

# ticks and labels
x_ticks_plot = np.arange(0, 65, 5)
x_labels_plot = np.arange(-200, 1100, 100)
y_ticks_plot = np.arange(-0.1, 0.4, 0.05)
y_labels_plot = (y_ticks_plot * 100).round()

plt.yticks(
    ticks=y_ticks_plot,
    labels=y_labels_plot,
    fontdict={"family": font, "size": 11},
)
plt.xticks(
    ticks=x_ticks_plot,
    labels=x_labels_plot,
    fontdict={"family": font, "size": 11},
    rotation=45,
)

##img
color_img = "teal"
mean_decoding_img_minus_chance = mean_decoding_img - chance
plt.plot(
    mean_decoding_img_minus_chance,
    color=color_img,
    linewidth=2,
    zorder=4,
    label="Images",
)
plt.fill_between(
    range(plot_timepoints),
    low_CI_img,
    high_CI_img,
    color=color_img,
    alpha=0.2,
    zorder=3,
)
significant_time_points_img = np.where(bool_decoding_img)[0]
plt.plot(
    significant_time_points_img,
    array_img[significant_time_points_img],
    "*",
    color=color_img,
    markersize=4,
)
peak_tick_halflength = 0.015
plt.plot(
    [
        np.argmax(mean_decoding_img_minus_chance),
        np.argmax(mean_decoding_img_minus_chance),
    ],
    [
        np.max(mean_decoding_img_minus_chance) - peak_tick_halflength,
        np.max(mean_decoding_img_minus_chance) + peak_tick_halflength,
    ],
    color="black",
    alpha=0.6,
    linestyle=":",
    linewidth=2,
)
plt.show()

##diff
color_diff = "black"
plt.plot(
    mean_decoding_diff,
    color=color_diff,
    linewidth=2,
    label="Images minus videos",
)
plt.fill_between(
    range(plot_timepoints),
    low_CI_diff,
    high_CI_diff,
    alpha=0.2,
    color=color_diff,
    zorder=2,
)
significant_time_points = np.where(bool_decoding_diff)[0]
plt.plot(
    significant_time_points,
    array_diff[significant_time_points],
    "*",
    color=color_diff,
    markersize=4,
    zorder=1,
)
plt.show()

# vertical and horizontal lines at (0)
plt.axvline(
    x=10,
    color="lightgrey",
    linestyle="solid",
    linewidth=2,
    alpha=0.5,
    zorder=0,
)
plt.axhline(
    y=0,
    color="lightgrey",
    linestyle="solid",
    linewidth=2,
    alpha=0.5,
    zorder=0,
)

### Plot params ###
# ticks, labels and legend
x_ticks_plot = np.arange(0, 70, 10)
x_labels_plot = np.arange(-200, 1200, 200)
y_ticks_plot = np.arange(-0.1, 0.4, 0.1)
y_labels_plot = (y_ticks_plot * 100).astype(int)

plt.yticks(
    ticks=y_ticks_plot,
    labels=y_labels_plot,
    fontdict={"family": font, "size": 11},
)
plt.xticks(
    ticks=x_ticks_plot,
    labels=x_labels_plot,
    fontdict={"family": font, "size": 11},
    rotation=45,
)

plt.tick_params(axis="both", direction="inout")
plt.xlabel("Time (ms)", fontdict={"family": font, "size": 14})
plt.ylabel(
    "Decoding accuracy minus chance (%)",
    fontdict={"family": font, "size": 14},
)
ax = plt.gca()
ax.tick_params(
    axis="x", which="both", length=6, width=2
)  # Adjust the length and width as needed
ax.tick_params(
    axis="y", which="both", length=6, width=2
)  # Adjust the length and width as needed
plt.xlim(0, plot_timepoints)

handles, labels = ax.get_legend_handles_labels()
plt.legend(
    [handles[1], handles[0], handles[2]],
    [labels[1], labels[0], labels[2]],
    bbox_to_anchor=(0.75, 0.7),
    frameon=False,
    loc="center",
    prop={"family": font, "size": 11},
)

# spines
ax.spines["bottom"].set_linewidth(2)  # Set x-axis spine linewidth
ax.spines["left"].set_linewidth(2)  # Set y-axis spine linewidth
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# final params

plt.tight_layout()
plt.show()
plt.savefig(plotDir_svg, format="svg", dpi=300, transparent=True)
plt.savefig(plotDir_png, format="png", transparent=True, bbox_inches="tight")
plt.close()
