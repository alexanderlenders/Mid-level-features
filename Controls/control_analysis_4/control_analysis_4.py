"""
This script contains the code for control analysis 4, where the image and
video responses for the same stimuli are correlated. Note that the correlation
is computed after applying MVNN, and after averaging across trials, and participants. The correlation is computed for each electrode separately, and then averaged across electrodes.

@author: Alexander Lenders
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import argparse
project_root = Path(__file__).resolve().parents[2]
print(project_root)
sys.path.append(str(project_root))
from EEG.Encoding.utils import load_eeg, vectorized_correlation, load_config

def c4(subj_list_img: list, subj_list_vid: list, eeg_dir: str, save_dir: str, font: str = "Arial"):
    """
    Function for control analysis 4, in which image and video responses are correlated
    across timepoints after applying MVNN and averaging across electrodes, trials, and participants.

    This analysis is only conducted for the test data set.
    """
    N_CHANNELS = 19  # Use same channels as in encoding analysis
    REGION = "posterior"  # Use same region as in encoding analysis
    FREQ = 50
    
    # First step: Load all the EEG data for all subjects (images)
    full_eeg_data_img = []
    for sub in subj_list_img:
        y_test, _ = load_eeg(
            sub, "test", REGION, FREQ, "images", eeg_dir=eeg_dir
        )
        y_train, _ = load_eeg(
            sub, "train", REGION, FREQ, "images", eeg_dir=eeg_dir
        )
        y_val, _ = load_eeg(
            sub, "val", REGION, FREQ, "images", eeg_dir=eeg_dir
        )

        # Concatenate train, val, and test data
        y = np.concatenate((y_train, y_val, y_test), axis=0)

        full_eeg_data_img.append(y)

    full_eeg_data_img = np.array(full_eeg_data_img)

    # Now average across participants
    full_eeg_data_img = full_eeg_data_img.mean(axis=0)

    # Second step: Load all the EEG data for all subjects (videos)
    full_eeg_data_vid = []
    for sub in subj_list_vid:
        y_test, _ = load_eeg(
            sub, "test", REGION, FREQ, "miniclips", eeg_dir=eeg_dir
        )
        y_train, _ = load_eeg(
            sub, "train", REGION, FREQ, "miniclips", eeg_dir=eeg_dir
        )
        y_val, _ = load_eeg(
            sub, "val", REGION, FREQ, "miniclips", eeg_dir=eeg_dir
        )

        # Concatenate train, val, and test data
        y = np.concatenate((y_train, y_val, y_test), axis=0)

        full_eeg_data_vid.append(y)

    full_eeg_data_vid = np.array(full_eeg_data_vid)
    # Now average across participants
    full_eeg_data_vid = full_eeg_data_vid.mean(axis=0)

    print(full_eeg_data_vid.shape, flush=True)
    print(full_eeg_data_img.shape, flush=True)

    # Number of timepoints
    timepoints = full_eeg_data_img.shape[2]
    print(timepoints, flush=True)

    # Third step: Compute the correlation between image and video responses for each timepoint
    corr = np.zeros((timepoints, N_CHANNELS))

    for tp in range(timepoints):
        y_img = full_eeg_data_img[:, :, tp]
        y_vid = full_eeg_data_vid[:, :, tp]

        correlation = vectorized_correlation(y_img, y_vid)

        print(correlation.shape, flush=True)

        corr[tp, :] = correlation

    # Average across electrodes
    corr = np.mean(corr, axis=1)

    # Fifth step: Plot correlation across timepoints
    timepoints = np.arange(60)  # for plotting

    ### Create the plot ###
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 4.3))  # just one plot

    ax.axvline(x=10, color="lightgrey", linestyle="solid", linewidth=1)
    ax.axhline(y=0, color="lightgrey", linestyle="solid", linewidth=1)

    x_tick_values_curves = np.arange(0, 65, 5)
    x_tick_labels_curves = [-200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # Plot
    ax.plot(
        timepoints,
        corr,
        color="black",
        linewidth=2,
    )

    ax.set_xlabel("Time (ms)", fontdict={"family": font, "size": 11})
    ax.set_ylabel("Pearson's r", fontdict={"family": font, "size": 11})

    ax.set_yticks(
        ticks=[-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        labels=[-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        fontsize=9,
        fontname=font,
    )

    ax.set_xticks(
        ticks=x_tick_values_curves,
        labels=x_tick_labels_curves,
        fontsize=9,
        fontname=font,
        rotation=45,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)

    ax.tick_params(axis="x", which="both", length=6, width=3)
    ax.tick_params(axis="y", which="both", length=6, width=3)

    ax.set_xlim(0, 60)

    # Sixth step: Save the plot
    plt.show()

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    plotDir = os.path.join(
        save_dir,
        f"plot_correlation_image_and_video_responses.svg"
    )
    plt.savefig(plotDir, dpi=300, format="svg", transparent=True)
    plotDir = os.path.join(
        save_dir,
        f"plot_correlation_image_and_video_responses.png"
    )
    plt.savefig(plotDir, dpi=300, format="png", transparent=True)

if __name__ == "__main__":

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
    eeg_dir = config.get(args.config, "eeg_dir")
    SAVE_DIR = "/scratch/alexandel91/mid_level_features/results/c4"

    subj_list_vid = [
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
    
    subj_list_img = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    # Run the function
    c4(
        subj_list_img,
        subj_list_vid,
        eeg_dir=eeg_dir,
        save_dir=SAVE_DIR,
        font=args.font,
    )

