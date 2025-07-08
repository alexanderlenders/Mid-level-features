"""
This script contains the code for control analysis 4, where the image and
video responses for the same stimuli are correlated. Note that the correlation
is computed after applying MVNN, and after averaging across trials, and participants. The correlation is computed for each electrode separately, and then averaged across electrodes.

@author: Alexander Lenders
"""

from utils import load_eeg, vectorized_correlation
import numpy as np
import os
import matplotlib.pyplot as plt


def c4(subj_list_img: list, subj_list_vid: list, eeg_dir: str, save_dir: str):
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

        full_eeg_data_img.append(y_test)

    full_eeg_data_img = np.array(full_eeg_data_img)

    # Now average across participants
    full_eeg_data_img = full_eeg_data_img.mean(axis=0)

    # Second step: Load all the EEG data for all subjects (videos)
    full_eeg_data_vid = []
    for sub in subj_list_vid:
        y_test, _ = load_eeg(
            sub, "test", REGION, FREQ, "miniclips", eeg_dir=eeg_dir
        )

        full_eeg_data_vid.append(y_test)

    full_eeg_data_vid = np.array(full_eeg_data_vid)
    # Now average across participants
    full_eeg_data_vid = full_eeg_data_vid.mean(axis=0)

    # Number of timepoints
    timepoints = full_eeg_data_img.shape[2]

    # Third step: Compute the correlation between image and video responses for each timepoint
    corr = np.zeros((timepoints, N_CHANNELS))

    for tp in range(timepoints):
        y_img = full_eeg_data_img[:, :, tp]
        y_vid = full_eeg_data_vid[:, :, tp]

        correlation = vectorized_correlation(y_img, y_vid)

        corr[tp, :] = correlation

    # Average across electrodes
    corr_avg = np.mean(corr, axis=1)

    # Fifth step: Plot correlation across timepoints
    num_timepoints_og = 70  # full epoch
    time_ms = np.arange(-400, 1000, 20)
    timepoints = np.arange(60)  # for plotting

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

    # Plot
    ax[0].plot(
        timepoints,
        corr,
        color="black",
        linewidth=2,
    )  # accuracy

    peak_tick_halflength = 0.015

    ### Set plotting parameters ###
    ax[0].set_xlabel("Time (ms)", fontdict={"family": font, "size": 11})
    ax[0].set_ylabel("Pearson's r", fontdict={"family": font, "size": 11})

    # yticks and labels
    ax[0].set_yticks(
        ticks=[-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        labels=[-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        fontsize=9,
        fontname=font,
    )

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

    # Sixth step: Save the results


if __name__ == "__main__":
    ...
