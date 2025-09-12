#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOISE CEILING

@author: Agnessa Karapetian
"""
import os
import numpy as np
import pickle
import argparse
from encoding_utils import vectorized_correlation


def noise_ceiling(sub, freq, region, workDir, ica, input_type):
    """
    Input:
    ----------
    I. Test, Training and Validation EEG data sets, which are already
    preprocessed + MVNN. The input are dictionaries, which include:
        a. EEG-Data (eeg_data, 5400 Videos x 19 Channels x 54 Timepoints)
        b. Video Categories (img_cat, 5400 x 1) - Each video has one specific ID
        c. Channel names (channels, 64 x 1 OR 19 x 1)
        d. Time (time, 54 x 1) - Downsampled timepoints of a video
        In case of the validation data set there are 900 videos instead of 5400.
    II. Video features
        a. video_features.pkl: Canny edges, World normals, Lighting, Scene Depth,
        Reflectance, Character Identity, Action Identity, Skeleton Position after
        PCA (if necessary), saved in a dictionary "video_features"
            - Dictionary contains matrix for each feature with the dimension
            num_videos x num_components
        b. exp_variance_pca.pkl: Explained variance for each feature after PCA
        with n_components.

    Returns
    ----------
    Noise ceiling, with its upper and lower bounds.

    Parameters
    ----------
    sub : int
        Subject number
    freq : int
          Downsampling frequency (default is 50)
    region : str
        The region for which the EEG data should be analyzed.
    workdir : str
        Trove or scratch
    ica: bool
        Whether ICA was used for preprocessing of the EEG data.
    input_type: str
        Miniclips or images
    """
    # -------------------------------------------------------------------------
    # STEP 1 LOAD DATA
    # -------------------------------------------------------------------------

    def load_eeg(sub, img_type, ica, workDir, region, freq, input_type):

        # Define the directory
        if workDir == "scratch":
            workDirFull = "/scratch/agnek95/Unreal/"
        elif workDir == "trove":
            workDirFull = "Z:/Unreal/"

        # will need modifications for images
        if input_type == "miniclips":
            if sub < 10:
                if ica:
                    folderDir = os.path.join(
                        workDirFull,
                        "{}_data".format(input_type)
                        + "/sub-0{}".format(sub)
                        + "/eeg/preprocessing/ica"
                        + "/"
                        + img_type
                        + "/"
                        + region
                        + "/",
                    )

                else:
                    folderDir = os.path.join(
                        workDirFull,
                        "{}_data".format(input_type)
                        + "/sub-0{}".format(sub)
                        + "/eeg/preprocessing/no_ica"
                        + "/"
                        + img_type
                        + "/"
                        + region
                        + "/",
                    )
                fileDir = (
                    "sub-0{}_seq_{}_{}hz_{}_prepared_epochs_redone.npy".format(
                        sub, img_type, freq, region
                    )
                )

            else:
                if ica:
                    folderDir = os.path.join(
                        workDirFull,
                        "{}_data".format(input_type)
                        + "/sub-{}".format(sub)
                        + "/eeg/preprocessing/ica"
                        + "/"
                        + img_type
                        + "/"
                        + region
                        + "/",
                    )
                else:
                    folderDir = os.path.join(
                        workDirFull,
                        "{}_data".format(input_type)
                        + "/sub-{}".format(sub)
                        + "/eeg/preprocessing/no_ica"
                        + "/"
                        + img_type
                        + "/"
                        + region
                        + "/",
                    )
                fileDir = (
                    "sub-{}_seq_{}_{}hz_{}_prepared_epochs_redone.npy".format(
                        sub, img_type, freq, region
                    )
                )

        elif input_type == "images":
            if sub < 10:
                folderDir = os.path.join(
                    workDirFull,
                    "{}_data".format(input_type)
                    + "/prepared"
                    + "/sub-0{}".format(sub)
                    + "/{}/{}/{}hz/".format(img_type, region, freq),
                )
                fileDir = (
                    "sub-0{}_seq_{}_{}hz_{}_prepared_epochs_redone.npy".format(
                        sub, img_type, freq, region
                    )
                )
            else:
                folderDir = os.path.join(
                    workDirFull,
                    "{}_data".format(input_type)
                    + "/prepared"
                    + "/sub-{}".format(sub)
                    + "/{}/{}/{}hz/".format(img_type, region, freq),
                )
                fileDir = (
                    "sub-{}_seq_{}_{}hz_{}_prepared_epochs_redone.npy".format(
                        sub, img_type, freq, region
                    )
                )

        total_dir = os.path.join(folderDir, fileDir)

        # Load EEG data
        data = np.load(total_dir, allow_pickle=True).item()

        eeg_data = data["eeg_data"]
        img_cat = data["img_cat"]

        del data

        # Split into conditions
        if input_type == "miniclips":
            n_conditions = len(np.unique(img_cat))
            _, n_channels, timepoints = eeg_data.shape
            n_trials = img_cat.shape[0]
            n_rep = round(n_trials / n_conditions)

            y_prep = np.zeros(
                (n_conditions, n_rep, n_channels, timepoints), dtype=float
            )

            for condition in range(n_conditions):
                idx = np.where(img_cat == np.unique(img_cat)[condition])
                y_prep[condition, :, :, :] = eeg_data[idx, :, :]

        elif input_type == "images":
            _, n_channels, timepoints = eeg_data.shape
            if img_type == "train":
                n_conditions = 1080
                n_rep = 5
            elif img_type == "test":
                n_conditions = 180
                n_rep = 30
            elif img_type == "val":
                n_conditions = 180
                n_rep = 5
            y_prep = eeg_data.reshape(
                n_conditions, n_rep, n_channels, timepoints
            )

        return y_prep

    if input_type == "miniclips":
        y_test = load_eeg(sub, "test", ica, workDir, region, freq, input_type)

    elif input_type == "images":
        y_test = load_eeg(sub, "test", ica, workDir, region, freq, input_type)

    # -------------------------------------------------------------------------
    # STEP 2 Calculate lower and upper noise ceilings by correlating EEG data
    # -------------------------------------------------------------------------

    # Set some variables
    np.random.seed(42)  # set random seed (for reproduction)
    num_perm = 100
    num_timepoints = y_test.shape[-1]
    lower_ceiling = np.zeros([num_perm, num_timepoints])
    upper_ceiling = np.zeros([num_perm, num_timepoints])
    y_test_reshaped = np.transpose(y_test, axes=[1, 0, 2, 3])

    # Calculate ceilings
    for p in range(num_perm):
        y_test_rand = np.random.permutation(y_test_reshaped)
        for t in range(num_timepoints):
            # correlate vector of activations (num images x 1) for the two divisions,
            # for every channel and timepoint, then avg over channels
            lower_ceiling[p, t] = np.mean(
                vectorized_correlation(
                    np.mean(y_test_rand[:15, :, :, t], axis=0),
                    np.mean(y_test_rand[15:, :, :, t], axis=0),
                )
            )
            upper_ceiling[p, t] = np.mean(
                vectorized_correlation(
                    np.mean(y_test_rand[:15, :, :, t], axis=0),
                    np.mean(y_test_rand[:, :, :, t], axis=0),
                )
            )

    # Average over permutations
    lower_ceiling_avg = np.mean(lower_ceiling, axis=0)
    upper_ceiling_avg = np.mean(upper_ceiling, axis=0)

    # -------------------------------------------------------------------------
    # STEP 3 Save noise ceiling
    # -------------------------------------------------------------------------
    # Save the dictionary

    if input_type == "images":
        saveDir = "Z:/Unreal/images_results/encoding/redone/noise_ceiling/"

    elif input_type == "miniclips":
        saveDir = "Z:/Unreal/Results/Encoding/redone/noise_ceiling/"

    fileDir_lower = (
        str(sub)
        + "_seq_"
        + str(freq)
        + "hz_"
        + region
        + "_lower_noise_ceiling"
        + ".pkl"
    )
    savefileDir = os.path.join(saveDir, fileDir_lower)

    with open(savefileDir, "wb") as f:
        pickle.dump(lower_ceiling_avg, f)

    fileDir_upper = (
        str(sub)
        + "_seq_"
        + str(freq)
        + "hz_"
        + region
        + "_upper_noise_ceiling"
        + ".pkl"
    )
    savefileDir = os.path.join(saveDir, fileDir_upper)

    with open(savefileDir, "wb") as f:
        pickle.dump(upper_ceiling_avg, f)


# -------------------------------------------------------------------------
# STEP 4 Run function
# -------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # add arguments / inputs
    parser.add_argument(
        "-s",
        "--sub",
        default=6,
        type=int,
        metavar="",
        help="subject from (1 to 24)",
    )
    parser.add_argument(
        "-f",
        "--freq",
        default=50,
        type=int,
        metavar="",
        help="downsampling frequency",
    )
    parser.add_argument(
        "-r",
        "--region",
        default="posterior",
        type=str,
        metavar="",
        help="Electrodes to be included, posterior (19) or wholebrain (64)",
    )
    parser.add_argument(
        "-d",
        "--workdir",
        default="trove",
        type=str,
        metavar="",
        help="Working directory type: scratch or trove",
    )
    parser.add_argument(
        "--ica",
        default=True,
        type=bool,
        metavar="",
        help="included artifact rejection using ica or not",
    )
    parser.add_argument(
        "-i",
        "--input_type",
        default="miniclips",
        type=str,
        metavar="",
        help="Font",
    )

    args = parser.parse_args()  # to get values for the arguments

    sub = args.sub
    freq = args.freq
    region = args.region
    workDir = args.workdir
    ica = args.ica
    input_type = args.input_type
    if input_type == "miniclips":
        subjects = [
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
        subjects = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    for sub in subjects:
        noise_ceiling(sub, freq, region, workDir, ica, input_type)
