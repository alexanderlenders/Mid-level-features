#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIME GENERALIZATION ANALYSIS - CONTROL ANALYSIS 8

@author: Alexander Lenders
"""
import os
import numpy as np
import torch
import pickle
import argparse
from sklearn.metrics import r2_score
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
print(project_root)
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
    parse_list,
    load_eeg,
    load_feature_set,
    load_alpha,
    OLS_pytorch,
    vectorized_correlation,
)

def time_gen(
    sub,
    freq,
    region,
    input_type,
    feat_dir,
    save_dir,
    eeg_dir,
    frame,
    feature_names,
):
    """
    Adapted encoding function for performing a time generalization encoding analysis.
    """
    if input_type == "images":
        featuresDir = os.path.join(
            feat_dir,
            f"img_features_frame_{frame}_redone_{len(feature_names)}_features_onehot.pkl",
        )
    elif input_type == "miniclips":
        featuresDir = os.path.join(
            feat_dir,
            f"video_features_avg_frame_redone_{len(feature_names)}.pkl",
        )

    features_dict = dict.fromkeys(feature_names)

    # Device agnostic code: Use gpu if possible, otherwise cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Number of channels
    if region == "wholebrain":
        n_channels = 64
    elif region == "posterior":
        n_channels = 19

    alpha_tp = False  # maybe add to function as a parameter above

    if input_type == "miniclips":
        y_train, timepoints = load_eeg(
            sub, "training", region, freq, input_type, eeg_dir=eeg_dir
        )

    elif input_type == "images":
        y_train, timepoints = load_eeg(
            sub, "train", region, freq, input_type, eeg_dir=eeg_dir
        )

    y_test, _ = load_eeg(
        sub, "test", region, freq, input_type, eeg_dir=eeg_dir
    )

    output_names = ("rmse_score", "correlation", "var_explained")

    # define matrix where to save the values
    regression_features = dict.fromkeys(feature_names)

    for feature in features_dict.keys():
        X_train, _, X_test = load_feature_set(feature, featuresDir)

        if alpha_tp is False:
            alpha = load_alpha(
                sub,
                freq,
                region,
                feature,
                input_type,
                feat_dir=save_dir,
                feat_len=len(feature_names),
            )

        output = dict.fromkeys(output_names)

        rmse = np.zeros((timepoints, timepoints, n_channels))
        corr = np.zeros((timepoints, timepoints, n_channels))
        var_explained = np.zeros((timepoints, timepoints, n_channels))

        for tp in range(timepoints):
            if alpha_tp is True:
                alpha = load_alpha(
                    sub,
                    freq,
                    region,
                    feature,
                    input_type,
                    feat_dir=save_dir,
                    timepoint=tp,
                    feat_len=len(feature_names),
                )

            y_train_tp = y_train[:, :, tp]
            regression = OLS_pytorch(alpha=alpha)

            try:
                regression.fit(X_train, y_train_tp, solver="cholesky")
            except Exception as error:
                print("Attention. Cholesky solver did not work: ", error)
                print("Trying the standard linalg.solver...")
                regression.fit(X_train, y_train_tp, solver="solve")
            
            for tp_test in range(timepoints):
                y_test_tp = y_test[:, :, tp_test]
                prediction = regression.predict(X_test)

                rmse[tp, tp_test, :] = regression.score(entry=X_test, y=y_test_tp)
                corr[tp, tp_test, :] = vectorized_correlation(prediction, y_test_tp)
                var_explained[tp, tp_test, :] = r2_score(y_test_tp, prediction, multioutput="raw_values")


        output["rmse_score"] = rmse
        output["correlation"] = corr
        output["var_explained"] = var_explained
        regression_features[feature] = output

    # -------------------------------------------------------------------------
    # STEP 2.8 Save hyperparameters and scores
    # -------------------------------------------------------------------------
    # Save the dictionary
    saveDir = os.path.join(save_dir, f"{input_type}")

    fileDir = (
        str(sub)
        + "_seq_"
        + str(freq)
        + "hz_"
        + region
        + f"time_gen_encoding_results_averaged_frame_before_mvnn_{len(feature_names)}_features_onehot"
        + ".pkl"
    )

    savefileDir = os.path.join(saveDir, fileDir)

    # Creating the directory if not existing
    if os.path.isdir(os.path.join(saveDir)) == False:  # if not a directory
        os.makedirs(os.path.join(saveDir))

    with open(savefileDir, "wb") as f:
        pickle.dump(regression_features, f)

    return regression_features


# -----------------------------------------------------------------------------
# STEP 1: Initialize variables
# -----------------------------------------------------------------------------

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
        "-i",
        "--input_type",
        default="images",
        type=str,
        metavar="",
        help="Font",
    )

    args = parser.parse_args()  # to get values for the arguments

    config = load_config(args.config_dir, args.config)

    freq = args.freq
    region = args.region
    input_type = args.input_type
    frame = config.getint(args.config, "img_frame")
    save_dir = config.get(args.config, "save_dir")
    feature_names = parse_list(config.get(args.config, "feature_names"))
    eeg_dir = config.get(args.config, "eeg_dir")

    # -------------------------------------------------------------------------
    # STEP 3 Run function
    # -------------------------------------------------------------------------
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
        feat_dir = config.get(args.config, "save_dir_feat_video")
    elif input_type == "images":
        subjects = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        feat_dir = config.get(args.config, "save_dir_feat_img")

    for sub in subjects:
        result = time_gen(
            sub,
            freq,
            region,
            input_type,
            feat_dir=feat_dir,
            save_dir=save_dir,
            eeg_dir=eeg_dir,
            frame=frame,
            feature_names=feature_names,
        )
