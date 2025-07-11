#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENCODING - DEEP NETS

This script implements the multivariate linear ridge regression for the EEG
data.

@author: Alexander Lenders, Agnessa Karapetian
"""
import os
import numpy as np
import torch
import pickle
import argparse
from utils import load_activation, load_alpha

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
print(project_root)
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
    load_features,
    OLS_pytorch,
    vectorized_correlation,
)


def encoding(input_type, feat_dir, cnn_dir, save_dir, frame):
    """
    Perform encoding (ridge regression) for predicting the unit activations
    in deep nets.

    Parameters
    ----------
    input_type : str
        Images or miniclips

    """
    # -------------------------------------------------------------------------
    # STEP 1 Define Variables
    # -------------------------------------------------------------------------
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
    feature_names = (
        "edges",
        "world_normal",
        "lighting",
        "scene_depth",
        "reflectance",
        "action",
        "skeleton",
    )

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
    
    explained_var_dir = os.path.join(cnn_dir, "pca")
    save_dir = os.path.join(save_dir, input_type)
    act_dir = os.path.join(cnn_dir, "prepared")

    features_dict = dict.fromkeys(feature_names)

    # Device agnostic code: Use gpu if possible, otherwise cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # STEP 2 Loop over all features and save best alpha hyperparameter
    # -------------------------------------------------------------------------
    output_names = (
        "rmse_score",
        "correlation",
        "rmse_average",
        "correlation_average",
    )

    # define matrix where to save the values
    regression_features = dict.fromkeys(feature_names)

    num_layers = len(layers_names)

    for feature in features_dict.keys():
        print(feature)
        X_train, _, X_test = load_features(feature, featuresDir)

        if explained_var_dir:
            alpha_dir_final = os.path.join(save_dir, "weighted")
        else:
            alpha_dir_final = os.path.join(save_dir, "unweighted")


        alpha = load_alpha(input_type, feature=feature, feat_dir=alpha_dir_final
        )

        output = dict.fromkeys(output_names)

        rmse_scores = {}
        corr_scores = {}

        for tp, l in enumerate(layers_names):
            print(l)

            y_train_tp = load_activation("training", l, act_dir)
            y_test_tp = load_activation("test", l, act_dir)

            regression = OLS_pytorch(alpha=alpha)
            try:
                regression.fit(X_train, y_train_tp, solver="cholesky")
            except Exception as error:
                print("Attention. Cholesky solver did not work: ", error)
                print("Trying the standard linalg.solver...")
                regression.fit(X_train, y_train_tp, solver="solve")
            prediction = regression.predict(X_test)
            rmse_score = regression.score(entry=X_test, y=y_test_tp)
            correlation = vectorized_correlation(prediction, y_test_tp)

            rmse_scores[l] = rmse_score
            corr_scores[l] = correlation

        if explained_var_dir:
            rmse_avg_chan = np.zeros((num_layers))
            corr_avg_chan = np.zeros((num_layers))

            for i, layer in enumerate(layers_names):

                explained_var_dir_layer = os.path.join(
                    explained_var_dir, layer, "explained_variance.pkl"
                )

                with open(explained_var_dir_layer, "rb") as file:
                    explained_var = pickle.load(file)

                explained_var = np.array(explained_var["explained_variance"])
                total_variance = np.sum(explained_var)

                rmse_it = rmse_scores[layer]
                corr_it = corr_scores[layer]

                rmse_avg_chan[i] = (
                    np.sum(rmse_it * explained_var) / total_variance
                )
                corr_avg_chan[i] = (
                    np.sum(corr_it * explained_var) / total_variance
                )

        output["rmse_score"] = rmse_scores
        output["correlation"] = corr_scores
        output["rmse_average"] = rmse_avg_chan
        output["correlation_average"] = corr_avg_chan

        regression_features[feature] = output

    # -------------------------------------------------------------------------
    # STEP 3 Save results
    # -------------------------------------------------------------------------
    # Save the dictionary
    fileDir = "encoding_layers_resnet.pkl"

    if explained_var_dir:
        resultsDir = os.path.join(save_dir, "weighted")
    else:
        resultsDir = os.path.join(save_dir, "unweighted")

    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)

    savefileDir = os.path.join(resultsDir, fileDir)

    with open(savefileDir, "wb") as f:
        pickle.dump(regression_features, f)

    return regression_features


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
        "--input_type",
        default="images",
        type=str,
        help="Images or miniclips",
        required=True,
    )

    args = parser.parse_args()  # to get values for the arguments

    config = load_config(args.config_dir, args.config)

    args = parser.parse_args()

    input_type = args.input_type
    frame = config.getint(args.config, "img_frame")
    save_dir = config.get(args.config, "save_dir_cnn")
    
    if input_type == "images":
        feat_dir = config.get(args.config, "feat_dir_cnn_img")
        cnn_dir = config.get(args.config, "cnn_dir_img")
    else:
        feat_dir = config.get(args.config, "feat_dir_cnn_vid")
        cnn_dir = config.get(args.config, "cnn_dir_vid")

    encoding(input_type, feat_dir, cnn_dir, save_dir, frame)

