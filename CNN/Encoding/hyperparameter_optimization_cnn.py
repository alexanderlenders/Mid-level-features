#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPERPARAMETER TUNING FOR ENCODING - DEEP NETS

This script tunes the hyperparameter alpha (<=> lambda) for ridge regression.

@author: Alexander Lenders, Agnessa Karapetian
"""
import os
import numpy as np
import torch
import pickle
import argparse
import sys
from encoding_deepnet_utils import no_pca_load_activation

# Add the project root (two levels up from this script) to `sys.path`
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from src.eeg_encoding.analysis_extraction.analysis.encoding_utils import (
    load_features,
    OLS_pytorch,
    vectorized_correlation,
)


def hyperparameter_tuning(
    layer_dir: str,
    featuresDir: str,
    saveDir: str,
    explained_var_dir: str = None,
):
    """
    Hyperparameter tuning for ridge regression for the DNN encoding.

    Parameters
    ----------
    layer_dir : str
        Directory where the activations for each layer are stored.
    explained_var_dir : str
        Directory where the explained variance for each layer is stored.
    featuresDir : str
        Directory where the features are stored.
    saveDir : str
        Directory where the hyperparameters are stored.
    """
    # set up some variables and paths
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
        "lightning",
        "scene_depth",
        "reflectance",
        "action",
        "skeleton",
    )
    features_dict = dict.fromkeys(feature_names)
    num_layers = len(layers_names)

    # Device agnostic code: Use gpu if possible, otherwise cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Hyperparameter space
    alpha_space = np.logspace(-5, 10, 10)

    output_names = (
        "rmse_score",
        "correlation",
        "best_alpha_rmse",
        "best_alpha_corr",
        "best_alpha_a_rmse",
        "best_alpha_a_corr",
    )

    # define matrix where to save the values
    regression_features = dict.fromkeys(feature_names)

    for feature in features_dict.keys():
        print(feature)
        X_train, X_val, _ = load_features(feature, featuresDir)
        output = dict.fromkeys(output_names)

        rmse_scores = {}
        corr_scores = {}

        for tp, l in enumerate(layers_names):
            print(l)

            y_train_tp = no_pca_load_activation("training", layer_dir, l)
            y_val_tp = no_pca_load_activation("validation", layer_dir, l)

            print(y_train_tp.shape)
            print(y_val_tp.shape)

            rmse = np.zeros((1, len(alpha_space), y_train_tp.shape[1]))
            corr = np.zeros((1, len(alpha_space), y_train_tp.shape[1]))

            for a in range(len(alpha_space)):
                alpha = alpha_space[a]
                regression = OLS_pytorch(alpha=alpha)
                try:
                    regression.fit(X_train, y_train_tp, solver="cholesky")
                except Exception as error:
                    print("Attention. Cholesky solver did not work: ", error)
                    print("Trying the standard linalg.solver...")
                    regression.fit(X_train, y_train_tp, solver="solve")
                prediction = regression.predict(entry=X_val)
                rmse_score = regression.score(entry=X_val, y=y_val_tp)
                correlation = vectorized_correlation(prediction, y_val_tp)

                rmse[0, a, :] = rmse_score
                corr[0, a, :] = correlation

            rmse_scores[l] = rmse
            corr_scores[l] = corr

        if explained_var_dir:

            rmse_avg_chan = np.zeros((num_layers, len(alpha_space)))
            corr_avg_chan = np.zeros((num_layers, len(alpha_space)))

            for i, layer in enumerate(layers_names):

                explained_var_dir_layer = os.path.join(
                    explained_var_dir, layer, "explained_variance.pkl"
                )

                with open(explained_var_dir_layer, "rb") as file:
                    explained_var = pickle.load(file)

                explained_var = np.array(explained_var["explained_variance"])
                total_variance = np.sum(explained_var)

                rmse_it = rmse_scores[layer][0]
                corr_it = corr_scores[layer][0]

                print(layer)
                print(explained_var.shape)
                print(rmse_it.shape)
                print(corr_it.shape)

                rmse_avg_chan[i, :] = (
                    np.sum(rmse_it * explained_var, axis=1) / total_variance
                )
                corr_avg_chan[i, :] = (
                    np.sum(corr_it * explained_var, axis=1) / total_variance
                )

        best_alpha_rmse_idx = rmse_avg_chan.argmin(axis=1)
        best_alpha_corr_idx = corr_avg_chan.argmax(axis=1)

        best_alpha_rmse = np.zeros((num_layers, 1))
        best_alpha_corr = np.zeros((num_layers, 1))

        # best alpha for each tp
        for tp in range(num_layers):
            best_alpha_rmse[tp] = alpha_space[best_alpha_rmse_idx[tp]]
            best_alpha_corr[tp] = alpha_space[best_alpha_corr_idx[tp]]

        # best alpha on average
        average_rmse = np.mean(rmse_avg_chan, axis=0)
        average_corr = np.mean(corr_avg_chan, axis=0)
        best_alpha_a_rmse_idx = average_rmse.argmin(axis=0)
        best_alpha_a_corr_idx = average_corr.argmax(axis=0)
        best_alpha_a_rmse = alpha_space[best_alpha_a_rmse_idx]
        best_alpha_a_corr = alpha_space[best_alpha_a_corr_idx]

        output["rmse_score"] = rmse
        output["correlation"] = corr
        output["best_alpha_rmse"] = best_alpha_rmse
        output["best_alpha_corr"] = best_alpha_corr
        output["best_alpha_a_rmse"] = best_alpha_a_rmse
        output["best_alpha_a_corr"] = best_alpha_a_corr

        regression_features[feature] = output

        # Save hyperparameters and scores
        if explained_var_dir:
            resultsDir = os.path.join(saveDir, "weighted")
        else:
            resultsDir = os.path.join(saveDir, "unweighted")

        if not os.path.exists(resultsDir):
            os.makedirs(resultsDir)

        fileDir = "hyperparameter_tuning_resnet.pkl"

        savefileDir = os.path.join(resultsDir, fileDir)

        with open(savefileDir, "wb") as f:
            pickle.dump(regression_features, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--layer_dir",
        type=str,
        help="Directory where the activations for each layer are stored.",
        required=True,
    )
    parser.add_argument(
        "--featuresDir",
        type=str,
        help="Directory where the features are stored.",
        required=True,
    )
    parser.add_argument(
        "--saveDir",
        type=str,
        help="Directory where the hyperparameters are stored.",
        required=True,
    )
    parser.add_argument(
        "--explained_var_dir",
        type=str,
        help="Directory where the explained variance for each layer is stored.",
        required=False,
    )

    args = parser.parse_args()
    layer_dir = args.layer_dir
    featuresDir = args.featuresDir
    saveDir = args.saveDir
    explained_var_dir = args.explained_var_dir

    hyperparameter_tuning(
        layer_dir=layer_dir,
        featuresDir=featuresDir,
        saveDir=saveDir,
        explained_var_dir=explained_var_dir,
    )
