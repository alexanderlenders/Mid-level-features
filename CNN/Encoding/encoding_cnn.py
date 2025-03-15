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
import sys
from encoding_deepnet_utils import no_pca_load_activation

# Add the project root (two levels up from this script) to `sys.path`
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from src.eeg_encoding.analysis_extraction.analysis.encoding_utils import (
    load_features,
    load_alpha,
    OLS_pytorch,
    vectorized_correlation,
)


def encoding(
    featuresDir: str,
    layer_dir: str,
    saveDir: str,
    alpha_dir: str,
    explained_var_dir: str = None,
):
    """
    Perform encoding (ridge regression) for predicting the unit activations
    in deep nets.

    Parameters
    ----------
    explained_var_dir : str
        Directory where the explained variance for each layer is stored.
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
        "lightning",
        "scene_depth",
        "reflectance",
        "action",
        "skeleton",
    )
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
            alpha_dir_final = os.path.join(alpha_dir, "weighted")
        else:
            alpha_dir_final = os.path.join(alpha_dir, "unweighted")

        alpha = load_alpha(
            alphaDir=alpha_dir_final, feature=feature, eeg=False
        )

        output = dict.fromkeys(output_names)

        rmse_scores = {}
        corr_scores = {}

        for tp, l in enumerate(layers_names):
            print(l)

            y_train_tp = no_pca_load_activation("training", layer_dir, l)
            y_test_tp = no_pca_load_activation("test", layer_dir, l)

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
        resultsDir = os.path.join(saveDir, "weighted")
    else:
        resultsDir = os.path.join(saveDir, "unweighted")

    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)

    savefileDir = os.path.join(resultsDir, fileDir)

    with open(savefileDir, "wb") as f:
        pickle.dump(regression_features, f)

    return regression_features


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
    parser.add_argument(
        "--alpha_dir",
        type=str,
        help="Directory where the best alpha hyperparameters are stored.",
        required=True,
    )

    args = parser.parse_args()
    layer_dir = args.layer_dir
    featuresDir = args.featuresDir
    saveDir = args.saveDir
    explained_var_dir = args.explained_var_dir
    alpha_dir = args.alpha_dir

    encoding(
        featuresDir=featuresDir,
        layer_dir=layer_dir,
        saveDir=saveDir,
        explained_var_dir=explained_var_dir,
        alpha_dir=alpha_dir,
    )
