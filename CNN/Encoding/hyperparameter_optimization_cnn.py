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
from EEG.Encoding.utils import load_features, OLS_pytorch, vectorized_correlation
from utils import load_activation


def hyperparameter_tuning(input_type, feat_dir="/home/agnek95/Encoding-midlevel-features/Results/Encoding/", exp_var_dir="/scratch/agnek95/Unreal/CNN_activations_redone", save_dir="/home/agnek95/Encoding-midlevel-features/Results/CNN_Encoding/"):
    """
    Hyperparameter tuning for ridge regression for the DNN encoding.

    Parameters
    ----------
    input_type : str
        Images or miniclips

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
        "lighting",
        "scene_depth",
        "reflectance",
        "action",
        "skeleton",
    )

    if input_type == "images":
        featuresDir = os.path.join(feat_dir, "images/7_features/img_features_frame_20_redone_7features_onehot.pkl")

        explained_var_dir = os.path.join(exp_var_dir, "2D_ResNet18/pca_90_percent/pca/")

        saveDir = os.path.join(save_dir, "2D_ResNet18/pca_90_percent/hyperparameters/")

    elif input_type == "miniclips":
        featuresDir = os.path.join(feat_dir, "miniclips/7_features/video_features_avg_frame_redone.pkl")

        explained_var_dir = os.path.join(exp_var_dir, "3D_ResNet18/pca_90_percent/pca/")

        saveDir = os.path.join(save_dir, "3D_ResNet18/pca_90_percent/hyperparameters/")

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

            y_train_tp = load_activation(input_type, "training", l)
            y_val_tp = load_activation(input_type, "validation", l)

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
        "--input_type",
        default="images",
        type=str,
        help="Images or miniclips",
        required=True,
    )
    parser.add_argument(
        "--feat_dir",
        default="/home/agnek95/Encoding-midlevel-features/Results/Encoding/",
        type=str,
        help="Directory where the features are stored",
    )
    parser.add_argument(
        "--exp_var_dir",
        default="/scratch/agnek95/Unreal/CNN_activations_redone",
        type=str,
        help="Directory where the explained variance is stored",
    )
    parser.add_argument(
        "--save_dir",
        default="/home/agnek95/Encoding-midlevel-features/Results/CNN_Encoding/",
        type=str,
        help="Directory where the results will be saved",
    )

    args = parser.parse_args()

    input_type = args.input_type
    feat_dir = args.feat_dir
    exp_var_dir = args.exp_var_dir
    save_dir = args.save_dir
    
    hyperparameter_tuning(input_type, feat_dir, exp_var_dir, save_dir)
