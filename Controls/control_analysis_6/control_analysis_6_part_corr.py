"""
This script implements the code for control analysis 6, i.e. partial correlation analysis.

For more information, on how the partial correlation is computed, cf. https://en.wikipedia.org/wiki/Partial_correlation.

@author: Alexander Lenders
"""

import os
import numpy as np
import pickle
import argparse
import sys
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
    parse_list,
)

def regress_out_ols(y, X_control):
    """Regress y on X_control using OLS, return residuals"""
    model = LinearRegression(fit_intercept=True)
    X_control = X_control.reshape(-1, 1)  # Ensure X_control is 2D
    model.fit(X_control, y)  # control is 1D here, reshape needed
    y_pred = model.predict(X_control)
    residuals = y - y_pred
    return residuals

def c6(
    sub: int,
    workDir: str,
    input_type: str,
    feature_names: list,
):
    """
    This function implements the computation of the partial correlations.
    """
    workDir = os.path.join(workDir, f"{input_type}")

    identifierDir = f"seq_50hz_posterior_encoding_results_averaged_frame_before_mvnn_{len(feature_names)}_features_onehot.pkl"

    fileDir = os.path.join(workDir, f"{sub}_{identifierDir}")

    # Load the encoding results for the partial correlation analysis
    encoding_results = np.load(fileDir, allow_pickle=True)

    # Create a list with the keys for the dictionary
    features_keys = [
        f"{', '.join(f)}" if isinstance(f, (tuple, list)) else str(f)
        for f in feature_names
    ]

    full_model_res = encoding_results[features_keys[-1]]["residuals"]
    full_model_corr = encoding_results[features_keys[-1]]["correlation"]
    # Average correlation across channels
    corr = np.mean(full_model_corr, axis=1)

    # Plot correlation of full model
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(corr, label='Full Model Correlation', color='blue')
    plt.title(f'Subject {sub} - Full Model Correlation')
    plt.xlabel('Time')
    plt.ylabel('Correlation')
    plt.legend()
    # Save in current directory
    plt.savefig(os.path.join(f"./full_model_correlation_subject_{sub}.png"))


    y_true = encoding_results[features_keys[-1]]["y_true"]
    # Swap axes of y_true
    y_true = np.swapaxes(y_true, 1, 2)  # Swap time and channel axes
    full_model_res = np.swapaxes(full_model_res, 0, 1)  # Swap time and channel axes

    y_true = np.mean(y_true, axis=2)  # Average across channels
    full_model_res = np.mean(full_model_res, axis=2)  # Average across

    predictions = y_true - full_model_res  # Compute the predictions

    # Compute mean prediction of y_true (baseline model)
    y_true_mean = np.mean(y_true, axis=0)

    # Plot true values, full model predictions and mean prediction
    import matplotlib.pyplot as plt

    for i in range(20):
        plt.figure(figsize=(10, 5))
        plt.plot(y_true_mean, label='Mean True Values', color='blue')
        plt.plot(predictions[i], label='Predictions', color='green')
        plt.plot(y_true[i], label='True Values', color='red')
        plt.title(f'Subject {sub} - Mean True Values vs Predictions')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.legend()
        # Save in current directory
        plt.savefig(os.path.join(f"./partial_correlation_subject_{i}.png"))
    
    raise ValueError

    results = {feature: None for feature in features_keys[:-1]}

    for feature in features_keys[:-1]:
        res = np.zeros((full_model_res.shape[0], full_model_res.shape[2]))
        for tp in range(full_model_res.shape[0]):
            for channel in range(full_model_res.shape[2]):

                feat_res = encoding_results[feature]["residuals"]

                # Regress the residuals of the full model on the residuals of the feature
                res_model = regress_out_ols(full_model_res[tp, :, channel], y_true[:, channel, tp] - feat_res[tp, :, channel])
                # Regress the true values on the residuals of the feature
                y_true_res = regress_out_ols(y_true[:, channel, tp], y_true[:, channel, tp] - feat_res[tp, :, channel])

                # Compute the Pearson correlation between the residuals
                corr, _ = pearsonr(res_model, y_true_res)

                res[tp, channel] = corr

        
        print(np.mean(res, axis=1))
        
        results[feature] = {
            "correlation": res
        }


    # Save the results
    feature_names = feature_names[:-1]  # Exclude the full feature set
    identifierDir = f"seq_50hz_posterior_encoding_results_averaged_frame_before_mvnn_{len(feature_names)}_features_onehot.pkl"

    fileDir = os.path.join(workDir, f"{sub}_{identifierDir}")

    with open(fileDir, "wb") as f:
        pickle.dump(results, f)


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
        "-i",
        "--input_type",
        default="images",
        type=str,
        metavar="",
        help="Font",
    )

    args = parser.parse_args()  # to get values for the arguments
    config = load_config(args.config_dir, args.config)
    workDir = config.get(args.config, "save_dir")
    feature_names = parse_list(config.get(args.config, "feature_names"))
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

    for sub in list_sub:
        print(f"Running partial correlation analysis for subject {sub}...")
        c6(sub, workDir, input_type, feature_names)
