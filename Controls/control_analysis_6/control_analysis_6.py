"""
This script implements the code for control analysis 6, i.e. variance partitioning.

@author: Alexander Lenders
"""

import os
import numpy as np
import pickle
import argparse
import sys
from pathlib import Path
from scipy.optimize import minimize
from functools import partial

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
    parse_list,
)


def get_constraints_idea_1(X_hat_tp: np.ndarray):
    """
    This function defines the constraints for idea 1 of the variance partitioning.

    X_hat_tp: np.ndarray, shape (n_models,) with uncorrected R^2 values for each model at a specific time point.

    Returns a constraint dict for the optimization problem.
    """

    # This basically formulates the constraint that the corrected full model's R^2 value
    # must be at least equal to the corrected smaller model's R^2 value
    def constraint(b, X_hat_tp, model_idx):
        return X_hat_tp[-1] + b[-1] - X_hat_tp[model_idx] - b[model_idx]

    co_1 = partial(constraint, X_hat_tp=X_hat_tp, model_idx=0)
    co_2 = partial(constraint, X_hat_tp=X_hat_tp, model_idx=1)
    co_3 = partial(constraint, X_hat_tp=X_hat_tp, model_idx=2)
    co_4 = partial(constraint, X_hat_tp=X_hat_tp, model_idx=3)
    co_5 = partial(constraint, X_hat_tp=X_hat_tp, model_idx=4)
    co_6 = partial(constraint, X_hat_tp=X_hat_tp, model_idx=5)
    co_7 = partial(constraint, X_hat_tp=X_hat_tp, model_idx=6)

    constraints = [
        {"type": "ineq", "fun": c}
        for c in [co_1, co_2, co_3, co_4, co_5, co_6, co_7]
    ]

    return constraints


def get_constraints_idea_2(X_hat_tp):
    """
    This function defines the constraints for idea 2 of the variance partitioning.

    X_hat_tp: np.ndarray, shape (n_models,) with uncorrected R^2 values for each model at a specific time point.
    Returns a constraint dict for the optimization problem.
    """

    def constraint(b, X_hat_tp, model_idx):
        return X_hat_tp[model_idx] + b[model_idx] - X_hat_tp[-1] - b[-1]

    co_1 = partial(constraint, X_hat_tp=X_hat_tp, model_idx=0)
    co_2 = partial(constraint, X_hat_tp=X_hat_tp, model_idx=1)
    co_3 = partial(constraint, X_hat_tp=X_hat_tp, model_idx=2)
    co_4 = partial(constraint, X_hat_tp=X_hat_tp, model_idx=3)
    co_5 = partial(constraint, X_hat_tp=X_hat_tp, model_idx=4)

    constraints = [
        {"type": "ineq", "fun": c} for c in [co_1, co_2, co_3, co_4, co_5]
    ]

    return constraints


def get_unbiased_vp(constraints: dict):
    """
    This functions estimates the bias terms for the variance partitioning
    analysis 1 and returns the estimates.
    """

    def objective(b):
        return np.sum(b**2)  # Minimize l2 norm of the bias parameters

    results = minimize(
        objective,
        x0=np.zeros(8),
        constraints=constraints,
        method="SLSQP",
        options={"disp": False, "ftol": 1e-8, "maxiter": 100000},
    )

    if not results.success:
        raise ValueError("Optimization failed: " + results.message)

    b_updated = results.x

    return b_updated


def get_unbiased_vp_idea_2(constraints):
    """
    This function estimates the bias terms for the variance partitioning
    analysis 2 and returns the estimates.
    """

    def objective(b):
        return np.sum(b**2)  # Minimize l2 norm of the bias parameters

    results = minimize(
        objective,
        x0=np.zeros(6),
        constraints=constraints,
        method="SLSQP",
        options={"disp": False, "ftol": 1e-8, "maxiter": 100000},
    )

    if not results.success:
        raise ValueError("Optimization failed: " + results.message)

    b_updated = results.x

    return b_updated


def c6(
    sub: int,
    workDir: str,
    input_type: str,
    feature_names: list,
    idea: int,
    partial_corr: bool = True,
):
    """
    This function implements variance partitioning. It should be run after
    fitting the encoding models, cf. run_c6.sh.
    """
    workDir = os.path.join(workDir, f"{input_type}")

    identifierDir = f"seq_50hz_posterior_encoding_results_averaged_frame_before_mvnn_{len(feature_names)}_features_onehot.pkl"

    fileDir = os.path.join(workDir, f"{sub}_{identifierDir}")

    # Load the encoding results for the variance partitioning analysis
    encoding_results = np.load(fileDir, allow_pickle=True)

    # Create a list with the keys for the dictionary
    features_keys = [
        f"{', '.join(f)}" if isinstance(f, (tuple, list)) else str(f)
        for f in feature_names
    ]

    # Get the uncorrected R^2 values for each feature
    X_hat_matrix = np.stack(
        [encoding_results[key]["var_explained"] for key in features_keys],
        axis=-1,
    )

    # Build results dictionary
    results = {}
    for i, feature in enumerate(features_keys[:-1]):
        results_tp = np.zeros(
            (X_hat_matrix.shape[0], X_hat_matrix.shape[1])
        )  # Initialize results for each time point
        for tp in range(X_hat_matrix.shape[0]):
            for channel in range(X_hat_matrix.shape[1]):
                X_hat_channel = X_hat_matrix[tp, channel, :]

                if idea == 1:
                    constraints = get_constraints_idea_1(X_hat_channel)
                    b = get_unbiased_vp(constraints)
                    partial_variance = (
                        X_hat_channel[-1] + b[-1] - X_hat_channel[i] - b[i]
                    )
                elif idea == 2:
                    constraints = get_constraints_idea_2(X_hat_channel)
                    b = get_unbiased_vp_idea_2(constraints)
                    partial_variance = (
                        X_hat_channel[i] + b[i] - X_hat_channel[-1] - b[-1]
                    )
                else:
                    raise ValueError(
                        "Unsupported idea version (must be 1 or 2)"
                    )

                results_tp[tp, channel] = partial_variance

        # Check if any results more negative than 1e-3 from 0
        if np.any(results_tp < -1e-3):
            raise ValueError(
                f"Partial variance for feature {feature} at some time points is negative: {results_tp[results_tp < -1e-3]}"
            )
        # Set all results to zero if they are negative
        partial_variance = np.maximum(results_tp, 0)

        print(f"Feature: {feature}, Partial Variance: {partial_variance}")

        if partial_corr:
            partial_correlation = np.sqrt(partial_variance)

        results[feature] = {
            "correlation": (
                partial_correlation if partial_corr else partial_variance
            )
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
        "-id",
        "--idea",
        type=int,
        metavar="",
        help="Font",
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

    idea = args.idea
    input_type = args.input_type

    PARTIAL_CORR = False

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
        print(f"Running variance partitioning for subject {sub}...")
        c6(sub, workDir, input_type, feature_names, idea, PARTIAL_CORR)
