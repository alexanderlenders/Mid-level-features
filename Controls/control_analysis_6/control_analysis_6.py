"""
This script implements variance partitioning.
"""
import os
import numpy as np
import pickle
import argparse
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
print(project_root)
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
    parse_list,
)

def c6(sub: int, workDir: str, input_type: str, feature_names: list, idea: int, partial_corr: bool = True):
    """
    This function implements variance partitioning. It should be run after
    fitting the encoding models, cf. run_c6.sh.
    """
    workDir = os.path.join(workDir, f"{input_type}")

    identifierDir = f"seq_50hz_posterior_encoding_results_averaged_frame_before_mvnn_{len(feature_names)}_features_onehot.pkl"

    fileDir = os.path.join(workDir, f"{sub}_{identifierDir}")
    encoding_results = np.load(fileDir, allow_pickle=True)

    # Get correlation results for the full model
    full_model_var = encoding_results[feature_names[-1]]["var_explained"]

    # Average over all channels
    full_model_var_averaged = np.mean(full_model_var, axis=1)

    # Initialize results dictionary
    results = {}
    for feature in feature_names[:-1]:
        # Get correlation results for the current feature set
        feature_var = encoding_results[feature]["var_explained"]

        # Average over all channels
        feature_var_averaged = np.mean(feature_var, axis=1)

        # Calculate variance explained by the current feature set
        if idea == 1:
            partial_variance = full_model_var_averaged - feature_var_averaged
        elif idea == 2: # Here the feature set contains more features than the model including only low and high-level features
            partial_variance = feature_var_averaged - full_model_var_averaged

        # Make sure that the variance is non-negative (by definition)
        partial_variance = np.maximum(partial_variance, 0)

        if partial_corr:
            partial_correlation = np.sqrt(partial_variance)

        # Store the results
        # We save this as correlation for consistency with the original code
        # This allows to reuse the following scripts
        results[feature] = {}
        results[feature]["correlation"] = partial_correlation if partial_corr else partial_variance
    
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

    PARTIAL_CORR = True

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
        print(f"Variance partitioning for subject {sub} completed.")

    

