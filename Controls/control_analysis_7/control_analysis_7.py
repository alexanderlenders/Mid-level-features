"""
This script contains the code for control analysis 7, which creates a correlation matrix between the different features.

@author: Alexander Lenders
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
import argparse
project_root = Path(__file__).resolve().parents[2]
print(project_root)
sys.path.append(str(project_root))
from EEG.Encoding.utils import vectorized_correlation, load_config, load_features, parse_list

def c7(feat_dir, input_type, save_dir: str, feature_names, frame, font: str = "Arial"):
    """
    Function for control analysis 7.
    The output is a correlation matrix plot showing the correlation coefficients for each feature.
    """
    if input_type == "images":
        featuresDir = os.path.join(
                feat_dir,
                f"img_features_frame_{frame}_redone_{len(feature_names)}_features_onehot.pkl",
            )
    else:
        featuresDir = os.path.join(
                feat_dir,
                f"video_features_avg_frame_redone_{len(feature_names)}.pkl",
            )
        
    feature_arrays = [] 

    # First step: Load all the features for images and videos
    for feature in sorted(feature_names):
        X_train, X_val, X_test = load_features(feature, featuresDir)
        X_concat = np.concatenate((X_train, X_val, X_test), axis=0)

        feature_arrays.append(X_concat)
    
    # Create correlation matrix
    num_features = len(feature_names)
    correlation_matrix = np.zeros((num_features, num_features))

    for i in range(num_features):
        for j in range(i, num_features):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                corr = vectorized_correlation(feature_arrays[i], feature_arrays[j])
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr

    # Plot with seaborn
    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        correlation_matrix,
        ax=ax_corr,
        cmap="jet",
        square=True,
        cbar_kws={"label": "Pearson's r"},
        xticklabels=sorted(feature_names),
        yticklabels=sorted(feature_names),
    )

    ax_corr.set_xticklabels(sorted(feature_names), rotation=45, ha='right', fontsize=9, fontname=font)
    ax_corr.set_yticklabels(sorted(feature_names), fontsize=9, fontname=font)

    # Title and labels
    ax_corr.set_title("Feature-Feature Correlation Matrix", fontsize=12, fontname=font)

    # Layout and save
    plt.tight_layout()
    plt.show()

    for ext in ["svg", "png"]:
        corr_path = os.path.join(save_dir, f"plot_feature_correlation_matrix_{input_type}.{ext}")
        fig_corr.savefig(corr_path, dpi=300, format=ext, transparent=True)


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
    parser.add_argument(
    "-f", "--font", default="Arial", type=str, metavar="", help="Font"
    )

    args = parser.parse_args()  # to get values for the arguments

    config = load_config(args.config_dir, args.config)
    input_type = args.input_type
    if input_type == "images":
        feat_dir = config.get(args.config, "save_dir_feat_img")
    else:
        feat_dir = config.get(args.config, "save_dir_feat_video")

    frame = config.getint(args.config, "img_frame")
    feature_names = parse_list(config.get(args.config, "feature_names"))

    SAVE_DIR = "/scratch/alexandel91/mid_level_features/results/c7"

    c7(feat_dir, input_type, SAVE_DIR, feature_names, frame, font=args.font)


