"""
This script contains the code for control analysis 7, which creates a correlation matrix between the different features. In this script, a "naive correlation" approach is used, i.e. Pearson's r is computed between the features across the whole stimulus set.

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
sys.path.append(str(project_root))
from EEG.Encoding.utils import (
    vectorized_correlation,
    load_config,
    load_features,
)


def c7(
    feat_dir: str,
    input_type: str,
    feature_names: list,
    save_dir: str,
    frame: int,
    feature_graphs: list,
    font: str = "Arial",
):
    """
    Function for control analysis 7.
    The output is a correlation matrix plot showing the correlation coefficients for each feature.
    """
    if input_type == "images":
        featuresDir = os.path.join(
            feat_dir,
            f"img_features_frame_{frame}_redone_7_features_onehot.pkl",
        )
    else:
        featuresDir = os.path.join(
            feat_dir,
            "video_features_avg_frame_redone_7.pkl",
        )

    feature_arrays = []

    sorted_lists = sorted(
        zip(feature_names, feature_graphs), key=lambda x: x[0]
    )

    # Unzip the sorted pairs back into two lists
    feature_names, feature_graphs = zip(*sorted_lists)
    feature_names = list(feature_names)
    feature_graphs = list(feature_graphs)

    # First step: Load all the features for images and videos
    for feature in feature_names:
        X_train, X_val, X_test = load_features(feature, featuresDir)
        X_concat = np.concatenate((X_train, X_val, X_test), axis=0)
        feature_arrays.append(X_concat)

    # Create correlation matrix
    num_features = len(feature_names)
    correlation_matrix = np.zeros((num_features, num_features))

    for i in range(num_features):
        for j in range(i, num_features):
            if i == j:
                correlation_matrix[i, j] = np.nan  # Diagonal elements are NaN
            else:
                corr = vectorized_correlation(
                    feature_arrays[i], feature_arrays[j]
                )
                corr = np.mean(corr)
                # Make sure that maximum correlation is 1
                if corr > 1:
                    corr = 1.0
                elif corr < -1:
                    corr = -1.0
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr

    # Plot with seaborn
    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))

    heatmap = sns.heatmap(
        correlation_matrix,
        ax=ax_corr,
        cmap="viridis",
        square=True,
        cbar_kws={"label": "Pearson's r"},
        xticklabels=feature_graphs,
        yticklabels=feature_graphs,
        mask=np.isnan(correlation_matrix),
    )

    # Increase colorbar label font size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(14)  # Label font size
    cbar.ax.tick_params(labelsize=12)  # Tick font size

    for spine in ax_corr.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")

    ax_corr.set_xticklabels(
        feature_graphs,
        rotation=45,
        ha="right",
        fontsize=12,
        fontname=font,
    )
    ax_corr.set_yticklabels(
        feature_graphs, fontsize=12, rotation=45, fontname=font
    )

    # Layout and save
    plt.tight_layout()
    plt.show()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for ext in ["svg", "png"]:
        corr_path = os.path.join(
            save_dir, f"plot_feature_correlation_matrix_{input_type}.{ext}"
        )
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

    # We should exclude action (as one cannot correlate action as it has only 6 values)
    feature_names = [
        "edges",
        "reflectance",
        "lighting",
        "world_normal",
        "scene_depth",
    ]
    feature_names_graph = [
        "Edges",
        "Reflectance",
        "Lighting",
        "Normals",
        "Depth",
    ]

    SAVE_DIR = "/scratch/alexandel91/mid_level_features/results/c7_1"

    c7(
        feat_dir,
        input_type,
        feature_names,
        SAVE_DIR,
        frame,
        feature_names_graph,
        font=args.font,
    )
