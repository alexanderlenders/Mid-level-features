"""
This script contains the code for control analysis 5, where the image and video annotations are correlated across the whole stimulus set.

@author: Alexander Lenders
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import argparse
from scipy.stats import pearsonr, PermutationMethod

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from EEG.Encoding.utils import (
    load_config,
    load_features,
    parse_list,
)


def c5(
    feat_dir_img: str,
    feat_dir_vid: str,
    save_dir: str,
    feature_names: list,
    frame: int,
    font: str = "Arial",
):
    """
    Function for control analysis 5, in which image and video features are correlated.
    The output is a plot showing the correlation coefficients for each feature.
    """
    featuresDir_img = os.path.join(
        feat_dir_img,
        f"img_features_frame_{frame}_redone_{len(feature_names)}_features_onehot.pkl",
    )

    featuresDir_vid = os.path.join(
        feat_dir_vid,
        f"video_features_avg_frame_redone_{len(feature_names)}.pkl",
    )

    features_dict = dict.fromkeys(feature_names)

    # First step: Load all the features for images and videos
    for feature in features_dict.keys():
        X_train_img, X_val_img, X_test_img = load_features(
            feature, featuresDir_img
        )
        X_train_vid, X_val_vid, X_test_vid = load_features(
            feature, featuresDir_vid
        )

        X_img = np.concatenate((X_train_img, X_val_img, X_test_img), axis=0)
        X_vid = np.concatenate((X_train_vid, X_val_vid, X_test_vid), axis=0)

        # Using permutation tests 
        method = PermutationMethod(
            n_resamples=1000,
            batch=None,
            rng=42
        )

        corr, pval = pearsonr(X_img.flatten(), X_vid.flatten(), method=method)
        # Make sure that maximum correlation is 1
        if corr > 1:
            corr = 1.0
        elif corr < -1:
            corr = -1.0

        features_dict[feature] = (corr, pval)

    # Sort features alphabetically (optional)
    features = sorted(features_dict.keys())
    feature_labels = [
        "Action",
        "Edges",
        "Lighting",
        "Reflectance",
        "Depth",
        "Skeleton",
        "Normals",
    ]
    correlations = [features_dict[feature][0] for feature in features]

    plt.close()
    fig, ax = plt.subplots(figsize=(6, 4.5))  # Adjust size as needed

    # Bar or point plot
    # ax.plot(
    #     features, correlations, "o-", color="black", linewidth=2, markersize=6
    # )  # Point plot
    colormap = plt.colormaps["Set2"]
    colors = [colormap(i) for i in range(len(features))]

    # Sort colors manually
    sorted_indices = [
        6,
        0,
        3,
        4,
        2,
        5,
        1
    ]
    sorted_color_dict = [colors[i] for i in sorted_indices]

    ax.bar(
        features, correlations, color=sorted_color_dict, edgecolor="black"
    )  # Or use a bar plot instead

    for i, feature in enumerate(features):
        corr, pval = features_dict[feature]
        
        if pval < 0.001:
            star = "***"
        elif pval < 0.01:
            star = "**"
        elif pval < 0.05:
            star = "*"
        else:
            star = ""
        
        if star:
            ax.text(
                i, correlations[i] + 0.02,  # slightly above the bar
                star,
                ha='center',
                fontsize=14,
                color='black'
            )

    # Labels and ticks
    ax.set_ylabel("Pearson's r", fontdict={"family": font, "size": 11})
    ax.set_xlabel("Feature", fontdict={"family": font, "size": 11})

    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(
        feature_labels, rotation=45, ha="right", fontsize=11, fontname=font
    )

    ax.set_yticks(
        ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        fontsize=9,
        fontname=font,
    )

    # Clean up plot appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(axis="x", which="both", length=6, width=2)
    ax.tick_params(axis="y", which="both", length=6, width=2)

    plt.tight_layout()
    plt.show()

    # Save figure
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for ext in ["svg", "png"]:
        plot_path = os.path.join(
            save_dir, f"plot_featurewise_correlations.{ext}"
        )
        plt.savefig(plot_path, dpi=300, format=ext, transparent=True)


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
        "-f", "--font", default="Arial", type=str, metavar="", help="Font"
    )

    args = parser.parse_args()  # to get values for the arguments

    config = load_config(args.config_dir, args.config)
    eeg_dir = config.get(args.config, "eeg_dir")
    feat_dir_img = config.get(args.config, "save_dir_feat_img")
    feat_dir_vid = config.get(args.config, "save_dir_feat_video")
    frame = config.getint(args.config, "img_frame")
    feature_names = parse_list(config.get(args.config, "feature_names"))

    # Hardcoded for now
    SAVE_DIR = "/scratch/alexandel91/mid_level_features/results/c5"

    c5(
        feat_dir_img,
        feat_dir_vid,
        SAVE_DIR,
        feature_names,
        frame,
        font=args.font,
    )
