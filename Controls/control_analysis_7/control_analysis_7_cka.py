"""
Control Analysis 7 (CKA):
Computes similarity between feature representations using
Linear Centered Kernel Alignment (CKA) as defined in:
Kornblith et al. (2019), Gretton et al. (2005)

@author: Alexander Lenders
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
import argparse

# Set project root and path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
    load_features,
)


# ----------------------------- CKA Functions -----------------------------

def center_gram_matrix(K: np.ndarray) -> np.ndarray:
    """Centers a kernel (Gram) matrix."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def hsic(Kc: np.ndarray, Lc: np.ndarray) -> float:
    """Hilbert-Schmidt Independence Criterion (HSIC)"""
    return np.sum(Kc * Lc)

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes linear CKA between two feature matrices.
    
    Args:
        X: [M x NX]
        Y: [M x NY]
    Returns:
        Scalar CKA similarity score
    """
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)

    K = X @ X.T
    L = Y @ Y.T

    Kc = center_gram_matrix(K)
    Lc = center_gram_matrix(L)

    hsic_xy = hsic(Kc, Lc)
    hsic_xx = hsic(Kc, Kc)
    hsic_yy = hsic(Lc, Lc)

    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


# ----------------------------- Main Function -----------------------------

def c7_cka(
    feat_dir: str,
    input_type: str,
    save_dir: str,
    feature_names: list,
    frame: int,
    feature_labels: list,
    font: str = "Arial",
):
    """Main function for control analysis 7 using CKA."""

    if input_type == "images":
        features_path = os.path.join(
            feat_dir,
            f"img_features_frame_{frame}_redone_7_features_onehot.pkl",
        )
    else:
        features_path = os.path.join(
            feat_dir,
            "video_features_avg_frame_redone_7.pkl",
        )

    # Sort features alphabetically for consistent layout
    feature_names, feature_labels = zip(*sorted(zip(feature_names, feature_labels)))
    feature_names, feature_labels = list(feature_names), list(feature_labels)

    # Load all feature matrices
    feature_arrays = []
    for feature in feature_names:
        X_train, X_val, X_test = load_features(feature, features_path)
        X = np.concatenate([X_train, X_val, X_test], axis=0)
        feature_arrays.append(X)

    num_features = len(feature_arrays)
    cka_matrix = np.zeros((num_features, num_features))

    # Compute CKA for each pair of features
    for i in range(num_features):
        for j in range(i, num_features):
            if i == j:
                cka_matrix[i, j] = np.nan  # No self-comparison
            else:
                sim = linear_cka(feature_arrays[i], feature_arrays[j])
                cka_matrix[i, j] = sim
                cka_matrix[j, i] = sim

    # Plot CKA similarity matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cka_matrix,
        ax=ax,
        cmap="viridis",
        square=True,
        cbar_kws={"label": "Linear CKA Similarity"},
        xticklabels=feature_labels,
        yticklabels=feature_labels,
        mask=np.isnan(cka_matrix),
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(14)
    cbar.ax.tick_params(labelsize=12)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

    ax.set_xticklabels(feature_labels, rotation=45, ha="right", fontsize=12, fontname=font)
    ax.set_yticklabels(feature_labels, rotation=45, fontsize=12, fontname=font)

    plt.tight_layout()
    plt.show()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    for ext in ["svg", "png"]:
        out_path = os.path.join(save_dir, f"cka_feature_similarity_matrix_{input_type}.{ext}")
        fig.savefig(out_path, dpi=300, format=ext, transparent=True)


# ----------------------------- Main Script -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, required=True, help="Path to config file directory.")
    parser.add_argument("--config", type=str, required=True, help="Config name.")
    parser.add_argument("-i", "--input_type", default="images", type=str, metavar="", help="Input type: images or video.")
    parser.add_argument("-f", "--font", default="Arial", type=str, metavar="", help="Font for plot.")

    args = parser.parse_args()

    config = load_config(args.config_dir, args.config)
    input_type = args.input_type

    if input_type == "images":
        feat_dir = config.get(args.config, "save_dir_feat_img")
    else:
        feat_dir = config.get(args.config, "save_dir_feat_video")

    frame = config.getint(args.config, "img_frame")

    feature_names = [
        "edges",
        "reflectance",
        "lighting",
        "world_normal",
        "scene_depth",
        "skeleton",
        "action",
    ]
    feature_names_graph = [
        "Edges",
        "Reflectance",
        "Lighting",
        "Normals",
        "Depth",
        "Skeleton",
        "Action",
    ]

    SAVE_DIR = "/scratch/alexandel91/mid_level_features/results/c7_3"

    c7_cka(
        feat_dir=feat_dir,
        input_type=input_type,
        save_dir=SAVE_DIR,
        feature_names=feature_names,
        frame=frame,
        feature_labels=feature_names_graph,
        font=args.font,
    )
