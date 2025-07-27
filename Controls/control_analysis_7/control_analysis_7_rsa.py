"""
This script contains the code for control analysis 7, which creates a dissimilarity matrix between different feature representations across the whole stimulus set using Kriegeskorte-style (Kriegeskorte et al., 2008) Representational Similarity Analysis (RSA).
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
import argparse
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr

# Project root and path setup
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
    load_features,
)


def compute_rdm_kriegeskorte(
    X: np.ndarray, strategy: str = "spearman"
) -> np.ndarray:
    """
    For different approaches/strategies on how to compute the RDM, see:
    - https://arxiv.org/abs/2411.14633 - Bo et al. (2024)
    - Kriegeskorte et al. (2008)

    """
    if strategy == "khosla":
        S = X @ X.T
        J = np.ones_like(S)
        return J - S
    elif strategy == "spearman":
        X_ranked = np.apply_along_axis(rankdata, 1, X)
        n = X.shape[0]
        RDM = np.zeros(
            (n, n)
        )  # Dissimilarity of 0 for diagonal (as identical)
        for i in range(n):
            for j in range(i + 1, n):
                rho, _ = pearsonr(X_ranked[i], X_ranked[j])
                dist = 1 - rho
                RDM[i, j] = RDM[j, i] = dist
        return RDM
    elif strategy == "pearson":
        n = X.shape[0]
        RDM = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r, _ = pearsonr(X[i], X[j])
                dist = 1 - r
                RDM[i, j] = RDM[j, i] = dist
        return RDM
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Supported strategies: 'khosla', 'spearman', 'pearson'."
        )


def rdm_to_vector_upper(rdm: np.ndarray) -> np.ndarray:
    """Extract upper triangle (without diagonal) from a matrix."""
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def rsa_kriegeskorte(
    X: np.ndarray, Y: np.ndarray, strategy: str = "spearman"
) -> float:
    """Computes RSA(X, Y) = τ(J - X Xᵗ, J - Y Yᵗ)"""
    RDM_X = compute_rdm_kriegeskorte(X, strategy=strategy)
    RDM_Y = compute_rdm_kriegeskorte(Y, strategy=strategy)

    vec_X = rdm_to_vector_upper(RDM_X)
    vec_Y = rdm_to_vector_upper(RDM_Y)

    if strategy == "spearman":
        # Use Spearman's rank correlation for RSA
        corr, _ = spearmanr(vec_X, vec_Y)
    elif strategy == "pearson":
        # Use Pearson correlation for RSA
        corr, _ = pearsonr(vec_X, vec_Y)
    elif strategy == "kendall":
        # Use Kendall's tau for RSA
        corr, _ = kendalltau(vec_X, vec_Y)
    return corr


def c7(
    feat_dir,
    input_type,
    save_dir: str,
    feature_names,
    frame,
    feature_graphs,
    font: str = "Arial",
    strategy: str = "spearman",
):
    """
    Perform RSA analysis between mid-level features using Kendall's τ.
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

    # Sort feature names and display names together
    sorted_lists = sorted(
        zip(feature_names, feature_graphs), key=lambda x: x[0]
    )
    feature_names, feature_graphs = zip(*sorted_lists)
    feature_names = list(feature_names)
    feature_graphs = list(feature_graphs)

    # Load features
    feature_arrays = []
    for feature in feature_names:
        X_train, X_val, X_test = load_features(feature, featuresDir)
        X_concat = np.concatenate((X_train, X_val, X_test), axis=0)
        feature_arrays.append(X_concat)

    # Compute RSA matrix
    num_features = len(feature_arrays)
    rsa_matrix = np.zeros((num_features, num_features))

    for i in range(num_features):
        for j in range(num_features):
            if i == j:
                rsa_matrix[i, j] = np.nan
            else:
                corr = rsa_kriegeskorte(
                    feature_arrays[i], feature_arrays[j], strategy=strategy
                )
                rsa_matrix[i, j] = corr

    # Plot matrix
    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
    if strategy == "spearman":
        label = "Spearman's ρ (RSA)"
    elif strategy == "pearson":
        label = "Pearson's r (RSA)"
    elif strategy == "kendall":
        label = "Kendall's τ (RSA)"

    heatmap = sns.heatmap(
        rsa_matrix,
        ax=ax_corr,
        cmap="viridis",
        square=True,
        cbar_kws={"label": label},
        xticklabels=feature_graphs,
        yticklabels=feature_graphs,
        mask=np.isnan(rsa_matrix),
    )

    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(14)
    cbar.ax.tick_params(labelsize=12)

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

    plt.tight_layout()
    plt.show()

    # Save output
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for ext in ["svg", "png"]:
        out_path = os.path.join(
            save_dir, f"rsa_feature_matrix_{input_type}.{ext}"
        )
        fig_corr.savefig(out_path, dpi=300, format=ext, transparent=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_dir",
        type=str,
        required=True,
        help="Directory to the configuration file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration section name.",
    )
    parser.add_argument(
        "-i",
        "--input_type",
        default="images",
        type=str,
        metavar="",
        help="Specify input type: 'images' or 'videos'",
    )
    parser.add_argument(
        "-f",
        "--font",
        default="Arial",
        type=str,
        metavar="",
        help="Font for plots",
    )

    args = parser.parse_args()
    config = load_config(args.config_dir, args.config)

    input_type = args.input_type
    feat_dir = (
        config.get(args.config, "save_dir_feat_img")
        if input_type == "images"
        else config.get(args.config, "save_dir_feat_video")
    )

    frame = config.getint(args.config, "img_frame")

    # Feature list
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

    SAVE_DIR = "/scratch/alexandel91/mid_level_features/results/c7_2"

    c7(
        feat_dir,
        input_type,
        SAVE_DIR,
        feature_names,
        frame,
        feature_names_graph,
        font=args.font,
    )
