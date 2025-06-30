#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANNOTATION PREPARATION AND PCA - VIDEOS - UNREAL ENGINE

This script prepares the annotations from the Unreal Engine by extracting the low-, mid-, and
high-level features from them. For the low-level feature, canny edges, the canny algorithm is applied.
Lastly, a PCA is performed on all features which have more than n=100 components.

Anaconda-environment on local machine: opencv_env

@author: Alexander Lenders, Agnessa Karapetian
"""
import argparse
import pickle
import numpy as np
import scipy.io
from tqdm import tqdm
import pandas as pd
from utils import (
    canny_edge,
    action,
    skeleton_pos,
    world_normals,
    lighting,
    depth,
    reflectance,
    pca,
    load_config,
    parse_list,
)
import os


def feature_extraction(
    videos_dir,
    annotations_dir,
    character_dir,
    action_dir,
    save_dir,
    n_components,
    pca_method,
    feature_names: list = None,
    start_frame: int = 10,
    end_frame: int = 19,
):
    """
    Standard Scaler and PCA are fitted only on the training data and applied
    to the training, test and validation data. By default this function makes
    use of the canny edge extraction function in the openCV module.

    Input:
    ----------
    a. Single frames of the videos (frame 10 to 18)
    b. Single frame annotations from Unreal Engine
    c. Metadata containing information about data split and character/action

    Returns
    ----------
    video_features.pkl: Canny edges, World normals, Lighting, Scene Depth,
    Reflectance, Action Identity, Skeleton Position after
    PCA (if necessary), saved in a dictionary "video_features"
        - Dictionary contains matrix for each feature with the dimension
        num_videos x num_components

    Parameters
    ----------
    videos_dir : str
        Directory with single frames as .jpg for every video
    annotations_dir: str
        Directory with single frame annotations as .jpg and .pkl
    character_dir: str
        Directory with meta data about character identity as .mat
    action_dir: str
        Directory with meta data about action identity as .csv
    save_dir : str
        Directory where to save the output of the function
    n_components: int
        Number of components for the PCA
    pca_method: str
        Whether to use a linearPCA ('linear') or KernelPCA ('nonlinear')
    """
    if feature_names is None:
        # Feature names
        feature_names = (
            "edges",
            "skeleton",
            "world_normal",
            "lighting",
            "scene_depth",
            "reflectance",
            "action",
        )

    # Number of videos
    num_videos = 1440
    num_frame = 9

    # Load meta data for character identity and action identity
    ## Character Identity
    meta_data = scipy.io.loadmat(character_dir)
    char_data = pd.DataFrame((meta_data.get("meta_data")))
    rows, cols = char_data.shape
    char_meta = pd.DataFrame(np.zeros((rows, cols)))

    for col in range(cols):
        extracted_values = [item.item() for item in char_data.iloc[:, col]]
        char_meta[char_meta.columns[col]] = extracted_values

    ## Action identity
    action_data = pd.read_csv(action_dir, header=None)
    # Recode the six different actions
    actions = {1: 0, 2: 1, 9: 2, 18: 3, 19: 4, 30: 5}
    action_data[1] = action_data[1].replace(actions)

    # Get information about train and test data
    split_data = np.array(char_meta.iloc[:, 1])
    # 0 -> training data
    # 1 -> validation data
    # 2 -> test data
    index = np.arange(0, num_videos)
    split_data = np.column_stack([index, split_data])
    train_data = split_data[:, 0][split_data[:, 1] == 0]
    val_data = split_data[:, 0][split_data[:, 1] == 1]
    test_data = split_data[:, 0][split_data[:, 1] == 2]

    pca_features = dict.fromkeys(feature_names)

    for feature in pca_features.keys():
        print(feature)

        datasets = []

        if feature == "edges":

            # for GRAY 390*520, where 390*520 dimension of video
            features_flattened = np.zeros((num_videos, 202800), dtype=float)

            for video in tqdm(range(num_videos)):
                features_flattened_frame = np.zeros((1, 202800), dtype=float)
                video_index = video + 1

                for i, frame in enumerate(range(start_frame, end_frame)):
                    feature_np = canny_edge(video_index, videos_dir, frame)
                    feature_flatten = feature_np.flatten()
                    features_flattened_frame = np.add(
                        features_flattened_frame, feature_flatten
                    )

                features_flattened_frame = np.divide(
                    features_flattened_frame, num_frame
                )
                features_flattened[video, :] = features_flattened_frame

            # Split data
            features_train = features_flattened[train_data]
            features_val = features_flattened[val_data]
            features_test = features_flattened[test_data]
            del features_flattened

            # PCA
            pca_features_train, pca_features_val, pca_features_test = pca(
                features_train,
                features_val,
                features_test,
                pca_method,
                n_components,
            )
            del features_train, features_val, features_test
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)

        elif feature == "skeleton":
            # 14 skeleton positions * 2 coordinates
            features_flattened = np.zeros((num_videos, 28), dtype=float)

            for video in tqdm(range(num_videos)):
                features_flattened_frame = np.zeros((1, 28), dtype=float)
                video_index = video + 1

                for i, frame in enumerate(range(start_frame, end_frame)):
                    feature_np = skeleton_pos(
                        video_index, annotations_dir, frame
                    )
                    feature_flatten = feature_np.flatten()
                    features_flattened_frame = np.add(
                        features_flattened_frame, feature_flatten
                    )

                features_flattened_frame = np.divide(
                    features_flattened_frame, num_frame
                )
                features_flattened[video, :] = features_flattened_frame

            # Split data
            features_train = features_flattened[train_data]
            features_val = features_flattened[val_data]
            features_test = features_flattened[test_data]
            del features_flattened

            datasets.append(features_train)
            datasets.append(features_val)
            datasets.append(features_test)

        elif feature == "world_normal":

            # for RGB 390*520*3, where 390*520 dimension of video
            features_flattened = np.zeros((num_videos, 608400), dtype=float)

            for video in tqdm(range(num_videos)):
                features_flattened_frame = np.zeros((1, 608400), dtype=float)
                video_index = video + 1

                for i, frame in enumerate(range(start_frame, end_frame)):
                    feature_np = world_normals(
                        video_index, annotations_dir, frame
                    )
                    feature_flatten = feature_np.flatten()
                    features_flattened_frame = np.add(
                        features_flattened_frame, feature_flatten
                    )

                features_flattened_frame = np.divide(
                    features_flattened_frame, num_frame
                )
                features_flattened[video, :] = features_flattened_frame

            # Split data
            features_train = features_flattened[train_data]
            features_val = features_flattened[val_data]
            features_test = features_flattened[test_data]
            del features_flattened

            # PCA
            pca_features_train, pca_features_val, pca_features_test = pca(
                features_train,
                features_val,
                features_test,
                pca_method,
                n_components,
            )
            del features_train, features_val, features_test
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)

        elif feature == "lighting":

            # for GRAY 390*520, where 390*520 dimension of video
            features_flattened = np.zeros((num_videos, 202800), dtype=float)

            for video in tqdm(range(num_videos)):

                video_index = video + 1
                features_flattened_frame = np.zeros((1, 202800), dtype=float)

                for i, frame in enumerate(range(start_frame, end_frame)):
                    feature_np = lighting(video_index, annotations_dir, frame)
                    feature_flatten = feature_np.flatten()
                    features_flattened_frame = np.add(
                        features_flattened_frame, feature_flatten
                    )

                features_flattened_frame = np.divide(
                    features_flattened_frame, num_frame
                )
                features_flattened[video, :] = features_flattened_frame

            # Split data
            features_train = features_flattened[train_data]
            features_val = features_flattened[val_data]
            features_test = features_flattened[test_data]
            del features_flattened

            # PCA
            pca_features_train, pca_features_val, pca_features_test = pca(
                features_train,
                features_val,
                features_test,
                pca_method,
                n_components,
            )
            del features_train, features_val, features_test
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)

        elif feature == "scene_depth":

            # for GRAY 390*520, where 390*520 dimension of video
            features_flattened = np.zeros((num_videos, 202800), dtype=float)

            for video in tqdm(range(num_videos)):

                video_index = video + 1
                features_flattened_frame = np.zeros((1, 202800), dtype=float)

                for i, frame in enumerate(range(start_frame, end_frame)):
                    feature_np = depth(video_index, annotations_dir, frame)
                    feature_flatten = feature_np.flatten()
                    features_flattened_frame = np.add(
                        features_flattened_frame, feature_flatten
                    )

                features_flattened_frame = np.divide(
                    features_flattened_frame, num_frame
                )
                features_flattened[video, :] = features_flattened_frame

            # Split data
            features_train = features_flattened[train_data]
            features_val = features_flattened[val_data]
            features_test = features_flattened[test_data]
            del features_flattened

            # PCA
            pca_features_train, pca_features_val, pca_features_test = pca(
                features_train,
                features_val,
                features_test,
                pca_method,
                n_components,
            )
            del features_train, features_val, features_test
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)

        elif feature == "reflectance":

            # for RGB 390*520*3, where 390*520 dimension of video
            features_flattened = np.zeros((num_videos, 608400), dtype=float)

            for video in tqdm(range(num_videos)):
                features_flattened_frame = np.zeros((1, 608400), dtype=float)
                video_index = video + 1

                for i, frame in enumerate(range(start_frame, end_frame)):
                    feature_np = reflectance(
                        video_index, annotations_dir, frame
                    )
                    feature_flatten = feature_np.flatten()
                    features_flattened_frame = np.add(
                        features_flattened_frame, feature_flatten
                    )

                features_flattened_frame = np.divide(
                    features_flattened_frame, num_frame
                )
                features_flattened[video, :] = features_flattened_frame

            # Split data
            features_train = features_flattened[train_data]
            features_val = features_flattened[val_data]
            features_test = features_flattened[test_data]
            del features_flattened

            # PCA
            pca_features_train, pca_features_val, pca_features_test = pca(
                features_train,
                features_val,
                features_test,
                pca_method,
                n_components,
            )
            del features_train, features_val, features_test
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)

        elif feature == "action":
            features_flattened = np.zeros(
                (num_videos, len(actions)), dtype=float
            )

            for img in tqdm(range(num_videos)):
                feature_np = action(img, action_data, actions)
                features_flattened[img, :] = feature_np

            # Split data
            features_train = features_flattened[train_data]
            features_val = features_flattened[val_data]
            features_test = features_flattened[test_data]
            del features_flattened

            datasets.append(features_train)
            datasets.append(features_val)
            datasets.append(features_test)

        pca_features[feature] = datasets

    # -------------------------------------------------------------------------
    # STEP 2.11 Save Output
    # -------------------------------------------------------------------------
    features_dir = os.path.join(
        save_dir, f"video_features_avg_frame_redone_{len(feature_names)}.pkl"
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(features_dir, "wb") as f:
        pickle.dump(pca_features, f)


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()

    config = load_config(args.config_dir, args.config)

    n_components = config.getint(args.config, "n_components")
    pca_method = config.get(args.config, "pca_method")
    videos_dir = config.get(args.config, "videos_dir")
    annotations_dir = config.get(args.config, "video_annotations_dir")
    action_dir = config.get(args.config, "action_metadata_dir")
    character_dir = config.get(args.config, "character_metadata_dir")
    save_dir = config.get(args.config, "save_dir_feat_video")
    feature_names = parse_list(config.get(args.config, "feature_names"))
    start_frame = config.getint(args.config, "start_frame")
    end_frame = config.getint(args.config, "end_frame")

    # -----------------------------------------------------------------------------
    # STEP 3: Run Function
    # -----------------------------------------------------------------------------
    feature_extraction(
        videos_dir,
        annotations_dir,
        character_dir,
        action_dir,
        save_dir,
        n_components,
        pca_method,
        feature_names=feature_names,
        start_frame=start_frame,
        end_frame=end_frame,
    )
