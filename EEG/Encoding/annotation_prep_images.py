#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANNOTATION PREPARATION AND PCA - IMAGES - UNREAL ENGINE

This script prepares the annotations from the Unreal Engine by extracting the low-, mid-, and
high-level features from them. For the low-level feature, canny edges, the canny algorithm is applied.
Lastly, a PCA is performed on all features which have more than n=100 components.

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
    images_dir: str,
    n_components: int,
    annotations_dir: str,
    character_dir: str,
    action_dir: str,
    save_dir: str,
    pca_method: str,
    frame: int,
    feature_names: list = None,
):
    """
    Standard Scaler and PCA are fitted only on the training data and applied
    to the training, test and validation data. By default this function makes
    use of the canny edge extraction function in the openCV module.

    Input:
    ----------
    a. Single frame (frame 20)
    b. Single frame annotations from Unreal Engine
    c. Metadata containing information about data split and character/action

    Returns
    ----------
    image_features.pkl: Canny edges, World normals, Lighting, Scene Depth,
    Reflectance, Action Identity, Skeleton Position after
    PCA (if necessary), saved in a dictionary "image_features"
        - Dictionary contains matrix for each feature with the dimension
        num_images x num_components

    Parameters
    ----------
    n_components: int
        Number of components for the PCA
    annotations_dir: str
        Directory with single frame annotations as .jpg and .pkl
    character_dir: str
        Directory with character information (for train/test/val split)
    action_dir: str
        Directory with action identity information
    save_dir: str
        Directory where to save the output of the function
    pca_method: str
        Whether to use a linearPCA ('linear') or KernelPCA ('nonlinear')
    frame: int
        Image frame where to get annotations
    feature_names: list, optional
        List of feature names to extract. If None, default features are used.
        Default is None, which uses all features:
        ['edges', 'skeleton', 'world_normal', 'lighting', 'scene_depth',
        'reflectance', 'action'].
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

    # Number of images
    num_images = 1440

    # Load meta data for character identity - for the train/test/val split
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
    index = np.arange(0, num_images)
    split_data = np.column_stack([index, split_data])
    train_data = split_data[:, 0][split_data[:, 1] == 0]
    val_data = split_data[:, 0][split_data[:, 1] == 1]
    test_data = split_data[:, 0][split_data[:, 1] == 2]

    # -------------------------------------------------------------------------
    # STEP 2.10 Get features for all images and apply PCA
    # -------------------------------------------------------------------------
    pca_features = dict.fromkeys(feature_names)

    for feature in pca_features.keys():

        datasets = []

        if feature == "edges":
            # for GRAY 390*520, where 390*520 dimension of image
            features_flattened = np.zeros((num_images, 202800), dtype=float)

            for img in tqdm(range(num_images)):
                # features_flattened_frame = np.zeros((1, 202800), dtype = float)
                img_index = img + 1

                # for i, frame in enumerate(range(10, 18)):
                feature_np = canny_edge(img_index, images_dir, frame)
                feature_flatten = feature_np.flatten()
                features_flattened[img, :] = feature_flatten

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
            features_flattened = np.zeros((num_images, 28), dtype=float)

            for img in tqdm(range(num_images)):
                img_index = img + 1

                feature_np = skeleton_pos(img_index, annotations_dir, frame)
                feature_flatten = feature_np.flatten()
                features_flattened[img, :] = feature_flatten

            # Split data
            features_train = features_flattened[train_data]
            features_val = features_flattened[val_data]
            features_test = features_flattened[test_data]
            del features_flattened

            datasets.append(features_train)
            datasets.append(features_val)
            datasets.append(features_test)

        elif feature == "world_normal":

            # for RGB 390*520*3, where 390*520 dimension of image
            features_flattened = np.zeros((num_images, 608400), dtype=float)

            for img in tqdm(range(num_images)):
                img_index = img + 1

                feature_np = world_normals(img_index, annotations_dir, frame)
                feature_flatten = feature_np.flatten()
                features_flattened[img, :] = feature_flatten

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

            # for GRAY 390*520, where 390*520 dimension of image
            features_flattened = np.zeros((num_images, 202800), dtype=float)

            for img in tqdm(range(num_images)):

                img_index = img + 1

                feature_np = lighting(img_index, annotations_dir, frame)
                feature_flatten = feature_np.flatten()
                features_flattened[img, :] = feature_flatten

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

            # for GRAY 390*520, where 390*520 dimension of image
            features_flattened = np.zeros((num_images, 202800), dtype=float)

            for img in tqdm(range(num_images)):

                img_index = img + 1

                feature_np = depth(img_index, annotations_dir, frame)
                feature_flatten = feature_np.flatten()
                features_flattened[img, :] = feature_flatten

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

            # for RGB 390*520*3, where 390*520 dimension of image
            features_flattened = np.zeros((num_images, 608400), dtype=float)

            for img in tqdm(range(num_images)):
                img_index = img + 1

                feature_np = reflectance(img_index, annotations_dir, frame)
                feature_flatten = feature_np.flatten()
                features_flattened[img, :] = feature_flatten

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
                (num_images, len(actions)), dtype=float
            )

            for img in tqdm(range(num_images)):
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
        save_dir,
        f"img_features_frame_{frame}_redone_{len(feature_names)}_features_onehot.pkl",
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(features_dir, "wb") as f:
        pickle.dump(pca_features, f)


if __name__ == "__main__":

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

    args = parser.parse_args()  # to get values for the arguments

    config = load_config(args.config_dir, args.config)

    n_components = config.getint(args.config, "n_components")
    pca_method = config.get(args.config, "pca_method")
    annotations_dir = config.get(args.config, "img_annotations_dir")
    action_dir = config.get(args.config, "action_metadata_dir")
    character_dir = config.get(args.config, "character_metadata_dir")
    save_dir = config.get(args.config, "save_dir_feat_img")
    feature_names = parse_list(config.get(args.config, "feature_names"))
    frame = config.getint(args.config, "img_frame")
    images_dir = config.get(args.config, "images_dir")

    feature_extraction(
        images_dir,
        n_components,
        annotations_dir,
        character_dir,
        action_dir,
        save_dir,
        pca_method,
        frame,
        feature_names=feature_names,
    )
