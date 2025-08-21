"""
Script to extract CNN activations from videos and save them in a specified directory.

@author: Alexander Lenders, Agnessa Karapetian
"""

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.io.video import read_video
from torchvision.models.video import r3d_18
from torchvision.models.video import R3D_18_Weights
from torchvision.transforms._presets import VideoClassification
from torchvision.transforms import InterpolationMode
import torch
import random
import os
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from EEG.Encoding.utils import (
    load_config,
)


def extract_activations(
    miniclips_dir: str, save_dir: str, seed: int, init: bool
):
    """
    Extracts activations from specified layers of a ResNet3D-18 model for a set of videos,
    and saves the extracted features for each layer as .pkl files.
    Loads a ResNet3D-18 model (optionally with pre-trained weights), preprocesses videos,
    extracts activations from specified intermediate layers, and saves the flattened
    activations for each video and layer.

    Input:
    ----------
    Directory containing videos to process. Videos should be named in the format
    '{video_number:04}.mp4'.

    Returns:
    ----------
    Saves a dictionary of extracted features for each specified layer as a .pkl file
    in the provided save directory. Each .pkl file contains a 2D numpy array of shape
    (num_videos, num_units), where num_units = C*H*W for the layer.

    Parameters
    ----------
    miniclips_dir : str
        Directory containing input videos.
    save_dir : str
        Directory to save extracted features.
    seed : int
        Random seed for reproducibility.
    init : bool
        If True, loads pre-trained weights for the model; otherwise uses random initialization.
    """
    # --------------------------------------
    # STEP 1: LOAD RESNET3D MODEL #
    # --------------------------------------
    # Set random seeds (especially important for random initialization)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Load pretrained weights
    if init:
        resnet_video = r3d_18(weights="KINETICS400_V1")
    else:
        resnet_video = r3d_18(weights=None, progress=True)

    resnet_video = resnet_video.to(device)

    # define transformation procedure for images (aka preprocessing)
    # preprocess = R3D_18_Weights.DEFAULT.transforms()

    # Define it manually as our original input has different shape than Kinetics400
    # And we want to keep the aspect ratio
    resize_size = (171, 128)  # Custom for our dataset
    crop_size = (112, 112)
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]
    interpolation = InterpolationMode.BILINEAR

    preprocess = VideoClassification(
        crop_size=crop_size,
        resize_size=resize_size,
        mean=mean,
        std=std,
        interpolation=interpolation,
    )

    # --------------------------------------
    # STEP 2: DEFINE DATA VARIABLES #
    # --------------------------------------
    # number of videos
    num_videos = sum(1 for file in os.scandir(miniclips_dir) if file.is_file())

    if num_videos != 1440:
        raise ValueError("Number of videos is not 1440")

    # number of frames
    num_frames = 9

    # --------------------------------------
    # STEP 3: EXTRACT UNIT ACTIVATIONS #
    # --------------------------------------
    return_layers = [
        "layer1.0.relu",
        "layer1.1.relu",
        "layer2.0.relu",
        "layer2.1.relu",
        "layer3.0.relu",
        "layer3.1.relu",
        "layer4.0.relu",
        "layer4.1.relu",
    ]

    # dimensions or num entries in each feature map in different layers
    num_col_1_0 = 56 * 56
    num_col_1_1 = 56 * 56
    num_col_2_0 = 28 * 28
    num_col_2_1 = 28 * 28
    num_col_3_0 = 14 * 14
    num_col_3_1 = 14 * 14
    num_col_4_0 = 7 * 7
    num_col_4_1 = 7 * 7

    num_feat_maps_1_0 = 64
    num_feat_maps_1_1 = 64
    num_feat_maps_2_0 = 128
    num_feat_maps_2_1 = 128
    num_feat_maps_3_0 = 256
    num_feat_maps_3_1 = 256
    num_feat_maps_4_0 = 512
    num_feat_maps_4_1 = 512

    # array for activations per video - First dimension refers to # of feature maps or channels
    layer1_0_features_s = np.zeros((64, num_col_1_0))
    layer1_1_features_s = np.zeros((64, num_col_1_1))
    layer2_0_features_s = np.zeros((128, num_col_2_0))
    layer2_1_features_s = np.zeros((128, num_col_2_1))
    layer3_0_features_s = np.zeros((256, num_col_3_0))
    layer3_1_features_s = np.zeros((256, num_col_3_1))
    layer4_0_features_s = np.zeros((512, num_col_4_0))
    layer4_1_features_s = np.zeros((512, num_col_4_1))

    # array for saving flattened activations across videos
    layer1_0_features = np.zeros((1440, num_col_1_0 * num_feat_maps_1_0))
    layer1_1_features = np.zeros((1440, num_col_1_1 * num_feat_maps_1_1))
    layer2_0_features = np.zeros((1440, num_col_2_0 * num_feat_maps_2_0))
    layer2_1_features = np.zeros((1440, num_col_2_1 * num_feat_maps_2_1))
    layer3_0_features = np.zeros((1440, num_col_3_0 * num_feat_maps_3_0))
    layer3_1_features = np.zeros((1440, num_col_3_1 * num_feat_maps_3_1))
    layer4_0_features = np.zeros((1440, num_col_4_0 * num_feat_maps_4_0))
    layer4_1_features = np.zeros((1440, num_col_4_1 * num_feat_maps_4_1))

    # Extract features
    train_nodes, eval_nodes = get_graph_node_names(resnet_video)

    # checker whether nodes are same for training and evaluation mode
    assert [t == e for t, e in zip(train_nodes, eval_nodes)]

    feature_extractor = create_feature_extractor(
        resnet_video, return_nodes=return_layers
    )

    print("Extracting features...")
    for img in tqdm(range(1, (num_videos + 1))):
        idx = img - 1

        # Get video directory (here referred to as image)
        image_file = str(img).zfill(4) + ".mp4"  # zfill: fill with zeros (4)
        image_dir = os.path.join(miniclips_dir, image_file)

        # Load video
        video, _, _ = read_video(image_dir, output_format="TCHW")

        video_preprocessed = preprocess(video)

        batch_t = video_preprocessed.unsqueeze(0).to(device)

        # Activate the evaluation mode of the DNN
        resnet_video.eval()

        # apply those features on image
        with torch.no_grad():
            out = feature_extractor(batch_t)

        for _, layer in enumerate(return_layers):
            # pick layer
            feat_maps = out[layer].squeeze(0).detach().cpu().numpy()

            if layer == "layer1.0.relu":
                for fm in range(len(list(feat_maps))):

                    flatten_fm_final = np.zeros((1, num_col_1_0))
                    num_frames = feat_maps.shape[1]
                    for frame in range(num_frames):
                        flatten_fm = feat_maps[fm, frame, :, :].flatten()
                        flatten_fm_final = np.add(flatten_fm_final, flatten_fm)

                    flatten_fm_final = np.divide(flatten_fm_final, num_frames)
                    layer1_0_features_s[fm, :] = flatten_fm_final

                layer1_0_features[idx, :] = layer1_0_features_s.flatten()

            elif layer == "layer1.1.relu":
                for fm in range(len(list(feat_maps))):

                    flatten_fm_final = np.zeros((1, num_col_1_1))
                    num_frames = feat_maps.shape[1]
                    for frame in range(num_frames):
                        flatten_fm = feat_maps[fm, frame, :, :].flatten()
                        flatten_fm_final = np.add(flatten_fm_final, flatten_fm)

                    flatten_fm_final = np.divide(flatten_fm_final, num_frames)
                    layer1_1_features_s[fm, :] = flatten_fm_final

                layer1_1_features[idx, :] = layer1_1_features_s.flatten()

            elif layer == "layer2.0.relu":
                for fm in range(len(list(feat_maps))):

                    flatten_fm_final = np.zeros((1, num_col_2_0))
                    num_frames = feat_maps.shape[1]
                    for frame in range(num_frames):
                        flatten_fm = feat_maps[fm, frame, :, :].flatten()
                        flatten_fm_final = np.add(flatten_fm_final, flatten_fm)

                    flatten_fm_final = np.divide(flatten_fm_final, num_frames)
                    layer2_0_features_s[fm, :] = flatten_fm_final

                layer2_0_features[idx, :] = layer2_0_features_s.flatten()

            elif layer == "layer2.1.relu":
                for fm in range(len(list(feat_maps))):

                    flatten_fm_final = np.zeros((1, num_col_2_1))
                    num_frames = feat_maps.shape[1]
                    for frame in range(num_frames):
                        flatten_fm = feat_maps[fm, frame, :, :].flatten()
                        flatten_fm_final = np.add(flatten_fm_final, flatten_fm)

                    flatten_fm_final = np.divide(flatten_fm_final, num_frames)
                    layer2_1_features_s[fm, :] = flatten_fm_final

                layer2_1_features[idx, :] = layer2_1_features_s.flatten()

            elif layer == "layer3.0.relu":
                for fm in range(len(list(feat_maps))):

                    flatten_fm_final = np.zeros((1, num_col_3_0))
                    num_frames = feat_maps.shape[1]
                    for frame in range(num_frames):
                        flatten_fm = feat_maps[fm, frame, :, :].flatten()
                        flatten_fm_final = np.add(flatten_fm_final, flatten_fm)

                    flatten_fm_final = np.divide(flatten_fm_final, num_frames)
                    layer3_0_features_s[fm, :] = flatten_fm_final

                layer3_0_features[idx, :] = layer3_0_features_s.flatten()

            elif layer == "layer3.1.relu":
                for fm in range(len(list(feat_maps))):

                    flatten_fm_final = np.zeros((1, num_col_3_1))
                    num_frames = feat_maps.shape[1]
                    for frame in range(num_frames):
                        flatten_fm = feat_maps[fm, frame, :, :].flatten()
                        flatten_fm_final = np.add(flatten_fm_final, flatten_fm)

                    flatten_fm_final = np.divide(flatten_fm_final, num_frames)
                    layer3_1_features_s[fm, :] = flatten_fm_final

                layer3_1_features[idx, :] = layer3_1_features_s.flatten()

            elif layer == "layer4.0.relu":
                for fm in range(len(list(feat_maps))):

                    flatten_fm_final = np.zeros((1, num_col_4_0))
                    num_frames = feat_maps.shape[1]
                    for frame in range(num_frames):
                        flatten_fm = feat_maps[fm, frame, :, :].flatten()
                        flatten_fm_final = np.add(flatten_fm_final, flatten_fm)

                    flatten_fm_final = np.divide(flatten_fm_final, num_frames)
                    layer4_0_features_s[fm, :] = flatten_fm_final

                layer4_0_features[idx, :] = layer4_0_features_s.flatten()

            elif layer == "layer4.1.relu":
                for fm in range(len(list(feat_maps))):

                    flatten_fm_final = np.zeros((1, num_col_4_1))
                    num_frames = feat_maps.shape[1]
                    for frame in range(num_frames):
                        flatten_fm = feat_maps[fm, frame, :, :].flatten()
                        flatten_fm_final = np.add(flatten_fm_final, flatten_fm)

                    flatten_fm_final = np.divide(flatten_fm_final, num_frames)
                    layer4_1_features_s[fm, :] = flatten_fm_final

                layer4_1_features[idx, :] = layer4_1_features_s.flatten()
    # --------------------------------------
    # STEP 4: SAVE FEATURES WITHOUT PCA #
    # --------------------------------------
    # Save each layer separately
    features = {
        "layer1.0.relu_1": layer1_0_features,
        "layer1.1.relu_1": layer1_1_features,
        "layer2.0.relu_1": layer2_0_features,
        "layer2.1.relu_1": layer2_1_features,
        "layer3.0.relu_1": layer3_0_features,
        "layer3.1.relu_1": layer3_1_features,
        "layer4.0.relu_1": layer4_0_features,
        "layer4.1.relu_1": layer4_1_features,
    }

    for layer in features.keys():
        features_dir = os.path.join(save_dir, f"features_resnet_{layer}.pkl")

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        with open(features_dir, "wb") as f:
            pickle.dump(features[layer], f)


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
        "--init",
        action="store_true",
        help="Whether to use pretrained weights or not.",
    )

    args = parser.parse_args()
    config = load_config(args.config_dir, args.config)

    if args.init:
        init = True
    else:
        init = False

    mp4_dir = config.get(args.config, "mp4_dir")
    save_dir = config.get(args.config, "save_dir_cnn_video")
    seed = 42

    # Run feature extraction
    extract_activations(
        miniclips_dir=mp4_dir,
        save_dir=save_dir,
        seed=seed,
        init=init,
    )
