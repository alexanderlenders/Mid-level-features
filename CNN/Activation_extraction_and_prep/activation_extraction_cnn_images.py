"""
Script to extract CNN activations from images and save them in a specified directory.

@author: Alexander Lenders
"""
# Import modules
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import torch
import random
from torch.autograd import Variable as V
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
import argparse
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
print(project_root)
sys.path.append(str(project_root))

from EEG.Encoding.utils import (
    load_config,
)

def extract_activations(images_dir, save_dir, seed=42, init=True, transform="vid"):
    # STEP 1: LOAD RESNET2D MODEL #
    # Set random seeds (especially important for random initialization)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set the architecture to use
    arch = "resnet18"

    model_file = "%s_places365.pth.tar" % arch

    if not os.access(model_file, os.W_OK):
        weight_url = (
            "http://places2.csail.mit.edu/models_places365/" + model_file
        )
        os.system("wget " + weight_url)

    # New syntax
    model = models.resnet18(num_classes=365, weights=None)
    if init:  # Initialize model with pre-trained weights
        save_dir = save_dir + "_pretrained"
        checkpoint = torch.load(
            model_file, map_location=lambda storage, loc: storage
        )
        state_dict = {
            str.replace(k, "module.", ""): v
            for k, v in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state_dict)

    model.eval()

    # STEP 2: DEFINE DATA VARIABLES #
    # number of images
    num_videos = 1440

    if transform == "vid":

        print("Downscaling the images to 112 x 112 pixels as in 3D ResNet18.")
        print("This corresponds to control analysis 11.")

        centre_crop = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.CenterCrop((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        centre_crop = transforms.Compose(
            [
                transforms.Resize(256), # Keep the aspect ratio
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    # --------------------------------------
    # STEP 3: EXTRACT UNIT ACTIVATIONS #
    # --------------------------------------
    return_layers = ["layer1.0.relu_1", "layer1.1.relu_1", "layer2.0.relu_1", "layer2.1.relu_1", "layer3.0.relu_1", "layer3.1.relu_1", "layer4.0.relu_1", "layer4.1.relu_1"]

    # Extract features
    train_nodes, eval_nodes = get_graph_node_names(model)

    # checker whether nodes are same for training and evaluation mode
    assert [t == e for t, e in zip(train_nodes, eval_nodes)]

    feature_extractor = create_feature_extractor(
        model, return_nodes=return_layers
    )

    # Preallocate arrays based on one image
    print("Preallocating arrays for feature extraction...")
    feature_arrays = {}

    example_path = os.path.join(images_dir, f"{1:04}_frame_20.jpg")
    image = Image.open(example_path)
    input_tensor = V(centre_crop(image).unsqueeze(0))

    with torch.no_grad():
        out = feature_extractor(input_tensor)

    for layer in return_layers:
        shape = out[layer].squeeze(0).numpy().shape  # (C, H, W)

        # Total number of units for this layer
        num_units = np.prod(shape)

        # Create a 2D array for this layer's features
        feature_arrays[layer] = np.zeros((num_videos, num_units), dtype=np.float32)

    # Loop through all images
    print("Extracting features...")
    for img in tqdm(range(1, (num_videos + 1))):
        idx = img - 1

        image_dir = os.path.join(images_dir, (f"{img:04}_frame_20.jpg"))

        # Load image
        image = Image.open(image_dir)

        # Preprocess image
        batch_t = V(centre_crop(image).unsqueeze(0))

        # apply those features on image
        with torch.no_grad():
            out = feature_extractor(batch_t)

        for layer in return_layers:
            feature_map = out[layer].squeeze(0).numpy()  # shape (C, H, W)
            flattened = feature_map.flatten()            # shape (C*H*W,)
            feature_arrays[layer][idx, :] = flattened

    # --------------------------------------
    # STEP 4: SAVE FEATURES WITHOUT PCA #
    # --------------------------------------

    for layer in feature_arrays.keys():
        features_dir = os.path.join(save_dir, f"features_resnet_{layer}.pkl")
        
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        with open(features_dir, "wb") as f:
            pickle.dump(feature_arrays[layer], f)

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
        '--init', 
        action='store_true', 
        help='Whether to use pretrained weights or not.'
    )
    parser.add_argument(
        '--transform',
        type=str,
        default="img",
        help="Transformation to apply 'img' or 'vid'."
    )

    args = parser.parse_args()
    config = load_config(args.config_dir, args.config)

    if args.init:
        init = True
    else:
        init = False

    transform = args.transform

    img_dir = config.get(args.config, "images_dir")
    save_dir = config.get(args.config, "save_dir_cnn_img")
    seed = config.getint(args.config, "seed")

    # Run feature extraction
    extract_activations(
        images_dir=img_dir,
        save_dir=save_dir,
        seed=seed,
        init=init,
        transform=transform
    )