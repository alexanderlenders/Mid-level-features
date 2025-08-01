from configparser import ConfigParser
import json
import os

# =============================================================================
# ADAPT THE FOLLOWING VARIABLES WHICH ARE SHARED ACROSS CONFIGS

config = ConfigParser()

# ADAPT ROOT_DIR!
root_dir = "/scratch/alexandel91/mid_level_features"

# =============================================================================
# The following directories are shared across all configurations
# =============================================================================
# Where the video frames are stored
videos_dir = os.path.join(root_dir, "stimuli", "miniclips", "frames")
# Where the mp4 videos are stored
mp4_dir = os.path.join(root_dir, "stimuli", "miniclips", "mp4")
# Where the video annotations are stored
video_annotations_dir = os.path.join(
    root_dir, "stimuli", "miniclips", "frame_annotations"
)
# Where the action metadata is stored
action_metadata_dir = os.path.join(root_dir, "features", "action_indices.csv")
# Where the character metadata is stored
character_metadata_dir = os.path.join(
    root_dir, "features", "meta_data_anim.mat"
)
# Where the image frames are stored
img_dir = os.path.join(root_dir, "stimuli", "images", "frames")
# Where the image annotations are stored
img_annotations_dir = os.path.join(
    root_dir, "stimuli", "images", "frame_annotations"
)
# Where the EEG data is stored
eeg_dir = os.path.join(root_dir, "data", "EEG")
# Where the noise ceiling is stored
noise_ceiling_dir = os.path.join(root_dir, "results", "noise_ceiling")

# =============================================================================
# Default encoding analysis
# =============================================================================
feature_names_default = [
    "edges",
    "reflectance",
    "lighting",
    "world_normal",
    "scene_depth",
    "skeleton",
    "action",
]
feature_names_graph_default = [
    "Edges",
    "Reflectance",
    "Lighting",
    "Normals",
    "Depth",
    "Skeleton",
    "Action",
]

save_dir_default = os.path.join(root_dir, "results", "EEG", "default")
save_dir_default_cnn = os.path.join(root_dir, "results", "CNN", "default")

save_dir_feat_img = os.path.join(root_dir, "features", "images", "default")
save_dir_feat_video = os.path.join(
    root_dir, "features", "miniclips", "default"
)
save_dir_cnn_img = os.path.join(
    root_dir, "data", "CNN", "2dresnet18_pretrained"
)
save_dir_cnn_video = os.path.join(root_dir, "data", "CNN", "3dresnet18")

config.add_section("default")
config.set("default", "feature_names", json.dumps(feature_names_default))
config.set(
    "default", "feature_names_graph", json.dumps(feature_names_graph_default)
)
config.set("default", "n_components", "100")
config.set("default", "pca_method", "linear")
config.set("default", "videos_dir", videos_dir)
config.set("default", "video_annotations_dir", video_annotations_dir)
config.set("default", "action_metadata_dir", action_metadata_dir)
config.set("default", "character_metadata_dir", character_metadata_dir)
config.set("default", "save_dir", save_dir_default)
config.set("default", "start_frame", "10")
config.set("default", "end_frame", "19")
config.set("default", "img_frame", "20")
config.set("default", "images_dir", img_dir)
config.set("default", "img_annotations_dir", img_annotations_dir)
config.set("default", "save_dir_feat_img", save_dir_feat_img)
config.set("default", "save_dir_feat_video", save_dir_feat_video)
config.set("default", "eeg_dir", eeg_dir)
config.set("default", "noise_ceiling_dir", noise_ceiling_dir)
config.set("default", "save_dir_cnn_img", save_dir_cnn_img)
config.set("default", "save_dir_cnn_video", save_dir_cnn_video)
config.set("default", "save_dir_cnn", save_dir_default_cnn)
config.set("default", "mp4_dir", mp4_dir)

# =============================================================================

# =============================================================================
# Control analysis 1 (First frame)
# =============================================================================
feature_names_c1 = ["skeleton"]
feature_names_graph_c1 = ["Skeleton"]
save_dir_c1 = os.path.join(root_dir, "results", "EEG", "control_1")
save_dir_feat_img = os.path.join(root_dir, "features", "images", "control_1")
save_dir_feat_video = os.path.join(
    root_dir, "features", "miniclips", "control_1"
)

config.add_section("control_1")
config.set("control_1", "feature_names", json.dumps(feature_names_default))
config.set(
    "control_1", "feature_names_graph", json.dumps(feature_names_graph_default)
)
config.set("control_1", "n_components", "100")
config.set("control_1", "pca_method", "linear")
config.set("control_1", "videos_dir", videos_dir)
config.set("control_1", "video_annotations_dir", video_annotations_dir)
config.set("control_1", "action_metadata_dir", action_metadata_dir)
config.set("control_1", "character_metadata_dir", character_metadata_dir)
config.set("control_1", "img_frame", "20")
config.set("control_1", "save_dir", save_dir_c1)
config.set("control_1", "start_frame", "10")
config.set("control_1", "end_frame", "11")
config.set("control_1", "save_dir_feat_video", save_dir_feat_video)
config.set("control_1", "eeg_dir", eeg_dir)
config.set("control_1", "noise_ceiling_dir", noise_ceiling_dir)

# =============================================================================
# Control analysis 2 (Last frame)
# =============================================================================
feature_names_c2 = ["skeleton"]
feature_names_graph_c2 = ["Skeleton"]
save_dir_c2 = os.path.join(root_dir, "results", "EEG", "control_2")
save_dir_feat_img = os.path.join(root_dir, "features", "images", "control_2")
save_dir_feat_video = os.path.join(
    root_dir, "features", "miniclips", "control_2"
)

config.add_section("control_2")
config.set("control_2", "feature_names", json.dumps(feature_names_default))
config.set(
    "control_2", "feature_names_graph", json.dumps(feature_names_graph_default)
)
config.set("control_2", "n_components", "100")
config.set("control_2", "pca_method", "linear")
config.set("control_2", "videos_dir", videos_dir)
config.set("control_2", "video_annotations_dir", video_annotations_dir)
config.set("control_2", "action_metadata_dir", action_metadata_dir)
config.set("control_2", "character_metadata_dir", character_metadata_dir)
config.set("control_2", "save_dir", save_dir_c2)
config.set("control_2", "start_frame", "18")
config.set("control_2", "img_frame", "20")
config.set("control_2", "end_frame", "19")
config.set("control_2", "save_dir_feat_video", save_dir_feat_video)
config.set("control_2", "eeg_dir", eeg_dir)
config.set("control_2", "noise_ceiling_dir", noise_ceiling_dir)

# =============================================================================
# Control analysis 3 (Use image annotations for miniclips EEG data)
# =============================================================================
feature_names_default = [
    "edges",
    "reflectance",
    "lighting",
    "world_normal",
    "scene_depth",
    "skeleton",
    "action",
]
feature_names_graph_default = [
    "Edges",
    "Reflectance",
    "Lighting",
    "Normals",
    "Depth",
    "Skeleton",
    "Action",
]

save_dir_c3 = os.path.join(root_dir, "results", "EEG", "control_3")
save_dir_feat_img = os.path.join(root_dir, "features", "images", "default")
save_dir_feat_video = os.path.join(
    root_dir, "features", "miniclips", "default"
)

config.add_section("control_3")
config.set("control_3", "feature_names", json.dumps(feature_names_default))
config.set(
    "control_3", "feature_names_graph", json.dumps(feature_names_graph_default)
)
config.set("control_3", "n_components", "100")
config.set("control_3", "pca_method", "linear")
config.set("control_3", "videos_dir", videos_dir)
config.set("control_3", "video_annotations_dir", video_annotations_dir)
config.set("control_3", "action_metadata_dir", action_metadata_dir)
config.set("control_3", "character_metadata_dir", character_metadata_dir)
config.set("control_3", "save_dir", save_dir_c3)
config.set("control_3", "start_frame", "10")
config.set("control_3", "end_frame", "19")
config.set("control_3", "img_frame", "20")
config.set("control_3", "images_dir", img_dir)
config.set("control_3", "img_annotations_dir", img_annotations_dir)
config.set("control_3", "save_dir_feat_img", save_dir_feat_img)
config.set("control_3", "save_dir_feat_video", save_dir_feat_video)
config.set("control_3", "eeg_dir", eeg_dir)
config.set("control_3", "noise_ceiling_dir", noise_ceiling_dir)

# =============================================================================
# Control analysis 6.1 (Variance partitioning - Idea 1)
# =============================================================================
# Feature names for variance partitioning idea 1
features = (
    "edges",
    "reflectance",
    "lighting",
    "world_normal",
    "scene_depth",
    "skeleton",
    "action",
)
# Leave-one-out combinations
feature_names_6_1 = [
    tuple(f for f in features if f != excluded) for excluded in features
]
# Append the full feature set
feature_names_6_1.append(features)

feature_names_graph_default = [
    "Edges",
    "Reflectance",
    "Lighting",
    "Normals",
    "Depth",
    "Skeleton",
    "Action",
]

save_dir_c6_1 = os.path.join(root_dir, "results", "EEG", "control_6_1")
save_dir_feat_img = os.path.join(root_dir, "features", "images", "default")
save_dir_feat_video = os.path.join(
    root_dir, "features", "miniclips", "default"
)

config.add_section("control_6_1")
config.set("control_6_1", "feature_names", json.dumps(feature_names_6_1))
config.set(
    "control_6_1",
    "feature_names_graph",
    json.dumps(feature_names_graph_default),
)
config.set("control_6_1", "n_components", "100")
config.set("control_6_1", "pca_method", "linear")
config.set("control_6_1", "videos_dir", videos_dir)
config.set("control_6_1", "video_annotations_dir", video_annotations_dir)
config.set("control_6_1", "action_metadata_dir", action_metadata_dir)
config.set("control_6_1", "character_metadata_dir", character_metadata_dir)
config.set("control_6_1", "save_dir", save_dir_c6_1)
config.set("control_6_1", "start_frame", "10")
config.set("control_6_1", "end_frame", "19")
config.set("control_6_1", "img_frame", "20")
config.set("control_6_1", "images_dir", img_dir)
config.set("control_6_1", "img_annotations_dir", img_annotations_dir)
config.set("control_6_1", "save_dir_feat_img", save_dir_feat_img)
config.set("control_6_1", "save_dir_feat_video", save_dir_feat_video)
config.set("control_6_1", "eeg_dir", eeg_dir)
config.set("control_6_1", "noise_ceiling_dir", noise_ceiling_dir)

# =============================================================================
# Control analysis 6.2 (Variance partitioning - Idea 2)
# =============================================================================
# Feature names for variance partitioning idea 2
low_and_high_feat = ("edges", "action")
mid_level_feat = (
    "reflectance",
    "lighting",
    "world_normal",
    "scene_depth",
    "skeleton",
)

feature_names_6_2 = [
    tuple(low_and_high_feat) + (mid_feat,) for mid_feat in mid_level_feat
]

feature_names_6_2.append(low_and_high_feat)

feature_names_graph_6_2 = [
    "Reflectance",
    "Lighting",
    "Normals",
    "Depth",
    "Skeleton",
]

save_dir_c6_2 = os.path.join(root_dir, "results", "EEG", "control_6_2")
save_dir_feat_img = os.path.join(root_dir, "features", "images", "default")
save_dir_feat_video = os.path.join(
    root_dir, "features", "miniclips", "default"
)

config.add_section("control_6_2")
config.set("control_6_2", "feature_names", json.dumps(feature_names_6_2))
config.set(
    "control_6_2", "feature_names_graph", json.dumps(feature_names_graph_6_2)
)
config.set("control_6_2", "n_components", "100")
config.set("control_6_2", "pca_method", "linear")
config.set("control_6_2", "videos_dir", videos_dir)
config.set("control_6_2", "video_annotations_dir", video_annotations_dir)
config.set("control_6_2", "action_metadata_dir", action_metadata_dir)
config.set("control_6_2", "character_metadata_dir", character_metadata_dir)
config.set("control_6_2", "save_dir", save_dir_c6_2)
config.set("control_6_2", "start_frame", "10")
config.set("control_6_2", "end_frame", "19")
config.set("control_6_2", "img_frame", "20")
config.set("control_6_2", "images_dir", img_dir)
config.set("control_6_2", "img_annotations_dir", img_annotations_dir)
config.set("control_6_2", "save_dir_feat_img", save_dir_feat_img)
config.set("control_6_2", "save_dir_feat_video", save_dir_feat_video)
config.set("control_6_2", "eeg_dir", eeg_dir)
config.set("control_6_2", "noise_ceiling_dir", noise_ceiling_dir)


# =============================================================================
# Control analysis 9
# =============================================================================
feature_names_default = [
    "edges",
    "reflectance",
    "lighting",
    "world_normal",
    "scene_depth",
    "skeleton",
    "action",
]
feature_names_graph_default = [
    "Edges",
    "Reflectance",
    "Lighting",
    "Normals",
    "Depth",
    "Skeleton",
    "Action",
]

save_dir_c9 = os.path.join(root_dir, "results", "EEG", "control_9")

save_dir_feat_img = os.path.join(root_dir, "features", "images", "default")
save_dir_feat_video = os.path.join(
    root_dir, "features", "miniclips", "default"
)

config.add_section("control_9")
config.set("control_9", "feature_names", json.dumps(feature_names_default))
config.set(
    "control_9", "feature_names_graph", json.dumps(feature_names_graph_default)
)
config.set("control_9", "n_components", "100")
config.set("control_9", "pca_method", "linear")
config.set("control_9", "videos_dir", videos_dir)
config.set("control_9", "video_annotations_dir", video_annotations_dir)
config.set("control_9", "action_metadata_dir", action_metadata_dir)
config.set("control_9", "character_metadata_dir", character_metadata_dir)
config.set("control_9", "save_dir", save_dir_c9)
config.set("control_9", "start_frame", "10")
config.set("control_9", "end_frame", "19")
config.set("control_9", "img_frame", "20")
config.set("control_9", "images_dir", img_dir)
config.set("control_9", "img_annotations_dir", img_annotations_dir)
config.set("control_9", "save_dir_feat_img", save_dir_feat_img)
config.set("control_9", "save_dir_feat_video", save_dir_feat_video)
config.set("control_9", "eeg_dir", eeg_dir)
config.set("control_9", "noise_ceiling_dir", noise_ceiling_dir)


# =============================================================================
# Control analysis 10
# =============================================================================
feature_names_default = [
    "edges",
    "reflectance",
    "lighting",
    "world_normal",
    "scene_depth",
    "skeleton",
    "action",
]
feature_names_graph_default = [
    "Edges",
    "Reflectance",
    "Lighting",
    "Normals",
    "Depth",
    "Skeleton",
    "Action",
]

save_dir_default = os.path.join(root_dir, "results", "EEG", "control_10")
save_dir_default_cnn = os.path.join(root_dir, "results", "CNN", "control_10")

save_dir_cnn_img = os.path.join(
    root_dir, "data", "CNN", "2dresnet18_pretrained", "control_10"
)
save_dir_cnn_video = os.path.join(
    root_dir, "data", "CNN", "3dresnet18", "control_10"
)
weights_dir = "/scratch/alexandel91/mid_level_features/results/CNN/training/ResNet18/checkpoints/resnet18-epoch=06-val_loss=3.42.ckpt"

config.add_section("control_10")
config.set("control_10", "feature_names", json.dumps(feature_names_default))
config.set(
    "control_10",
    "feature_names_graph",
    json.dumps(feature_names_graph_default),
)
config.set("control_10", "n_components", "100")
config.set("control_10", "pca_method", "linear")
config.set("control_10", "videos_dir", videos_dir)
config.set("control_10", "video_annotations_dir", video_annotations_dir)
config.set("control_10", "action_metadata_dir", action_metadata_dir)
config.set("control_10", "character_metadata_dir", character_metadata_dir)
config.set("control_10", "save_dir", save_dir_default)
config.set("control_10", "start_frame", "10")
config.set("control_10", "end_frame", "19")
config.set("control_10", "img_frame", "20")
config.set("control_10", "images_dir", img_dir)
config.set("control_10", "img_annotations_dir", img_annotations_dir)
config.set("control_10", "save_dir_feat_img", save_dir_feat_img)
config.set("control_10", "save_dir_feat_video", save_dir_feat_video)
config.set("control_10", "eeg_dir", eeg_dir)
config.set("control_10", "noise_ceiling_dir", noise_ceiling_dir)
config.set("control_10", "save_dir_cnn_img", save_dir_cnn_img)
config.set("control_10", "save_dir_cnn_video", save_dir_cnn_video)
config.set("control_10", "save_dir_cnn", save_dir_default_cnn)
config.set("control_10", "mp4_dir", mp4_dir)
config.set("control_10", "kinetics_weights_dir", weights_dir)


# =============================================================================
# Control analysis 11
# =============================================================================
feature_names_default = [
    "edges",
    "reflectance",
    "lighting",
    "world_normal",
    "scene_depth",
    "skeleton",
    "action",
]
feature_names_graph_default = [
    "Edges",
    "Reflectance",
    "Lighting",
    "Normals",
    "Depth",
    "Skeleton",
    "Action",
]

save_dir_default = os.path.join(root_dir, "results", "EEG", "control_11")
save_dir_default_cnn = os.path.join(root_dir, "results", "CNN", "control_11")

save_dir_cnn_img = os.path.join(
    root_dir, "data", "CNN", "2dresnet18_pretrained", "control_11"
)
save_dir_cnn_video = os.path.join(
    root_dir, "data", "CNN", "3dresnet18", "control_11"
)

config.add_section("control_11")
config.set("control_11", "feature_names", json.dumps(feature_names_default))
config.set(
    "control_11",
    "feature_names_graph",
    json.dumps(feature_names_graph_default),
)
config.set("control_11", "n_components", "100")
config.set("control_11", "pca_method", "linear")
config.set("control_11", "videos_dir", videos_dir)
config.set("control_11", "video_annotations_dir", video_annotations_dir)
config.set("control_11", "action_metadata_dir", action_metadata_dir)
config.set("control_11", "character_metadata_dir", character_metadata_dir)
config.set("control_11", "save_dir", save_dir_default)
config.set("control_11", "start_frame", "10")
config.set("control_11", "end_frame", "19")
config.set("control_11", "img_frame", "20")
config.set("control_11", "images_dir", img_dir)
config.set("control_11", "img_annotations_dir", img_annotations_dir)
config.set("control_11", "save_dir_feat_img", save_dir_feat_img)
config.set("control_11", "save_dir_feat_video", save_dir_feat_video)
config.set("control_11", "eeg_dir", eeg_dir)
config.set("control_11", "noise_ceiling_dir", noise_ceiling_dir)
config.set("control_11", "save_dir_cnn_img", save_dir_cnn_img)
config.set("control_11", "save_dir_cnn_video", save_dir_cnn_video)
config.set("control_11", "save_dir_cnn", save_dir_default_cnn)
config.set("control_11", "mp4_dir", mp4_dir)


# =============================================================================
# Control analysis 12
# =============================================================================
feature_names_default = [
    "edges",
    "reflectance",
    "lighting",
    "world_normal",
    "scene_depth",
    "skeleton",
    "action",
]
feature_names_graph_default = [
    "Edges",
    "Reflectance",
    "Lighting",
    "Normals",
    "Depth",
    "Skeleton",
    "Action",
]

save_dir_default = os.path.join(root_dir, "results", "EEG", "control_12")
save_dir_default_cnn = os.path.join(root_dir, "results", "CNN", "control_12")

save_dir_feat_img = os.path.join(root_dir, "features", "images", "default")
save_dir_feat_video = os.path.join(
    root_dir, "features", "miniclips", "default"
)
save_dir_cnn_img = os.path.join(
    root_dir, "data", "CNN", "2dresnet18_pretrained"
)
save_dir_cnn_video = os.path.join(root_dir, "data", "CNN", "3dresnet18")

config.add_section("control_12")
config.set("control_12", "feature_names", json.dumps(feature_names_default))
config.set(
    "control_12", "feature_names_graph", json.dumps(feature_names_graph_default)
)
config.set("control_12", "n_components", "100")
config.set("control_12", "pca_method", "linear")
config.set("control_12", "videos_dir", videos_dir)
config.set("control_12", "video_annotations_dir", video_annotations_dir)
config.set("control_12", "action_metadata_dir", action_metadata_dir)
config.set("control_12", "character_metadata_dir", character_metadata_dir)
config.set("control_12", "save_dir", save_dir_default)
config.set("control_12", "start_frame", "10")
config.set("control_12", "end_frame", "19")
config.set("control_12", "img_frame", "20")
config.set("control_12", "images_dir", img_dir)
config.set("control_12", "img_annotations_dir", img_annotations_dir)
config.set("control_12", "save_dir_feat_img", save_dir_feat_img)
config.set("control_12", "save_dir_feat_video", save_dir_feat_video)
config.set("control_12", "eeg_dir", eeg_dir)
config.set("control_12", "noise_ceiling_dir", noise_ceiling_dir)
config.set("control_12", "save_dir_cnn_img", save_dir_cnn_img)
config.set("control_12", "save_dir_cnn_video", save_dir_cnn_video)
config.set("control_12", "save_dir_cnn", save_dir_default_cnn)
config.set("control_12", "mp4_dir", mp4_dir)

with open("./config.ini", "w") as f:
    config.write(f)
