from configparser import ConfigParser
import json
import os

# =============================================================================
# ADAPT THE FOLLOWING VARIABLES WHICH ARE SHARED ACROSS CONFIGS

config = ConfigParser()

# Please adapt
root_dir = "/scratch/alexandel91/mid_level_features"

# Where the video frames are stored
videos_dir = os.path.join(root_dir, "stimuli", "miniclips", "frames")
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


# =============================================================================
# Default encoding analysis
# =============================================================================
feature_names_default = [
    "edges",
    "skeleton",
    "world_normal",
    "lighting",
    "scene_depth",
    "reflectance",
    "action",
]
save_dir_default = os.path.join(root_dir, "results", "EEG", "default")
save_dir_feat_img = os.path.join(root_dir, "features", "images", "default")
save_dir_feat_video = os.path.join(
    root_dir, "features", "miniclips", "default"
)

config.add_section("default")
config.set("default", "feature_names", json.dumps(feature_names_default))
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

# =============================================================================

# =============================================================================
# Control analysis 1 (First frame)
# =============================================================================
feature_names_c1 = ["skeleton"]
save_dir_c1 = os.path.join(root_dir, "results", "EEG", "control_1")
save_dir_feat_img = os.path.join(root_dir, "features", "images", "control_1")
save_dir_feat_video = os.path.join(
    root_dir, "features", "miniclips", "control_1"
)

config.add_section("control_1")
config.set("control_1", "feature_names", json.dumps(feature_names_c1))
config.set("control_1", "n_components", "100")
config.set("control_1", "pca_method", "linear")
config.set("control_1", "videos_dir", videos_dir)
config.set("control_1", "video_annotations_dir", video_annotations_dir)
config.set("control_1", "action_metadata_dir", action_metadata_dir)
config.set("control_1", "character_metadata_dir", character_metadata_dir)
config.set("control_1", "save_dir", save_dir_c1)
config.set("control_1", "start_frame", "10")
config.set("control_1", "end_frame", "11")
config.set("control_1", "save_dir_feat_video", save_dir_feat_video)
config.set("control_1", "eeg_dir", eeg_dir)

# =============================================================================
# Control analysis 2 (Last frame)
# =============================================================================
feature_names_c2 = ["skeleton"]
save_dir_c2 = os.path.join(root_dir, "results", "EEG", "control_2")
save_dir_feat_img = os.path.join(root_dir, "features", "images", "control_2")
save_dir_feat_video = os.path.join(
    root_dir, "features", "miniclips", "control_2"
)

config.add_section("control_2")
config.set("control_2", "feature_names", json.dumps(feature_names_c2))
config.set("control_2", "n_components", "100")
config.set("control_2", "pca_method", "linear")
config.set("control_2", "videos_dir", videos_dir)
config.set("control_2", "video_annotations_dir", video_annotations_dir)
config.set("control_2", "action_metadata_dir", action_metadata_dir)
config.set("control_2", "character_metadata_dir", character_metadata_dir)
config.set("control_2", "save_dir", save_dir_c2)
config.set("control_2", "start_frame", "18")
config.set("control_2", "end_frame", "19")
config.set("control_2", "save_dir_feat_video", save_dir_feat_video)
config.set("control_2", "eeg_dir", eeg_dir)

with open("./config.ini", "w") as f:
    config.write(f)
