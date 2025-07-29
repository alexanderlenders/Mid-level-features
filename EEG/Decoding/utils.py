"""
Utility functions for the EEG decoding analysis.
"""

import os
import numpy as np


def load_eeg(
    sub: int,
    img_type: str,
    region: str,
    freq: int,
    input_type: str,
    eeg_dir: str,
):
    """
    Utility function to load the EEG data for a given subject and input type (video or image).
    """

    # Define the directory
    workDirFull = eeg_dir

    # load mvnn data
    if input_type == "miniclips":
        if sub < 10:
            folderDir = os.path.join(
                workDirFull,
                "{}".format(input_type)
                + "/sub-0{}".format(sub)
                + "/eeg/preprocessing/ica"
                + "/"
                + img_type
                + "/"
                + region
                + "/",
            )
            fileDir = "sub-0{}_seq_{}_{}hz_{}.npy".format(
                sub, img_type, freq, region
            )

        else:
            folderDir = os.path.join(
                workDirFull,
                "{}".format(input_type)
                + "/sub-{}".format(sub)
                + "/eeg/preprocessing/ica"
                + "/"
                + img_type
                + "/"
                + region
                + "/",
            )
            fileDir = "sub-{}_seq_{}_{}hz_{}.npy".format(
                sub, img_type, freq, region
            )

    elif input_type == "images":
        if sub < 10:
            folderDir = os.path.join(
                workDirFull,
                "{}".format(input_type)
                + "/prepared"
                + "/sub-0{}".format(sub)
                + "/{}/{}/{}hz/".format(img_type, region, freq),
            )
            fileDir = "{}_img_data_{}hz_sub_00{}.npy".format(
                img_type, freq, sub
            )
        else:
            folderDir = os.path.join(
                workDirFull,
                "{}".format(input_type)
                + "/prepared"
                + "/sub-{}".format(sub)
                + "/{}/{}/{}hz/".format(img_type, region, freq),
            )
            fileDir = "{}_img_data_{}hz_sub_0{}.npy".format(
                img_type, freq, sub
            )

    total_dir = os.path.join(folderDir, fileDir)

    # Load EEG data
    data = np.load(total_dir, allow_pickle=True).item()

    eeg_data = data["eeg_data"]
    img_cat = data["img_cat"]
    time = data["time"]

    return eeg_data, img_cat, time
