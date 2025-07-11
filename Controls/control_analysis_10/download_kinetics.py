"""
Script to download Kinetics-400 dataset using torchvision.datasets.Kinetics.
"""
from torchvision.datasets import Kinetics
import argparse
import os

def download_kinetics400(
    root_dir: str = "./kinetics400",
    split: str = "train",
    num_download_workers: int = 4,
):
    """
    Download and the Kinetics-400 dataset using torchvision.datasets.Kinetics. All .mp4 videos (around 10s) will be downloaded.
    """
    print(f"Downloading Kinetics-400 split='{split}' to '{root_dir}'...")

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    kinetics_dataset = Kinetics(
        root=root_dir,
        frames_per_clip=1,
        num_classes="400",
        split=split,
        frame_rate=1,
        step_between_clips=1,
        download=True,
        num_download_workers=num_download_workers,
        num_workers=1,
    )

    print(f"Done downloading {split} split. Dataset size: {len(kinetics_dataset)} clips.")
    return kinetics_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kinetics-400")
    parser.add_argument("--root_dir", type=str, default="/scratch/alexandel91/mid_level_features/kinetics_400", help="Root directory for dataset")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--num_download_workers", type=int, default=4, help="Parallel download workers")

    args = parser.parse_args()

    download_kinetics400(
        root_dir=args.root_dir,
        split=args.split,
        num_download_workers=args.num_download_workers,
    )