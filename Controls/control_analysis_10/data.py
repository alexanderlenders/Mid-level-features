"""
# Dataset for Kinetics-400 video frames (Control analysis 10).
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from decord import VideoReader
from decord import cpu
import pytorch_lightning as pl
import glob
from torchvision.transforms.functional import to_pil_image

class KineticsFrameDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train', class_names=None):
        self.transform = transform
        self.samples = []
        self.classes = sorted(os.listdir(data_dir))
        if class_names is not None:
            # Filter classes based on provided class names
            self.classes = sorted([cls for cls in self.classes if cls in class_names])

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        for cls in self.classes:
            video_paths = glob.glob(os.path.join(data_dir, split, cls, '*.mp4'))
            for vp in video_paths:
                self.samples.append((vp, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        # Hardcode cpu for now
        vr = VideoReader(video_path, ctx=cpu(0))  

        if len(vr) == 0:
            raise ValueError(f"Failed to read video: {video_path}")

        mid_idx = len(vr) // 2
        frame = vr[mid_idx].asnumpy()  # (H, W, C), dtype=uint8

        # Convert to PIL so we can use torchvision transforms
        frame = to_pil_image(frame)

        if self.transform:
            frame = self.transform(frame)

        return frame, label
    
class KineticsFrameDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64, num_workers=4, class_names=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_names = class_names

        self.transform = transforms.Compose([
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    def setup(self, stage=None):
        self.train_dataset = KineticsFrameDataset(self.data_dir, transform=self.transform, split='train', class_names=self.class_names)
        self.val_dataset = KineticsFrameDataset(self.data_dir, transform=self.transform, split='val', class_names=self.class_names)
        self.test_dataset = KineticsFrameDataset(self.data_dir, transform=self.transform, split='test', class_names=self.class_names)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, pin_memory=True)
