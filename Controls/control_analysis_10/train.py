"""
Script to train the ResNet-18 model on Kinetics-400 video frames (control analysis 10).
"""

import argparse
import pytorch_lightning as pl
from model import ResNetClassifier
from data import KineticsFrameDataModule
from utils import set_random_seeds
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import os

def main(data_dir, batch_size, num_workers, max_epochs, lr, gpus, seed, save_dir, dev=False, class_names=None):
    # Check if the save directory exists, if not create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set random seeds
    set_random_seeds(seed)

    dm = KineticsFrameDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        class_names=class_names
    )

    model = ResNetClassifier(lr=lr, num_classes=len(class_names))

    # Define logger
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name="resnet18_kinetics",
        log_graph=False
    )

    # Determine model checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(save_dir, 'checkpoints'),
        filename='resnet18-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        every_n_epochs=10,
        mode='min',
        save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus,
        logger=logger,
        deterministic=True,
        callbacks=[checkpoint_callback],
        fast_dev_run=dev,
    )

    trainer.fit(model, datamodule=dm)

    # Test the model
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Kinetics-400 root folder')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='logs', help='Directory to save logs and checkpoints')
    parser.add_argument('--dev', action='store_true', help='Run in development mode (fast_dev_run)')

    args = parser.parse_args()

    # Hardcode the classes to include in the dataset
    # class_names = [
    #     "playing_guitar",
    #     "riding_horse",
    #     "playing_violin",
    #     "riding_motorcycle",
    #     "playing_drum",
    #     "welding",
    #     "cooking_egg",
    #     "frying_vegetables",
    #     "brushing_teeth",
    #     "cutting_vegetables",
    #     "playing_tennis",
    #     "skiing_crosscountry",
    #     "snowboarding",
    #     "swimming_breast_stroke",
    #     "juggling_balls",
    #     "baking_cookies",
    #     "blowing_glass",
    #     "driving_car",
    #     "shooting_basketball",
    #     "archery"
    # ]

    class_names = None

    data_dir = args.data_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    max_epochs = args.max_epochs
    lr = args.lr
    gpus = args.gpus
    seed = args.seed
    save_dir = args.save_dir
    dev = args.dev

    main(data_dir, batch_size, num_workers, max_epochs, lr, gpus, seed, save_dir, dev, class_names)
