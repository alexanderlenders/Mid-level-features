"""
Model class (ResNet-18) for Kinetics-400 video frames (Control analysis 10).
"""

import torch
from torch import nn
from torchvision.models import resnet18
import lightning as pl
import torch.nn.functional as F


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=400, lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet18(weights=None)
        # Replace the final fclayer (1000 classes) with num_classes using the initialized weights
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_acc", acc, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, sync_dist = True)
        self.log("val_acc", acc, sync_dist = True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
