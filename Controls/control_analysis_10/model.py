"""
Model class (ResNet-18) for Kinetics-400 video frames (Control analysis 10).
"""
import torch
from torch import nn
from torchvision.models import resnet18
import pytorch_lightning as pl
import torch.nn.functional as F

class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=400, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet18(pretrained=False)
        # Replace the final fclayer (1000 classes) with num_classes using the initialized weights
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
