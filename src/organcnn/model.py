"""OrganCNN: a compact 3D CNN for volumetric organ classification.

Designed from scratch by Habib Ur Rehman for OrganMNIST3D (MedMNIST v2):
multi-class classification of 28x28x28 CT bounding-box volumes into 11
body-organ classes.

The architecture deliberately follows the compact VesselCNN family shape
(three 3x3x3 conv blocks and a global-average-pooling head), but the task
is different: this is an 11-way classifier, and several labels encode
left/right anatomy. That has a concrete training consequence recorded in
scripts/train.py: the octahedral flip/permutation augmentation used for
VesselCNN is NOT label-preserving here, because flipping a volume can turn
a right kidney into a left kidney. OrganCNN therefore trains without
geometric augmentation.

Labels (MedMNIST v2 / OrganMNIST3D official order):
  0 liver
  1 kidney-right
  2 kidney-left
  3 femur-right
  4 femur-left
  5 bladder
  6 heart
  7 lung-right
  8 lung-left
  9 spleen
 10 pancreas
"""
import torch
import torch.nn as nn


class OrganCNN(nn.Module):
    """3 conv blocks (16 -> 32 -> 64 channels) + GAP classifier head.

    Input: (N, 1, 28, 28, 28) float tensor, normalized with the
    OrganMNIST3D train-split mean/std (see organcnn.infer). The GAP head
    keeps the parameter count near 105k against only 971 training
    volumes; a flattened MLP head would add roughly 1.4M parameters.
    """

    def __init__(self, in_ch=1, num_classes=11, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_ch, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(dropout * 0.5),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(dropout * 0.5),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        # 28^3 -> 14^3 -> 7^3 -> GAP -> 64 features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model():
    return OrganCNN()
