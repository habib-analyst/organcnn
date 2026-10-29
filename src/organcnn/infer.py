"""Inference helpers for OrganCNN.

Preprocessing contract (identical to training/evaluation):
  uint8 volume (28, 28, 28) -> add channel dim -> /255
  -> normalize with the OrganMNIST3D train-split mean/std
     (TRAIN_MEAN / TRAIN_STD below, computed on the full train split
     with chunked float64 accumulation and re-verified by
     examples/verify_metrics.py).

Labels: MedMNIST v2 OrganMNIST3D order, see organcnn.model.
"""
from __future__ import annotations

import os
from typing import Union

import numpy as np
import torch

from .model import OrganCNN, build_model

# Full-train-split statistics (971 volumes), computed with chunked
# float64 accumulation over the /255-scaled intensities. Do not
# recompute in float32 over the whole split in one reduction (the
# family's documented trap).
TRAIN_MEAN = 0.50092
TRAIN_STD = 0.28022
LABELS = (
    "liver",
    "kidney-right",
    "kidney-left",
    "femur-right",
    "femur-left",
    "bladder",
    "heart",
    "lung-right",
    "lung-left",
    "spleen",
    "pancreas",
)
VOL_SIZE = 28

VolumeLike = Union[str, os.PathLike, np.ndarray]


def load_model(weights_path: str | os.PathLike, device: str = "cpu") -> OrganCNN:
    """Load an OrganCNN checkpoint (.safetensors or release .pt) in eval mode."""
    model = build_model()
    path = str(weights_path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path)
    else:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def preprocess(volume: VolumeLike) -> torch.Tensor:
    """One volume -> (1, 1, 28, 28, 28) normalized float tensor.

    Accepts a uint8/float numpy array of shape (28, 28, 28) or a path
    to a .npy file holding one.
    """
    if isinstance(volume, (str, os.PathLike)):
        arr = np.load(str(volume))
    else:
        arr = np.asarray(volume)
    if arr.shape != (VOL_SIZE, VOL_SIZE, VOL_SIZE):
        raise ValueError(f"expected a (28, 28, 28) volume, got shape {arr.shape}")
    x = arr.astype(np.float32) / 255.0
    x = (x - TRAIN_MEAN) / TRAIN_STD
    return torch.from_numpy(x)[None, None]


def predict_proba(model: OrganCNN, volume: VolumeLike) -> dict:
    """Return {label: probability} for all 11 classes."""
    x = preprocess(volume)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].tolist()
    return {LABELS[i]: probs[i] for i in range(len(LABELS))}


def predict(model: OrganCNN, volume: VolumeLike) -> dict:
    """Return {'label', 'confidence', 'probabilities'} for one volume."""
    probs = predict_proba(model, volume)
    label = max(probs, key=probs.get)
    return {"label": label, "confidence": probs[label], "probabilities": probs}
