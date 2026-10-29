"""OrganCNN — compact 3D CNN for OrganMNIST3D 11-class organ classification.

A research/benchmark model release (not a medical device). See the model
card (README.md) for verified metrics and limitations.
"""
from .infer import LABELS, TRAIN_MEAN, TRAIN_STD, load_model, predict, predict_proba, preprocess
from .model import OrganCNN, build_model

__version__ = "0.1.0"
__all__ = [
    "OrganCNN",
    "build_model",
    "load_model",
    "predict",
    "predict_proba",
    "preprocess",
    "LABELS",
    "TRAIN_MEAN",
    "TRAIN_STD",
]
