from dataclasses import dataclass
from pathlib import Path


# These are the seven emotion classes in the FER-2013 dataset.
# Folder names inside data/train and data/test should match these names.
EMOTION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


@dataclass
class TrainConfig:
    # Main dataset folder.
    # Expected structure:
    # data/train/<emotion_name>/
    # data/test/<emotion_name>/
    data_dir: Path = Path("data")

    # FER-2013 images are 48 x 48 grayscale images.
    image_size: int = 48

    # Number of images processed together in one training step.
    batch_size: int = 64

    # Fraction of the training folder reserved for validation.
    val_fraction: float = 0.15

    # Proposed_Model_2 uses 130 training epochs.
    epochs: int = 130

    # Proposed_Model_2 uses learning rate 0.0001.
    learning_rate: float = 0.0001

    # Small regularization value to reduce overfitting.
    weight_decay: float = 0.0001

    # Smooth labels slightly so the model is less overconfident on noisy FER labels.
    label_smoothing: float = 0.05

    # 0.5 means square-root class weighting: helpful, but less aggressive than inverse frequency.
    class_weight_power: float = 0.5

    # Stop training if validation accuracy does not improve for this many epochs.
    patience: int = 15

    # Reduce learning rate after validation accuracy plateaus.
    lr_scheduler_patience: int = 3
    min_learning_rate: float = 0.00001

    # Number of background workers used for loading images.
    num_workers: int = 2

    # Seed makes random operations more repeatable.
    seed: int = 99
