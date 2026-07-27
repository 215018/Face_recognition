# Loads FER-2013 grayscale images for the Proposed_Model_2 baseline without landmark heatmaps.

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from fer_cnn.config import EMOTION_CLASSES, TrainConfig


def build_baseline_loaders(config: TrainConfig):
    train_dir = config.data_dir / "train"
    test_dir = config.data_dir / "test"

    # Training preprocessing for Proposed_Model_2 baseline.
    # HF=True means horizontal flip augmentation is applied during training only.
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # Validation and test preprocessing do not use augmentation.
    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=train_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transform)

    class_names = train_dataset.classes

    missing_classes = sorted(set(EMOTION_CLASSES) - set(class_names))
    if missing_classes:
        print(f"Warning: missing classes: {missing_classes}")

    val_fraction = getattr(config, "val_fraction", 0.15)
    val_count = int(len(train_dataset) * val_fraction)
    train_count = len(train_dataset) - val_count

    generator = torch.Generator().manual_seed(config.seed)
    shuffled_indices = torch.randperm(len(train_dataset), generator=generator).tolist()

    train_indices = shuffled_indices[:train_count]
    val_indices = shuffled_indices[train_count:]

    train_data = Subset(train_dataset, train_indices)
    val_data = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader, test_loader, class_names