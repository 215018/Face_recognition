# Loads FER-2013 images and precomputed MediaPipe heatmaps for early-fusion CNN training.

from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from fer_cnn.config import EMOTION_CLASSES, TrainConfig


class LandmarkHeatmapDataset(datasets.ImageFolder):
    # Dataset that returns a 2-channel tensor: grayscale face image + landmark heatmap.
    def __init__(
        self,
        root: str | Path,
        image_size: int,
        data_root: Path,
        heatmap_root: Path,
        train: bool = False,
    ):
        super().__init__(root=root)

        self.image_size = image_size
        self.data_root = data_root
        self.heatmap_root = heatmap_root
        self.train = train

        # FER-2013 images are loaded as 48x48 grayscale tensors.
        self.image_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def _heatmap_path(self, image_path: Path):
        # Finds the saved heatmap path matching the original image path.
        relative_path = image_path.relative_to(self.data_root)
        return (self.heatmap_root / relative_path).with_suffix(".pt")

    def _augment(self, fused_tensor: torch.Tensor):
        if torch.rand(1).item() < 0.5:
            fused_tensor = torch.flip(fused_tensor, dims=[2])

        padding = max(2, self.image_size // 12)
        _, height, width = fused_tensor.shape
        padded = F.pad(fused_tensor, (padding, padding, padding, padding), mode="reflect")

        top = torch.randint(0, padding * 2 + 1, (1,)).item()
        left = torch.randint(0, padding * 2 + 1, (1,)).item()
        return padded[:, top : top + height, left : left + width]

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image_path = Path(image_path)

        # Open image as RGB first so torchvision transforms work consistently.
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)

        heatmap_path = self._heatmap_path(image_path)

        # Load precomputed 468-landmark heatmap.
        if heatmap_path.exists():
            heatmap_tensor = torch.load(heatmap_path)
        else:
            heatmap_tensor = torch.zeros((1, self.image_size, self.image_size))

        # Early fusion: channel 1 is grayscale image, channel 2 is heatmap.
        fused_tensor = torch.cat([image_tensor, heatmap_tensor], dim=0)

        if self.train:
            fused_tensor = self._augment(fused_tensor)

        return fused_tensor, label


def build_loaders(config: TrainConfig):
    train_dir = config.data_dir / "train"
    test_dir = config.data_dir / "test"
    heatmap_root = config.data_dir / "processed" / "heatmaps"

    train_dataset = LandmarkHeatmapDataset(
        root=train_dir,
        image_size=config.image_size,
        data_root=config.data_dir,
        heatmap_root=heatmap_root,
        train=True,
    )

    val_dataset = LandmarkHeatmapDataset(
        root=train_dir,
        image_size=config.image_size,
        data_root=config.data_dir,
        heatmap_root=heatmap_root,
        train=False,
    )

    test_data = LandmarkHeatmapDataset(
        root=test_dir,
        image_size=config.image_size,
        data_root=config.data_dir,
        heatmap_root=heatmap_root,
        train=False,
    )

    class_names = train_dataset.classes

    missing_classes = sorted(set(EMOTION_CLASSES) - set(class_names))
    if missing_classes:
        print(f"Warning: missing classes: {missing_classes}")

    val_count = int(len(train_dataset) * config.val_fraction)
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
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader, test_loader, class_names
