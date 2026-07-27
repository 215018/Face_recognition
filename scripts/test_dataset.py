import sys
from pathlib import Path

# Add src folder to Python path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fer_cnn.config import TrainConfig
from fer_cnn.dataset import build_loaders


def main():
    # Create config using your data folder.
    config = TrainConfig(data_dir=Path("data"), batch_size=4)

    # Build train, validation, and test loaders.
    train_loader, val_loader, test_loader, class_names = build_loaders(config)

    # Print detected emotion classes.
    print("Classes:", class_names)

    # Take one batch from training loader.
    images, labels = next(iter(train_loader))

    # Check fused input shape.
    print("Image batch shape:", images.shape)

    # Check label shape.
    print("Label batch shape:", labels.shape)

    # Print labels.
    print("Labels:", labels)

    print("Dataset test completed successfully.")


if __name__ == "__main__":
    main()