import sys
from pathlib import Path

import torch

# Add src folder to Python path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fer_cnn.model import ProposedModel2LMCNN


def main():
    # Create model with 7 emotion output classes.
    model = ProposedModel2LMCNN(num_classes=7)

    # Create fake batch: 4 samples, 2 channels, 48x48 image size.
    fake_input = torch.randn(4, 2, 48, 48)

    # Run model forward pass.
    output = model(fake_input)

    # Output should be: 4 samples x 7 classes.
    print("Input shape:", fake_input.shape)
    print("Output shape:", output.shape)

    print("Model test completed successfully.")


if __name__ == "__main__":
    main()