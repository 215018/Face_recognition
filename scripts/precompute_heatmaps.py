# Precomputes and saves super-resolution-assisted MediaPipe heatmaps for all FER-2013 images.

import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fer_cnn.heatmaps import gaussian_heatmap
from fer_cnn.landmarks import MediaPipeLandmarkDetector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_output_path(image_path: Path, data_root: Path, output_root: Path):
    # Keeps the same folder structure as data/train and data/test.
    relative_path = image_path.relative_to(data_root)
    return (output_root / relative_path).with_suffix(".pt")


def main():
    data_root = Path("data")
    output_root = Path("data/processed/heatmaps")
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = [
        path
        for path in data_root.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print("Total images found:", len(image_paths))

    detector = MediaPipeLandmarkDetector()

    for image_path in tqdm(image_paths, desc="Precomputing heatmaps"):
        output_path = get_output_path(image_path, data_root, output_root)

        # Skip image if heatmap is already saved.
        if output_path.exists():
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Detect 468 landmarks using super-resolution-assisted image.
        points = detector.detect(image_path, output_size=48)

        # If landmarks fail, save zero heatmap.
        if len(points) == 0:
            heatmap = torch.zeros((1, 48, 48))
        else:
            heatmap = gaussian_heatmap(size=48, points=points, sigma=2.0)

        torch.save(heatmap, output_path)

    detector.close()

    print("Heatmaps saved to:", output_root)


if __name__ == "__main__":
    main()