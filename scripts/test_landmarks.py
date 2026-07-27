import sys
from pathlib import Path

# Add src folder to Python path so we can import fer_cnn modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fer_cnn.heatmaps import gaussian_heatmap
from fer_cnn.landmarks import MediaPipeLandmarkDetector


# Choose one real image from your dataset.
# You can change this path to any image you want to test.
IMAGE_PATH = Path("data/train/angry/Training_3908.jpg")


def main():
    print("Testing image:", IMAGE_PATH)

    if not IMAGE_PATH.exists():
        print("Image path does not exist.")
        print("Please change IMAGE_PATH in scripts/test_landmarks.py")
        return

    # Create MediaPipe detector.
    detector = MediaPipeLandmarkDetector()

    # Detect all 468 MediaPipe landmarks.
    points = detector.detect(IMAGE_PATH, output_size=48)

    # Close detector after use.
    detector.close()

    print("Number of landmark points detected:", len(points))

    if len(points) == 0:
        print("No face detected. Try another clearer face image.")
        return

    # Convert landmark points into one Gaussian heatmap.
    heatmap = gaussian_heatmap(size=48, points=points)

    print("Heatmap shape:", heatmap.shape)
    print("Heatmap minimum value:", heatmap.min().item())
    print("Heatmap maximum value:", heatmap.max().item())

    print("Landmark and heatmap test completed successfully.")


if __name__ == "__main__":
    main()