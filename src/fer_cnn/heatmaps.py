# MediaPipe Face Mesh detects 468 landmark points from each face image. This project uses all 468 points to preserve detailed facial geometry. These points are converted into Gaussian heatmaps and fused with the grayscale image for CNN-based expression recognition.
import math

import numpy as np
import torch


def gaussian_heatmap(size: int, points: list[tuple[float, float]], sigma: float = 2.0):
    # Create coordinate grid for the output heatmap.
    yy, xx = np.mgrid[0:size, 0:size]

    # Start with an empty heatmap.
    heatmap = np.zeros((size, size), dtype=np.float32)

    # Add one Gaussian blob for every landmark point.
    for x, y in points:
        if math.isnan(x) or math.isnan(y):
            continue

        heatmap += np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)
        )

    # Normalize heatmap values to range 0 to 1.
    max_value = float(heatmap.max())
    if max_value > 0:
        heatmap = heatmap / max_value

    # Convert to PyTorch tensor with shape: 1 x 48 x 48.
    return torch.from_numpy(heatmap).unsqueeze(0)