# Detects 468 MediaPipe landmarks using super-resolution-assisted upscaling.

from pathlib import Path

import numpy as np
from PIL import Image

from fer_cnn.super_resolution import super_resolve_for_detection


class MediaPipeLandmarkDetector:
    # Uses MediaPipe Face Mesh to detect all 468 facial landmark points.
    def __init__(self):
        import mediapipe as mp

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )

    def detect(
        self,
        image_path: str | Path,
        output_size: int = 48,
        detection_size: int = 224,
    ):
        # Open FER image and convert to RGB for MediaPipe.
        image = Image.open(image_path).convert("RGB")

        # Super-resolve/upscale only for landmark detection.
        detection_image = super_resolve_for_detection(
            image,
            target_size=detection_size,
        )

        # Convert image to NumPy array for MediaPipe.
        image_array = np.asarray(detection_image)

        # Run MediaPipe Face Mesh.
        results = self.face_mesh.process(image_array)

        # Return empty list if no face is detected.
        if not results.multi_face_landmarks:
            return []

        landmarks = results.multi_face_landmarks[0].landmark
        points = []

        # Convert normalized MediaPipe points back to 48x48 heatmap space.
        for landmark in landmarks:
            x = landmark.x * output_size
            y = landmark.y * output_size
            points.append((x, y))

        return points

    def close(self):
        # Release MediaPipe resources.
        self.face_mesh.close()