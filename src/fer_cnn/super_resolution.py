# Applies super-resolution-assisted upscaling before MediaPipe landmark detection.

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def super_resolve_for_detection(
    image: Image.Image,
    target_size: int = 224,
    model_path: str | Path = "models/EDSR_x4.pb",
):
    # If EDSR model exists, use deep-learning super-resolution.
    model_path = Path(model_path)

    if model_path.exists():
        import cv2

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(model_path))
        sr.setModel("edsr", 4)

        image_array = np.asarray(image.convert("RGB"))
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        super_resolved_bgr = sr.upsample(image_bgr)
        super_resolved_rgb = cv2.cvtColor(super_resolved_bgr, cv2.COLOR_BGR2RGB)

        super_resolved_image = Image.fromarray(super_resolved_rgb)
        return super_resolved_image.resize((target_size, target_size))

    # Fallback: high-quality resize + sharpen.
    return image.resize(
        (target_size, target_size),
        Image.Resampling.LANCZOS,
    ).filter(ImageFilter.SHARPEN)