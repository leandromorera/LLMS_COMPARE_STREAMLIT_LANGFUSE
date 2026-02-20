"""CPU-only feature extraction for receipt images."""
import os
from pathlib import Path

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def image_basic_stats(image_path: str) -> dict:
    """Return width, height, aspect_ratio, file_size_kb."""
    path = Path(image_path)
    file_size_kb = path.stat().st_size / 1024.0

    if _PIL_AVAILABLE:
        with PILImage.open(image_path) as img:
            width, height = img.size
    elif _CV2_AVAILABLE:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = img.shape[:2]
    else:
        width, height = 0, 0

    aspect_ratio = round(width / height, 4) if height > 0 else 0.0
    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": round(file_size_kb, 2),
    }


def blur_variance_of_laplacian(image_path: str) -> float:
    """Return sharpness proxy via variance of Laplacian. Higher = sharper."""
    if not _CV2_AVAILABLE:
        return -1.0
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1.0
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return float(laplacian.var())


def brightness_contrast(image_path: str) -> dict:
    """Return mean brightness and contrast std on 0-1 scale."""
    if _CV2_AVAILABLE:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"brightness": -1.0, "contrast": -1.0}
        import numpy as np
        arr = img.astype(float) / 255.0
        return {
            "brightness": round(float(arr.mean()), 4),
            "contrast": round(float(arr.std()), 4),
        }
    return {"brightness": -1.0, "contrast": -1.0}
