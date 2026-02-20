from __future__ import annotations

from pathlib import Path



import cv2

import numpy as np

from PIL import Image



def blur_variance_of_laplacian(image_path: Path) -> float:

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:

        return float("nan")

    return float(cv2.Laplacian(img, cv2.CV_64F).var())



def brightness_contrast(image_path: Path) -> dict:

    with Image.open(image_path) as im:

        im = im.convert("L")

        arr = np.asarray(im, dtype=np.float32) / 255.0

    return {

        "brightness_mean": float(arr.mean()),

        "contrast_std": float(arr.std()),

    }
