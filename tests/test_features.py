"""Tests for src/features.py — image quality metrics using sample data."""
from pathlib import Path

import pytest

from src.features import blur_variance_of_laplacian, brightness_contrast

SAMPLE_IMAGES = Path(__file__).parent.parent / "data" / "sample" / "images"
FAKE_IMAGE = SAMPLE_IMAGES / "X00016469622.png"
REAL_IMAGE = SAMPLE_IMAGES / "X00016469623.png"


def _skip_if_missing(path: Path):
    if not path.exists():
        pytest.skip(f"Sample image not found: {path}")


class TestBlurVarianceOfLaplacian:
    def test_returns_positive_float_fake(self):
        _skip_if_missing(FAKE_IMAGE)
        result = blur_variance_of_laplacian(FAKE_IMAGE)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_returns_positive_float_real(self):
        _skip_if_missing(REAL_IMAGE)
        result = blur_variance_of_laplacian(REAL_IMAGE)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_sharp_image_has_higher_variance_than_blurred(self, tmp_path):
        """Blurring an image should reduce the Laplacian variance."""
        import cv2
        import numpy as np
        from PIL import Image

        _skip_if_missing(REAL_IMAGE)

        # Save a blurred version
        img = cv2.imread(str(REAL_IMAGE), cv2.IMREAD_GRAYSCALE)
        blurred = cv2.GaussianBlur(img, (51, 51), 0)
        blurred_path = tmp_path / "blurred.png"
        cv2.imwrite(str(blurred_path), blurred)

        sharp_var = blur_variance_of_laplacian(REAL_IMAGE)
        blurred_var = blur_variance_of_laplacian(blurred_path)
        assert sharp_var > blurred_var


class TestBrightnessContrast:
    def test_returns_expected_keys_fake(self):
        _skip_if_missing(FAKE_IMAGE)
        result = brightness_contrast(FAKE_IMAGE)
        assert "brightness_mean" in result
        assert "contrast_std" in result

    def test_returns_expected_keys_real(self):
        _skip_if_missing(REAL_IMAGE)
        result = brightness_contrast(REAL_IMAGE)
        assert "brightness_mean" in result
        assert "contrast_std" in result

    def test_brightness_in_valid_range(self):
        # Values are normalised to [0.0, 1.0]
        _skip_if_missing(FAKE_IMAGE)
        result = brightness_contrast(FAKE_IMAGE)
        assert 0.0 <= result["brightness_mean"] <= 1.0

    def test_contrast_non_negative(self):
        _skip_if_missing(REAL_IMAGE)
        result = brightness_contrast(REAL_IMAGE)
        assert result["contrast_std"] >= 0.0

    def test_white_image_has_high_brightness(self, tmp_path):
        # brightness_contrast normalises to [0.0, 1.0] (divides by 255)
        from PIL import Image
        import numpy as np

        white_path = tmp_path / "white.png"
        Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8)).save(white_path)
        result = brightness_contrast(white_path)
        assert result["brightness_mean"] > 0.99

    def test_black_image_has_low_brightness(self, tmp_path):
        from PIL import Image
        import numpy as np

        black_path = tmp_path / "black.png"
        Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(black_path)
        result = brightness_contrast(black_path)
        assert result["brightness_mean"] < 10
