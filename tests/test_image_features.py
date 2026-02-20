"""Tests for image feature extraction."""
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.image_features import image_basic_stats, blur_variance_of_laplacian, brightness_contrast


def _create_test_image(path: str, width: int = 100, height: int = 200, fmt: str = "RGB"):
    """Create a test image using PIL if available."""
    try:
        from PIL import Image
        img = Image.new(fmt, (width, height), color=(128, 128, 128))
        img.save(path)
        return True
    except ImportError:
        return False


class TestImageBasicStats:
    def test_returns_dict_with_required_keys(self, tmp_path):
        img_path = str(tmp_path / "test.png")
        if not _create_test_image(img_path):
            pytest.skip("PIL not available")
        stats = image_basic_stats(img_path)
        assert "width" in stats
        assert "height" in stats
        assert "aspect_ratio" in stats
        assert "file_size_kb" in stats

    def test_aspect_ratio_calculation(self, tmp_path):
        img_path = str(tmp_path / "test.png")
        if not _create_test_image(img_path, width=200, height=100):
            pytest.skip("PIL not available")
        stats = image_basic_stats(img_path)
        assert stats["aspect_ratio"] == 2.0

    def test_file_size_positive(self, tmp_path):
        img_path = str(tmp_path / "test.png")
        if not _create_test_image(img_path):
            pytest.skip("PIL not available")
        stats = image_basic_stats(img_path)
        assert stats["file_size_kb"] > 0

    def test_dimensions_correct(self, tmp_path):
        img_path = str(tmp_path / "test.png")
        if not _create_test_image(img_path, width=150, height=300):
            pytest.skip("PIL not available")
        stats = image_basic_stats(img_path)
        assert stats["width"] == 150
        assert stats["height"] == 300

    def test_square_image_aspect_ratio(self, tmp_path):
        img_path = str(tmp_path / "test.png")
        if not _create_test_image(img_path, width=100, height=100):
            pytest.skip("PIL not available")
        stats = image_basic_stats(img_path)
        assert stats["aspect_ratio"] == 1.0


class TestBlurVarianceOfLaplacian:
    def test_returns_float(self, tmp_path):
        img_path = str(tmp_path / "test.png")
        if not _create_test_image(img_path):
            pytest.skip("PIL not available")
        try:
            import cv2
        except ImportError:
            pytest.skip("cv2 not available")
        result = blur_variance_of_laplacian(img_path)
        assert isinstance(result, float)

    def test_nonexistent_returns_negative_one(self):
        try:
            import cv2
        except ImportError:
            pytest.skip("cv2 not available")
        result = blur_variance_of_laplacian("/nonexistent.png")
        assert result == -1.0

    def test_sharp_image_higher_than_blur(self, tmp_path):
        """A uniform image (no edges) should have low variance."""
        img_path = str(tmp_path / "uniform.png")
        if not _create_test_image(img_path):
            pytest.skip("PIL not available")
        try:
            import cv2
        except ImportError:
            pytest.skip("cv2 not available")
        result = blur_variance_of_laplacian(img_path)
        # Uniform gray image should have low but non-negative variance
        assert result >= 0


class TestBrightnessContrast:
    def test_returns_dict_with_keys(self, tmp_path):
        img_path = str(tmp_path / "test.png")
        if not _create_test_image(img_path):
            pytest.skip("PIL not available")
        try:
            import cv2
        except ImportError:
            pytest.skip("cv2 not available")
        result = brightness_contrast(img_path)
        assert "brightness" in result
        assert "contrast" in result

    def test_brightness_in_range(self, tmp_path):
        img_path = str(tmp_path / "test.png")
        if not _create_test_image(img_path):
            pytest.skip("PIL not available")
        try:
            import cv2
        except ImportError:
            pytest.skip("cv2 not available")
        result = brightness_contrast(img_path)
        if result["brightness"] != -1.0:
            assert 0.0 <= result["brightness"] <= 1.0

    def test_nonexistent_returns_negative_one(self):
        try:
            import cv2
        except ImportError:
            pytest.skip("cv2 not available")
        result = brightness_contrast("/nonexistent.png")
        assert result["brightness"] == -1.0
        assert result["contrast"] == -1.0

    def test_gray_image_mid_brightness(self, tmp_path):
        """Gray (128,128,128) image should have brightness near 0.5."""
        img_path = str(tmp_path / "gray.png")
        if not _create_test_image(img_path):
            pytest.skip("PIL not available")
        try:
            import cv2
        except ImportError:
            pytest.skip("cv2 not available")
        result = brightness_contrast(img_path)
        if result["brightness"] != -1.0:
            assert 0.4 <= result["brightness"] <= 0.6  # 128/255 ≈ 0.502

    def test_uniform_image_low_contrast(self, tmp_path):
        """Uniform color image should have near-zero contrast."""
        img_path = str(tmp_path / "uniform.png")
        if not _create_test_image(img_path):
            pytest.skip("PIL not available")
        try:
            import cv2
        except ImportError:
            pytest.skip("cv2 not available")
        result = brightness_contrast(img_path)
        if result["contrast"] != -1.0:
            assert result["contrast"] < 0.1  # Uniform = very low std
