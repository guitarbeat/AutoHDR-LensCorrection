import numpy as np
import sys
import os

# Add the project root to sys.path so we can import backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.evaluation.metrics import (
    calculate_mae,
    calculate_psnr,
    calculate_ssim,
    evaluate_metrics,
)


class TestMetrics:
    def test_calculate_mae_identical(self):
        """Test MAE with identical images (should be 0)."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mae = calculate_mae(img, img)
        assert mae == 0.0

    def test_calculate_mae_different(self):
        """Test MAE with completely different images."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mae = calculate_mae(img1, img2)
        assert mae == 255.0

    def test_calculate_mae_shape_mismatch(self):
        """Test MAE with shape mismatch (should resize)."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create a smaller image
        img2 = np.zeros((50, 50, 3), dtype=np.uint8)
        # The function resizes img_gt (second arg) to match img_pred (first arg)
        # So if we pass different sizes, it should handle it without error
        # and if content is "same" (zeros), error should be 0
        mae = calculate_mae(img1, img2)
        assert mae == 0.0

    def test_calculate_psnr_identical(self):
        """Test PSNR with identical images (should be 100.0)."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        psnr = calculate_psnr(img, img)
        assert psnr == 100.0

    def test_calculate_psnr_different(self):
        """Test PSNR with different images."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # MSE = 255^2
        # PSNR = 20 * log10(255 / 255) = 20 * log10(1) = 0
        psnr = calculate_psnr(img1, img2)
        assert psnr == 0.0

    def test_calculate_ssim_identical(self):
        """Test SSIM with identical images (should be 1.0)."""
        # Use random data to ensure non-trivial stats
        np.random.seed(42)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        ssim = calculate_ssim(img, img)
        assert np.isclose(ssim, 1.0)

    def test_evaluate_metrics_identical(self):
        """Test evaluate_metrics wrapper with identical images."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        metrics = evaluate_metrics(img, img)
        assert metrics["mae"] == 0.0
        assert metrics["ssim"] == 1.0
        assert metrics["psnr"] == 100.0
        assert metrics["estimated_kaggle_score"] == 100.0

    def test_evaluate_metrics_score_calc(self):
        """Test evaluate_metrics score calculation."""
        # MAE = 50
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 50
        metrics = evaluate_metrics(img1, img2)
        assert metrics["mae"] == 50.0
        # Score = max(0, 100 - 50*2) = 0
        assert metrics["estimated_kaggle_score"] == 0.0

        # MAE = 25
        img3 = np.ones((100, 100, 3), dtype=np.uint8) * 25
        metrics = evaluate_metrics(img1, img3)
        assert metrics["mae"] == 25.0
        # Score = 100 - 25*2 = 50
        assert metrics["estimated_kaggle_score"] == 50.0

    def test_single_pixel(self):
        """Test with single pixel images."""
        img1 = np.array([[[0]]], dtype=np.uint8)
        img2 = np.array([[[255]]], dtype=np.uint8)
        mae = calculate_mae(img1, img2)
        assert mae == 255.0
