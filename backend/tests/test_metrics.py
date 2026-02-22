import numpy as np
import pytest
import sys
import os

# Add the project root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.evaluation.metrics import calculate_mae

def test_calculate_mae_same_shape():
    """Test MAE calculation when images have the same shape."""
    img1 = np.ones((10, 10, 3), dtype=np.uint8) * 10
    img2 = np.ones((10, 10, 3), dtype=np.uint8) * 20
    # MAE should be |10 - 20| = 10
    mae = calculate_mae(img1, img2)
    assert mae == 10.0

def test_calculate_mae_different_shape():
    """Test MAE calculation when images have different shapes."""
    # Predicted image is 100x100
    img_pred = np.zeros((100, 100, 3), dtype=np.uint8)
    # Ground truth is smaller
    img_gt = np.zeros((50, 50, 3), dtype=np.uint8)

    # img_gt should be resized to (100, 100)
    # Both are zeros, so MAE should be 0
    mae = calculate_mae(img_pred, img_gt)
    assert mae == 0.0

def test_calculate_mae_different_shape_resize_logic():
    """Test that resizing logic works as expected with constant values."""
    # Predicted is 100x100 with value 10
    img_pred = np.ones((100, 100, 3), dtype=np.uint8) * 10
    # GT is 50x50 with value 20
    img_gt = np.ones((50, 50, 3), dtype=np.uint8) * 20

    # Resizing a constant color image should result in the same constant color image
    # So |10 - 20| = 10
    mae = calculate_mae(img_pred, img_gt)
    assert mae == 10.0

def test_calculate_mae_exact_match():
    """Test MAE is 0 for identical images."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mae = calculate_mae(img, img)
    assert mae == 0.0
