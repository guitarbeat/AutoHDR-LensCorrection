import sys
import os
import numpy as np

# Add the project root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.evaluation.metrics import calculate_mae

def test_calculate_mae_same_shape():
    """Test MAE calculation with images of the same shape."""
    img_pred = np.zeros((100, 100, 3), dtype=np.uint8)
    img_gt = np.zeros((100, 100, 3), dtype=np.uint8)

    mae = calculate_mae(img_pred, img_gt)
    assert mae == 0.0

    img_pred = np.ones((100, 100, 3), dtype=np.uint8) * 10
    img_gt = np.ones((100, 100, 3), dtype=np.uint8) * 20

    mae = calculate_mae(img_pred, img_gt)
    assert mae == 10.0

def test_calculate_mae_different_shape():
    """Test MAE calculation with images of different shapes."""
    # Predict: (100, 100, 3), GT: (50, 50, 3)
    img_pred = np.zeros((100, 100, 3), dtype=np.uint8)
    img_gt = np.zeros((50, 50, 3), dtype=np.uint8)

    # Both are zeros, so MAE should be 0.0
    mae = calculate_mae(img_pred, img_gt)
    assert mae == 0.0

def test_calculate_mae_different_shape_values():
    """Test MAE calculation with different shapes and specific values."""
    # Predict: (100, 100, 3) with value 10
    img_pred = np.ones((100, 100, 3), dtype=np.uint8) * 10
    # GT: (50, 50, 3) with value 20
    img_gt = np.ones((50, 50, 3), dtype=np.uint8) * 20

    # GT will be resized to (100, 100, 3). Since it's constant, it remains 20.
    # MAE = |10 - 20| = 10
    mae = calculate_mae(img_pred, img_gt)
    # Allow small floating point error if interpolation causes it,
    # but for constant images it should be exact or very close.
    assert abs(mae - 10.0) < 1e-5

def test_calculate_mae_different_aspect_ratio():
    """Test MAE calculation with different aspect ratios."""
    # Predict: (100, 200, 3) -> Height 100, Width 200
    img_pred = np.zeros((100, 200, 3), dtype=np.uint8)
    # GT: (50, 50, 3)
    img_gt = np.zeros((50, 50, 3), dtype=np.uint8)

    # Both are zeros, so MAE should be 0.0
    mae = calculate_mae(img_pred, img_gt)
    assert mae == 0.0
