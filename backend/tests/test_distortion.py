import numpy as np
import cv2
from backend.core.distortion import undistort_image


def test_undistort_image_basic():
    # Create a simple 100x100 black image
    h, w = 100, 100
    image = np.zeros((h, w, 3), dtype=np.uint8)
    # Draw a white circle in the center
    cv2.circle(image, (w // 2, h // 2), 30, (255, 255, 255), -1)

    # Dummy distortion coefficients: [k1, k2, k3, p1, p2]
    # Small coefficients to avoid extreme warping
    distortion_coeffs = [0.1, 0.01, 0.001, 0.0, 0.0]

    corrected_image, info = undistort_image(image, distortion_coeffs)

    assert isinstance(corrected_image, np.ndarray)
    assert isinstance(info, dict)
    assert "original_shape" in info
    assert "new_shape" in info
    assert "roi" in info
    assert info["original_shape"] == (w, h)

    # Check that the corrected image is not empty
    assert corrected_image.size > 0
    assert corrected_image.shape[2] == 3


def test_undistort_image_with_camera_matrix():
    h, w = 100, 100
    image = np.zeros((h, w, 3), dtype=np.uint8)
    distortion_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion

    # defined camera matrix
    focal_length = 100
    center_x = 50
    center_y = 50
    camera_matrix = np.array(
        [[focal_length, 0, center_x], [0, focal_length, center_y], [0, 0, 1]],
        dtype=np.float32,
    )

    corrected_image, info = undistort_image(
        image, distortion_coeffs, camera_matrix=camera_matrix
    )

    # Since distortion is 0, the image should remain roughly the same,
    # but getOptimalNewCameraMatrix might slightly crop or not depending on alpha=1
    assert corrected_image.shape[0] <= h
    assert corrected_image.shape[1] <= w

    # Check info
    assert info["original_shape"] == (w, h)


def test_undistort_image_cropping():
    # Test that cropping logic works.
    # We'll use coefficients that we know produce valid ROI
    h, w = 200, 200
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    # Some distortion
    distortion_coeffs = [-0.2, 0.1, 0, 0, 0]

    corrected_image, info = undistort_image(image, distortion_coeffs)

    roi = info["roi"]
    x, y, w_roi, h_roi = roi

    # If ROI is valid (w_roi > 0 and h_roi > 0), the output should be cropped to it
    if w_roi > 0 and h_roi > 0:
        assert corrected_image.shape[0] == h_roi
        assert corrected_image.shape[1] == w_roi
    else:
        # If ROI is empty, it returns the full uncropped result (based on code logic)
        # But getOptimalNewCameraMatrix with alpha=1 usually returns valid ROI for reasonable distortion
        pass


def test_undistort_image_grayscale():
    # Test with grayscale image (2D array)
    h, w = 100, 100
    image = np.zeros((h, w), dtype=np.uint8)
    distortion_coeffs = [0.1, 0.1, 0, 0, 0]

    corrected_image, info = undistort_image(image, distortion_coeffs)

    assert len(corrected_image.shape) == 2 or (
        len(corrected_image.shape) == 3 and corrected_image.shape[2] == 1
    )
    # cv2.undistort might return same number of channels
    assert corrected_image.shape[0] > 0
    assert corrected_image.shape[1] > 0
