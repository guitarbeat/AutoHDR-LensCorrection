import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2

# Add the project root to sys.path so we can import backend.core.distortion
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.core.distortion import undistort_image

class TestDistortion(unittest.TestCase):
    def setUp(self):
        # Create a dummy image (100x100, 3 channels)
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Dummy distortion coefficients [k1, k2, k3, p1, p2]
        self.distortion_coeffs = [0.1, 0.1, 0.0, 0.0, 0.0]

    def test_undistort_execution_with_custom_camera_matrix(self):
        """Test that undistort_image executes correctly with a custom camera matrix."""
        # Create a custom camera matrix
        focal_length = 100
        center_x = 50
        center_y = 50
        custom_matrix = np.array([
            [focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1]
        ], dtype=np.float32)

        # Run the function
        corrected_image, info = undistort_image(self.image, self.distortion_coeffs, camera_matrix=custom_matrix)

        # Basic assertions
        self.assertIsInstance(corrected_image, np.ndarray)
        self.assertIsInstance(info, dict)
        self.assertEqual(info['original_shape'], (100, 100))
        # Note: Depending on distortion, output shape might change due to cropping logic

        # Verify the custom matrix was used (indirectly, by mocking)
        with patch('backend.core.distortion.cv2.getOptimalNewCameraMatrix') as mock_get_optimal:
            # Setup mock return values to avoid errors downstream if possible,
            # though we only care about the call arguments here.
            # getOptimalNewCameraMatrix returns (new_camera_matrix, roi)
            mock_get_optimal.return_value = (custom_matrix, (0, 0, 100, 100))

            with patch('backend.core.distortion.cv2.undistort') as mock_undistort:
                mock_undistort.return_value = self.image

                undistort_image(self.image, self.distortion_coeffs, camera_matrix=custom_matrix)

                # Check getOptimalNewCameraMatrix called with custom_matrix
                args, _ = mock_get_optimal.call_args
                # args[0] is cameraMatrix
                np.testing.assert_array_equal(args[0], custom_matrix)

    def test_undistort_execution_default_matrix(self):
        """Test that undistort_image executes correctly without a camera matrix (default)."""
        corrected_image, info = undistort_image(self.image, self.distortion_coeffs, camera_matrix=None)

        self.assertIsInstance(corrected_image, np.ndarray)
        self.assertIsInstance(info, dict)
        self.assertEqual(info['original_shape'], (100, 100))

if __name__ == '__main__':
    unittest.main()
