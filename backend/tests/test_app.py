import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from fastapi.testclient import TestClient

# Add parent directory to path to import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Mock dependencies that are heavy or irrelevant to the image loading logic
# We need to mock them before importing app to avoid initialization side effects
sys.modules["core.hardware"] = MagicMock()
sys.modules["core.distortion"] = MagicMock()
sys.modules["models.detector"] = MagicMock()
sys.modules["evaluation.metrics"] = MagicMock()

# Setup the mocks specifically
mock_hardware = MagicMock()
mock_hardware.system_hardware.get_tensor_device.return_value = "cpu"
sys.modules["core.hardware"].system_hardware = mock_hardware.system_hardware

mock_detector_instance = MagicMock()
# detector(img) -> [coeffs]
# We need the result to be convertible to list
mock_tensor = MagicMock()
mock_tensor.cpu.return_value.numpy.return_value.tolist.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
mock_detector_instance.return_value = [mock_tensor]
sys.modules["models.detector"].load_model.return_value = mock_detector_instance

# Undistort returns (image, info)
# The image must be a numpy array because app.py accesses .shape
dummy_corrected = np.zeros((100, 100, 3), dtype=np.uint8)
sys.modules["core.distortion"].undistort_image.return_value = (dummy_corrected, {})

sys.modules["evaluation.metrics"].evaluate_metrics.return_value = {"score": 0.99}

# Now we can import app
from backend.app import app

client = TestClient(app)

class TestCorrectImage(unittest.TestCase):

    def test_invalid_image_content(self):
        """
        Test uploading a file that is not an image (e.g. text file).
        This should result in a 400 error (after fix).
        Currently it might crash or 500.
        """
        response = client.post(
            "/correct",
            files={"file": ("test.txt", b"This is clearly not an image", "text/plain")}
        )

        # After fix, this should be 400
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Invalid image file"})

    def test_valid_image_content(self):
        """
        Test uploading a valid image.
        """
        # Create a real dummy image using opencv
        import cv2
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, img_encoded = cv2.imencode('.jpg', img)
        img_bytes = img_encoded.tobytes()

        response = client.post(
            "/correct",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["width"], 100)
        self.assertEqual(data["height"], 100)

if __name__ == "__main__":
    unittest.main()
