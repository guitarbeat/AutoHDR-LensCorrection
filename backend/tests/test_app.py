import sys
import os
import io
import cv2
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.detector import DistortionDetector

# Patch load_model to avoid downloading weights during import
# We also mock system_hardware to avoid hardware checks
with patch('models.detector.load_model') as mock_load,      patch('core.hardware.system_hardware') as mock_hw:

    mock_model = MagicMock()
    mock_load.return_value = mock_model

    # Configure mock hardware
    mock_hw.get_tensor_device.return_value = torch.device('cpu')

    from app import app

client = TestClient(app)

def create_dummy_image():
    # Create a simple 100x100 black image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded_img = cv2.imencode('.jpg', img)
    return io.BytesIO(encoded_img.tobytes())

class TestApp:
    def test_correct_image_tensor_mismatch_fails(self):
        """
        Test that simulates the current bug: detector returns a 4D tensor,
        but app expects a list of 5 floats.
        This verifies the current code is broken given the current detector implementation.
        """
        # Mock detector to return a tensor of shape (B, 2, 14, 14) like the current DistortionDetector
        with patch('app.detector') as mock_det:
             # Create a tensor that simulates the output (B, 2, 14, 14)
             tensor_output = torch.randn(1, 2, 14, 14)
             mock_det.return_value = tensor_output

             img_file = create_dummy_image()

             # Expect ValueError during unpacking
             with pytest.raises(ValueError) as excinfo:
                 client.post("/correct", files={"file": ("test.jpg", img_file, "image/jpeg")})

             # Verify it's the unpacking error
             assert "unpack" in str(excinfo.value)

    def test_correct_image_happy_path_mocked(self):
        """
        Test that verifies app works if detector returns 5 coefficients.
        """
        with patch('app.detector') as mock_det:
             # Return tensor of shape (B, 5)
             tensor_output = torch.tensor([[0.1, 0.2, 0.0, 0.0, 0.0]])
             mock_det.return_value = tensor_output

             img_file = create_dummy_image()
             response = client.post("/correct", files={"file": ("test.jpg", img_file, "image/jpeg")})

             assert response.status_code == 200
             data = response.json()
             assert data["status"] == "success"
             assert "metrics" in data

    def test_correct_image_integration_real_model(self):
        """
        Test that uses the real DistortionDetector class (with pretrained=False)
        to verify that the app and model interfaces match.
        """
        # Create a real instance (no weights download)
        real_model = DistortionDetector(pretrained=False)
        real_model.eval()

        # Patch the global detector object in app module with the real model
        with patch('app.detector', real_model):
             img_file = create_dummy_image()
             response = client.post("/correct", files={"file": ("test.jpg", img_file, "image/jpeg")})

             if response.status_code != 200:
                 print(response.json())

             assert response.status_code == 200
             data = response.json()
             assert data["status"] == "success"
             assert "metrics" in data
