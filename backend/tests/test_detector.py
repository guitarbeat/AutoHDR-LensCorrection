import sys
import os
import torch
import pytest

# Add backend directory to sys.path so we can import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector import DistortionDetector

def test_detector_output_shape():
    """
    Test that the DistortionDetector outputs the correct shape:
    (Batch Size, 5 Coefficients: k1, k2, k3, p1, p2)
    """
    model = DistortionDetector(pretrained=False) # Use False to speed up test (no download)
    model.eval()

    # Create dummy input: (Batch=1, Channels=3, Height=224, Width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    # The crucial check: ensure output shape is (1, 5)
    assert output.shape == (1, 5), f"Expected shape (1, 5), but got {output.shape}"

def test_detector_output_values():
    """
    Test that the model outputs valid float values.
    """
    model = DistortionDetector(pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224) # Batch size of 2

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 5), f"Expected shape (2, 5), but got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
