import torch
import sys
import os

# Add backend directory to sys.path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.detector import DistortionDetector


def test_output_shape():
    """
    Test that the DistortionDetector model outputs 5 coefficients:
    [k1, k2, k3, p1, p2].
    """
    model = DistortionDetector(
        pretrained=False
    )  # Use pretrained=False to speed up test
    model.eval()

    # Create a dummy input tensor: Batch size 1, 3 channels, 224x224
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    # Check that the output shape is (1, 5)
    assert output.shape == (
        1,
        5,
    ), f"Expected output shape (1, 5), but got {output.shape}"


def test_forward_pass_values():
    """
    Test that the forward pass returns valid float values.
    """
    model = DistortionDetector(pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 5)
    assert not torch.isnan(output).any(), "Output contains NaNs"
    assert not torch.isinf(output).any(), "Output contains Infs"
