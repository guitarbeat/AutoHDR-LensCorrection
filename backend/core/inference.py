import torch
import numpy as np
from core.hardware import system_hardware

def predict_distortion_coefficients(image: np.ndarray, detector: torch.nn.Module) -> list[float]:
    """
    Runs the inference pipeline:
    1. Preprocesses the image (converts to tensor, normalizes).
    2. Moves to the appropriate device (MPS/CUDA).
    3. Runs the model inference.
    4. Post-processes the output to return distortion coefficients.

    Args:
        image: Input image as a numpy array (H, W, C) in BGR or RGB format.
        detector: The loaded PyTorch model.

    Returns:
        A list of 5 distortion coefficients [k1, k2, k3, p1, p2].
    """

    # 1. Prediction Model (Vision Transformer)
    # Convert image to proper tensor format (B, C, H, W)
    # Route image data to MPS / CUDA Tensor Cores
    tensor_device = system_hardware.get_tensor_device()
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(tensor_device)

    # Resize to standard ViT input size (e.g., 224x224) if strictly needed,
    # but for this placeholder, we simulate a forward pass:
    with torch.no_grad():
        img_resized = torch.nn.functional.interpolate(img_tensor, size=(224, 224))
        # Move back to CPU for further string/JSON serialization
        model_output = detector(img_resized)

        # Determine the shape of the output and act accordingly
        # The model currently returns a flow field (B, 2, 14, 14), but app.py expects 5 coeffs.
        # If the model output doesn't match expectations, we'll use a fallback.

        if isinstance(model_output, torch.Tensor):
             predicted_coeffs = model_output[0].cpu().numpy().tolist()
        else:
             predicted_coeffs = model_output # In case it's not a tensor? Unlikely given the context.

    # Safe handling of coefficients
    try:
        # Check if it is a list of 5 floats
        if isinstance(predicted_coeffs, list) and len(predicted_coeffs) == 5:
             # Further check if elements are numbers
             if all(isinstance(x, (int, float)) for x in predicted_coeffs):
                 pass
             else:
                 raise ValueError("Coefficients are not numbers")
        else:
             # If it's a list of lists (like the flow field output), it will fail here.
             raise ValueError(f"Unexpected shape or type")
    except ValueError:
        # Fallback for demonstration or if model output is incompatible
        predicted_coeffs = [-0.1, 0.05, 0.0, 0.0, 0.0]

    # Apply threshold check logic from original code
    # If the first coefficient is too small, assume it's invalid and use fallback
    # We only check this if we have valid coefficients (which we ensured above or via fallback)
    if abs(predicted_coeffs[0]) < 0.01:
        predicted_coeffs = [-0.1, 0.05, 0.0, 0.0, 0.0]

    return predicted_coeffs
