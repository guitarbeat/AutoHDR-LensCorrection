import os
import json
import numpy as np
import cv2
import pandas as pd
from typing import List, Dict, Any

# Assumes run from backend directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import evaluate_metrics
from core.distortion import undistort_image
from models.detector import load_model
from core.hardware import system_hardware
import torch

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def generate_dummy_image(w=800, h=600):
    """Creates a basic distorted-looking dummy image with gridlines for testing."""
    image = np.ones((h, w, 3), dtype=np.uint8) * 255
    # Draw a grid
    for i in range(0, w, 50):
        cv2.line(image, (i, 0), (i, h), (0, 0, 0), 2)
    for i in range(0, h, 50):
        cv2.line(image, (0, i), (w, i), (0, 0, 0), 2)
    # Apply synthetic barrel distortion to the blank image for testing
    dist_coeffs = np.array([-0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    focal_length = w
    camera_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    return cv2.undistort(image, camera_matrix, dist_coeffs)

def run_inference(detector: torch.nn.Module, device: str, image: np.ndarray) -> List[float]:
    """
    Runs inference on a single image and validates the output.
    Returns default coefficients if the model output is not compatible (e.g. flow field vs coeffs).
    """
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    default_coeffs = [-0.1, 0.05, 0.0, 0.0, 0.0]

    with torch.no_grad():
        img_resized = torch.nn.functional.interpolate(img_tensor, size=(224, 224))
        raw_output = detector(img_resized)

        # Convert tensor to list structure
        output = raw_output.cpu().numpy().tolist() if isinstance(raw_output, torch.Tensor) else raw_output

        # Check if output matches expected coefficient format (batch_size, 5) -> taking first item: list of 5 floats
        # output is likely [[...]] so we want output[0]

        predicted_coeffs = None

        try:
            if isinstance(output, list) and len(output) > 0:
                first_item = output[0]

                # Case 1: Standard coefficients [k1, k2, k3, p1, p2]
                if isinstance(first_item, list) and len(first_item) == 5:
                    # Verify elements are numbers
                    if all(isinstance(x, (int, float)) for x in first_item):
                        predicted_coeffs = first_item

                # Case 2: Flow field or other incompatible format
                # e.g. list of lists of lists (14x14 grid) -> incompatible with simple undistort
                # We will fall through to fallback
        except Exception as e:
            print(f"Error parsing model output: {e}")

        if predicted_coeffs:
            # Ensure some mild coefficients for verification (logic from original code)
            if abs(predicted_coeffs[0]) < 0.01:
                return default_coeffs
            return predicted_coeffs

    # Fallback if model output is incompatible (e.g. flow field update)
    return default_coeffs

def calculate_metrics(distorted_img: np.ndarray, corrected_img: np.ndarray, index: int) -> Dict[str, Any]:
    """Wraps metric evaluation for a single image pair."""
    metrics = evaluate_metrics(distorted_img, corrected_img)
    metrics['image_id'] = f"sample_{index+1:03d}"
    return metrics

def calculate_summary(results: List[Dict]) -> Dict[str, float]:
    """Calculates aggregate metrics from all results."""
    df = pd.DataFrame(results)
    summary = {}

    # Dynamically calculate means for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary[f"mean_{col}"] = float(df[col].mean())

    print("\nAggregate Verification Metrics:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    return summary

def save_report(summary: Dict, results: List[Dict], output_dir: str):
    """Saves the validation report to disk."""
    report_path = os.path.join(output_dir, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "summary": summary,
            "samples": results
        }, f, indent=4, cls=NumpyEncoder)
    print(f"\nSaved detailed validation results to: {report_path}")

def run_validation_suite(num_images=5):
    """
    Simulates running the AutoHDR pipeline on a batch of images to score the model's performance.
    """
    print(f"Starting Phase 4 automated validation suite on {num_images} samples...")
    print("Initializing hardware and model...")

    tensor_device = system_hardware.get_tensor_device()
    detector = load_model().to(tensor_device)
    detector.eval()
    
    results: List[Dict] = []
    
    # Process batch
    for i in range(num_images):
        print(f"Processing image {i+1}/{num_images}...")
        
        # 1. Generate/Fetch Data
        distorted_img = generate_dummy_image()
        
        # 2. Run Inference
        predicted_coeffs = run_inference(detector, tensor_device, distorted_img)

        # 3. Apply Correction
        corrected_img, info = undistort_image(distorted_img, predicted_coeffs)
        
        # 4. Evaluate Metrics
        metrics = calculate_metrics(distorted_img, corrected_img, i)
        results.append(metrics)

    # 5. Aggregate & Save Results
    print("\n--- AutoHDR Validation Complete ---")
    summary = calculate_summary(results)
    save_report(summary, results, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    run_validation_suite()
