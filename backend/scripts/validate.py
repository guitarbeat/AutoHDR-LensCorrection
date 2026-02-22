import os
import json
import numpy as np
import cv2
import pandas as pd
from typing import List, Dict

# Assumes run from backend directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import evaluate_metrics
from core.distortion import undistort_image
from models.detector import load_model, predict_distortion_parameters
from core.hardware import system_hardware
import torch

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

def run_validation_suite(num_images=5):
    """
    Simulates running the AutoHDR pipeline on a batch of images to score the model's performance
    based on the strictly requested metrics (edge alignment, line straightness, etc.).
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
        
        # 1. Fetch data (In real setup: from Kaggle dataframe paths; here: generate dummy)
        distorted_img = generate_dummy_image()
        
        # 2. Predict Model Coefficients on Tensor Device
        predicted_coeffs = predict_distortion_parameters(detector, distorted_img, tensor_device)

        # 3. Apply Correction Pipeline on I/O Device / standard numpy runtime
        corrected_img, info = undistort_image(distorted_img, predicted_coeffs)
        
        # 4. Run Evaluation Metrics
        metrics = evaluate_metrics(distorted_img, corrected_img)
        metrics['image_id'] = f"sample_{i+1:03d}"
        results.append(metrics)

    # 5. Aggregate Results
    print("\n--- AutoHDR Validation Complete ---")
    df = pd.DataFrame(results)
    
    # Calculate geometric accuracy averages
    summary = {
        "mean_edge_alignment": df['edge_alignment'].mean(),
        "mean_line_straightness": df['line_straightness'].mean(),
        "mean_gradient_orientation": df['gradient_orientation'].mean(),
        "mean_structural_similarity": df['structural_similarity'].mean(),
        "mean_pixel_accuracy": df['pixel_accuracy'].mean(),
        "mean_psnr": df['psnr'].mean(),
        "mean_rpe": df['rpe'].mean(),
    }
    
    print("\nAggregate Verification Metrics:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
        
    # Export to JSON
    report_path = os.path.join(os.path.dirname(__file__), "validation_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "summary": summary,
            "samples": results
        }, f, indent=4)
    print(f"\nSaved detailed validation results to: {report_path}")

if __name__ == "__main__":
    run_validation_suite()
