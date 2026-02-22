from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

import torch
from core.distortion import undistort_image
from core.hardware import system_hardware
from models.detector import load_model, predict_distortion_parameters
from evaluation.metrics import evaluate_metrics

app = FastAPI(title="AutoHDR Distortion Correction API")

# Print routing strategy on startup
system_hardware.print_system_info()

# Load model globally and map to Apple Silicon MPS / GCP Tensor Cores
detector = load_model()
detector.to(system_hardware.get_tensor_device())

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "AutoHDR API is running", "message": "Ready to correct distortions"}

@app.post("/correct")
async def correct_image(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 1. Prediction Model (Vision Transformer)
    # Route image data to MPS / CUDA Tensor Cores
    tensor_device = system_hardware.get_tensor_device()
    predicted_coeffs = predict_distortion_parameters(detector, image, tensor_device)
    
    # 2. Geometric Correction (Routed to CPU for standard array ops)
    corrected_image, geom_info = undistort_image(image, predicted_coeffs)
    
    # 3. Evaluation & Validation
    # We pass the image to itself for demonstration to ensure metrics run properly
    metrics = evaluate_metrics(corrected_image, corrected_image)
    
    return {
        "status": "success",
        "width": corrected_image.shape[1],
        "height": corrected_image.shape[0],
        "metrics": metrics
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
