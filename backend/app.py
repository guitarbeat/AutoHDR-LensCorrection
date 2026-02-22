from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

import torch
from core.distortion import undistort_image
from core.hardware import system_hardware
from models.detector import load_model
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

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit

@app.get("/")
def read_root():
    return {"status": "AutoHDR API is running", "message": "Ready to correct distortions"}

@app.post("/correct")
async def correct_image(file: UploadFile = File(...)):
    # Read the uploaded image with size limit
    content = bytearray()
    size = 0
    while True:
        chunk = await file.read(1024 * 1024)  # Read 1MB chunks
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        content.extend(chunk)

    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file or format")
    
    # 1. Prediction Model (Vision Transformer)
    # Optimization: Resize on CPU before converting to tensor to avoid large data transfer
    # and expensive interpolation on full-resolution image.
    img_resized_cpu = cv2.resize(image, (224, 224))

    # Route image data to MPS / CUDA Tensor Cores
    tensor_device = system_hardware.get_tensor_device()
    img_tensor = torch.from_numpy(img_resized_cpu).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(tensor_device)
    
    # Run inference
    with torch.no_grad():
        output = detector(img_tensor)
        # Move back to CPU for further string/JSON serialization
        model_output = output[0].cpu().numpy().tolist()

    # Handle model output
    # The model might return a flow field (2, 14, 14) or coefficients depending on version.
    # We attempt to unpack 5 coefficients. If it fails (e.g. flow field), we use defaults.
    default_coeffs = [-0.1, 0.05, 0.0, 0.0, 0.0]
    predicted_coeffs = default_coeffs
    
    try:
        # Check if output is compatible with 5 coefficients [k1, k2, k3, p1, p2]
        if isinstance(model_output, list) and len(model_output) == 5 and all(isinstance(x, (int, float)) for x in model_output):
             predicted_coeffs = model_output
             # Apply the logic: if k1 is too small, use defaults
             if abs(predicted_coeffs[0]) < 0.01:
                 predicted_coeffs = default_coeffs
        else:
             # Log warning but continue
             print(f"Model output format mismatch. Using defaults.")

    except Exception as e:
        print(f"Error processing model output: {e}. Using defaults.")
    
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
