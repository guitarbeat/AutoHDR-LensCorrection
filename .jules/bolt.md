## 2024-05-22 - Backend Image Processing Bottleneck
**Learning:** Resizing large images (e.g., 12MP) on the GPU using PyTorch 'interpolate' is significantly slower than CPU-based 'cv2.resize' due to the overhead of transferring the full-resolution image to the GPU.
**Action:** When the model input is small (e.g., 224x224), always resize on CPU using optimized libraries like OpenCV before converting to tensors and transferring to the device.
