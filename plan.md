Here is the markdown text explaining your goals, objectives, and plan of action. You can copy and paste this directly into your AI agent to give it the necessary context to help you build the project.

# AutoHDR Hackathon Project: Automated Lens Distortion Correction

## 1. Project Goal

To build a highly accurate, automated lens distortion correction pipeline that rectifies barrel and pincushion distortions inherent to wide-angle real estate photography. By applying rigorous geometric correction rather than generative AI pixel hallucinations, the project aims to ensure real estate listings maintain absolute physical truth and strictly adhere to tightening MLS visual compliance regulations.

## 2. Core Objectives

* **Single-Image Rectification:** Develop a machine learning model capable of performing blind distortion correction on a single image, eliminating the need for traditional multi-image calibration grids or checkerboards.
* **Biomedical Translation:** Implement an algorithm that accurately estimates inverse distortion coefficients based on the Brown-Conrady model (, , ) or utilizes a model-independent spatial mapping approach derived from techniques used in medical endoscopy.
* **Hardware-Specific Execution:** Optimize the computational workload specifically for the hackathon's provided hardware, leveraging the ASUS Ascent GX10 workstations configured with the NVIDIA DGX Spark architecture and power-efficient Arm CPU cores.



## 3. Plan of Action

* **Phase 1: Environment & Full-Stack Setup**
* Initialize a hybrid architecture separating the web application from the computer vision engine.
* Build the frontend using Next.js, utilizing the App Router within a `src/app` directory structure to handle user uploads and before/after image comparisons.
* Set up a Python/PyTorch backend to execute the OpenCV logic and neural network inferences.


* **Phase 2: Model Engineering**
* Adapt distortion correction algorithms standard in biomedical imaging (such as those used for correcting circumferential motion in catheters or wide-angle endoscopes) to architectural photography.
* Utilize a Convolutional Neural Network (CNN) or a ResNet-based architecture to detect curved structural lines and predict the necessary displacement fields.


* **Phase 3: Hardware Routing & Optimization**
* Program the application to dynamically route computational tasks. Assign lightweight tasks like EXIF extraction and initial image parsing to the Arm CPU cores.


* Route the heavy matrix calculations for the deep learning inference to the NVIDIA DGX Spark tensor cores to achieve the ultra-low latency demanded by AutoHDR.




* **Phase 4: Evaluation & Validation**
* Integrate automated evaluation metrics into the backend to score the model's outputs.
* Validate the geometric accuracy of the corrected images against the distorted inputs using strict, industry-standard metrics including Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Reprojection Error (RPE).