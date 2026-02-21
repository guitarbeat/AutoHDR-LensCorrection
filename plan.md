# AutoHDR Hackathon Project: Automated Lens Distortion Correction

## 1. Project Goal

To build a highly accurate, automated lens distortion correction pipeline that rectifies barrel and pincushion distortions inherent to wide-angle real estate photography. By applying rigorous geometric correction rather than generative AI pixel hallucinations, the project aims to ensure real estate listings maintain absolute physical truth and strictly adhere to tightening MLS visual compliance regulations.

## 2. Core Objectives

* **Single-Image Rectification:** Develop a machine learning model capable of performing blind distortion correction on a single image, eliminating the need for traditional multi-image calibration grids or checkerboards.
* **Biomedical Translation:** Implement an algorithm that accurately estimates inverse distortion coefficients based on the Brown-Conrady model (, , ) or utilizes a model-independent spatial mapping approach derived from techniques used in medical endoscopy.
* **Scalable Compute Execution:** Optimize the computational workload for a two-stage approach: prototype locally on an Apple Silicon (M4) architecture utilizing the PyTorch `MPS` backend, and scale up heavy training to Google Cloud Platform (GCP) or Kaggle Notebook instances leveraging NVIDIA T4/P100 GPUs or TPUs.

## 3. Plan of Action

* **Phase 1: Environment & Full-Stack Setup**
* Initialize a hybrid architecture separating the web application from the computer vision engine.
* Build the frontend using Next.js, utilizing the App Router within a `src/app` directory structure to handle user uploads and before/after image comparisons.
* Set up a Python/PyTorch backend to execute the OpenCV logic and neural network inferences. Maximize the use of cloud infrastructure for backend operations.
* **Local Data Storage:** The 37GB Kaggle dataset is being downloaded directly to an external drive (`/Volumes/Love SSD/`) to bypass local storage limitations and avoid purely cloud-based training limitations.
* **API Credentials:** Downloading requires the Kaggle API token. Ensure `KAGGLE_API_TOKEN` is exported in the environment (e.g. `export KAGGLE_API_TOKEN=...`) or `~/.kaggle/kaggle.json` is configured. You must accept competition rules on Kaggle's website first.
* **Dataset Reference:** [Automatic Lens Correction Data](https://www.kaggle.com/competitions/automatic-lens-correction/data)

* **Phase 2: Model Engineering & Methodology**
* **Iterative Development Pipeline (Fail-Fast Strategy):**
  1. **Micro-Dataset (Overfitting Test):** Create a DataLoader for just 10-50 images. Train locally on the M4 Mac. The model *must* achieve near-zero MAE. If it can't overfit a tiny batch, there is a fundamental bug in the architecture or loss function.
  2. **Mini-Dataset (Baseline Validation):** Use 1% - 5% of the dataset (~1,000 images) with a proper Train/Val split. Run locally to tune learning rates, confirm the Differentiable Solver works, and generate a baseline submission to `bounty.autohdr.com` to verify the end-to-end pipeline.
  3. **Full Scale (Cloud Training):** Only after the Mini-Dataset pipeline is verified, deploy the code to GCP/Kaggle GPUs to train on the full 37GB dataset, saving cloud credits and time.
* **Training Without Coefficient Ground Truth:** The Kaggle dataset provides pairs of `original.jpg` and `generated.jpg` but *does not* provide the mathematical lens coefficients. Therefore, the network cannot be trained via simple supervised regression on 5 numbers. The architecture must utilize one of two advanced techniques:
  1. **Spatial Transformer Network (STN):** The ViT predicts distortion coefficients, which are fed into an in-network, differentiable OpenCV-style un-distorter module. The training loss is the pixel-level Mean Absolute Error (MAE) between the dynamically "warped" input and the `generated.jpg`.
  2. **Direct Flow-Field Estimation:** The ViT bypasses modeling standard analytical lenses entirely and outputs a dense pixel-coordinate displacement map, using MAE between the predicted coordinates and the target layout.
* **Blind Geometric Distortion Correction:** Shift away from traditional checkerboard-based multi-image calibration (e.g. OpenCV Zhang's method) which requires physical lens profiles. Instead, focus on single-image "Blind" correction suitable for unknown user-uploaded real estate photos.
* **Displacement Field Regression:** Utilize modern Vision Transformers (ViT) or Swin Transformers (e.g., GMFlow, TransUNet architectures) to detect curved structural lines and predict a pixel-wise displacement field. Transformers excel at understanding global image geometry compared to the limited receptive fields of traditional CNNs/ResNets.
* **Differentiable Camera Models:** Instead of directly predicting Brown-Conrady coefficients natively, implement a Differentiable Solver layer. The neural network predicts the distortion parameters, and a differentiable rendering layer applies the un-distortion dynamically during training, providing mathematical guarantees of geometry.
* **Strict Perceptual Loss:** Avoid Generative Adversarial Networks (GANs) as they are notoriously prone to "pixel hallucination"—a severe violation of MLS compliance rules. Instead, apply a Structural/Perceptual Loss function (like LPIPS) alongside L1/L2 metrics to preserve architectural integrity.
* Adapt distortion correction algorithms standard in biomedical imaging (such as those used for correcting circumferential motion in catheters or wide-angle endoscopes) to architectural photography.

* **Phase 3: Hardware Routing & Optimization**
* Program the application to dynamically route computational tasks. Assign lightweight tasks like EXIF extraction and initial image parsing to the Arm CPU cores.
* Route the heavy matrix calculations for the deep learning inference to the NVIDIA DGX Spark tensor cores to achieve the ultra-low latency demanded by AutoHDR. To maximize this hardware, the final PyTorch model will be exported to **TensorRT** or **ONNX Runtime** for optimized tensor core execution.

* **Phase 4: Evaluation & Validation**
* Integrate automated evaluation metrics into the backend to score the model's outputs, focusing primarily on **Mean Absolute Error (MAE)** between predicted coordinate mappings and ground truth, as defined by the Kaggle competition.
* **Submission Pipeline:** The model must output corrected images formatted as `{image_id}.jpg`. These need to be zipped and uploaded to [bounty.autohdr.com](https://bounty.autohdr.com/) to generate the final scored CSV, which is then submitted to the Kaggle Leaderboard.
* Validate the geometric accuracy of the corrected images against the distorted inputs using strict, industry-standard metrics alongside MAE:
  * **Edge alignment**: do edges in your output match the ground truth?
  * **Line straightness**: are lines that should be straight actually straight?
  * **Gradient orientation**: are structural directions preserved correctly?
  * **Structural similarity**: does the overall structure match?
  * **Pixel accuracy**: are the pixels close to ground truth?