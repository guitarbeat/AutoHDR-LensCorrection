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
* Set up a Python/PyTorch backend to execute the OpenCV logic and neural network inferences. Maximize the use of cloud infrastructure for backend operations.
* **Cloud-First Data & Processing Strategy:** Avoid downloading massive datasets locally. Utilize Kaggle Notebooks (Kernels) or Google Colab (with Google Drive integration) to process data directly in the cloud.
* **Selective Data Access:** Use the Kaggle API for targeted file downloads using the CLI, or use pandas (`nrows`, `usecols`) to load only necessary data subsets into memory.
* **Dataset Reference:** [Automatic Lens Correction Data](https://www.kaggle.com/competitions/automatic-lens-correction/data)


* **Phase 2: Model Engineering & Methodology**
* **Blind Geometric Distortion Correction:** Shift away from traditional checkerboard-based multi-image calibration (e.g. OpenCV Zhang's method) which requires physical lens profiles. Instead, focus on single-image "Blind" correction suitable for unknown user-uploaded real estate photos.
* **Displacement Field Regression:** Utilize a Convolutional Neural Network (CNN) or a ResNet-based architecture to detect curved structural lines and predict a pixel-wise displacement field (flow map) to straighten the image (as researched in CVPR 2019: *"Blind Geometric Distortion Correction on Images Through Deep Learning"*).
* **Alternative Approaches:** Evaluate **Hybrid Calibration Networks** (predicting Brown-Conrady coefficients via CNN to feed into classical OpenCV un-distorters) or **Generative Adversarial Networks (GANs)** containing a Discriminator trained to strictly penalize non-rectilinear architectural curves.
* Adapt distortion correction algorithms standard in biomedical imaging (such as those used for correcting circumferential motion in catheters or wide-angle endoscopes) to architectural photography.


* **Phase 3: Hardware Routing & Optimization**
* Program the application to dynamically route computational tasks. Assign lightweight tasks like EXIF extraction and initial image parsing to the Arm CPU cores.


* Route the heavy matrix calculations for the deep learning inference to the NVIDIA DGX Spark tensor cores to achieve the ultra-low latency demanded by AutoHDR.




* **Phase 4: Evaluation & Validation**
* Integrate automated evaluation metrics into the backend to score the model's outputs.
* Validate the geometric accuracy of the corrected images against the distorted inputs using strict, industry-standard metrics:
  * **Edge alignment**: do edges in your output match the ground truth?
  * **Line straightness**: are lines that should be straight actually straight?
  * **Gradient orientation**: are structural directions preserved correctly?
  * **Structural similarity**: does the overall structure match?
  * **Pixel accuracy**: are the pixels close to ground truth?