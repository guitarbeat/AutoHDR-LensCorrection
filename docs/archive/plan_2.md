# AutoHDR Hackathon — Second Brain

> **Hackathon Start:** Feb 21, 2026 ~5:00 PM CST
> **Deadline:** Feb 22, 2026 ~5:00 PM CST (24 hours)
> **Competition:** [Kaggle — Automatic Lens Correction](https://www.kaggle.com/competitions/automatic-lens-correction)
> **Scoring Portal:** [bounty.autohdr.com](https://bounty.autohdr.com/)

---

## 1. Project Goal

Build an automated lens distortion correction pipeline for wide-angle real estate photography. Use rigorous geometric correction (not generative AI hallucinations) to maintain MLS compliance. Deliver a Kaggle leaderboard score.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | Kaggle (`automatic-lens-correction`) |
| Location | `/Volumes/Love SSD/` (external SSD) |
| Train pairs | 23,119 (`{uuid}_g{n}_original.jpg` ↔ `{uuid}_g{n}_generated.jpg`) |
| Test images | 1,000 (`{uuid}_g{n}.jpg`) |
| Resolution | 1367 × 2048 |
| Mean pixel diff | ~12.66 (originals vs generated) |
| Total size | ~37 GB |

**Naming Convention Discovery:** The initial codebase assumed `pair_X/original.jpg` subdirectories. The actual dataset uses flat files with UUID-based naming. This was the first bug caught during triage and required rewriting `dataloader.py`.

---

## 3. Submission Pipeline

```text
Corrected images (.jpg) → zip → upload to bounty.autohdr.com → download submission.csv → submit to Kaggle
```

The scoring is external — we never compute the final Kaggle score ourselves. bounty.autohdr.com evaluates a **composite metric**: edge alignment, line straightness, gradient orientation, structural similarity, AND pixel accuracy. This is NOT pure MAE — optimizing only L1 loss may produce blurry results that score poorly on edge/line metrics.

---

## 3.5. Timeline & Decision Points

*Current time: ~7:00 PM CST. Deadline: 5:00 PM tomorrow (~22 hrs remaining.*

| Checkpoint | Target | Decision |
|---|---|---|
| **9 PM tonight** | v3 + per-image optimization both scored | Which heuristic direction to pursue? |
| **Midnight** | U-Net first full-training checkpoint scored | Is U-Net converging fast enough to be viable? |
| **2 PM tomorrow** | Final method chosen, submission polished | Commit to best approach, stop exploring |
| **4 PM tomorrow** | Demo rehearsed, buffer for upload issues | Pencils down |
| **5 PM tomorrow** | Hard deadline | Submit |

**Bail-out rules:**

- If U-Net epoch time > 30 min → switch to Kaggle T4 GPU or abandon U-Net
- If no method beats v1 (24.00) by midnight → focus on per-image optimization with more candidates
- If per-image optimization proxy doesn't correlate with scoring → try training-pair MAE as proxy instead

---

## 4. Phase Log

### Phase 0: Triage & Dataset Check ✅

#### Completed ~5:30 PM

- Verified dataset exists on external SSD (46,238 train files, 1,000 test)
- Confirmed naming conventions differ from codebase assumptions
- Discovered the original ViT model in `detector.py` was **never trained** — outputs random noise
- Discovered `app.py` used hardcoded placeholder coefficients and compared corrected images to themselves for "evaluation"

> **Lesson:** Always audit scaffolded code before building on top of it. The previous session's walkthrough claimed the model was working, but an actual codebase audit revealed it was entirely non-functional.

### Heuristic Iterations

#### Submission Log

| Submission | Approach | Mean Score | Key Metrics | Status |
| :--- | :--- | :--- | :--- | :--- |
| `submission_heuristic.csv` | **Global Baseline (v1)**: Grid search over 50 pairs to find best single setup (`k1=-0.17`, `k2=0.35`, `alpha=0`). | 24.00 | max: 93.05, 17.8% zeros, 5% > 50 | Baseline |
| `submission_v2.csv` | **Adaptive (v2)**: Pick best of 6 candidates per image using Hough line variance, with `alpha=1`. | 3.45 | 44% new zeros | **FAILED** (`alpha=1` + bad proxy) |
| `submission_v3.csv` | **Adaptive (v3)**: Pick best of 5 (no identity) with `alpha=0` using Laplacian variance. | 21.21 | 23.4% zeros | Regression (proxy mismatch) |
| `submission_phase2.csv` | **Phase 2 (Nelder-Mead)**: Continuous per-image optimization using Hough line straightness. | 18.89 | 19.8% zeros, 2.6% > 50 | **FAILED** (proxy anti-correlation) |
| `submission_v4.csv` | **Dimension-Aware (v4)**: Classify by dimensions + tight grid search per bucket. | **29.64** | max: 94.39, 14.3% zeros, 17.9% > 50 | **SUCCESS** (New Best) |

> **Pattern:** Per-image proxy selection (Hough, Laplacian) doesn't work. Both v2 and v3 regressed. The proxies don't correlate with actual scoring. v1's dumb global coefficients remain the best.

#### Hard Rules (learned the hard way)

1. **Always `alpha=0`** — `alpha=1` introduces black borders → scoring penalizes heavily
2. **Never output uncorrected originals** — scoring compares against *generated* ground truth (which IS corrected). Identity = maximum distance
3. **Scoring is composite** — edge alignment + line straightness + gradient orientation + structural similarity + pixel accuracy. NOT just MAE. Optimizing pixel distance alone may produce blurry results that fail on edge metrics.
4. **v1 is the floor** — 24.00 mean. Every new attempt must beat this or it's a regression
5. **Score everything ASAP** — proxies (Hough lines, Laplacian, MAE) may not correlate with actual bounty scores. Only real scores tell the truth.

#### What's been tried

- **Global coefficients** (v1): k1=-0.17, k2=0.35 from grid search. Solid baseline, but 17.8% zeros
- **Per-image preset selection** (v2): 6 presets including identity, Hough lines as metric. Catastrophic failure — 442 new zeros
- **Per-image candidate selection** (v3): 5 candidates (all correction, no identity), Laplacian variance as metric. v1's alpha=0 approach. 56% chose k1=-0.20, k2=0.40

#### Available scripts

| Script | Description |
|---|---|
| `heuristic_baseline.py` | Phase 1 grid search + apply (v1, mean 24.00) |
| `heuristic_v4.py` | **Current Best**: Dimension-aware heuristic (v4, mean 29.64) |

*(Note: Defunct iterations like `v2`, `v3`, and `optimize_per_image.py` were deleted from the repository because they proved proxy metrics anti-correlate with the hidden bounty score.)*

#### Phase 1.5: v4 Iteration Roadmap

Because v4 represents our highest expected-value approach with zero hardware bottlenecks, we will continue iterating on it locally while U-Net trains on Akash.

1. **Zero Clustering Analysis**: The diagnostic script revealed heavily cropped dimensions (e.g., `2048x1357` and `2048x1361`) were failing catastrophically (MAE > 15 up to 66% of the time).
2. **Bucket Expansion**: Using Sequential Thinking, we deduced *why*: the optical center ($C_x, C_y$) is assumed to be $w/2, h/2$. When an image is heavily cropped asymmetrically, this assumption breaks. We expanded to 7 scientifically-derived buckets:
   - `standard` (h=1367)
   - `near_standard_tall` / `near_standard_short` (safe crops)
   - `moderate_crop` (1360-1364, mild coefficient reduction)
   - `heavy_crop` (<1360, catastrophic unmooring, near-identity)
   - `portrait_standard` / `portrait_cropped`
3. **Finer Grid Search**: Our current best combinations were found using a coarse `0.01` step size for `k1` and `k2`. We will run a micro-grid search with a `0.001` step size centered directly over the new buckets.
4. **Sub-pattern Discovery**: Check if images with identical dimensions but drastically different internal aspect-ratios act as sub-modes requiring distinct alpha constants.

### Phase 2: Per-Image Coefficient Optimization (Abandoned) ❌

Attempted Nelder-Mead optimization on individual images:

- **Proxy**: Hough line straightness / Laplacian variance.
- **Result**: Failed scoring (18.89). Proxies simply do not correlate well enough with the bounty's multi-metric scoring (Structural Similarity + Edge Alignment).
- **Why it Failed (The Theory)**: The Brown-Conrady distortion model parameters (`k1`, `k2`) are **intrinsic camera properties**. They belong to the physical lens, *not* the image content. By allowing an optimizer to adjust `k1` and `k2` per-image to find straight lines, we were effectively hallucinating a new physical camera lens for every single photograph. If an image contained naturally curved architecture (like a modern staircase), the Nelder-Mead optimizer would aggressively warp the mathematical geometry of the image to force those curves to become straight, completely destroying the structural integrity and pixel MAE of the result.
- **Why `v4` Worked (The Literature)**: In photogrammetry, lens distortion must be calibrated per camera. Our `v4` (Dimension-Aware) algorithm succeeded because image dimensions strongly correlate with specific camera sensors and crop factors in the dataset. By locking `k1` and `k2` to fixed values *per dimension bucket*, we implicitly calibrated the intrinsic parameters for the different physical cameras used in the dataset, without letting image content (extrinsic factors) confuse the geometry. (Ref: *Brown, D. C. (1966). Decentering distortion of lenses. Photogrammetric Engineering*).

### Phase 3: U-Net Training 🔄

- **Original Plan**: Train a localized deep neural network (U-Net) to predict pixel-level distortion maps natively using PyTorch MPS on Apple Silicon.
- **Result**: **FAILED (Hardware)**. The M2 MPS backend throws `NotImplementedError` for `ConvTranspose2d` grouped operations required by certain architectures, and fallback CPU execution is 950x slower (40hrs per epoch vs 4 mins).

---

## 6. Current Bottlenecks

- **Hardware Limits**: The local M2 MacBook GPU (MPS backend) freezes when compiling complex models like U-Net (`NotImplementedError` for certain grouped convolutions).
- **Strategy Shift**: Deep learning locally on a Mac for this dataset size is functionally impossible within our 24h deadline. We must either limit the dataset to ~5,000 images overnight (1.5 hrs/epoch) or abandon U-Net and submit our `v4` dimension-aware heuristic.

---

## 7. Execution Runbooks

### Akash Network Deployment (Overnight Compute)

Due to local hardware bottlenecks and Kaggle instability, we are utilizing the **Akash Network** decentralized cloud for our 24-hour training run.

**Workflow:**

1. Generate PyTorch SDL manifest.
2. Accept provider bid ($<0.50/hr).
3. Container pulls Jupyter/Dataset dynamically via startup curl script.
4. Stream logs.

---

## 6. File / Storage Map

Due to the 37GB size of the dataset, files are strictly split between the local MacBook hard drive (code) and the external SSD (data/weights).

### MacBook Internal Hard Drive (Code & Configuration)

```text
/Users/aaron/Desktop/AutoHDR/
├── .env                             ← Holds KAGGLE_API_TOKEN and AKASH_API_KEY
├── plan.md                          ← This file (second brain)
├── demo.py                          ← Gradio web UI
├── requirements.txt                 ← Python dependencies
├── backend/
│   ├── core/                        ← Reusable modules (dataset, hardware, distortion)
│   ├── evaluation/                  ← Metrics calculators (MAE, PSNR, SSIM)
│   ├── scripts/                     ← Execution scripts
│   │   ├── akash/                   ← `akash_deploy.ts` 
│   │   ├── kaggle/                  ← `Train-UNet-Lens-Correction.ipynb`
│   │   ├── heuristics/              ← `heuristic_v4.py` and diagnostic tools
│   │   └── local_training/          ← `train.py` and inference tools
│   └── outputs/                     ← Lightweight outputs (CSVs, logs - NO ZIPS)
```

### External SSD (Heavy Data & Checkpoints)

```text
/Volumes/Love SSD/
├── lens-correction-train-cleaned/   ← 23K training pairs (30GB+)
├── test-originals/                  ← 1,000 test images
├── AutoHDR_Checkpoints/             <- PyTorch .pt weight files
└── AutoHDR_Submissions/             <- Gigabyte-heavy zip files (submission_v*.zip)
```

---

## 8. What Was Cut (and Why)

| Cut | Reason |
|---|---|
| Next.js frontend | Overkill for a hackathon; Gradio does the same in 1 file |
| ViT / Vision Transformers | Too slow to train in 24h; flow-field output was incompatible with `cv2.undistort` |
| TensorRT / ONNX export | Premature optimization; inference speed isn't the bottleneck |
| GCP setup | Kaggle Notebooks give free T4 GPUs with zero config |
| DGX Spark routing | Hardware not available during the hackathon |
| LPIPS perceptual loss | Added complexity for marginal gain; competition metric is pure MAE |
| FastAPI backend (`app.py`) | Was scaffolded but non-functional; replaced by direct script execution |
| `detector.py` (ViT model) | Never trained, outputs random noise; replaced by U-Net in `train.py` |

---

## 9. Gotchas & Bugs Encountered

1. **Dataloader filename parsing:** Initial code assumed `pair_X/` subdirectories. Actual data uses flat `{uuid}_g{n}_original.jpg` naming. Wasted 30 minutes debugging "0 samples found."

2. **ViT model was never trained:** The original codebase had a ViT with random weights. The walkthrough document from the prior session incorrectly claimed it was functional. Always verify with an actual forward pass.

3. **`app.py` self-comparison:** The evaluation endpoint compared `corrected_image` to itself, guaranteeing perfect metrics. This masked the fact that nothing was working.

4. **MPS `pin_memory` warning:** PyTorch's `pin_memory=True` is not supported on Apple Silicon MPS. It doesn't crash, but emits a warning every epoch. Set `pin_memory=False` to suppress.

5. **SSD I/O bottleneck:** Loading 46K filenames from the SSD for `os.listdir()` takes several seconds. The `--max-train` flag was added to `train.py` to limit the initial scan.

6. **Image dimension mismatch:** Some OpenCV `undistort` outputs have slightly different dimensions than inputs due to `getOptimalNewCameraMatrix`. Always resize back to the original dimensions before saving.

---

### Open Investigations & Next Steps

#### 1. Zero-Score Analysis (Partially Solved)

- **Problem**: 17.8% of images (178/1000) scored perfectly zero in v1.
- **Breakthrough**: Analysis revealed these images universally had **non-standard dimensions** (e.g., 1363x2048, 1371x2048) or **portrait orientations** (2048x1360).
- **Fix**: The `v4` dimension-aware classifier (categorizing images into standard, near-standard, nonstandard, portrait) reduced zeros from 178 to 143 and boosted the overall mean to **29.64**.
- **Remaining Question**: 143 images are still scoring zero. What characterizes these remaining failures?

#### 2. The Kaggle U-Net

- MPS was too slow (hanging). Code is fully prepped (`backend/scripts/kaggle_unet_train.py`) with data augmentation and Sobel edge-aware loss.
- Ready to be launched on Kaggle T4 GPUs.
