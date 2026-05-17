# 🌙 DLP May 2025 || NPPE-3: Low-Light Image Denoising & 4× Super-Resolution

> **Kaggle Competition** · Aug 20–23, 2025 · Late Submission · Scored on PSNR (higher is better)

Train an **RRDBNet** (Residual-in-Residual Dense Block Network) from scratch to jointly denoise and upscale noisy low-light images by a factor of 4×, restoring fine detail and improving visibility.

---

## 🎯 Problem Statement

Given a **low-resolution, noisy, low-light** input image, produce a **high-resolution, clean** output that is 4× larger in each spatial dimension. The two degradation types — noise and low resolution — are addressed jointly in a single end-to-end model.

**Evaluation metric:** Peak Signal-to-Noise Ratio (PSNR) — higher is better.

```
PSNR = 10 × log₁₀(MAX² / MSE)
```

Where `MAX` is the maximum possible pixel value (255) and `MSE` is the mean squared error between the predicted and ground-truth high-resolution images. PSNR is computed per image and averaged across the test set.

---

## 🗂️ Dataset

### Files

| File / Folder | Description |
|---|---|
| `train/train/` | Low-resolution noisy training images (LR input) |
| `train/gt/` | Corresponding high-resolution clean training images (HR target) |
| `val/val/` | Low-resolution noisy validation images |
| `val/gt/` | Corresponding high-resolution clean validation images |
| `test/` | Low-resolution noisy test images (no ground truth) |
| `sample_submission.csv` | Submission template in the correct format |
| `submission.py` | Helper script to generate `submission.csv` from predicted images |

### Data Characteristics

- **Task:** Joint denoising + 4× super-resolution
- **Input:** Low-light, noisy, low-resolution RGB images (PNG / JPG)
- **Target:** Clean, high-resolution RGB images (4× spatial scale)
- **Channels:** 3 (RGB), converted to float tensors in [0, 1]
- **Pairing:** LR and HR images are sorted and matched by filename index

---

## 🔍 Approach

The solution trains **RRDBNet** — the generator backbone of ESRGAN — end-to-end on paired LR/HR image sets using pixel-level L1 loss. The model learns to simultaneously remove noise and hallucinate high-frequency detail at 4× scale.

**Why RRDBNet?**
- Designed specifically for super-resolution tasks
- Residual-in-Residual Dense Blocks (RRDB) allow very deep feature reuse without gradient vanishing
- Residual scaling (×0.2) at both block and sub-block levels stabilizes training
- Two-stage PixelShuffle upsampling cleanly handles the 4× scale factor (2× + 2×)
- Proven state-of-the-art on blind SR benchmarks (ESRGAN lineage)

**Training strategy:**

```
LR noisy image  →  RRDBNet  →  Predicted HR  →  L1 loss vs. clean HR GT
                                                        ↓
                                               Adam optimizer step
                                                        ↓
                                        Best val PSNR checkpoint saved
```

---

## 🏗️ Model Architecture

### Overview

```
Input (3 × H × W)  [low-res, noisy]
        │
  ┌─────▼─────────┐
  │  conv_first    │   Conv2d(3 → nf, 3×3)
  └─────┬─────────┘
        │ feat
        ├──────────────────────────────────────┐  (skip connection)
  ┌─────▼─────────────────────────────────┐   │
  │  nb × RRDB blocks                     │   │
  │  ┌─────────────────────────────────┐  │   │
  │  │  RRDB                           │  │   │
  │  │  ├── RDB1 (5 dense conv layers) │  │   │
  │  │  ├── RDB2                       │  │   │
  │  │  └── RDB3  → ×0.2 + input       │  │   │
  │  └─────────────────────────────────┘  │   │
  └─────┬─────────────────────────────────┘   │
        │ trunk                               │
  ┌─────▼─────────┐                           │
  │  trunk_conv    │   Conv2d(nf → nf)        │
  └─────┬─────────┘                           │
        └──────────── + ──────────────────────┘
                       │ feat + trunk
  ┌────────────────────▼───────────────────┐
  │  Upsample ×2 (stage 1)                 │
  │  upconv1: Conv2d(nf → nf×4, 3×3)       │
  │  PixelShuffle(2): nf×4 → nf            │
  │  LeakyReLU(0.2)                        │
  └────────────────────┬───────────────────┘
  ┌────────────────────▼───────────────────┐
  │  Upsample ×2 (stage 2)                 │
  │  upconv2: Conv2d(nf → nf×4, 3×3)       │
  │  PixelShuffle(2): nf×4 → nf            │
  │  LeakyReLU(0.2)                        │
  └────────────────────┬───────────────────┘
  ┌────────────────────▼───────────────────┐
  │  hr_conv: Conv2d(nf → nf, 3×3) + LReLU │
  │  conv_last: Conv2d(nf → 3, 3×3)        │
  └────────────────────┬───────────────────┘
        │
Output (3 × 4H × 4W)  [denoised, high-res]
```

### Residual Dense Block (RDB)

Each RDB stacks 5 convolutional layers with dense connections — every layer receives the concatenated outputs of all previous layers as input:

```
x → conv1(x)             → x1
x, x1 → conv2(...)       → x2
x, x1, x2 → conv3(...)   → x3
x, x1, x2, x3 → conv4(…) → x4
x, x1, x2, x3, x4 → conv5 → x5

Output = x5 × 0.2 + x   (residual scaling)
```

Each RRDB chains three RDBs with its own outer residual:

```
Output = RDB3(RDB2(RDB1(x))) × 0.2 + x
```

### Model Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `in_nc` | 3 | Input channels (RGB) |
| `out_nc` | 3 | Output channels (RGB) |
| `nf` | 64 | Base feature channel count |
| `nb` | 23 | Number of RRDB blocks |
| `gc` | 32 | Growth channels inside each RDB |
| Scale factor | 4× | Two sequential PixelShuffle(2) steps |

---

## ⚙️ Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Loss function | L1 (Mean Absolute Error) |
| Batch size | 4 |
| Epochs | 40 |
| Device | CUDA (2× Tesla T4) |
| Validation metric | PSNR (skimage) |
| Checkpoint | Best validation PSNR saved as `best_model.pth` |

### Data Pipeline

```python
DenoiseSRDataset(lr_dir, hr_dir)
  └── Loads paired LR/HR images sorted by filename
  └── Converts to RGB PIL Images
  └── Applies TF.to_tensor()  →  float tensors in [0, 1]
  └── Returns {"lr": tensor, "hr": tensor}
```

No random augmentation is applied — the model learns purely from pixel-aligned pairs.

### Validation

After each epoch, PSNR is computed on the validation set using `skimage.metrics.peak_signal_noise_ratio` and the best checkpoint is retained:

```python
psnr_score = psnr(hr_np, pred_np, data_range=1.0)
```

---

## 🚀 Inference & Submission

At inference time, the saved best checkpoint is loaded and run on the test LR images. Predictions are saved as image files and converted to the submission format using the provided `submission.py` helper:

```
test LR images
      │
[best_model.pth loaded]
      │
[RRDBNet.forward()]
      │
[Predicted HR images → OUTPUT_DIR]
      │
[submission.py]  →  21F3002062.csv
```

**Key paths:**

| Path | Purpose |
|---|---|
| `/kaggle/input/dlp-may-2025-nppe-3/archive/` | Dataset root |
| `/kaggle/working/best_model.pth` | Best validation checkpoint |
| `/kaggle/working/submission_images/` | Predicted HR images |
| `/kaggle/working/21F3002062.csv` | Final submission CSV |

---

## 📊 Evaluation

PSNR is computed between the model's predicted HR image and the hidden ground-truth HR image for each test sample:

```
PSNR (dB) = 10 × log₁₀(1.0² / MSE)    [with pixel range normalized to [0, 1]]
```

Higher PSNR indicates lower distortion. Typical benchmarks:

| PSNR Range | Quality |
|---|---|
| > 40 dB | Excellent — nearly indistinguishable from reference |
| 30–40 dB | Good — minor artifacts |
| 20–30 dB | Fair — visible degradation |
| < 20 dB | Poor |


---

## 📦 Requirements

| Package | Purpose |
|---|---|
| `torch` | Model definition, training, inference |
| `torchvision` | `TF.to_tensor()` for image preprocessing |
| `Pillow` | Image loading and RGB conversion |
| `numpy` | Array operations |
| `pandas` | CSV I/O |
| `scikit-image` | `peak_signal_noise_ratio` for PSNR computation |
| `tqdm` | Progress bars during training and inference |

**Hardware:** 2× NVIDIA Tesla T4 GPUs via Kaggle Notebooks (internet disabled — all assets loaded from Kaggle input).
