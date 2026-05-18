> 🌐 English | [中文](README_CN.md)

# COVID-19 Lung CT Segmentation — U-Net Based Medical Image Segmentation

> **Disclaimer: This is a personal learning project for research purposes only. Not intended for clinical diagnosis.**
>
> This README is long by design. It documents not just "how to use it", but "what went wrong", "why things were changed", and the complete journey of a learner going from confused to working results in the field of medical image segmentation. If you're learning too, hopefully this saves you some pain.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Predecessor: Medic\_Project and Its Limitations](#2-predecessor-medic_project-and-its-limitations)
3. [What This Refactor Fixed (The Debugging Journey)](#3-what-this-refactor-fixed-the-debugging-journey)
4. [Architecture Details](#4-architecture-details)
5. [Loss Functions and Math](#5-loss-functions-and-math)
6. [Training Pipeline](#6-training-pipeline)
7. [Dataset and Data Augmentation](#7-dataset-and-data-augmentation)
8. [Results](#8-results)
9. [Setup and Usage](#9-setup-and-usage)
10. [Project Structure](#10-project-structure)

---

## 1. Project Overview

This project uses a **U-Net** convolutional neural network to perform **semantic segmentation** on lung CT scans from COVID-19 patients, automatically identifying infected regions (Ground Glass Opacity) in CT images.

### Final Metrics (500 epochs)

| Metric | Value |
|--------|-------|
| Best Epoch | 439 |
| Best Train Loss | 0.1694 |
| Val Dice | **0.811** |
| Val IoU | **0.683** |
| Training Time | ~2 hours (4× RTX A6000) |

### Tech Stack

| Category | Technology |
|----------|-----------|
| Deep Learning | PyTorch 2.x |
| Distributed Training | PyTorch DDP (DistributedDataParallel) |
| Mixed Precision | `torch.amp.autocast` + `GradScaler` |
| Data Augmentation | Albumentations (synchronized transforms) |
| Optimizer | Adam |
| LR Scheduler | CosineAnnealingLR |
| Image Processing | Pillow, NumPy |
| Visualization | Matplotlib |
| Logging | Python logging + CSV metrics |

---

## 2. Predecessor: Medic\_Project and Its Limitations

Before this version, there was an earlier iteration called `Medic_Project` (November 2025). At the time, I had just started learning medical image segmentation. The code ran without crashing, but training results were terrible — I initially blamed the dataset. After seriously reviewing the code, I found several deeply buried bugs, each of them fundamental.

Here are the specific issues in `Medic_Project`, ordered by severity:

---

### Bug 1: Sigmoid Inside `forward()` + `binary_cross_entropy` = Numerical Instability, AMP Crash

**Original code (`unet_model.py`):**

```python
# Last line of forward()
return torch.sigmoid(self.final_conv(x))   # outputs are already probabilities

# combined_loss
def combined_loss(pred, target, alpha=0.5):
    dice_loss_val = dice_loss(pred, target)
    bce_loss = F.binary_cross_entropy(pred, target)  # expects probability input
    return alpha * dice_loss_val + (1 - alpha) * bce_loss
```

**The problem:**

`F.binary_cross_entropy` expects sigmoid-activated probabilities in [0, 1]. This works in FP32, but is **numerically catastrophic** — when logit magnitudes are large, sigmoid pushes values to near 0 or 1, and taking `log` produces `-inf`.

Even worse: enabling **AMP (Automatic Mixed Precision)**, the limited precision of FP16 makes this combination almost guarantee `NaN` loss, causing training to immediately collapse.

**The fix:**

```python
# forward() returns raw logits — no sigmoid
return self.final_conv(x)

# Use BCEWithLogitsLoss (numerically stable via log-sum-exp trick)
bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pos_weight)
# Apply sigmoid externally when probabilities are needed
pred_prob = torch.sigmoid(pred_logits)
```

`binary_cross_entropy_with_logits` uses the log-sum-exp trick internally:

$$\text{BCE}(x, y) = \max(x, 0) - x \cdot y + \log(1 + e^{-|x|})$$

This is numerically equivalent to `sigmoid + log` but never produces `inf` at extreme values.

---

### Bug 2: Ignoring Class Imbalance — The Model Learned to "Do Nothing"

In CT scans, infected regions (foreground) typically occupy only **2–5%** of the image, with background dominating.

**The original code had no mechanism to handle class imbalance.** This meant the model could achieve 95%+ pixel accuracy and a fairly low loss simply by predicting all-black (all background) for every input — what I later called "lazy learning." Early training consistently showed loss dropping quickly to ~0.3 and then plateauing, because the model had learned to predict nothing.

**Fix: Dynamic `pos_weight`**

```python
# Dynamically compute negative/positive ratio, clamped at 100 to prevent extremes
neg = (target == 0).float().sum()
pos = (target == 1).float().sum()
pos_weight = torch.clamp(neg / (pos + 1e-6), min=1.0, max=100.0)

bce_loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
```

`pos_weight` tells the model: missing one positive pixel is N times worse than falsely marking one negative pixel (where N ≈ neg/pos ratio). This forces the model to take the 2–5% foreground seriously.

---

### Bug 3: Desynchronized Data Augmentation — Image Flipped, Mask Wasn't

**Original `data_loader.py`:**

```python
self.to_tensor = transforms.ToTensor()
# mask processed separately...

# In __getitem__:
image = self.to_tensor(image)
mask = self.to_tensor(mask)
```

Using `torchvision.transforms`, each call to `RandomHorizontalFlip`, `RandomRotation`, etc. **independently samples its own random parameters**.

This means: for the same sample, the image might be horizontally flipped while the mask is not. Or the image rotated 30° while the mask rotates -15°. **Image and annotation are completely misaligned** — the model is being trained on noise.

**Fix: Switch to `albumentations` (synchronized transforms)**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})  # same random state applied to both
```

`albumentations` is designed for segmentation tasks — all transforms in a `Compose` share a single random state, **guaranteeing image and mask stay in sync**.

---

### Bug 4: Wrong U-Net Channel Sizes, Insufficient Model Capacity

**Original feature channels: `features=[64, 128, 128, 256]`**

In the original U-Net paper, channels double at each downsampling step: `[64, 128, 256, 512]`. The original code had 128 for both the second and third encoder layers — no doubling — capping feature extraction capacity in the middle layers.

**Fix:** `features=[64, 128, 256, 512]`, with the bottleneck expanding to 1024. Total parameters grew from ~8M to **31M**, significantly improving representational power.

---

### Bug 5: No Validation Metrics — Training Was Blind

The original code only tracked training loss, with no insight into validation performance. The direct consequence: no way to detect overfitting, and no principled way to select the best checkpoint.

**Fix:** After each epoch, compute **Dice coefficient** and **IoU** on the validation set. Save best model based on validation loss.

---

### Bug 6: Duplicate Logging Handlers in DDP

`setup_logging()` appended new handlers to the root logger every time it was called without clearing old ones. In a DDP multi-process environment where each process calls it once, every log line printed 4 times.

```python
# Fix: clear all existing handlers first
root = logging.getLogger()
root.handlers.clear()
```

---

## 3. What This Refactor Fixed (The Debugging Journey)

> The following reconstructs the debugging timeline. It was painful. It was also where I learned the most.

### Stage 1: Loss Becomes NaN After Enabling AMP (May 13, early morning)

On the first attempt at 500-epoch training, the first few epochs looked normal, then loss suddenly jumped to `nan` and the entire run was ruined.

Investigation:
1. Suspected learning rate too large → reduced to 1e-4 → still `nan`
2. Suspected bad data (all-black images) → checked dataset → no issues
3. Disabled AMP → **`nan` disappeared** → isolated to AMP + sigmoid + BCE combination
4. Found `binary_cross_entropy_with_logits` in PyTorch docs, understood the log-sum-exp trick
5. Moved sigmoid out of `forward()`, switched to `BCEWithLogitsLoss` → problem gone

This step took about half a day, mostly stuck on "why does disabling AMP fix it."

### Stage 2: Loss Stalls at 0.3, Predictions Are All Black (May 13, afternoon)

Loss dropped normally now, but stalled around 0.3. Every prediction image was solid black — nothing predicted.

Investigation:
1. Visualized several predictions — all black, confirmed model was doing "lazy learning"
2. Computed foreground pixel ratio across dataset: **average ~3.2%**
3. Recognized class imbalance problem, added dynamic `pos_weight`
4. Re-trained — loss dropped more slowly (heavier penalty), but predictions started showing white regions
5. Iteratively tuned `pos_weight` clamp (tried 10, 50, 100), settled on 100

### Stage 3: Segmentation Boundaries Blurry, Dice Stuck Below 0.7 (May 13 night – May 18)

After adding `pos_weight`, the model predicted rough regions but boundaries were blurry, Dice hovering below 0.7.

Investigation:
1. Audited data augmentation → found image and mask flip directions were inconsistent (torchvision independent random calls)
2. Switched to `albumentations`, unified augmentation pipeline, re-trained
3. Fixed channel sizes from `[64, 128, 128, 256]` to `[64, 128, 256, 512]` for proper capacity
4. Added `Dropout2d(p=0.1)` to prevent overfitting
5. Added `CosineAnnealingLR` to slowly decay learning rate in later epochs, enabling finer convergence

Re-ran 500 epochs on May 18. Final Val Dice: **0.811**, IoU: **0.683**, boundary sharpness clearly improved.

---

## 4. Architecture Details

### U-Net Structure

U-Net was proposed by Ronneberger et al. in 2015. Its defining feature is the **encoder-decoder architecture with skip connections**.

```
Input (1, 512, 512)
    │
    ▼
[Encoder]
DoubleConv(1→64)    ─────────────────────────────────────────────┐ skip_1
MaxPool ↓                                                         │
DoubleConv(64→128)  ─────────────────────────────────────┐ skip_2 │
MaxPool ↓                                                │        │
DoubleConv(128→256) ─────────────────────────┐ skip_3   │        │
MaxPool ↓                                    │          │        │
DoubleConv(256→512) ─────────────┐ skip_4   │          │        │
MaxPool ↓                        │          │          │        │
    │                            │          │          │        │
[Bottleneck]                     │          │          │        │
DoubleConv(512→1024)             │          │          │        │
    │                            │          │          │        │
[Decoder]                        │          │          │        │
ConvTranspose(1024→512) + cat ←──┘          │          │        │
DoubleConv(1024→512)                        │          │        │
ConvTranspose(512→256)  + cat ←─────────────┘          │        │
DoubleConv(512→256)                                    │        │
ConvTranspose(256→128)  + cat ←────────────────────────┘        │
DoubleConv(256→128)                                             │
ConvTranspose(128→64)   + cat ←──────────────────────────────────┘
DoubleConv(128→64)
    │
Conv2d(64→1, kernel=1)
    │
Output logits (1, 512, 512)
```

**Total parameters: 31,036,481 (~31M)**

### DoubleConv Block

```python
class DoubleConv(nn.Module):
    """Conv2d → BN → ReLU → Conv2d → BN → ReLU → [Dropout2d]"""
```

Each DoubleConv applies two 3×3 convolutions with padding=1 (preserving spatial dimensions):

$$\text{output} = \text{ReLU}(\text{BN}(\text{Conv}(\text{ReLU}(\text{BN}(\text{Conv}(x))))))$$

**BatchNorm** normalizes each mini-batch's feature map, accelerating convergence and mitigating vanishing gradients:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

**Dropout2d(p=0.1)** zeros out entire channels randomly, preventing feature co-adaptation and acting as regularization.

### Skip Connections

Skip connections are U-Net's core innovation. The encoder loses spatial detail during downsampling; the decoder struggles to recover precise boundaries during upsampling. Skip connections directly concatenate encoder feature maps into the decoder, **preserving spatial detail**.

```python
# Align spatial dimensions with bilinear interpolation if needed
if x.shape != skip.shape:
    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
# Concatenate along channel dimension
x = torch.cat((skip, x), dim=1)
```

---

## 5. Loss Functions and Math

This project uses a combined **Dice Loss + BCEWithLogitsLoss** objective.

### Dice Loss

The Dice coefficient measures overlap between two sets:

$$\text{Dice} = \frac{2 |P \cap G|}{|P| + |G|} = \frac{2 \sum_i p_i g_i}{\sum_i p_i + \sum_i g_i}$$

where $P$ is the prediction and $G$ is the ground truth. Higher is better (max = 1).

**Dice Loss** = $1 - \text{Dice}$ (lower is better):

$$\mathcal{L}_\text{Dice} = 1 - \frac{2 \sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}$$

$\epsilon = 10^{-6}$ prevents division by zero on empty masks.

Dice Loss is **naturally robust to class imbalance** — it only measures overlap in the positive region, independent of the overall pixel distribution.

### BCEWithLogitsLoss (with pos\_weight)

$$\mathcal{L}_\text{BCE}(x, y) = -\left[ w_+ \cdot y \log\sigma(x) + (1-y)\log(1-\sigma(x)) \right]$$

where $\sigma(x) = \frac{1}{1+e^{-x}}$ and $w_+$ is the positive class weight.

**Dynamic pos\_weight:**

$$w_+ = \text{clamp}\left(\frac{N_\text{neg}}{N_\text{pos}}, 1, 100\right)$$

When foreground is 3%, $w_+ \approx 32$: missing one foreground pixel costs as much as 32 false background detections.

### Combined Loss

$$\mathcal{L} = \alpha \cdot \mathcal{L}_\text{Dice} + (1 - \alpha) \cdot \mathcal{L}_\text{BCE}, \quad \alpha = 0.5$$

They complement each other:
- Dice Loss focuses on overall region overlap, less sensitive to boundaries
- BCE Loss operates at per-pixel level, providing finer gradient signal

---

## 6. Training Pipeline

### Distributed Training (DDP)

Uses PyTorch **DistributedDataParallel** across 4× RTX A6000 (47.4 GB each).

How DDP works:
1. Each GPU holds a complete model replica
2. Forward passes run independently on each GPU
3. During backprop, gradients are **all-reduced** (ring-allreduce) across all GPUs and averaged
4. All GPU parameters stay in strict sync

```python
model = DDP(model, device_ids=[rank])
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
```

### Automatic Mixed Precision (AMP)

FP16 for most forward/backward computation, FP32 only where needed (e.g., BN statistics).

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

with autocast(device_type='cuda'):
    logits = model(images)
    loss = combined_loss(logits, masks)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

`GradScaler` addresses FP16 gradient underflow: scales up the loss before backprop, then unscales before the optimizer step, keeping small gradients from vanishing to zero.

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Clips the L2 norm of all parameter gradients to 1.0, preventing gradient explosion — especially important in medical imaging where sparse masks produce uneven gradient signals.

### Learning Rate Schedule (CosineAnnealingLR)

$$\eta_t = \eta_\text{min} + \frac{1}{2}(\eta_\text{max} - \eta_\text{min})\left(1 + \cos\frac{t\pi}{T}\right)$$

Initial LR = 0.001, decaying to $10^{-5}$, allowing finer convergence in the loss landscape's detailed regions during later epochs.

### Validation Metrics

After each epoch, evaluated on the validation set (`val_images/` + `val_masks/`):

- **Dice coefficient**: measures segmentation overlap
- **IoU (Intersection over Union)**: $\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$

Relationship: $\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$ — IoU is always slightly lower than Dice.

---

## 7. Dataset and Data Augmentation

### Dataset

[COVID-19 CT segmentation dataset](https://medicalsegmentation.com/covid19/):
- Training set: 2,729 × 512×512 lung CT slices (grayscale) with binary masks
- Validation set: 409 slices

In masks: `255` (white) = lesion area, `0` (black) = background.

### Augmentation Pipeline (albumentations)

Applied synchronously to image and mask during training only (no augmentation at validation):

```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(p=0.2),           # simulates tissue deformation
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.3                             # simulates different scanner settings
    ),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})
```

`ElasticTransform` is particularly valuable for medical images — tissue deforms elastically, not rigidly.

---

## 8. Results

### Training Curve (500 epochs, May 18)

| Epoch | Train Loss | Val Dice | Val IoU | Best? |
|-------|-----------|----------|---------|-------|
| 1 | 0.605 | — | — | ✓ |
| 50 | ~0.35 | ~0.75 | ~0.60 | |
| 100 | ~0.28 | ~0.78 | ~0.64 | |
| 200 | ~0.24 | ~0.80 | ~0.67 | |
| 351 | 0.182 | 0.807 | 0.678 | ✓ |
| 398 | 0.178 | 0.815 | 0.689 | ✓ |
| 424 | 0.175 | 0.807 | 0.677 | ✓ |
| **439** | **0.169** | **0.811** | **0.683** | **✓ Final best** |
| 500 | 0.193 | 0.812 | 0.684 | |

After epoch ~440, the model was essentially converged — loss oscillated between 0.17–0.19 with no meaningful improvement.

### Sample Predictions

Below are representative predictions from the best model (epoch 439), stored in `results/val_best/`:

- `results/val_best/pred_Jun_coronacases_case1_119.png`
- `results/val_best/pred_Jun_coronacases_case3_113.png`
- `results/val_best/pred_Jun_coronacases_case5_140.png`
- `results/val_best/pred_Jun_coronacases_case9_104.png`
- `results/val_best/pred_Jun_radiopaedia_10_85902_3_case12_107.png`
- `results/val_best/pred_Jun_radiopaedia_14_85914_0_case13_47.png`

| Output Directory | Description |
|-----------------|-------------|
| `results/val_best/` | Best model predictions (epoch 439) |
| `results/val_epoch500/` | Final epoch predictions |
| `results/val_epoch500_final/` | Full validation set, final model |
| `results/val_final/` | Full validation set, last run |

---

## 9. Setup and Usage

### Install Dependencies

```bash
pip install torch torchvision
pip install albumentations
pip install numpy pillow pandas matplotlib
```

### Training

```bash
# Default: single machine, 4 GPUs
python src/train_ddp.py

# Custom parameters
python src/train_ddp.py \
    --epochs 500 \
    --batch_size 8 \
    --lr 0.001 \
    --features 64 128 256 512 \
    --dropout_p 0.1 \
    --img_size 512
```

Training automatically:
- Saves checkpoints every 10 epochs → `checkpoints/model_epoch_N.pth`
- Saves best model on val loss improvement → `checkpoints/best_model.pth`
- Writes training log → `log/training_TIMESTAMP.log`
- Writes CSV metrics → `metrics/epoch_metrics_TIMESTAMP.csv`

### Inference

```bash
# Single image prediction
python src/predict.py \
    --model_path checkpoints/best_model.pth \
    --image_path data_set/CT_COVID/val_images/example.png \
    --output_dir results/my_test

# Batch prediction with ground truth comparison
python src/predict.py \
    --model_path checkpoints/best_model.pth \
    --image_path data_set/CT_COVID/val_images/ \
    --mask_path data_set/CT_COVID/val_masks/ \
    --output_dir results/my_test \
    --threshold 0.5
```

### Monitor Training

```bash
# Follow training log in real time
tail -f log/training_*.log

# Watch loss values
tail -f metrics/epoch_metrics_*.csv
```

---

## 10. Project Structure

```
Medical-Image-Segmentation-For-COVID-19-By-Unet-STUDY-ONLY-/
├── src/
│   ├── unet_model.py        # U-Net model, loss functions, evaluation metrics
│   ├── data_loader.py       # Dataset class, albumentations augmentation
│   ├── train_ddp.py         # DDP distributed training script
│   ├── predict.py           # Single/batch inference script
│   ├── gui.py               # tkinter GUI
│   └── test/                # Unit tests
│
├── data_set/
│   └── CT_COVID/
│       ├── frames/          # Training CT images (2,729 slices)
│       ├── masks/           # Training masks
│       ├── val_images/      # Validation images (409 slices)
│       └── val_masks/       # Validation masks
│
├── checkpoints/             # Model weights (.pth, in .gitignore)
├── log/                     # Training/inference logs (.log, in .gitignore)
├── metrics/                 # Training metrics CSV (in .gitignore)
├── results/                 # Prediction output images
│   ├── val_best/            # Best model (epoch 439) predictions
│   ├── val_epoch500/        # Epoch 500 predictions
│   └── ...
├── Help/                    # Supplementary documentation
└── .gitignore
```

---

## References

- Ronneberger O, Fischer P, Brox T. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
- Milletari F, Navab N, Ahmadi S A. *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*. 3DV 2016. (source of Dice Loss)
- COVID-19 CT Segmentation Dataset: [medicalsegmentation.com/covid19](https://medicalsegmentation.com/covid19/)
- PyTorch AMP Documentation: [pytorch.org/docs/stable/amp.html](https://pytorch.org/docs/stable/amp.html)
- Albumentations: [albumentations.ai](https://albumentations.ai)

---

*Last updated: 2026-05-19*
*Status: Training complete, results validated*
