> 🌐 English | [中文](README_CN.md)

# COVID-19 Lung CT Segmentation — U-Net Based Medical Image Segmentation

> **Disclaimer: This is a personal learning project for research purposes only. Not intended for clinical diagnosis.**
>
> 本项目用于学习医学图像分割。**请勿用于任何临床用途。**

---

## ⚠️ Read this first

### 1. Historical results cannot be reproduced from this repo

The 500-epoch training run described in the "Results" section was done on a **multi-GPU DDP server that has since been decommissioned**. That environment's `checkpoints/`, `log/` and `metrics/` were never committed (they are in `.gitignore`), and the raw dataset is not in the repo either.

**Consequence:** the numbers in that table are a historical record, **not** something you can reproduce by cloning this repo. Treat them as unverified.

### 2. What *can* be verified today

The repository does contain one real output set — `results/val_epoch500_final/` (6 original CT slices + their ground-truth masks + predictions). Recomputing Dice/IoU directly from those PNGs gives:

| Case | Dice | IoU |
|---|---|---|
| case1 | 0.796 | 0.661 |
| case3 | 0.885 | 0.794 |
| case5 | 0.806 | 0.675 |
| case6 | 0.863 | 0.759 |
| case8 | 0.861 | 0.757 |
| case10 | 0.884 | 0.793 |
| **mean** | **0.849** | **0.740** |

These are in the same range as the claimed 0.811/0.683, which supports that the original training was real. They are **not** an independent validation — 6 cases, no held-out split guarantee.

### 3. Fixes applied (2026-10)

An earlier version of this README described a refactor that existed **only in the README** — the `src/` code had not been changed at all. That gap has now been closed. The following are implemented in code and covered by tests:

| Fix | File |
|---|---|
| `forward()` returns raw logits; loss uses `BCEWithLogits` + dynamic `pos_weight` | `unet_model.py` |
| Added `dice_coeff` / `iou_score` metrics computed over the whole set | `unet_model.py` |
| Paired augmentation that actually randomises per sample (was a no-op) | `data_loader.py` |
| Mask existence check + sorted file list + train/val split helper | `data_loader.py` |
| Checkpoint stores the model config; inference rebuilds the same architecture | `checkpoint.py` |
| Legacy state_dict-only `.pth` files auto-infer their architecture | `checkpoint.py` |
| Validation loop, best-model selection by val Dice, AMP, cosine LR, seeding | `train_ddp.py` |
| DDP logging: rank prefix + child processes reconfigure logging (was silent) | `train_ddp.py` |
| `--features` accepts `64 128 256 512` (was parsed into a char list) | `train_ddp.py` |
| Inference can rebuild from a checkpoint, `--threshold`, `--mask_path` metrics | `predict.py` |
| GUI loads via the shared checkpoint loader; threshold slider actually works | `gui.py` |
| Tests: runnable via `pytest` or directly, with real assertions (20 tests) | `src/test/`, `conftest.py` |

**Not** applied: `albumentations` is *not* used. Augmentation uses `torchvision.transforms.v2`, which is paired-safe and needs no extra dependency.

---

## 1. Project Overview

A **U-Net** convolutional network performs semantic segmentation on lung CT scans, identifying infected regions (Ground Glass Opacity).

### Tech stack (as implemented in this repo)

| Category | Technology |
|----------|-----------|
| Deep Learning | PyTorch 2.x |
| Distributed Training | `DistributedDataParallel` + `DistributedSampler` |
| Mixed Precision | `torch.amp.autocast` + `GradScaler` (opt-in: `--amp`) |
| Augmentation | `torchvision.transforms.v2` (paired image/mask transforms) |
| Optimizer | Adam |
| LR Scheduler | `CosineAnnealingLR` (opt-in: `--scheduler cosine`) |
| Class imbalance | Dynamic `pos_weight` for BCE (on by default) |
| Image processing | Pillow, NumPy |
| Logging | Python `logging` (rank-tagged) + CSV metrics |
| Tests | pytest |

---

## 2. Model, Loss and Metrics

### U-Net (`src/unet_model.py`)

Standard U-Net: 4 encoder stages, bottleneck, 4 decoder stages with skip connections, 1×1 output conv.

```
features = [64, 128, 256, 512]   # bottleneck expands to 1024
total parameters = 31,036,481
```

Encoder channels double at every stage — the original code had `[64, 128, 128, 256]` (no doubling in the middle), which capped capacity at ~8.5M params.

> Both `train_ddp.py` and `predict.py` now read the channel list from a single constant (`DEFAULT_FEATURES`) or from the checkpoint, so training and inference can no longer disagree.

### Important: `forward()` returns logits, not probabilities

```python
# forward() ends with:
return self.final_conv(x)        # raw logits — NO sigmoid

# use predict_proba() when you need [0, 1] probabilities
probs = predict_proba(logits)    # torch.sigmoid(logits)
```

Applying sigmoid inside `forward()` and then feeding it into `F.binary_cross_entropy` (as the original did) is **not** numerically catastrophic in FP32 — PyTorch clamps the internal `log` at −100 — but it does produce weaker, saturated gradients. Keeping logits separate is the correct contract, and it is what makes `BCEWithLogitsLoss` (log-sum-exp) available:

$$\text{BCE}(x, y) = \max(x, 0) - x \cdot y + \log(1 + e^{-|x|})$$

### Class imbalance

Infected regions occupy only ~2–5% of pixels, so an unweighted loss lets the model minimise loss by predicting all-background. `compute_pos_weight()` derives `pos_weight = neg / pos`, clamped to `[1, 100]`:

```python
pos_weight = compute_pos_weight(target)   # clamp(neg / pos, 1, 100)
loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
```

Disable with `--no-class_balance`.

### Combined loss

```python
combined_loss = alpha * dice_loss + (1 - alpha) * bce_with_logits
# alpha defaults to 0.5
```

### Metrics

`dice_coeff()` and `iou_score()` accumulate intersection / prediction / target counts over the **entire evaluation set** and divide once at the end. Per-image averaging would over-weight tiny lesions. Both take logits and binarise at a threshold (default 0.5).

---

## 3. Data and Augmentation

### Dataset (`src/data_loader.py`)

* Reads `frames/*.png` and looks up the same filename under `masks/`
* **Fails fast** if any image has no mask, listing the first few offenders
* Sorts the file list so runs are order-stable
* Converts to grayscale, resizes to `--img_size` (image: bilinear, mask: nearest), normalises image to `[-1, 1]`, binarises mask to `{0, 1}`

### Paired augmentation

```python
train_transform = T.Compose([
    T.ToImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.NEAREST, fill=0),
    T.ToDtype(torch.float32, scale=True),
])

image, mask = train_transform(image, mask)   # both passed together
```

Because the image and mask are passed **in a single call**, `transforms.v2` samples the random parameters once and applies them to both — they cannot desynchronise.

> **Regression note.** The original code tried to synchronise them with `torch.manual_seed(torch.initial_seed())` before each transform. `initial_seed()` returns a fixed value that does not advance, so every sample was reset to the *same* RNG state — the augmentation was a **no-op**, and it also clobbered the global RNG. `test_augmentation_is_actually_random` in `src/test/test_data_loader.py` guards against a relapse.

Rotation uses NEAREST for both inputs so the binary mask keeps hard edges. Flips are exact, so they are unaffected.

### Train/val split

By default 15% of the training set is held out for validation:

```bash
--val_split 0.15      # 0 disables validation entirely
```

The split is deterministic (seeded permutation). The validation subset is re-wrapped with the **non-augmenting** transform. You can also point at a separate directory:

```bash
--val_images_dir .../val_images --val_masks_dir .../val_masks
```

---

## 4. Training Pipeline (`src/train_ddp.py`)

Per epoch:

1. `train_sampler.set_epoch(epoch)` so shuffling differs per epoch
2. Forward/backward under optional AMP autocast, `scaler.step`, `scaler.update`
3. Optional `scheduler.step()`
4. **Validation on rank 0 only**, wrapped in `dist.barrier()` before and after — the other ranks would otherwise race ahead into the next epoch's gradient all-reduce and hang
5. Best model saved when val Dice improves (falls back to train loss if validation is disabled)
6. Periodic checkpoints every `--save_every` epochs

### Checkpoints carry their architecture

```python
{
  "model_state_dict": ...,
  "config": {"in_channels": 1, "out_channels": 1, "features": [64, 128, 256, 512]},
  "epoch": 439,
  "metrics": {"train_loss": ..., "val_loss": ..., "val_dice": ..., "val_iou": ...},
  "optimizer_state_dict": ...,
}
```

Previously only `state_dict()` was saved, so `predict.py` had to *guess* the architecture — it hard-coded the default `[64,128,256,512]` while training defaulted to `[64,128,128,256]`, and loading a trained model raised `size mismatch`. Inference now rebuilds from `config`.

`load_checkpoint()` also accepts legacy state_dict-only files and infers `features` / `in_channels` / `out_channels` from the weight shapes:

```
WARNING - 检测到旧格式 state_dict，已推断结构:
  {'in_channels': 1, 'out_channels': 1, 'features': [64, 128, 256, 512]}
```

### Logging

Log lines are tagged with the rank, and every spawned process reconfigures logging (a `spawn`ed child is a fresh interpreter and does **not** inherit the parent's logging config — the original code therefore wrote no training output to the log file at all).

Handler lists are cleared on reconfiguration, so a DDP run no longer prints every line `world_size` times.

### Reproducibility

`--seed` (default 42) seeds `random`, NumPy and PyTorch; each rank uses `seed + rank`.

---

## 5. Historical Results (⚠️ unverified — see the notice at the top)

Recorded at the time, on the now-decommissioned DDP server. Not reproducible from this repo.

| Epoch | Train Loss | Val Dice | Val IoU | Best? |
|-------|-----------|----------|---------|-------|
| 1 | 0.605 | — | — | ✓ |
| 50 | ~0.35 | ~0.75 | ~0.60 | |
| 100 | ~0.28 | ~0.78 | ~0.64 | |
| 200 | ~0.24 | ~0.80 | ~0.67 | |
| 351 | 0.182 | 0.807 | 0.678 | ✓ |
| 398 | 0.178 | 0.815 | 0.689 | ✓ |
| 424 | 0.175 | 0.807 | 0.677 | ✓ |
| **439** | **0.169** | **0.811** | **0.683** | **✓ final best** |
| 500 | 0.193 | 0.812 | 0.684 | |

### About the images that used to be linked here

An earlier version of this README presented six files from `results/val_best/` as "representative predictions from the best model". Those six files were **entirely black** — the model had predicted no foreground at all. They were removed from the repository in a follow-up commit (`Remove incorrect black prediction images from val_best`), but the README kept linking to them.

If you want to look at real predictions, use `results/val_epoch500_final/`, which holds `orig_*`, `mask_*` and `pred_*` for six cases and is what the Dice/IoU table in the notice above was computed from.

**Directories that do not exist in this repo:** `results/val_best/`, `results/val_epoch500/`, `results/val_final/` — the README previously listed all of them.

---

## 6. Setup and Usage

### Install

```bash
pip install -r requirements.txt
```

No `albumentations` needed.

### Prepare data

`data_set/` is gitignored, so you must supply it:

```
data_set/CT_COVID/
├── frames/     # training CT images (*.png)
└── masks/      # masks with matching filenames (*.png)
```

### Train

```bash
# Single machine, all visible GPUs
python src/train_ddp.py

# Custom parameters
python src/train_ddp.py \
    --epochs 500 \
    --batch_size 8 \
    --lr 0.001 \
    --features 64 128 256 512 \
    --val_split 0.15 \
    --amp \
    --scheduler cosine \
    --seed 42
```

Outputs:

* checkpoints → `checkpoints/model_epoch_N.pth`, `checkpoints/best_model.pth`
* log → `log/training_TIMESTAMP.log`
* per-epoch CSV → `metrics/epoch_metrics_TIMESTAMP.csv`
* optional per-batch CSV → `metrics/batch_metrics_TIMESTAMP.csv` (`--log_batch_metrics`)

### Inference

```bash
# Single image
python src/predict.py \
    --model_path checkpoints/best_model.pth \
    --image_path data_set/CT_COVID/frames/case1.png \
    --output_dir results/my_test

# Whole directory, with ground-truth metrics
python src/predict.py \
    --model_path checkpoints/best_model.pth \
    --image_path data_set/CT_COVID/val_images/ \
    --mask_path  data_set/CT_COVID/val_masks/ \
    --output_dir results/my_test \
    --threshold 0.5
```

`--mask_path` prints mean Dice and IoU; `--threshold` controls binarisation (the GUI slider does the same).

### GUI

```bash
python src/gui.py     # loads checkpoints/best_model.pth
```

### Tests

```bash
pytest                              # all 20 tests, ~2 s, synthetic data only
python src/test/test_unet_model.py  # or run any test file directly
```

Tests do not touch `data_set/`.

### Monitor training

```bash
tail -f log/training_*.log
cat metrics/epoch_metrics_*.csv
python src/test/analyze_training.py --metrics_file metrics/epoch_metrics_TIMESTAMP.csv
```

---

## 7. Project Structure

```
Medical-Image-Segmentation-For-COVID-19-By-Unet-STUDY-ONLY-/
├── src/
│   ├── unet_model.py        # U-Net, losses, Dice/IoU metrics
│   ├── data_loader.py       # Dataset, paired transforms.v2 augmentation, split helper
│   ├── checkpoint.py        # Save/load checkpoints incl. model config
│   ├── train_ddp.py         # DDP training with validation, AMP, LR schedule
│   ├── predict.py           # Single/batch inference, optional Dice/IoU
│   ├── gui.py               # tkinter GUI
│   └── test/
│       ├── _test_utils.py           # sys.path bootstrap + synthetic data generator
│       ├── test_unet_model.py
│       ├── test_data_loader.py
│       ├── test_model_save.py
│       ├── test_training_pipeline.py
│       └── analyze_training.py      # CSV metrics plotting tool (not a test)
│
├── data_set/                # NOT in repo (gitignored) — bring your own
├── checkpoints/             # gitignored
├── log/                     # gitignored
├── metrics/                 # gitignored
├── results/
│   └── val_epoch500_final/  # The only surviving result set: orig/mask/pred × 6
├── Help/                    # Supplementary notes
├── conftest.py              # pytest path setup
├── pytest.ini
├── requirements.txt
└── .gitignore
```

---

## 8. Debugging History (learning notes)

> Kept for reference. These are the issues found while going from the first non-working attempt to a working pipeline. Several were fixed only in the 2026-10 pass described at the top.

**1. Sigmoid inside `forward()` + `F.binary_cross_entropy`**
Mixing an activated output with `binary_cross_entropy` is a classic mismatch. In FP32 it does not explode (PyTorch clamps `log` at −100), but gradients saturate badly. Fix: return logits, use `binary_cross_entropy_with_logits`.

**2. Class imbalance → "learn to predict nothing"**
Early runs showed loss dropping to ~0.3 and flattening: the model had settled on all-background. Fix: dynamic `pos_weight`.

**3. Desynchronised augmentation**
The image and its mask were transformed independently, so flips/rotations did not line up and the model trained on noise. Fix: a single paired call into `transforms.v2`.

**4. Wrong encoder channel sizes**
`[64, 128, 128, 256]` does not double at the middle stage. Fix: `[64, 128, 256, 512]` (≈8.5M → 31M params), with `DEFAULT_FEATURES` as the single source of truth.

**5. No validation metrics**
Training loss alone gives no way to detect overfitting or to choose a checkpoint. Fix: per-epoch val Dice/IoU, best model selected by val Dice.

**6. Duplicated logging handlers in DDP**
`setup_logging()` kept appending handlers, so lines repeated per process. Fix: drop existing handlers before adding new ones. (The follow-on problem — spawned children not logging at all — is fixed by reconfiguring logging inside each rank.)

**7. `unet_model.py` default vs. `train_ddp.py` default**
These two disagreed, so `predict.py` could never load a trained checkpoint (`size mismatch`, 31,036,481 vs 8,568,513 params). Fix: config travels with the checkpoint, plus shape-based inference for legacy files.

---

## References

- Ronneberger O, Fischer P, Brox T. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
- Milletari F, Navab N, Ahmadi S A. *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*. 3DV 2016. (source of Dice Loss)
- COVID-19 CT Segmentation Dataset: [medicalsegmentation.com/covid19](https://medicalsegmentation.com/covid19/)
- PyTorch AMP: [pytorch.org/docs/stable/amp.html](https://pytorch.org/docs/stable/amp.html)
- torchvision transforms v2: [pytorch.org/vision/stable/transforms.html](https://pytorch.org/vision/stable/transforms.html)
