#!/usr/bin/env python3
"""端到端小规模训练测试：数据 -> 模型 -> 损失 -> 反向 -> 保存 -> 重新推理。

用合成数据，不依赖体积庞大的 data_set/，几秒钟内跑完。
"""

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from _test_utils import SRC_DIR, make_synthetic_dataset  # noqa: F401
from checkpoint import load_model_from_checkpoint, save_checkpoint
from data_loader import COVIDCTDataset, get_data_transforms, split_train_val
from unet_model import Unet, combined_loss, compute_pos_weight, dice_coeff

SIZE = 64
FEATURES = [8, 16]
CONFIG = {"in_channels": 1, "out_channels": 1, "features": FEATURES}


def _loader(n=8, transform=None, batch_size=4, shuffle=False):
    images_dir, masks_dir = make_synthetic_dataset(n=n, size=SIZE, seed=7)
    ds = COVIDCTDataset(images_dir, masks_dir, transform=transform, img_size=SIZE)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def test_forward_backward_updates_weights():
    train_transform, _ = get_data_transforms()
    loader = _loader(transform=train_transform)

    model = Unet(**CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = [p.detach().clone() for p in model.parameters()]
    images, masks = next(iter(loader))

    loss = combined_loss(model(images), masks, alpha=0.5,
                         pos_weight=compute_pos_weight(masks))
    assert torch.isfinite(loss), f"损失是 NaN/Inf: {loss}"

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    after = list(model.parameters())
    changed = sum(1 for b, a in zip(before, after) if not torch.equal(b, a))
    assert changed > 0, "optimizer.step() 之后没有任何参数被更新"


def test_loss_decreases_on_overfit_single_batch():
    """在单个 batch 上反复拟合，损失应当明显下降（验证梯度方向正确）。"""
    train_transform, _ = get_data_transforms()
    images_dir, masks_dir = make_synthetic_dataset(n=4, size=SIZE, seed=11)
    ds = COVIDCTDataset(images_dir, masks_dir, transform=train_transform, img_size=SIZE)
    subset = Subset(ds, [0])
    loader = DataLoader(subset, batch_size=1)
    images, masks = next(iter(loader))

    model = Unet(**CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(12):
        loss = combined_loss(model(images), masks, alpha=0.5,
                             pos_weight=compute_pos_weight(masks))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"损失没有下降: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_full_epoch_with_validation_metrics():
    train_transform, val_transform = get_data_transforms()
    full = _loader(n=12, transform=train_transform).dataset
    train_subset, val_subset = split_train_val(full, val_split=0.25, seed=3)

    val_source = COVIDCTDataset(full.images_dir, full.masks_dir,
                                transform=val_transform, img_size=SIZE)
    val_ds = Subset(val_source, val_subset.indices)

    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    model = Unet(**CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for images, masks in train_loader:
        loss = combined_loss(model(images), masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # 验证：整批累计 Dice 应落在 [0, 1]
    model.eval()
    total_dice, n = 0.0, 0
    with torch.no_grad():
        for images, masks in val_loader:
            total_dice += dice_coeff(model(images), masks)
            n += 1
    mean_dice = total_dice / max(n, 1)
    assert 0.0 <= mean_dice <= 1.0, f"验证 Dice 越界: {mean_dice}"


def test_checkpoint_roundtrip_after_training():
    train_transform, _ = get_data_transforms()
    loader = _loader(transform=train_transform)
    model = Unet(**CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    images, masks = next(iter(loader))
    combined_loss(model(images), masks).backward()
    optimizer.step()

    path = Path(tempfile.mkdtemp(prefix="pipeline_ckpt_")) / "best_model.pth"
    save_checkpoint(path, model, config=CONFIG, optimizer=optimizer,
                    epoch=1, metrics={"val_dice": 0.5})

    reloaded, ckpt = load_model_from_checkpoint(path, device="cpu")
    assert ckpt["config"] == CONFIG

    reloaded.eval()
    with torch.no_grad():
        out = reloaded(images)
    assert out.shape == (images.shape[0], 1, SIZE, SIZE)


if __name__ == "__main__":
    for fn in [test_forward_backward_updates_weights,
               test_loss_decreases_on_overfit_single_batch,
               test_full_epoch_with_validation_metrics,
               test_checkpoint_roundtrip_after_training]:
        fn()
        print(f"✓ {fn.__name__}")
    print("\n所有训练流程测试通过！")
