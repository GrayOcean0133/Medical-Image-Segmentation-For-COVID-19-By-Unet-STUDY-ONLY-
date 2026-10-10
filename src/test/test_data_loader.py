#!/usr/bin/env python3
"""数据加载器测试：增强必须随机、且 image/mask 必须同步。

直接运行:  python src/test/test_data_loader.py
"""

import torch

from _test_utils import SRC_DIR, make_synthetic_dataset  # noqa: F401
from data_loader import COVIDCTDataset, get_data_transforms, split_train_val

SIZE = 64


def _dataset(n=12, transform=None):
    images_dir, masks_dir = make_synthetic_dataset(n=n, size=SIZE, seed=1)
    return COVIDCTDataset(images_dir, masks_dir, transform=transform, img_size=SIZE)


def test_files_are_sorted_and_counts_match():
    ds = _dataset(n=7)
    assert len(ds) == 7
    assert ds.image_files == sorted(ds.image_files), "文件名未排序，顺序会随文件系统变化"


def test_augmentation_is_actually_random():
    """回归测试：旧实现每个样本都被重置到同一随机状态，增强等于没做。"""
    train_transform, _ = get_data_transforms()
    ds = _dataset(n=12, transform=train_transform)

    signatures = set()
    for i in range(len(ds)):
        image, _ = ds[i]
        signatures.add((round(float(image.sum()), 4),
                        round(float(image[0, :8, :8].sum()), 4)))

    assert len(signatures) > 1, (
        f"{len(ds)} 个样本只产生了 {len(signatures)} 种增强结果，"
        "说明随机增强没有生效"
    )


def test_image_and_mask_stay_aligned():
    """增强后前景像素与图像亮区必须仍然重合（旧实现会让两者错位）。"""
    train_transform, _ = get_data_transforms()
    ds = _dataset(n=8, transform=train_transform)

    for i in range(len(ds)):
        image, mask = ds[i]
        bright = (image > 0.2).float()
        intersection = (bright * mask).sum()
        union = ((bright + mask) > 0).sum()
        iou = float(intersection / union) if union > 0 else 1.0
        assert iou > 0.95, f"第 {i} 个样本图像与掩码错位，IoU={iou:.3f}"


def test_shapes_dtypes_and_value_range():
    train_transform, val_transform = get_data_transforms()
    for transform in (train_transform, val_transform):
        ds = _dataset(n=2, transform=transform)
        image, mask = ds[0]
        assert image.shape == (1, SIZE, SIZE), f"图像形状异常: {image.shape}"
        assert mask.shape == (1, SIZE, SIZE), f"掩码形状异常: {mask.shape}"
        assert image.dtype == torch.float32
        assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}, "掩码未二值化"


def test_missing_mask_raises_with_clear_error(tmp_path):
    images_dir, masks_dir = make_synthetic_dataset(n=3, size=SIZE, seed=2)
    import os
    # 删掉一个掩码，构造数据不完整的情形
    os.remove(os.path.join(str(masks_dir), sorted(os.listdir(masks_dir))[0]))

    try:
        COVIDCTDataset(images_dir, masks_dir, transform=None, img_size=SIZE)
    except FileNotFoundError as e:
        assert "找不到同名掩码" in str(e)
    else:
        raise AssertionError("缺掩码时应该抛出 FileNotFoundError")


def test_split_train_val():
    ds = _dataset(n=10)
    train_subset, val_subset = split_train_val(ds, val_split=0.2, seed=42)
    assert len(val_subset) == 2
    assert len(train_subset) == 8
    # 两个子集不应有交集
    assert set(train_subset.indices).isdisjoint(set(val_subset.indices))

    same, none = split_train_val(ds, val_split=0.0)
    assert none is None and same is ds


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    for fn in [test_files_are_sorted_and_counts_match,
               test_augmentation_is_actually_random,
               test_image_and_mask_stay_aligned,
               test_shapes_dtypes_and_value_range,
               test_split_train_val]:
        fn()
        print(f"✓ {fn.__name__}")

    test_missing_mask_raises_with_clear_error(Path(tempfile.mkdtemp()))
    print("✓ test_missing_mask_raises_with_clear_error")
    print("\n所有数据加载测试通过！")
