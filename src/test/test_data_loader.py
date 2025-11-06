#!/usr/bin/env python3
# 测试数据加载器

import torch
from data_loader import COVIDCTDataset, get_data_transforms, COVID_UNHEALTHY_LUNGS, COVID_UNHEALTHY_LUNGS_MASK

def test_data_loader():
    print("=" * 50)
    print("测试数据加载器")
    print("=" * 50)

    # 获取transforms
    train_transform, val_transform = get_data_transforms()

    print(f"\n训练集transform: {train_transform}")
    print(f"验证集transform: {val_transform}")

    # 创建数据集
    try:
        dataset = COVIDCTDataset(
            images_dir=COVID_UNHEALTHY_LUNGS,
            masks_dir=COVID_UNHEALTHY_LUNGS_MASK,
            transform=train_transform,
            img_size=512
        )
        print(f"\n数据集创建成功！")
        print(f"数据集大小: {len(dataset)}")

        # 测试加载第一个样本
        if len(dataset) > 0:
            image, mask = dataset[0]
            print(f"\n第一个样本信息:")
            print(f"  图像shape: {image.shape}")
            print(f"  图像dtype: {image.dtype}")
            print(f"  图像值范围: [{image.min():.4f}, {image.max():.4f}]")
            print(f"  掩码shape: {mask.shape}")
            print(f"  掩码dtype: {mask.dtype}")
            print(f"  掩码唯一值: {torch.unique(mask).tolist()}")

            # 验证数据格式
            assert image.dim() == 3, f"图像应该是3维张量 (C, H, W)，实际: {image.dim()}"
            assert mask.dim() == 3, f"掩码应该是3维张量 (C, H, W)，实际: {mask.dim()}"
            assert image.shape[1:] == (512, 512), f"图像尺寸应该是512x512，实际: {image.shape[1:]}"
            assert mask.shape[1:] == (512, 512), f"掩码尺寸应该是512x512，实际: {mask.shape[1:]}"

            print("\n✓ 所有验证通过!")
        else:
            print("\n警告: 数据集为空")

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = test_data_loader()
    exit(0 if success else 1)
