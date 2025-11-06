#!/usr/bin/env python3
# 测试完整训练流程（单GPU版本）

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import COVIDCTDataset, get_data_transforms, COVID_UNHEALTHY_LUNGS, COVID_UNHEALTHY_LUNGS_MASK
from unet_model import Unet, combined_loss

def test_training_pipeline():
    print("=" * 50)
    print("测试完整训练流程")
    print("=" * 50)

    # 检查GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 创建模型
    model = Unet(
        in_channels=1,
        out_channels=1,
        features=[64, 128, 256, 512]
    ).to(device)

    print(f"模型已加载到 {device}")

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 加载数据
    train_transform, val_transform = get_data_transforms()

    train_dataset = COVIDCTDataset(
        images_dir=COVID_UNHEALTHY_LUNGS,
        masks_dir=COVID_UNHEALTHY_LUNGS_MASK,
        transform=train_transform,
        img_size=512
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # 小批次用于测试
        shuffle=True,
        num_workers=0,  # 单进程测试
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"\n数据集大小: {len(train_dataset)}")
    print(f"批次数量: {len(train_loader)}")

    # 测试一个训练步骤
    print(f"\n开始测试训练步骤...")

    model.train()
    total_batches = min(3, len(train_loader))  # 只测试前3个batch

    for batch_idx, (images, masks) in enumerate(train_loader):
        if batch_idx >= total_batches:
            break

        print(f"\nBatch {batch_idx + 1}/{total_batches}")

        # 移到GPU
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        print(f"  图像shape: {images.shape}, 设备: {images.device}")
        print(f"  掩码shape: {masks.shape}, 设备: {masks.device}")

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)

        print(f"  输出shape: {outputs.shape}")

        # 计算损失
        loss = combined_loss(outputs, masks)

        print(f"  损失值: {loss.item():.4f}")

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        print(f"  ✓ 训练步骤完成")

    print("\n" + "=" * 50)
    print("训练流程测试通过！")
    print("=" * 50)

    return True

if __name__ == "__main__":
    try:
        success = test_training_pipeline()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
