#!/usr/bin/env python3
# 测试UNet模型

import torch
from unet_model import Unet, combined_loss

def test_unet_model():
    print("=" * 50)
    print("测试UNet模型")
    print("=" * 50)

    # 创建模型
    model = Unet(
        in_channels=1,
        out_channels=1,
        features=[64, 128, 256, 512]
    )

    print(f"\n模型创建成功！")

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    # 创建随机输入测试前向传播
    batch_size = 4
    img_size = 512

    print(f"\n测试前向传播...")
    print(f"  批次大小: {batch_size}")
    print(f"  图像尺寸: {img_size}x{img_size}")

    try:
        # 创建随机输入
        x = torch.randn(batch_size, 1, img_size, img_size)
        print(f"  输入shape: {x.shape}")

        # 前向传播
        with torch.no_grad():
            output = model(x)

        print(f"  输出shape: {output.shape}")
        print(f"  输出值范围: [{output.min():.4f}, {output.max():.4f}]")

        # 验证输出shape
        assert output.shape == (batch_size, 1, img_size, img_size), \
            f"输出shape错误，期望 ({batch_size}, 1, {img_size}, {img_size})，实际 {output.shape}"

        # 验证输出值在0-1之间（sigmoid输出）
        assert output.min() >= 0 and output.max() <= 1, \
            f"输出值应该在[0, 1]之间，实际范围 [{output.min()}, {output.max()}]"

        print("\n✓ 前向传播测试通过!")

        # 测试损失函数
        print(f"\n测试损失函数...")
        target = torch.randint(0, 2, (batch_size, 1, img_size, img_size)).float()

        loss = combined_loss(output, target)
        print(f"  损失值: {loss.item():.4f}")

        assert not torch.isnan(loss), "损失值为NaN"
        assert not torch.isinf(loss), "损失值为Inf"

        print("\n✓ 损失函数测试通过!")

        # 测试反向传播
        print(f"\n测试反向传播...")
        model.train()
        x = torch.randn(batch_size, 1, img_size, img_size)
        target = torch.randint(0, 2, (batch_size, 1, img_size, img_size)).float()

        output = model(x)
        loss = combined_loss(output, target)
        loss.backward()

        # 检查梯度
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "模型参数没有梯度"

        print("✓ 反向传播测试通过!")

        print("\n" + "=" * 50)
        print("所有测试通过！")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_unet_model()
    exit(0 if success else 1)
