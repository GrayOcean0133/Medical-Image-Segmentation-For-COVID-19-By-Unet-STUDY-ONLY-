#!/usr/bin/env python
"""
测试脚本：验证模型保存功能是否正常工作
"""
import torch
from pathlib import Path
from src.unet_model import Unet

# 项目根目录 (test目录的父目录的父目录)
PROJECT_ROOT = Path(__file__).parent.parent.parent

def test_model_save():
    """测试模型保存功能"""
    print("=" * 60)
    print("测试模型保存功能")
    print("=" * 60)

    # 1. 创建一个简单的模型
    print("\n1. 创建U-Net模型...")
    model = Unet(in_channels=1, out_channels=1)
    print("   ✓ 模型创建成功")

    # 2. 确保checkpoints目录存在
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    print(f"\n2. 检查点目录: {checkpoint_dir}")
    print(f"   ✓ 目录已创建")

    # 3. 保存模型state_dict（best_model格式）
    best_model_path = checkpoint_dir / "best_model.pth"
    print(f"\n3. 保存best_model到: {best_model_path}")
    try:
        torch.save(model.state_dict(), str(best_model_path))
        print("   ✓ 保存成功")
    except Exception as e:
        print(f"   ✗ 保存失败: {e}")
        return False

    # 4. 保存完整checkpoint（epoch格式）
    checkpoint_path = checkpoint_dir / "model_epoch_10.pth"
    print(f"\n4. 保存checkpoint到: {checkpoint_path}")
    try:
        checkpoint = {
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'loss': 0.05
        }
        torch.save(checkpoint, str(checkpoint_path))
        print("   ✓ 保存成功")
    except Exception as e:
        print(f"   ✗ 保存失败: {e}")
        return False

    # 5. 验证文件是否存在
    print("\n5. 验证保存的文件...")
    if best_model_path.exists():
        size = best_model_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ best_model.pth 存在 ({size:.2f} MB)")
    else:
        print("   ✗ best_model.pth 不存在")
        return False

    if checkpoint_path.exists():
        size = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ model_epoch_10.pth 存在 ({size:.2f} MB)")
    else:
        print("   ✗ model_epoch_10.pth 不存在")
        return False

    # 6. 测试加载模型
    print("\n6. 测试加载模型...")
    try:
        test_model = Unet(in_channels=1, out_channels=1)
        test_model.load_state_dict(torch.load(best_model_path))
        print("   ✓ 模型加载成功")
    except Exception as e:
        print(f"   ✗ 模型加载失败: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ 所有测试通过！模型保存功能正常工作。")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_model_save()
    exit(0 if success else 1)
