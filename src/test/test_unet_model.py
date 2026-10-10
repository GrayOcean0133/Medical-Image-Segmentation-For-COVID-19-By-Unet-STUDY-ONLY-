#!/usr/bin/env python3
"""UNet 模型测试：前向形状、损失、指标、反向传播。

直接运行:  python src/test/test_unet_model.py
用 pytest: pytest src/test/test_unet_model.py
"""

import torch

from _test_utils import SRC_DIR  # noqa: F401  (确保 src/ 在 sys.path 上)
from unet_model import (DEFAULT_FEATURES, Unet, combined_loss,
                        compute_pos_weight, dice_coeff, iou_score, predict_proba)

BATCH, CHANNELS, SIZE = 2, 1, 64


def _tiny_unet(features=(8, 16)):
    return Unet(in_channels=CHANNELS, out_channels=1, features=list(features))


def test_default_features_doubling():
    """默认 encoder 通道数应当逐层翻倍，否则中间层容量被压扁。"""
    expected = [DEFAULT_FEATURES[0] * (2 ** i) for i in range(len(DEFAULT_FEATURES))]
    assert DEFAULT_FEATURES == expected, f"默认 features 未逐层翻倍: {DEFAULT_FEATURES}"


def test_forward_shape_and_logits():
    """forward 返回原始 logits（不夹在 [0,1]），sigmoid 只应在 predict_proba 里。"""
    model = _tiny_unet()
    x = torch.randn(BATCH, CHANNELS, SIZE, SIZE)

    with torch.no_grad():
        logits = model(x)
        proba = predict_proba(logits)

    assert logits.shape == (BATCH, 1, SIZE, SIZE), f"输出形状错误: {logits.shape}"
    assert not torch.allclose(logits, proba), "forward 似乎已经做了 sigmoid"
    assert proba.min() >= 0 and proba.max() <= 1, "概率值应落在 [0, 1]"


def test_losses_are_finite_and_backward():
    model = _tiny_unet()
    x = torch.randn(BATCH, CHANNELS, SIZE, SIZE)
    # 故意造出极不平衡的目标（前景约 1/64），验证 pos_weight 路径不炸
    target = torch.zeros(BATCH, 1, SIZE, SIZE)
    target[:, :, :4, :4] = 1.0

    pos_weight = compute_pos_weight(target)
    assert pos_weight.item() >= 1.0, "pos_weight 应 >= 1"

    logits = model(x)
    loss = combined_loss(logits, target, alpha=0.5, pos_weight=pos_weight)
    assert torch.isfinite(loss), f"损失不是有限值: {loss}"

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "反向传播后没有任何梯度"
    assert any(g.abs().sum() > 0 for g in grads), "所有梯度都是 0"


def test_metrics_range_and_perfect_match():
    """指标应落在 [0,1]；预测与真值完全一致时 Dice/IoU 应为 1。"""
    target = torch.zeros(BATCH, 1, SIZE, SIZE)
    target[:, :, 10:20, 10:20] = 1.0

    # 造一组能把正负样本分得很开的 logits
    logits = torch.where(target > 0.5,
                         torch.full_like(target, 10.0),
                         torch.full_like(target, -10.0))

    dice = dice_coeff(logits, target)
    iou = iou_score(logits, target)
    assert abs(dice - 1.0) < 1e-3, f"Dice 应为 1，实际 {dice}"
    assert abs(iou - 1.0) < 1e-3, f"IoU 应为 1，实际 {iou}"

    # 全负预测：Dice 应为 0（smooth 项之外没有交集）
    empty = torch.full_like(target, -10.0)
    assert dice_coeff(empty, target) < 1e-2
    assert iou_score(empty, target) < 1e-2


def test_gradient_flow_through_all_blocks():
    """确认编码器/瓶颈/解码器都拿到了梯度。"""
    model = _tiny_unet()
    x = torch.randn(1, CHANNELS, SIZE, SIZE)
    target = torch.zeros(1, 1, SIZE, SIZE)
    target[:, :, :8, :8] = 1.0

    combined_loss(model(x), target).backward()

    for name in ("encoder", "bottleneck", "decoder", "final_conv"):
        module = getattr(model, name)
        params = list(module.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params), \
            f"{name} 没有收到有效梯度"


if __name__ == "__main__":
    for fn in [test_default_features_doubling, test_forward_shape_and_logits,
               test_losses_are_finite_and_backward, test_metrics_range_and_perfect_match,
               test_gradient_flow_through_all_blocks]:
        fn()
        print(f"✓ {fn.__name__}")
    print("\n所有模型测试通过！")
