#!/usr/bin/env python3
"""检查点保存/加载测试。

重点验证：推理端能根据检查点里的结构配置重建出完全一致的网络，
以及历史遗留的「只存 state_dict」的 .pth 依然能加载。
"""

import tempfile
from pathlib import Path

import torch

from _test_utils import SRC_DIR  # noqa: F401
from checkpoint import (build_model, infer_config_from_state_dict,
                        load_checkpoint, load_model_from_checkpoint,
                        save_checkpoint)
from unet_model import Unet

CONFIG = {"in_channels": 1, "out_channels": 1, "features": [8, 16]}


def _model(config=None):
    return Unet(**(config or CONFIG))


def _tmp_path(name="ckpt.pth"):
    return Path(tempfile.mkdtemp(prefix="ckpt_test_")) / name


def test_roundtrip_preserves_weights_and_config():
    model = _model()
    path = _tmp_path()
    save_checkpoint(path, model, config=CONFIG, epoch=7, metrics={"val_dice": 0.8})

    loaded, ckpt = load_model_from_checkpoint(path, device="cpu")

    assert ckpt["config"] == CONFIG, f"结构配置未正确保存: {ckpt['config']}"
    assert ckpt["epoch"] == 7
    assert ckpt["metrics"]["val_dice"] == 0.8

    for (name_a, param_a), (name_b, param_b) in zip(
            model.state_dict().items(), loaded.state_dict().items()):
        assert name_a == name_b
        assert torch.equal(param_a, param_b), f"权重 {name_a} 未正确恢复"


def test_legacy_state_dict_can_be_inferred():
    """旧格式 best_model.pth（裸 state_dict）应能自动推断出结构。"""
    model = _model(config={"in_channels": 1, "out_channels": 1,
                           "features": [16, 32, 64]})
    path = _tmp_path("legacy_best_model.pth")
    torch.save(model.state_dict(), path)

    loaded, ckpt = load_model_from_checkpoint(path, device="cpu")
    assert ckpt["config"]["features"] == [16, 32, 64], ckpt["config"]
    assert sum(p.numel() for p in loaded.parameters()) == \
        sum(p.numel() for p in model.parameters())


def test_infer_config_reads_channel_counts():
    model = _model(config={"in_channels": 3, "out_channels": 5, "features": [8, 16]})
    config = infer_config_from_state_dict(model.state_dict())
    assert config["in_channels"] == 3
    assert config["out_channels"] == 5
    assert config["features"] == [8, 16]


def test_mismatched_architecture_fails_loudly():
    """用错的 features 加载必须报错，而不是静默产出垃圾结果。"""
    model = _model(config=CONFIG)
    path = _tmp_path()
    save_checkpoint(path, model, config=CONFIG)

    wrong = build_model({"in_channels": 1, "out_channels": 1,
                         "features": [16, 32]}, device="cpu")
    try:
        wrong.load_state_dict(load_checkpoint(path)["model_state_dict"])
    except RuntimeError as e:
        assert "size mismatch" in str(e)
    else:
        raise AssertionError("结构不匹配时应该抛出 RuntimeError")


def test_saved_checkpoint_supports_inference():
    """保存 -> 重新加载 -> 前向，应当能跑通并给出正确形状。"""
    model = _model()
    path = _tmp_path()
    save_checkpoint(path, model, config=CONFIG)

    loaded, _ = load_model_from_checkpoint(path, device="cpu")
    with torch.no_grad():
        out = loaded(torch.randn(1, 1, 32, 32))
    assert out.shape == (1, 1, 32, 32)


if __name__ == "__main__":
    for fn in [test_roundtrip_preserves_weights_and_config,
               test_legacy_state_dict_can_be_inferred,
               test_infer_config_reads_channel_counts,
               test_mismatched_architecture_fails_loudly,
               test_saved_checkpoint_supports_inference]:
        fn()
        print(f"✓ {fn.__name__}")
    print("\n所有检查点测试通过！")
