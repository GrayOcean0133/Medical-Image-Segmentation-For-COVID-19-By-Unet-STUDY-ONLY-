"""模型检查点读写。

只存 state_dict 是不够的：权重本身不知道网络结构，推理端必须知道
features / in_channels / out_channels 才能把模型搭回来。
所以这里统一把结构配置一起存进 checkpoint。
"""

import logging
import re
from pathlib import Path

import torch

from unet_model import DEFAULT_FEATURES, Unet

CONFIG_KEY = "config"
STATE_KEY = "model_state_dict"


def _safe_torch_load(path, map_location="cpu"):
    """兼容新旧 torch：优先按 weights_only 安全模式加载。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # torch 版本过旧，没有 weights_only 参数
        return torch.load(path, map_location=map_location)
    except Exception:
        # 检查点里含有非纯张量对象时回退
        return torch.load(path, map_location=map_location, weights_only=False)


def save_checkpoint(path, model, config=None, optimizer=None, epoch=None,
                    metrics=None, extra=None):
    """保存检查点。

    Args:
        model: 已训练的 nn.Module（DDP 包装的请传 model.module）
        config: 结构配置 dict，至少含 in_channels / out_channels / features
        metrics: 该 epoch 的评估指标，用于事后分析
    """
    payload = {
        STATE_KEY: model.state_dict(),
        CONFIG_KEY: dict(config or {}),
        "epoch": epoch,
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
    return path


_ENC_RE = re.compile(r"^encoder\.(\d+)\.double_conv\.0\.weight$")


def infer_config_from_state_dict(state_dict):
    """从 state_dict 的权重形状反推网络结构。

    用于加载历史遗留的、只存了 state_dict 的 .pth 文件。
    """
    features = []
    for key, value in state_dict.items():
        m = _ENC_RE.match(key)
        if m:
            features.append((int(m.group(1)), int(value.shape[0])))
    if not features:
        raise ValueError("无法从 state_dict 推断 features：找不到 encoder.*.double_conv.0.weight")

    features.sort(key=lambda x: x[0])
    feature_list = [c for _, c in features]

    first = state_dict.get("encoder.0.double_conv.0.weight")
    in_channels = int(first.shape[1]) if first is not None else 1

    final = state_dict.get("final_conv.weight")
    out_channels = int(final.shape[0]) if final is not None else 1

    return {"in_channels": in_channels, "out_channels": out_channels,
            "features": feature_list}


def load_checkpoint(path, map_location="cpu"):
    """读取检查点，返回统一结构的 dict。"""
    raw = _safe_torch_load(path, map_location=map_location)

    # 新格式：带 config 的完整检查点
    if isinstance(raw, dict) and STATE_KEY in raw:
        config = dict(raw.get(CONFIG_KEY) or {})
        if not config:
            config = infer_config_from_state_dict(raw[STATE_KEY])
            logging.warning("检查点缺少 config，已从权重形状推断: %s", config)
        return {
            STATE_KEY: raw[STATE_KEY],
            CONFIG_KEY: config,
            "epoch": raw.get("epoch"),
            "metrics": dict(raw.get("metrics") or {}),
            "optimizer_state_dict": raw.get("optimizer_state_dict"),
        }

    # 旧格式：直接存的 state_dict（best_model.pth 那种）
    if isinstance(raw, dict):
        config = infer_config_from_state_dict(raw)
        logging.warning("检测到旧格式 state_dict，已推断结构: %s", config)
        return {STATE_KEY: raw, CONFIG_KEY: config, "epoch": None, "metrics": {}}

    raise ValueError(f"无法识别的检查点格式: {path}")


def build_model(config, device="cpu"):
    """按配置构造 U-Net 并搬到设备上（未加载权重）。"""
    model = Unet(
        in_channels=int(config.get("in_channels", 1)),
        out_channels=int(config.get("out_channels", 1)),
        features=list(config.get("features") or DEFAULT_FEATURES),
    )
    return model.to(device)


def load_model_from_checkpoint(path, device="cpu"):
    """一站式：读检查点 -> 造模型 -> 载权重 -> eval 模式。"""
    ckpt = load_checkpoint(path, map_location=device)
    model = build_model(ckpt[CONFIG_KEY], device=device)
    model.load_state_dict(ckpt[STATE_KEY])
    model.eval()
    return model, ckpt
