"""共享测试工具：让 `python src/test/xxx.py` 与 `pytest` 都能正确导入 src/ 下的模块。

同时提供一个生成临时 CT-like 数据的小工具，避免测试依赖体积巨大的 data_set/。
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# 把 src/ 加进 sys.path（src/test/xxx.py -> src/）
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def make_synthetic_dataset(n=8, size=64, seed=0):
    """生成一对临时图像/掩码目录，返回 (images_dir, masks_dir)。

    前景只占很小一块，模拟 CT 中病灶区域占比很低的情形。
    """
    rng = np.random.default_rng(seed)
    root = Path(tempfile.mkdtemp(prefix="covid_ct_synth_"))
    images_dir = root / "frames"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        image = rng.integers(0, 60, size=(size, size), dtype=np.uint8)
        # 一块不对称的亮区，便于检测翻转/旋转是否作用到图上
        image[0:size // 4, 0:size // 2] = 200
        # 掩码就是这块亮区：变换后两者必须依然重合
        mask = (image > 150).astype(np.uint8) * 255

        Image.fromarray(image).save(images_dir / f"case{i}.png")
        Image.fromarray(mask).save(masks_dir / f"case{i}.png")

    return images_dir, masks_dir
