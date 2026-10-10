"""pytest 配置：把 src/ 加入 sys.path，使测试既能 `pytest` 跑，
也能直接 `python src/test/xxx.py` 跑。"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
