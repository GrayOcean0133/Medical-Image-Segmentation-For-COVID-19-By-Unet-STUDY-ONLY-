# 训练指标监控系统使用指南

## 概述

新的训练指标监控系统可以详细记录训练过程中的所有关键指标，包括Loss值、学习率等，并提供可视化分析工具，帮助你深入理解模型训练过程。

## 功能特性

### 1. 自动记录训练指标
- **Epoch级别**：每个epoch的平均loss、最小/最大batch loss、学习率、训练时间等
- **Batch级别（可选）**：每个batch的详细loss和学习率记录

### 2. 实时进度显示
- 训练进度百分比
- 当前batch的loss值
- 预计剩余时间（ETA）

### 3. CSV文件导出
- 结构化的CSV格式，方便后续分析
- 时间戳命名，不会覆盖历史数据

### 4. 可视化分析
- Loss曲线图
- 学习率变化图
- 训练时间统计
- 趋势分析

---

## 快速开始

### 基础训练（只记录Epoch级别）

```bash
python src/train_ddp.py
```

这会：
- ✅ 每10个batch输出一次进度（默认）
- ✅ 自动记录每个epoch的统计信息
- ✅ 生成`metrics/epoch_metrics_TIMESTAMP.csv`文件

### 详细训练（记录Batch级别）

```bash
python src/train_ddp.py --log_batch_metrics
```

这会：
- ✅ 记录每个epoch的统计信息
- ✅ **额外**记录每个batch的详细loss值
- ✅ 生成两个文件：
  - `metrics/epoch_metrics_TIMESTAMP.csv`
  - `metrics/batch_metrics_TIMESTAMP.csv`

⚠️ **注意**：Batch级别记录会生成大量数据（200 epochs × 171 batches ≈ 34,000行）

### 调整日志输出频率

```bash
# 每5个batch输出一次（更频繁）
python src/train_ddp.py --log_interval 5

# 每50个batch输出一次（更少）
python src/train_ddp.py --log_interval 50
```

---

## 训练时看到什么

### 1. 启动信息
```
工作目录: /workspace/GrayOcean/code/Medic_Project
检查点保存目录: .../checkpoints
结果保存目录: .../results
日志保存目录: .../log
训练指标保存目录: .../metrics
================================================================================
训练配置:
  Epochs: 200
  Batch Size (per GPU): 4
  Learning Rate: 0.001
  ...
日志配置:
  日志输出间隔: 每10个batch
  记录Batch级别指标: 否
================================================================================
Epoch指标记录文件: .../metrics/epoch_metrics_20251106_103015.csv
================================================================================
```

### 2. 训练过程
```
Epoch [1/200] - 进度 0.0% (0/171) - Batch Loss: 0.523415 - LR: 0.00100000
Epoch [1/200] - 进度 5.8% (10/171) - Batch Loss: 0.478203 - LR: 0.00100000
...
Epoch [1/200] - 进度 58.5% (100/171) - Batch Loss: 0.312456 - LR: 0.00100000
...
================================================================================
Epoch [1/200] 完成
平均损失: 0.425316
最小Batch损失: 0.285432
最大Batch损失: 0.623571
学习率: 0.00100000
本epoch用时: 14.52秒
预计剩余时间: 0.80小时
================================================================================
```

### 3. 模型保存
```
✓ 已保存检查点: model_epoch_10.pth
✓ 新的最佳模型! Loss: 0.325416 (提升: 0.012345)
✓ 已保存最佳模型: best_model.pth
```

---

## 生成的数据文件

### 目录结构
```
Medic_Project/
├── metrics/                                    # 训练指标目录
│   ├── epoch_metrics_20251106_103015.csv      # Epoch级别指标
│   └── batch_metrics_20251106_103015.csv      # Batch级别指标（可选）
├── log/
│   └── training_20251106_103015.log           # 详细日志
└── analysis/                                   # 分析结果（运行分析脚本后）
    ├── training_analysis_epoch_metrics_*.png
    ├── training_stats_epoch_metrics_*.png
    └── training_report_epoch_metrics_*.txt
```

### Epoch Metrics CSV 格式

| 列名 | 说明 | 示例 |
|------|------|------|
| Epoch | Epoch编号 | 1, 2, 3, ... |
| Avg_Loss | 该epoch的平均loss | 0.425316 |
| Min_Batch_Loss | 该epoch中最小的batch loss | 0.285432 |
| Max_Batch_Loss | 该epoch中最大的batch loss | 0.623571 |
| Learning_Rate | 当前学习率 | 0.00100000 |
| Epoch_Time_Seconds | 该epoch训练时间（秒） | 14.52 |
| Best_Loss_So_Far | 目前为止的最佳loss | 0.425316 |
| Is_Best_Model | 是否为最佳模型 | True/False |
| Timestamp | 时间戳 | 2025-11-06 10:30:15 |

### Batch Metrics CSV 格式

| 列名 | 说明 | 示例 |
|------|------|------|
| Epoch | Epoch编号 | 1 |
| Batch | Batch编号 | 0, 1, 2, ... |
| Loss | 该batch的loss值 | 0.523415 |
| Learning_Rate | 当前学习率 | 0.00100000 |
| Timestamp | 时间戳 | 2025-11-06 10:30:15 |

---

## 训练后分析

### 1. 运行自动分析脚本

```bash
# 分析最新的epoch级别数据
python analyze_training.py

# 分析特定的metrics文件
python analyze_training.py --metrics_file metrics/epoch_metrics_20251106_103015.csv

# 分析batch级别数据
python analyze_training.py --batch_metrics
```

### 2. 生成的分析文件

#### a) Loss曲线图 (training_analysis_*.png)
包含4个子图：
1. **Training Loss Curve**：平均loss和最佳loss对比
2. **Loss Range**：每个epoch的loss波动范围
3. **Learning Rate Schedule**：学习率变化曲线
4. **Time per Epoch**：每个epoch的训练时间

#### b) 统计分析图 (training_stats_*.png)
包含2个子图：
1. **Loss Improvement per Epoch**：每个epoch的loss改善幅度
2. **Batch Loss Stability**：batch间loss的稳定性分析

#### c) 文本报告 (training_report_*.txt)
```
================================================================================
训练分析报告
================================================================================
数据文件: metrics/epoch_metrics_20251106_103015.csv
生成时间: 2025-11-06 12:30:45

训练概览:
  总Epoch数: 200
  总训练时间: 0.80小时
  平均每Epoch时间: 14.40秒

Loss统计:
  初始Loss: 0.523415
  最终Loss: 0.067721
  最佳Loss: 0.065432 (Epoch 195)
  Loss改善: 0.455694 (87.06%)

学习率:
  初始学习率: 0.00100000
  最终学习率: 0.00100000

Top 5最佳Epochs:
  Epoch 195: Loss=0.065432, LR=0.00100000
  Epoch 198: Loss=0.067215, LR=0.00100000
  Epoch 192: Loss=0.067721, LR=0.00100000
  ...
```

---

## 手动分析数据

### 使用Python (pandas)

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('metrics/epoch_metrics_20251106_103015.csv')

# 查看基本统计
print(df.describe())

# 绘制loss曲线
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Avg_Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('my_loss_curve.png')

# 找出loss最低的5个epochs
best_epochs = df.nsmallest(5, 'Avg_Loss')
print(best_epochs)
```

### 使用Excel

1. 打开CSV文件：`metrics/epoch_metrics_*.csv`
2. 插入图表 → 折线图
3. 选择数据范围：
   - X轴：Epoch列
   - Y轴：Avg_Loss列
4. 自定义图表样式

---

## 常见问题

### Q1: 为什么看不到实时的batch loss？
**A**: 默认每10个batch输出一次。可以通过`--log_interval 1`每个batch都输出，但会产生大量日志。

### Q2: batch_metrics文件太大怎么办？
**A**:
- 如果只需要了解训练趋势，epoch级别的数据就足够了
- batch级别数据主要用于：
  - 调试训练不稳定问题
  - 分析过拟合现象
  - 研究batch size影响

### Q3: 如何对比两次训练的结果？
**A**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取两次训练的数据
df1 = pd.read_csv('metrics/epoch_metrics_run1.csv')
df2 = pd.read_csv('metrics/epoch_metrics_run2.csv')

# 绘制对比图
plt.figure(figsize=(10, 6))
plt.plot(df1['Epoch'], df1['Avg_Loss'], label='Run 1')
plt.plot(df2['Epoch'], df2['Avg_Loss'], label='Run 2')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Comparison')
plt.savefig('comparison.png')
```

### Q4: metrics文件会自动覆盖吗？
**A**: 不会。每次训练都会创建带时间戳的新文件，所以可以保留历史训练记录。

### Q5: 如何删除旧的metrics文件？
**A**:
```bash
# 只保留最近7天的metrics
find metrics/ -name "*.csv" -mtime +7 -delete

# 只保留最新的3个metrics文件
cd metrics
ls -t epoch_metrics_*.csv | tail -n +4 | xargs rm -f
```

---

## 高级使用

### 1. 添加自定义指标

编辑`train_ddp.py`中的MetricsLogger类：

```python
# 在MetricsLogger._init_epoch_logger()中添加新列
headers = [
    'Epoch',
    'Avg_Loss',
    'Val_Loss',  # 新增：验证集loss
    'Train_Acc',  # 新增：训练准确率
    ...
]

# 在log_epoch()中传入新数据
def log_epoch(self, epoch, avg_loss, val_loss, train_acc, ...):
    row = [epoch, avg_loss, val_loss, train_acc, ...]
    ...
```

### 2. 实时监控脚本

```python
# watch_training.py
import pandas as pd
import time
from pathlib import Path

def watch_latest_metrics():
    metrics_dir = Path("metrics")
    files = list(metrics_dir.glob("epoch_metrics_*.csv"))
    latest = max(files, key=lambda p: p.stat().st_mtime)

    print(f"监控文件: {latest}")
    last_size = 0

    while True:
        try:
            current_size = latest.stat().st_size
            if current_size > last_size:
                df = pd.read_csv(latest)
                latest_row = df.iloc[-1]
                print(f"\nEpoch {latest_row['Epoch']}: "
                      f"Loss={latest_row['Avg_Loss']:.6f}, "
                      f"LR={latest_row['Learning_Rate']:.8f}")
                last_size = current_size
            time.sleep(5)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    watch_latest_metrics()
```

---

## 总结

### 基础用户（只关心最终结果）
```bash
python src/train_ddp.py
# 训练完成后
python analyze_training.py
```

### 学习用户（想了解训练细节）
```bash
# 记录详细的batch数据
python src/train_ddp.py --log_batch_metrics --log_interval 5

# 分析epoch数据
python analyze_training.py

# 分析batch数据
python analyze_training.py --batch_metrics
```

### 研究用户（深度分析）
```bash
# 导出数据后使用Python/Excel进行自定义分析
import pandas as pd
df = pd.read_csv('metrics/epoch_metrics_*.csv')
# 自定义分析...
```

---

**Happy Training! 🚀**

如有问题，请查看日志文件：`log/training_*.log`
