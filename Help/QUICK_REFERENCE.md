# 快速参考卡片

## 训练命令

```bash
# 基础训练（推荐）
python src/train_ddp.py

# 详细记录（学习AI必备）
python src/train_ddp.py --log_batch_metrics --log_interval 5

# 快速测试（10个epochs）
python src/train_ddp.py --epochs 10
```

## 分析命令

```bash
# 自动分析最新训练
python analyze_training.py

# 分析batch数据
python analyze_training.py --batch_metrics

# 查看生成的图表
cd analysis && ls -lh
```

## 实时查看

```bash
# 查看训练日志
tail -f log/training_*.log

# 查看metrics数据
tail -f metrics/epoch_metrics_*.csv
```

## 数据文件位置

```
metrics/epoch_metrics_TIMESTAMP.csv  # Epoch统计（总是生成）
metrics/batch_metrics_TIMESTAMP.csv  # Batch详情（--log_batch_metrics）
log/training_TIMESTAMP.log           # 完整日志
analysis/training_analysis_*.png     # 分析图表
analysis/training_report_*.txt       # 文本报告
```

## 训练时看到的信息

```
Epoch [10/200] - 进度 5.8% (10/171) - Batch Loss: 0.072156 - LR: 0.00100000
                 ↑       ↑       ↑         ↑                      ↑
              进度百分比  batch编号  当前batch的loss           学习率
```

## CSV文件格式

### epoch_metrics_*.csv
| Epoch | Avg_Loss | Min_Batch_Loss | Max_Batch_Loss | Learning_Rate | Epoch_Time_Seconds | Best_Loss_So_Far | Is_Best_Model |
|-------|----------|----------------|----------------|---------------|--------------------|--------------------|---------------|
| 1 | 0.425316 | 0.285432 | 0.623571 | 0.00100000 | 14.52 | 0.425316 | True |

### batch_metrics_*.csv
| Epoch | Batch | Loss | Learning_Rate |
|-------|-------|------|---------------|
| 1 | 0 | 0.523415 | 0.00100000 |

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 200 | 训练轮数 |
| `--batch_size` | 4 | 每个GPU的batch大小 |
| `--lr` | 0.001 | 学习率 |
| `--log_interval` | 10 | 每N个batch输出一次 |
| `--log_batch_metrics` | False | 记录每个batch详情 |

## Python快速分析

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('metrics/epoch_metrics_TIMESTAMP.csv')

# 绘制loss曲线
plt.plot(df['Epoch'], df['Avg_Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('my_loss.png')

# 查看最佳epochs
print(df.nsmallest(5, 'Avg_Loss'))
```

## 常用命令组合

```bash
# 训练 → 分析一条龙
python src/train_ddp.py && python analyze_training.py

# 详细训练 → 详细分析
python src/train_ddp.py --log_batch_metrics && \
python analyze_training.py && \
python analyze_training.py --batch_metrics
```

---
**详细文档**: `METRICS_GUIDE.md` | **完整报告**: `REALTIME_MONITORING_SUMMARY.md`


## [-> 返回README](../README.md)
