# 训练系统使用说明 - 已更新为每个Batch输出

## ⚠️ 重要更新

**默认行为已修改**：现在训练时会**每个batch都输出**日志信息，让你实时看到每一步的loss变化！

---

## 快速开始

### 基础训练（每个batch都显示）

```bash
python src/train_ddp.py
```

**你会看到**：
```
Epoch [1/200] - 进度 0.0% (0/171) - Batch Loss: 0.523415 - LR: 0.00100000
Epoch [1/200] - 进度 0.6% (1/171) - Batch Loss: 0.498234 - LR: 0.00100000
Epoch [1/200] - 进度 1.2% (2/171) - Batch Loss: 0.476123 - LR: 0.00100000
Epoch [1/200] - 进度 1.8% (3/171) - Batch Loss: 0.461567 - LR: 0.00100000
...（每个batch都会输出）
```

这样你可以：
- ✅ 实时看到每一步的loss变化
- ✅ 立即发现训练问题
- ✅ 学习loss的波动规律
- ✅ 更好地理解训练过程

---

## 调整输出频率

### 如果觉得输出太频繁

```bash
# 每10个batch输出一次（减少输出）
python src/train_ddp.py --log_interval 10

# 每50个batch输出一次（更少输出）
python src/train_ddp.py --log_interval 50
```

### 如果需要记录详细数据到CSV

```bash
# 除了屏幕显示，还保存每个batch的数据到CSV文件
python src/train_ddp.py --log_batch_metrics
```

⚠️ 注意：`--log_batch_metrics` 会生成约34,000行的CSV文件（200 epochs × 171 batches）

---

## 完整参数说明

| 参数 | 默认值 | 说明 | 示例 |
|------|--------|------|------|
| `--log_interval` | **1** | 屏幕输出间隔（每N个batch） | `--log_interval 10` |
| `--log_batch_metrics` | False | 是否保存batch数据到CSV | `--log_batch_metrics` |
| `--epochs` | 200 | 训练轮数 | `--epochs 100` |
| `--batch_size` | 4 | 每个GPU的batch大小 | `--batch_size 8` |
| `--lr` | 0.001 | 学习率 | `--lr 0.0001` |

---

## 使用场景

### 场景1：学习训练过程（推荐新手）

```bash
# 使用默认设置，每个batch都看得到
python src/train_ddp.py --epochs 10
```

**适合**：
- 第一次训练
- 学习AI模型
- 调试训练问题
- 观察loss变化规律

### 场景2：长时间训练（推荐经验用户）

```bash
# 减少输出，避免日志太长
python src/train_ddp.py --log_interval 20 --epochs 200
```

**适合**：
- 正式训练
- 长时间运行
- 已经熟悉训练过程

### 场景3：深度分析（推荐研究）

```bash
# 记录所有batch数据，用于事后分析
python src/train_ddp.py --log_batch_metrics --epochs 200
```

**适合**：
- 研究训练动态
- 分析过拟合
- 调优超参数

---

## 输出内容详解

### 每个Batch输出的信息

```
Epoch [15/200] - 进度 5.8% (10/171) - Batch Loss: 0.072156 - LR: 0.00100000
  ↑       ↑         ↑       ↑  ↑           ↑                     ↑
  │       │         │       │  │           │                     └─ 当前学习率
  │       │         │       │  │           └─ 这个batch的loss值
  │       │         │       │  └─ 总batch数
  │       │         │       └─ 当前batch编号
  │       │         └─ 完成百分比
  │       └─ 总epoch数
  └─ 当前epoch
```

### 每个Epoch结束时的总结

```
================================================================================
Epoch [15/200] 完成
平均损失: 0.072316          ← 这个epoch所有batch的平均loss
最小Batch损失: 0.065432      ← 最好的batch
最大Batch损失: 0.083571      ← 最差的batch（用于判断稳定性）
学习率: 0.00100000          ← 当前学习率
本epoch用时: 14.52秒         ← 训练时间
预计剩余时间: 0.75小时       ← ETA预估
================================================================================
```

---

## 日志文件

### 屏幕输出 vs 文件保存

| 类型 | 位置 | 内容 | 控制参数 |
|------|------|------|----------|
| **屏幕日志** | 终端显示 | 按`log_interval`显示 | `--log_interval 1` |
| **完整日志** | `log/training_*.log` | 所有日志（无论interval） | 自动保存 |
| **Epoch CSV** | `metrics/epoch_metrics_*.csv` | 每个epoch统计 | 自动保存 |
| **Batch CSV** | `metrics/batch_metrics_*.csv` | 每个batch详情 | `--log_batch_metrics` |

**重点**：
- 屏幕输出可以通过`--log_interval`控制
- 但日志文件`log/training_*.log`总是包含所有信息
- CSV文件只记录关键统计数据

---

## 实时查看技巧

### 在训练过程中另开终端

```bash
# 实时查看完整日志（包括所有batch）
tail -f log/training_*.log

# 只看epoch总结
tail -f log/training_*.log | grep "Epoch \["

# 实时查看metrics数据
tail -f metrics/epoch_metrics_*.csv
```

### 使用screen或tmux后台运行

```bash
# 使用screen
screen -S training
python src/train_ddp.py
# 按 Ctrl+A 然后按 D 离开
# 查看: screen -r training

# 使用tmux
tmux new -s training
python src/train_ddp.py
# 按 Ctrl+B 然后按 D 离开
# 查看: tmux attach -t training
```

---

## 常见问题

### Q: 输出太多，看不过来怎么办？

A: 增加`--log_interval`：
```bash
python src/train_ddp.py --log_interval 20
```

### Q: 怎么只看重要信息？

A: 使用grep过滤：
```bash
# 只看epoch总结
python src/train_ddp.py 2>&1 | grep "Epoch \["

# 只看保存模型的信息
python src/train_ddp.py 2>&1 | grep "已保存"
```

### Q: 为什么修改默认为每个batch输出？

A: **为了更好地学习AI**：
- 看到每个batch的变化，理解训练动态
- 及时发现问题（loss爆炸、NaN等）
- 更直观地感受模型学习过程
- 对新手更友好

### Q: 这会影响训练速度吗？

A: **几乎不会**：
- 打印到屏幕的开销极小（<0.01%）
- 主要瓶颈是GPU计算，不是日志输出
- 如果担心，可以设置`--log_interval 10`

### Q: 如何保存屏幕输出？

A:
```bash
# 方法1：重定向到文件
python src/train_ddp.py 2>&1 | tee my_training_output.txt

# 方法2：使用nohup
nohup python src/train_ddp.py > training_output.log 2>&1 &
```

---

## 对比：新 vs 旧

| 特性 | 旧默认 | 新默认 |
|------|--------|--------|
| 屏幕输出间隔 | 每10个batch | **每1个batch** |
| 信息详细度 | 中等 | **高** |
| 适合人群 | 有经验的用户 | **所有用户，特别是学习者** |
| 是否可调整 | ✅ `--log_interval N` | ✅ `--log_interval N` |

---

## 推荐工作流

### 新手学习流程

```bash
# 1. 先用少量epochs测试（观察每个batch）
python src/train_ddp.py --epochs 5

# 2. 分析结果
python analyze_training.py

# 3. 正式训练（可以适当减少输出）
python src/train_ddp.py --log_interval 5 --epochs 200

# 4. 最终分析
python analyze_training.py
```

### 进阶用户流程

```bash
# 直接正式训练（减少屏幕输出，但保存详细batch数据）
python src/train_ddp.py --log_interval 20 --log_batch_metrics --epochs 200

# 训练后深度分析
python analyze_training.py
python analyze_training.py --batch_metrics
```

---

## 快速命令参考

```bash
# 最详细（默认）
python src/train_ddp.py

# 适中（每10个batch）
python src/train_ddp.py --log_interval 10

# 简洁（每50个batch）
python src/train_ddp.py --log_interval 50

# 详细+保存batch数据
python src/train_ddp.py --log_batch_metrics

# 简洁+保存batch数据
python src/train_ddp.py --log_interval 20 --log_batch_metrics
```

---

## 总结

### 新默认设置的优势

✅ **实时监控**：每个batch都能看到
✅ **及时发现问题**：loss异常立即可见
✅ **学习友好**：更好地理解训练过程
✅ **灵活调整**：随时可以增加interval

### 何时调整interval

- `--log_interval 1`：学习、调试、短期训练
- `--log_interval 10-20`：正式训练、已熟悉过程
- `--log_interval 50+`：超长训练、只关心最终结果

---

**立即开始训练，实时观察loss变化！** 🚀

```bash
cd /workspace/GrayOcean/code/Medic_Project
python src/train_ddp.py --epochs 10
```


## [-> 返回README](../README.md)
