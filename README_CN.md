> 🌐 [English](README.md) | 中文

# 基于 U-Net 的 COVID-19 肺部 CT 图像分割

> **免责声明：本项目为个人学习项目，仅供研究学习使用，不可用于临床诊断。**
>
> **Disclaimer: This is a personal learning project for research purposes only. Not intended for clinical diagnosis.**

---

## ⚠️ 请先读这一段

### 1. 历史训练结果无法从本仓库复现

"训练结果"一节里那次 500 epoch 的训练，是在一台**已经归档/下线的多卡 DDP 服务器**上完成的。那台机器上的 `checkpoints/`、`log/`、`metrics/` 都没有随仓库提交（它们在 `.gitignore` 里），原始数据集也不在仓库中。

**所以：那张表是历史记录，不是"clone 下来就能复现"的结果。请把它当作未经验证的数据。**

### 2. 现在还能验证的东西

仓库里确实留了一份真实输出：`results/val_epoch500_final/`（6 例原始 CT + 对应真值掩码 + 预测结果）。直接对这些 PNG 重算 Dice/IoU：

| 病例 | Dice | IoU |
|---|---|---|
| case1 | 0.796 | 0.661 |
| case3 | 0.885 | 0.794 |
| case5 | 0.806 | 0.675 |
| case6 | 0.863 | 0.759 |
| case8 | 0.861 | 0.757 |
| case10 | 0.884 | 0.793 |
| **平均** | **0.849** | **0.740** |

这些数值和当年声称的 0.811 / 0.683 在同一量级，可以佐证**当年的训练是真实发生过的**。但请注意：这 6 例**不构成独立验证**（样本太少，也无法保证它们是严格留出的测试集）。

### 3. 2026-10 修复记录

早期版本的 README 描述了一次"重构"，但那次重构**只存在于 README 里**——`src/` 下的代码一行都没改。现在这个缺口已经补上。以下改动均已落到代码，并有测试覆盖：

| 修复内容 | 文件 |
|---|---|
| `forward()` 返回原始 logits；损失改用 `BCEWithLogits` + 动态 `pos_weight` | `unet_model.py` |
| 新增 `dice_coeff` / `iou_score`（全验证集累计计算） | `unet_model.py` |
| 数据增强真正随机化（旧实现是空转），且 image/mask 严格同步 | `data_loader.py` |
| 掩码缺失快速报错、文件排序、训练/验证切分工具 | `data_loader.py` |
| 检查点保存模型结构配置，推理端按配置重建网络 | `checkpoint.py` |
| 旧格式（只有 state_dict）的 `.pth` 可自动推断结构并加载 | `checkpoint.py` |
| 验证循环、按 val Dice 选最优模型、AMP、余弦退火、随机种子 | `train_ddp.py` |
| DDP 日志：带 rank 前缀 + 子进程重新配置日志（旧代码子进程根本没日志） | `train_ddp.py` |
| `--features` 支持 `64 128 256 512`（旧代码会解析成字符列表） | `train_ddp.py` |
| 推理可从检查点重建结构，支持 `--threshold`、`--mask_path` 算指标 | `predict.py` |
| GUI 改用统一检查点加载；阈值滑块真正生效（旧代码在 0/1 结果上再比阈值） | `gui.py` |
| 测试可用 `pytest` 或直接运行，含真实断言（共 20 个） | `src/test/`、`conftest.py` |

**没有**做的：**没有**使用 `albumentations`。增强改用 `torchvision.transforms.v2`，它天生支持成对变换，且不需要额外依赖。

---

## 一、项目简介

使用 **U-Net** 卷积网络对肺部 CT 做语义分割，自动标出感染区域（磨玻璃影）。

### 技术栈（以本仓库代码实际实现为准）

| 类别 | 技术 |
|------|------|
| 深度学习 | PyTorch 2.x |
| 分布式训练 | `DistributedDataParallel` + `DistributedSampler` |
| 混合精度 | `torch.amp.autocast` + `GradScaler`（需显式加 `--amp`） |
| 数据增强 | `torchvision.transforms.v2`（image/mask 成对变换） |
| 优化器 | Adam |
| 学习率调度 | `CosineAnnealingLR`（需显式加 `--scheduler cosine`） |
| 类别不平衡 | BCE 动态 `pos_weight`（默认开启） |
| 图像处理 | Pillow、NumPy |
| 日志 | Python `logging`（带 rank 标记）+ CSV 指标 |
| 测试 | pytest |

---

## 二、模型、损失与指标

### U-Net（`src/unet_model.py`）

标准 U-Net：4 层编码器、瓶颈层、4 层解码器（含跳跃连接）、1×1 输出卷积。

```
features = [64, 128, 256, 512]   # 瓶颈层扩展到 1024
总参数量 = 31,036,481
```

编码器通道逐层翻倍。原始代码用的是 `[64, 128, 128, 256]`——中间那层没有翻倍，等于把中间层容量压扁了，总参数只有约 850 万。

> 现在通道列表统一由 `DEFAULT_FEATURES` 常量或检查点提供，训练和推理**不可能再对不上**。

### 关键约定：`forward()` 返回 logits，不返回概率

```python
# forward() 最后一行：
return self.final_conv(x)        # 原始 logits，不做 sigmoid

# 需要 [0, 1] 概率时用这个：
probs = predict_proba(logits)    # torch.sigmoid(logits)
```

在 `forward()` 里做 sigmoid 再喂给 `F.binary_cross_entropy`（原始写法）在 FP32 下**不会**爆炸——PyTorch 内部把 `log` 钳制在 −100——但梯度会严重饱和。把 logits 独立出来才是正确契约，也正是这样才能用上数值稳定的 log-sum-exp 形式：

$$\text{BCE}(x, y) = \max(x, 0) - x \cdot y + \log(1 + e^{-|x|})$$

### 类别不平衡

病灶前景通常只占像素的 2%~5%。不加权时，模型只要"全预测背景"就能把 loss 压得很低。`compute_pos_weight()` 按 `负/正` 比例给出权重并钳制在 `[1, 100]`：

```python
pos_weight = compute_pos_weight(target)   # clamp(neg / pos, 1, 100)
loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
```

用 `--no-class_balance` 关闭。

### 组合损失

```python
combined_loss = alpha * dice_loss + (1 - alpha) * bce_with_logits
# alpha 默认 0.5
```

### 评估指标

`dice_coeff()` 和 `iou_score()` 在**整个评估集**上累计交集/预测/真值后一次性相除。如果改成"逐图算完再平均"，小病灶样本会被大病灶样本稀释。

---

## 三、数据与数据增强

### 数据集（`src/data_loader.py`）

* 读 `frames/*.png`，到 `masks/` 下找同名文件
* **缺掩码立即报错**，并列出前几个缺失的文件名（旧代码要训练到一半才抛 `FileNotFoundError`）
* 文件名排序，保证多次运行顺序一致
* 灰度化 → 缩放到 `--img_size`（图像双线性 / 掩码最近邻）→ 图像归一化到 `[-1, 1]` → 掩码二值化为 `{0, 1}`

### 成对增强

```python
train_transform = T.Compose([
    T.ToImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.NEAREST, fill=0),
    T.ToDtype(torch.float32, scale=True),
])

image, mask = train_transform(image, mask)   # 两者一起传入
```

因为图像和掩码是**在同一次调用**里传入的，`transforms.v2` 只采样一次随机参数并同时作用到两者，**不可能错位**。

> **回归说明。** 原始代码用 `torch.manual_seed(torch.initial_seed())` 在每个样本前重置随机状态。`initial_seed()` 返回的是固定的初始值、不会推进，于是每个样本都被重置回**同一个**随机状态——增强等于完全没做，而且还会污染全局 RNG。`src/test/test_data_loader.py` 里的 `test_augmentation_is_actually_random` 就是防止这个问题复发。

旋转对图像和掩码都用 NEAREST，避免插值把二值掩码的边缘糊掉；翻转不涉及插值，不受影响。

### 训练/验证切分

默认从训练集留出 15% 做验证：

```bash
--val_split 0.15      # 设为 0 则关闭验证
```

切分是确定性的（带种子的随机排列）。验证子集会被**不带增强**的 transform 重新包一层。也可以直接指定独立目录：

```bash
--val_images_dir .../val_images --val_masks_dir .../val_masks
```

---

## 四、训练流程（`src/train_ddp.py`）

每个 epoch：

1. `train_sampler.set_epoch(epoch)`，保证每个 epoch 打乱顺序不同
2. 前向/反向（可选 AMP autocast），`scaler.step` / `scaler.update`
3. 可选 `scheduler.step()`
4. **验证只在 rank 0 上跑**，前后用 `dist.barrier()` 同步——否则其它 rank 会提前冲进下一个 epoch 的梯度同步，直接挂起
5. val Dice 提升时保存最优模型（没有验证集则退回用训练损失判断）
6. 每 `--save_every` 个 epoch 保存一次检查点

### 检查点自带结构信息

```python
{
  "model_state_dict": ...,
  "config": {"in_channels": 1, "out_channels": 1, "features": [64, 128, 256, 512]},
  "epoch": 439,
  "metrics": {"train_loss": ..., "val_loss": ..., "val_dice": ..., "val_iou": ...},
  "optimizer_state_dict": ...,
}
```

以前只存 `state_dict()`，推理端只能**猜**结构：`predict.py` 硬编码默认 `[64,128,256,512]`，而训练默认的是 `[64,128,128,256]`，于是加载训练好的模型直接 `size mismatch` 报错。现在推理端按 `config` 重建。

`load_checkpoint()` 同时兼容旧格式（只有裸 state_dict）的文件，会从权重形状反推结构：

```
WARNING - 检测到旧格式 state_dict，已推断结构:
  {'in_channels': 1, 'out_channels': 1, 'features': [64, 128, 256, 512]}
```

**这一点对"服务器已经归档、只剩几个 `.pth`"的情况特别有用——不需要知道当年的配置也能加载。**

### 日志

日志行带 rank 前缀；每个 spawn 出来的子进程都会**重新配置** logging（spawn 的子进程是全新解释器，**不会**继承父进程的 logging 配置——原始代码因此从来没有把任何训练输出写进日志文件）。

重新配置前会清空已有 handler，所以 DDP 下不会每条日志重复 `world_size` 遍。

### 可复现性

`--seed`（默认 42）会同时设置 `random`、NumPy、PyTorch 的种子；各 rank 使用 `seed + rank`。

---

## 五、历史训练结果（⚠️ 未经验证，见开头说明）

当年在那台已归档的 DDP 服务器上记录的数据，**无法从本仓库复现**。

| Epoch | Train Loss | Val Dice | Val IoU | 是否最优 |
|-------|-----------|----------|---------|-------|
| 1 | 0.605 | — | — | ✓ |
| 50 | ~0.35 | ~0.75 | ~0.60 | |
| 100 | ~0.28 | ~0.78 | ~0.64 | |
| 200 | ~0.24 | ~0.80 | ~0.67 | |
| 351 | 0.182 | 0.807 | 0.678 | ✓ |
| 398 | 0.178 | 0.815 | 0.689 | ✓ |
| 424 | 0.175 | 0.807 | 0.677 | ✓ |
| **439** | **0.169** | **0.811** | **0.683** | **✓ 最终最优** |
| 500 | 0.193 | 0.812 | 0.684 | |

### 关于以前在这里链接的那几张图

早期版本的 README 把 `results/val_best/` 里的 6 个文件当作"最优模型的代表性预测"展示。那 6 张图**是全黑的**——模型一个前景像素都没预测出来。它们在后续提交里被删除（`Remove incorrect black prediction images from val_best`），但 README 一直在链接它们。

想看真实预测，请用 `results/val_epoch500_final/`，里面有 6 组 `orig_*` / `mask_*` / `pred_*`，也是上面那张 Dice/IoU 表的数据来源。

**本仓库中并不存在的目录：** `results/val_best/`、`results/val_epoch500/`、`results/val_final/` —— 旧 README 把这几个都列进去了。

---

## 六、安装与使用

### 安装依赖

```bash
pip install -r requirements.txt
```

**不需要** `albumentations`。

### 准备数据

`data_set/` 被 gitignore 了，需要自己准备：

```
data_set/CT_COVID/
├── frames/     # 训练用 CT 图像（*.png）
└── masks/      # 同名掩码（*.png）
```

### 训练

```bash
# 单机，使用全部可见 GPU
python src/train_ddp.py

# 自定义参数
python src/train_ddp.py \
    --epochs 500 \
    --batch_size 8 \
    --lr 0.001 \
    --features 64 128 256 512 \
    --val_split 0.15 \
    --amp \
    --scheduler cosine \
    --seed 42
```

输出：

* 检查点 → `checkpoints/model_epoch_N.pth`、`checkpoints/best_model.pth`
* 日志 → `log/training_TIMESTAMP.log`
* 每 epoch 指标 CSV → `metrics/epoch_metrics_TIMESTAMP.csv`
* 可选每 batch 指标 CSV → `metrics/batch_metrics_TIMESTAMP.csv`（加 `--log_batch_metrics`）

### 推理

```bash
# 单张图像
python src/predict.py \
    --model_path checkpoints/best_model.pth \
    --image_path data_set/CT_COVID/frames/case1.png \
    --output_dir results/my_test

# 整个目录，并计算真值指标
python src/predict.py \
    --model_path checkpoints/best_model.pth \
    --image_path data_set/CT_COVID/val_images/ \
    --mask_path  data_set/CT_COVID/val_masks/ \
    --output_dir results/my_test \
    --threshold 0.5
```

`--mask_path` 会打印平均 Dice / IoU；`--threshold` 控制二值化阈值（GUI 的滑块等价于此）。

### GUI

```bash
python src/gui.py     # 读取 checkpoints/best_model.pth
```

### 测试

```bash
pytest                              # 20 个测试，约 2 秒，只用合成数据
python src/test/test_unet_model.py  # 也可以直接运行任意测试文件
```

测试**不依赖** `data_set/`。

### 监控训练

```bash
tail -f log/training_*.log
cat metrics/epoch_metrics_*.csv
python src/test/analyze_training.py --metrics_file metrics/epoch_metrics_TIMESTAMP.csv
```

---

## 七、项目结构

```
Medical-Image-Segmentation-For-COVID-19-By-Unet-STUDY-ONLY-/
├── src/
│   ├── unet_model.py        # U-Net、损失函数、Dice/IoU 指标
│   ├── data_loader.py       # Dataset、成对 transforms.v2 增强、切分工具
│   ├── checkpoint.py        # 检查点读写（含模型结构配置）
│   ├── train_ddp.py         # DDP 训练：验证、AMP、学习率调度
│   ├── predict.py           # 单张/批量推理，可选 Dice/IoU
│   ├── gui.py               # tkinter 图形界面
│   └── test/
│       ├── _test_utils.py           # sys.path 引导 + 合成数据生成
│       ├── test_unet_model.py
│       ├── test_data_loader.py
│       ├── test_model_save.py
│       ├── test_training_pipeline.py
│       └── analyze_training.py      # 指标 CSV 绘图工具（不是测试）
│
├── data_set/                # 不在仓库中（gitignore），需自备
├── checkpoints/             # gitignore
├── log/                     # gitignore
├── metrics/                 # gitignore
├── results/
│   └── val_epoch500_final/  # 唯一幸存的真实结果：orig/mask/pred 各 6 张
├── Help/                    # 补充说明
├── conftest.py              # pytest 路径配置
├── pytest.ini
├── requirements.txt
└── .gitignore
```

---

## 八、调试历程（学习笔记）

> 留存备查。这些是从"第一版跑不起来"到"能跑出结果"过程中踩到的坑。其中若干项直到 2026-10 那次修复才真正落地。

**1. `forward()` 里的 sigmoid + `F.binary_cross_entropy`**
把已经激活过的输出再喂给 `binary_cross_entropy` 是典型的接口错配。FP32 下不会炸（PyTorch 把 `log` 钳制在 −100），但梯度饱和很严重。修法：返回 logits，改用 `binary_cross_entropy_with_logits`。

**2. 类别不平衡 → "学会什么都不预测"**
早期训练里 loss 很快掉到 ~0.3 然后就不动了：模型已经收敛到"全部预测背景"。修法：动态 `pos_weight`。

**3. 数据增强错位**
图像和掩码各自独立变换，翻转/旋转角度对不上，等于拿噪声训练。修法：一次调用同时传入两者。

**4. 编码器通道数不对**
`[64, 128, 128, 256]` 在中间那层没有翻倍。修法：`[64, 128, 256, 512]`（约 850 万 → 3100 万参数），并用 `DEFAULT_FEATURES` 作为唯一来源。

**5. 没有验证指标**
只看训练损失，既发现不了过拟合，也无法合理地挑选检查点。修法：每个 epoch 算 val Dice/IoU，按 val Dice 选最优。

**6. DDP 下日志 handler 重复**
`setup_logging()` 反复追加 handler，导致每行日志按进程数重复。修法：添加前先清空。（由此引出的第二个问题——spawn 出来的子进程完全没有日志——通过在每个 rank 内重新配置 logging 解决。）

**7. `unet_model.py` 默认值与 `train_ddp.py` 默认值不一致**
两者不同，导致 `predict.py` 永远加载不了训练出的模型（`size mismatch`，31,036,481 vs 8,568,513 参数）。修法：结构配置随检查点一起保存，并为旧文件提供按形状推断。

---

## 参考文献

- Ronneberger O, Fischer P, Brox T. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
- Milletari F, Navab N, Ahmadi S A. *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*. 3DV 2016.（Dice Loss 来源）
- COVID-19 CT Segmentation Dataset: [medicalsegmentation.com/covid19](https://medicalsegmentation.com/covid19/)
- PyTorch AMP： [pytorch.org/docs/stable/amp.html](https://pytorch.org/docs/stable/amp.html)
- torchvision transforms v2： [pytorch.org/vision/stable/transforms.html](https://pytorch.org/vision/stable/transforms.html)
