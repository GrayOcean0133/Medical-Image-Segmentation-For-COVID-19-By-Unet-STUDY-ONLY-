# COVID-19 肺部 CT 图像分割 —— 基于 U-Net 的医学影像语义分割

> **项目性质：个人学习项目，仅供研究参考，不可用于临床诊断。**
>
> 这个 README 写得很长。因为它记录了不只是"怎么用"，还记录了"踩了哪些坑"、"为什么这么改"，以及一个初学者在医学图像分割领域从一头雾水到跑出结果的完整过程。如果你也在学，希望能少走一些弯路。

---

## 目录

1. [项目简介](#1-项目简介)
2. [前身项目 Medic\_Project 及其局限性](#2-前身项目-medic_project-及其局限性)
3. [本次重构解决了什么问题（纠错过程）](#3-本次重构解决了什么问题纠错过程)
4. [技术架构详解](#4-技术架构详解)
5. [损失函数与数学原理](#5-损失函数与数学原理)
6. [训练流程](#6-训练流程)
7. [数据集与数据增强](#7-数据集与数据增强)
8. [实验结果](#8-实验结果)
9. [环境配置与使用方法](#9-环境配置与使用方法)
10. [文件结构](#10-文件结构)

---

## 1. 项目简介

本项目使用 **U-Net** 卷积神经网络，对 COVID-19 患者的肺部 CT 图像进行**语义分割**，目标是自动分割出 CT 图像中受感染的肺部区域（Ground Glass Opacity，磨玻璃影）。

### 核心指标（最终版本，500 epochs）

| 指标 | 数值 |
|------|------|
| 最佳 Epoch | 439 |
| 最佳 Train Loss | 0.1694 |
| Val Dice | **0.811** |
| Val IoU | **0.683** |
| 训练时长 | ~2 小时（4× RTX A6000）|

### 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.x |
| 分布式训练 | PyTorch DDP（DistributedDataParallel）|
| 混合精度 | `torch.amp.autocast` + `GradScaler` |
| 数据增强 | Albumentations（同步增强）|
| 优化器 | Adam |
| 学习率调度 | CosineAnnealingLR |
| 图像处理 | Pillow, NumPy |
| 可视化 | Matplotlib |
| 日志 | Python logging + CSV 指标记录 |

---

## 2. 前身项目 Medic\_Project 及其局限性

在本项目之前，有一个结构几乎相同的前身版本 `Medic_Project`（2025 年 11 月）。当时刚开始学医学图像分割，代码能跑起来，但训练结果很差，一度以为是数据集的问题。后来认真 review 代码，发现里面埋着好几颗雷，而且每一颗都是根本性的错误。

以下是 `Medic_Project` 中存在的具体问题，按严重程度排列：

---

### 问题 1：Sigmoid 放在 `forward()` 里 + 使用 `binary_cross_entropy` ——数值不稳定，AMP 直接爆炸

**原始代码（`unet_model.py`）：**

```python
# forward() 最后一行
return torch.sigmoid(self.final_conv(x))   # 输出已经是概率

# combined_loss
def combined_loss(pred, target, alpha=0.5):
    dice_loss_val = dice_loss(pred, target)
    bce_loss = F.binary_cross_entropy(pred, target)  # 期望概率输入
    return alpha * dice_loss_val + (1 - alpha) * bce_loss
```

**问题在哪里：**

PyTorch 中，`F.binary_cross_entropy` 期望输入是已经过 sigmoid 的概率值（0~1）。这在 FP32 下能跑，但**数值极度不稳定**——当 logit 绝对值较大时，sigmoid 会把值压到接近 0 或 1，再取 log 会产生 `-inf`。

更致命的是：启用 **AMP（自动混合精度）** 之后，FP16 的精度范围有限，这种组合几乎必然产生 `NaN` loss，导致训练直接崩掉。

**正确做法：**

```python
# forward() 返回 logits，不做 sigmoid
return self.final_conv(x)

# loss 使用 BCEWithLogitsLoss（内部 log-sum-exp 技巧，数值稳定）
bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pos_weight)
# 外部需要概率时，手动 sigmoid
pred_prob = torch.sigmoid(pred_logits)
```

`binary_cross_entropy_with_logits` 在内部使用 log-sum-exp 技巧：

$$\text{BCE}(x, y) = \max(x, 0) - x \cdot y + \log(1 + e^{-|x|})$$

这在数值上等价于 `sigmoid + log`，但不会在极端值时产生 `inf`。

---

### 问题 2：忽略类别不平衡 —— 模型学会了"偷懒"

CT 图像中，受感染区域（前景）通常只占整张图的 **2%~5%**，背景占绝大多数。

**原始代码没有处理类别不平衡。** 这意味着：模型完全可以通过"无论输入什么，输出全黑（全预测为背景）"来获得 95%+ 的像素准确率，并且 loss 也相当低。训练早期经常看到 loss 迅速下降到 0.3 左右然后不动了，就是因为模型在"摆烂"——它学会了什么都不预测。

**修复方案：动态 `pos_weight`**

```python
# 动态计算正负样本比，上限 100 防止极端值
neg = (target == 0).float().sum()
pos = (target == 1).float().sum()
pos_weight = torch.clamp(neg / (pos + 1e-6), min=1.0, max=100.0)

bce_loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
```

`pos_weight` 告诉模型：漏检一个正样本，惩罚是误报一个负样本的 N 倍（N ≈ 负/正比例）。这迫使模型必须认真对待那 2%~5% 的前景区域。

---

### 问题 3：数据增强不同步 —— 图像翻了，Mask 没翻

**原始 `data_loader.py` 的增强方式：**

```python
# 对图像应用 transforms
self.to_tensor = transforms.ToTensor()
# 对 mask 也单独处理...

# __getitem__ 里
image = self.to_tensor(image)
mask = self.to_tensor(mask)
```

用的是 `torchvision.transforms`。这个库的 `RandomHorizontalFlip`、`RandomRotation` 等变换在**每次调用时独立生成随机参数**。

这意味着：对同一个样本，图像可能被水平翻转了，但 mask 没有翻转。或者图像旋转了 30°，mask 旋转了 -15°。**图像和标注完全对不上**，等于在给模型喂噪声。

**修复方案：改用 `albumentations`（同步增强）**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})  # 同一随机参数同时作用于图像和 mask
```

`albumentations` 专为分割任务设计，`Compose` 里的所有变换共享同一随机状态，图像和 mask **保证同步**。

---

### 问题 4：U-Net 通道数配置错误，模型容量不足

**原始特征通道数：`features=[64, 128, 128, 256]`**

标准 U-Net 论文中，每经过一次下采样，通道数翻倍：`[64, 128, 256, 512]`。原始代码中第二层和第三层都是 128，通道数没有翻倍，模型在中间层的特征提取能力受限。

**修复：** `features=[64, 128, 256, 512]`，同时在瓶颈层扩展到 1024，使模型总参数量从约 8M 增加到 **31M**，特征表达能力显著提升。

---

### 问题 5：没有验证集指标，训练是盲目的

原始代码只记录训练集 Loss，完全不知道模型在验证集上的表现。最直接的后果是：不知道模型什么时候开始过拟合，不知道哪个 checkpoint 是真正"最好"的。

**修复：** 每个 epoch 结束后在验证集上计算 **Dice 系数** 和 **IoU**，以验证集 Loss 为基准保存最佳模型。

---

### 问题 6：logging 重复注册 handler，日志重复输出

`setup_logging()` 每次调用都会往 `root logger` 上追加新的 handler，而不清除旧的。在 DDP 多进程环境下，多个进程都调用一次，日志就重复打印 4 遍。

```python
# 修复：先清除所有旧 handler
root = logging.getLogger()
root.handlers.clear()
```

---

## 3. 本次重构解决了什么问题（纠错过程）

> 以下按时间线还原调试过程，比较痛苦，但也是学到东西最多的地方。

### 阶段一：AMP 开启后 Loss 变 NaN（5月13日凌晨）

第一次跑 500 epoch 时，前几个 epoch 看起来正常，然后 loss 突然跳到 `nan`，整个训练废掉。

排查过程：
1. 以为是学习率太大 → 调小到 1e-4 → 还是 nan
2. 以为是数据有问题（存在全黑图片）→ 检查数据集，没问题
3. 关掉 AMP → **不再出现 nan** → 锁定是 AMP + sigmoid + BCE 的组合问题
4. 查 PyTorch 文档，找到 `binary_cross_entropy_with_logits`，理解了 log-sum-exp 技巧
5. 把 sigmoid 从 `forward()` 移出，改用 `BCEWithLogitsLoss` → 问题消失

这一步花了大约半天时间，主要卡在"为什么关掉 AMP 就好了"这个问题上。

### 阶段二：Loss 下降后停滞在 0.3，预测全是黑图（5月13日下午）

Loss 能正常下降了，但到 0.3 左右就不动了。跑出来的预测图全是黑色，啥都没预测出来。

排查过程：
1. 可视化了几张预测结果，全黑，确认模型在"摆烂"
2. 统计了数据集的前景像素比例：**平均约 3.2%**
3. 意识到类别不平衡问题，加入动态 `pos_weight`
4. 重新训练，Loss 下降变慢了（因为惩罚变重），但预测图开始出现白色区域
5. 逐渐调整 `pos_weight` 上限（试过 10、50、100），最终保留 clamp 到 100

### 阶段三：分割边界很模糊，Dice 上不去（5月13日晚 - 5月18日）

加了 pos_weight 后，模型能预测出大致区域，但边界很糊，Dice 在 0.7 以下徘徊。

排查过程：
1. 检查数据增强 → 发现图像和 mask 的翻转方向不一致（torchvision 独立随机）
2. 换成 `albumentations`，统一增强，重新训练
3. 同时把通道数从 `[64, 128, 128, 256]` 改为 `[64, 128, 256, 512]`，增加模型容量
4. 加入 `Dropout2d(p=0.1)` 防止过拟合
5. 加入 `CosineAnnealingLR` 让学习率在训练后期慢慢衰减，使模型在精细区域收敛

5月18日重新跑 500 epoch，最终 Val Dice 达到 **0.811**，IoU **0.683**，边界清晰度明显提升。

---

## 4. 技术架构详解

### U-Net 整体结构

U-Net 是 Ronneberger 等人在 2015 年提出的医学图像分割网络，其最大特点是**编码器-解码器 + 跳跃连接（Skip Connection）**。

```
输入 (1, 512, 512)
    │
    ▼
[编码器 Encoder]
DoubleConv(1→64)   ──────────────────────────────────────────────┐ skip_1
MaxPool ↓                                                         │
DoubleConv(64→128) ──────────────────────────────────────┐ skip_2 │
MaxPool ↓                                                │        │
DoubleConv(128→256)──────────────────────────┐ skip_3   │        │
MaxPool ↓                                    │          │        │
DoubleConv(256→512)──────────────┐ skip_4   │          │        │
MaxPool ↓                        │          │          │        │
    │                            │          │          │        │
[瓶颈层 Bottleneck]              │          │          │        │
DoubleConv(512→1024)             │          │          │        │
    │                            │          │          │        │
[解码器 Decoder]                 │          │          │        │
ConvTranspose(1024→512) + cat ←──┘          │          │        │
DoubleConv(1024→512)                        │          │        │
ConvTranspose(512→256)  + cat ←─────────────┘          │        │
DoubleConv(512→256)                                    │        │
ConvTranspose(256→128)  + cat ←────────────────────────┘        │
DoubleConv(256→128)                                             │
ConvTranspose(128→64)   + cat ←──────────────────────────────────┘
DoubleConv(128→64)
    │
Conv2d(64→1, kernel=1)
    │
输出 logits (1, 512, 512)
```

**总参数量：31,036,481（约 31M）**

### DoubleConv 模块

```python
class DoubleConv(nn.Module):
    """
    双卷积块：Conv2d → BN → ReLU → Conv2d → BN → ReLU → [Dropout2d]
    """
```

每个 DoubleConv 做两次 3×3 卷积，padding=1 保持空间尺寸不变：

$$\text{output} = \text{ReLU}(\text{BN}(\text{Conv}(\text{ReLU}(\text{BN}(\text{Conv}(x))))))$$

**BatchNorm** 归一化每个 mini-batch 的 feature map，加速收敛，缓解梯度消失：

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

**Dropout2d(p=0.1)** 随机将整个 channel 置为 0，防止特征共适应（co-adaptation），起正则化作用。

### 跳跃连接（Skip Connection）

跳跃连接是 U-Net 的核心创新。编码器在下采样时会损失细节信息，解码器在上采样时难以恢复精确边界。跳跃连接直接将编码器对应层的 feature map 拼接（concat）到解码器，**保留了空间细节**。

```python
# 上采样后尺寸不一致时，用双线性插值对齐
if x.shape != skip.shape:
    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
# 沿通道维拼接
x = torch.cat((skip, x), dim=1)
```

---

## 5. 损失函数与数学原理

本项目使用 **Dice Loss + BCEWithLogitsLoss** 的组合损失函数。

### Dice Loss

Dice 系数来自医学图像分割领域，衡量两个集合的重叠程度：

$$\text{Dice} = \frac{2 |P \cap G|}{|P| + |G|} = \frac{2 \sum_i p_i g_i}{\sum_i p_i + \sum_i g_i}$$

其中 $P$ 是预测结果，$G$ 是 Ground Truth。Dice 系数越高越好（最大为 1）。

**Dice Loss** = $1 - \text{Dice}$（越低越好）：

$$\mathcal{L}_\text{Dice} = 1 - \frac{2 \sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}$$

加入 $\epsilon = 10^{-6}$ 防止分母为零（空 mask 时）。

Dice Loss 的特点：**天然对类别不平衡不敏感**，因为它只关注正样本区域的重叠，不依赖整体像素分布。

### BCEWithLogitsLoss（带 pos_weight）

$$\mathcal{L}_\text{BCE}(x, y) = -\left[ w_+ \cdot y \log\sigma(x) + (1-y)\log(1-\sigma(x)) \right]$$

其中 $\sigma(x) = \frac{1}{1+e^{-x}}$ 是 sigmoid，$w_+$ 是正样本权重（`pos_weight`）。

**pos_weight 的动态计算：**

$$w_+ = \text{clamp}\left(\frac{N_\text{neg}}{N_\text{pos}}, 1, 100\right)$$

当前景占 3% 时，$w_+ \approx 32$，意味着漏检一个前景像素相当于误检 32 个背景像素的代价。

### 组合损失

$$\mathcal{L} = \alpha \cdot \mathcal{L}_\text{Dice} + (1 - \alpha) \cdot \mathcal{L}_\text{BCE}, \quad \alpha = 0.5$$

两者互补：
- Dice Loss 关注分割区域的整体重叠，对边界不敏感
- BCE Loss 关注每个像素的分类准确性，提供更细粒度的梯度信号

---

## 6. 训练流程

### 分布式训练（DDP）

使用 PyTorch **DistributedDataParallel**，4 块 RTX A6000（47.4 GB each）并行训练。

DDP 核心原理：
1. 每个 GPU 持有一份完整的模型副本
2. 前向传播各自独立在本地 GPU 上进行
3. 反向传播时，梯度通过 **all-reduce（ring-allreduce 算法）** 在所有 GPU 间同步求均值
4. 所有 GPU 的参数保持严格同步

```python
model = DDP(model, device_ids=[rank])
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
```

### 混合精度训练（AMP）

使用 FP16 计算大部分前向传播和反向传播，只在必要时（如 BN 统计量）使用 FP32。

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

with autocast(device_type='cuda'):
    logits = model(images)
    loss = combined_loss(logits, masks)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

`GradScaler` 解决 FP16 梯度下溢问题：先将 loss 放大（scale），反向传播后再缩小（unscale），使小梯度不会变成 0。

### 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

将所有参数梯度的 L2 范数裁剪到 1.0 以内，防止梯度爆炸，在医学图像（mask 稀疏、梯度信号不均匀）场景下尤其重要。

### 学习率调度（CosineAnnealingLR）

$$\eta_t = \eta_\text{min} + \frac{1}{2}(\eta_\text{max} - \eta_\text{min})\left(1 + \cos\frac{t\pi}{T}\right)$$

初始 LR = 0.001，最终衰减至 $10^{-5}$，使模型在后期以更小步长在 loss landscape 的细节区域精细收敛。

### 验证指标

每个 epoch 在验证集（`val_images/` + `val_masks/`）上计算：

- **Dice 系数**：衡量分割区域重叠
- **IoU（Intersection over Union）**：$\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$

两者关系：$\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$，IoU 通常略低于 Dice。

---

## 7. 数据集与数据增强

### 数据集

[COVID-19 CT segmentation dataset](https://medicalsegmentation.com/covid19/)，包含：
- 训练集：2729 张 512×512 肺部 CT 切片（灰度图）及对应二值 mask
- 验证集：409 张切片

mask 中 `255`（白色）= 病变区域，`0`（黑色）= 背景。

### 数据增强（albumentations）

训练时对图像和 mask 同步应用以下增强（只在训练集，验证集不增强）：

```python
A.Compose([
    A.HorizontalFlip(p=0.5),        # 水平翻转
    A.VerticalFlip(p=0.3),          # 垂直翻转
    A.RandomRotate90(p=0.5),        # 随机 90° 旋转
    A.ElasticTransform(p=0.2),      # 弹性形变（模拟组织形态变化）
    A.RandomBrightnessContrast(     # 随机亮度/对比度（模拟不同扫描仪）
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.3
    ),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})
```

`ElasticTransform` 对医学图像特别有用，因为组织形态是弹性变化的，而不是刚性旋转/翻转。

---

## 8. 实验结果

### 训练曲线（500 epochs，5月18日）

| Epoch | Train Loss | Val Dice | Val IoU | 是否最佳 |
|-------|-----------|----------|---------|--------|
| 1 | 0.605 | - | - | ✓ |
| 50 | ~0.35 | ~0.75 | ~0.60 | |
| 100 | ~0.28 | ~0.78 | ~0.64 | |
| 200 | ~0.24 | ~0.80 | ~0.67 | |
| 351 | 0.182 | 0.807 | 0.678 | ✓ |
| 398 | 0.178 | 0.815 | 0.689 | ✓ |
| 424 | 0.175 | 0.807 | 0.677 | ✓ |
| **439** | **0.169** | **0.811** | **0.683** | **✓ 最终最佳** |
| 500 | 0.193 | 0.812 | 0.684 | |

后期（epoch 440+）模型基本收敛，loss 在 0.17~0.19 之间震荡，说明继续训练意义不大。

### 预测结果示例

以下为最佳模型（epoch 439）在验证集上的部分预测结果，保存于 `results/val_best/`：

- `results/val_best/pred_Jun_coronacases_case1_119.png`
- `results/val_best/pred_Jun_coronacases_case5_140.png`
- `results/val_best/pred_Jun_coronacases_case9_104.png`
- `results/val_best/pred_Jun_radiopaedia_10_85902_3_case12_107.png`

对比不同阶段预测（均来自同一张验证图）：

| 输出目录 | 说明 |
|---------|------|
| `results/val_best/` | epoch 439 最佳模型预测 |
| `results/val_epoch500/` | epoch 500 最终模型预测 |
| `results/val_epoch500_final/` | 最终模型完整验证集预测 |
| `results/val_final/` | 最后一次训练完整预测 |

---

## 9. 环境配置与使用方法

### 依赖安装

```bash
pip install torch torchvision
pip install albumentations
pip install numpy pillow pandas matplotlib
```

### 训练

```bash
# 单机 4 GPU 训练（默认）
python src/train_ddp.py

# 自定义参数
python src/train_ddp.py \
    --epochs 500 \
    --batch_size 8 \
    --lr 0.001 \
    --features 64 128 256 512 \
    --dropout_p 0.1 \
    --img_size 512
```

训练过程中会自动：
- 每 10 epoch 保存一次 checkpoint → `checkpoints/model_epoch_N.pth`
- 出现更低 val loss 时保存最佳模型 → `checkpoints/best_model.pth`
- 写入训练日志 → `log/training_TIMESTAMP.log`
- 写入 CSV 指标 → `metrics/epoch_metrics_TIMESTAMP.csv`

### 预测

```bash
# 对单张图像预测
python src/predict.py \
    --model_path checkpoints/best_model.pth \
    --image_path data_set/CT_COVID/val_images/example.png \
    --output_dir results/my_test

# 对整个目录批量预测，并与 ground truth 对比
python src/predict.py \
    --model_path checkpoints/best_model.pth \
    --image_path data_set/CT_COVID/val_images/ \
    --mask_path data_set/CT_COVID/val_masks/ \
    --output_dir results/my_test \
    --threshold 0.5
```

### 查看训练进度

```bash
# 实时跟踪日志
tail -f log/training_*.log

# 实时查看 loss 数据
tail -f metrics/epoch_metrics_*.csv
```

---

## 10. 文件结构

```
Medical-Image-Segmentation-For-COVID-19-By-Unet-STUDY-ONLY-/
├── src/
│   ├── unet_model.py        # U-Net 模型定义、损失函数、评估指标
│   ├── data_loader.py       # 数据集类、albumentations 增强
│   ├── train_ddp.py         # DDP 分布式训练主脚本
│   ├── predict.py           # 单张/批量预测脚本
│   ├── gui.py               # tkinter GUI 界面
│   └── test/                # 单元测试
│
├── data_set/
│   └── CT_COVID/
│       ├── frames/          # 训练集 CT 图像（2729 张）
│       ├── masks/           # 训练集掩码
│       ├── val_images/      # 验证集图像（409 张）
│       └── val_masks/       # 验证集掩码
│
├── checkpoints/             # 模型权重（.pth，已加入 .gitignore）
├── log/                     # 训练/预测日志（.log，已加入 .gitignore）
├── metrics/                 # 训练指标 CSV（已加入 .gitignore）
├── results/                 # 预测结果图像
│   ├── val_best/            # 最佳模型（epoch 439）预测结果
│   ├── val_epoch500/        # epoch 500 预测结果
│   └── ...
├── Help/                    # 补充文档
└── .gitignore
```

---

## 参考

- Ronneberger O, Fischer P, Brox T. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
- Milletari F, Navab N, Ahmadi S A. *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*. 3DV 2016.（Dice Loss 来源）
- COVID-19 CT Segmentation Dataset: [medicalsegmentation.com/covid19](https://medicalsegmentation.com/covid19/)
- PyTorch AMP 文档: [pytorch.org/docs/stable/amp.html](https://pytorch.org/docs/stable/amp.html)
- Albumentations: [albumentations.ai](https://albumentations.ai)

---

*最后更新：2026-05-19*
*项目状态：训练完成，结果已验证*
