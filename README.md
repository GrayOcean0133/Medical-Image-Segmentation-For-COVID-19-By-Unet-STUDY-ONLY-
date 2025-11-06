# Medical Image Segmentation For COVID-19 By U-Net

基于 U-Net 深度学习模型的 COVID-19 肺部 CT 图像分割项目（学习实践项目）

## 项目简介

使用 U-Net 神经网络对 COVID-19 患者的肺部 CT 图像进行自动分割，识别感染区域。支持分布式训练、实时预测和图形化界面操作。

## 主要特性

- ✅ **U-Net 模型**: 经典的医学图像分割架构
- ✅ **分布式训练**: 支持多GPU DDP训练加速
- ✅ **完整日志系统**: Epoch/Batch级别的训练指标记录
- ✅ **可视化分析**: 训练曲线分析和可视化工具
- ✅ **图形界面**: tkinter实现的GUI预测界面
- ✅ **相对路径**: 代码可移植，适合学习参考

## 项目结构

```
Medical-Image-Segmentation-For-COVID-19-By-Unet-STUDY-ONLY-/
├── src/                           # 源代码目录
│   ├── unet_model.py             # U-Net模型定义
│   ├── data_loader.py            # 数据加载器
│   ├── train_ddp.py              # 分布式训练脚本
│   ├── predict.py                # 单图像预测脚本
│   ├── gui.py                    # 图形界面
│   └── test/                     # 测试和工具
│       ├── test_unet_model.py
│       ├── test_data_loader.py
│       ├── test_training_pipeline.py
│       ├── test_model_save.py
│       └── analyze_training.py   # 训练分析可视化
├── TRAINING_USAGE.md             # 训练使用指南
├── LOGGING_REPORT.md             # 日志系统说明
├── METRICS_GUIDE.md              # 指标记录说明
├── REALTIME_MONITORING_SUMMARY.md # 实时监控说明
├── QUICK_REFERENCE.md            # 快速参考
└── README.md                     # 本文件
```

## 环境要求

```bash
Python 3.8+
PyTorch 1.12+
torchvision
numpy
Pillow
pandas
matplotlib
seaborn
```

## 使用方法

### 1. 准备数据集

将数据集放置在项目根目录：
```
data_set/
└── CT_COVID/
    ├── frames/     # CT图像
    └── masks/      # 标注掩码
```

### 2. 训练模型

```bash
# 单机多卡训练（推荐）
cd src
python train_ddp.py --epochs 200 --batch_size 4 --lr 0.001

# 查看更多参数
python train_ddp.py --help
```

### 3. 预测

```bash
# 单张图像预测
python predict.py --model_path ../checkpoints/best_model.pth --image_path test.png

# 批量预测
python predict.py --model_path ../checkpoints/best_model.pth --image_path /path/to/images/
```

### 4. 图形界面

```bash
python gui.py
```

### 5. 分析训练结果

```bash
cd src/test
python analyze_training.py
```

## 技术栈

- **深度学习框架**: PyTorch
- **模型架构**: U-Net (编码器-解码器结构)
- **损失函数**: Dice Loss + Binary Cross-Entropy
- **优化器**: Adam
- **分布式训练**: DDP (DistributedDataParallel)
- **数据增强**: 随机翻转、旋转

## 模型说明

**U-Net 架构参数：**
- 输入通道: 1 (灰度CT图像)
- 输出通道: 1 (二分类分割)
- 特征层: [64, 128, 256, 512]
- 图像尺寸: 512×512

## 训练输出

训练过程会生成以下文件：
- `checkpoints/`: 模型检查点文件
- `log/`: 训练日志文件
- `metrics/`: CSV格式的训练指标
- `results/`: 预测结果图像
- `analysis/`: 训练曲线分析图

## 测试

```bash
cd src/test

# 测试模型
python test_unet_model.py

# 测试数据加载
python test_data_loader.py

# 测试训练流程
python test_training_pipeline.py
```

## 注意事项

⚠️ **学习用途**: 本项目仅供学习研究使用
⚠️ **路径配置**: 使用相对路径，修改数据集路径请编辑 `src/data_loader.py`
⚠️ **GPU要求**: 建议使用GPU训练，CPU训练速度较慢
⚠️ **显存需求**: 建议至少4GB显存 (batch_size=4)

## 参考文档

详细使用说明请查看：
- [训练使用指南](TRAINING_USAGE.md)
- [日志系统说明](LOGGING_REPORT.md)
- [指标记录说明](METRICS_GUIDE.md)
- [快速参考](QUICK_REFERENCE.md)

## License

本项目仅供学习研究使用。

---

**Last Updated**: 2025-11-06
