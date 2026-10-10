import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from unet_model import predict_proba
from checkpoint import load_checkpoint, build_model
from data_loader import PROJECT_ROOTS

# 配置日志系统
def setup_logging():
    """配置预测日志系统"""
    log_dir = PROJECT_ROOTS / "log"
    log_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'prediction_{timestamp}.log'

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(str(log_file), mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    return log_file


def load_image_tensor(image_path, img_size=512):
    """读图 -> 灰度 -> 统一尺寸 -> 归一化，返回 (1, 1, H, W) 张量。"""
    image = Image.open(image_path).convert("L")

    if image.size != (img_size, img_size):
        image = image.resize((img_size, img_size), Image.BILINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transform(image).unsqueeze(0)


def predict_single_image(model, image_path, device, img_size=512):
    """返回 [0, 1] 的概率图 numpy 数组，形状 (H, W)。

    注意：这里返回概率而不是硬分割，阈值交给调用方，
    这样 GUI 的阈值滑块才真正起作用。
    """
    image_tensor = load_image_tensor(image_path, img_size).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probability = predict_proba(logits)

    return probability.squeeze().cpu().numpy()


def evaluate_pair(pred_mask, gt_mask, smooth=1e-6):
    """对单个样本算 Dice / IoU，返回 (dice, iou)。"""
    pred = pred_mask.astype(bool).reshape(-1)
    gt = gt_mask.astype(bool).reshape(-1)

    intersection = np.logical_and(pred, gt).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)
    union = np.logical_or(pred, gt).sum()
    iou = (intersection + smooth) / (union + smooth)
    return float(dice), float(iou)


def main():
    """主预测函数"""
    # 配置日志
    log_file = setup_logging()

    logging.info("=" * 80)
    logging.info("COVID-19肺部CT分割 - 模型预测")
    logging.info("=" * 80)
    logging.info(f"日志文件: {log_file}")

    # 解析参数
    parser = argparse.ArgumentParser(description="COVID-19肺部CT分割预测")

    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型路径（结构配置从检查点里读）')
    parser.add_argument('--image_path', type=str, required=True,
                        help='待预测的图像或目录路径')
    parser.add_argument('--mask_path', type=str, default=None,
                        help='可选：与图像同名的真值掩码目录，给了就计算 Dice / IoU')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='输出目录（默认: results）')
    parser.add_argument('--img_size', type=int, default=512,
                        help='图像尺寸（默认: 512）')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='二值化阈值（默认: 0.5）')

    args = parser.parse_args()

    # 检查模型文件
    if not os.path.exists(args.model_path):
        logging.error(f"模型文件不存在: {args.model_path}")
        sys.exit(1)

    logging.info(f"模型路径: {args.model_path}")
    logging.info(f"图像尺寸: {args.img_size}x{args.img_size}")
    logging.info(f"二值化阈值: {args.threshold}")

    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logging.info(f"使用设备: {device} ({torch.cuda.get_device_name(0)})")
    else:
        logging.info(f"使用设备: {device} (CPU)")

    # 加载模型（按检查点里保存的结构配置重建网络）
    try:
        logging.info("正在加载模型...")
        ckpt = load_checkpoint(args.model_path, map_location=device)
        model = build_model(ckpt['config'], device=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        logging.info(f"模型结构: {ckpt['config']}")
        if ckpt.get('epoch'):
            logging.info(f"检查点 epoch: {ckpt['epoch']}")
        if ckpt.get('metrics'):
            logging.info(f"检查点指标: {ckpt['metrics']}")

        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"模型加载成功，参数量: {total_params:,}")
    except Exception as e:
        logging.error(f"模型加载失败: {str(e)}", exc_info=True)
        sys.exit(1)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"输出目录: {output_dir}")

    # 获取图像列表
    try:
        if os.path.isfile(args.image_path):
            image_paths = [args.image_path]
            logging.info(f"输入: 单张图像 - {args.image_path}")
        else:
            if not os.path.exists(args.image_path):
                logging.error(f"输入路径不存在: {args.image_path}")
                sys.exit(1)

            image_paths = sorted(
                os.path.join(args.image_path, f)
                for f in os.listdir(args.image_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            )
            logging.info(f"输入: 目录 - {args.image_path}")
            logging.info(f"找到 {len(image_paths)} 张图像")

        if len(image_paths) == 0:
            logging.warning("未找到图像文件")
            sys.exit(0)

    except Exception as e:
        logging.error(f"读取图像路径失败: {str(e)}", exc_info=True)
        sys.exit(1)

    # 开始预测
    logging.info("=" * 80)
    logging.info("开始预测...")
    logging.info("=" * 80)

    success_count = 0
    fail_count = 0
    dice_scores = []
    iou_scores = []

    for idx, image_path in enumerate(image_paths, 1):
        try:
            logging.info(f"[{idx}/{len(image_paths)}] 处理: {os.path.basename(image_path)}")

            # 预测（返回概率图）
            probability = predict_single_image(model, image_path, device, args.img_size)
            pred_mask = (probability > args.threshold).astype(np.uint8)

            # 保存二值掩码
            base_name = os.path.basename(image_path)
            output_path = output_dir / f"pred_{base_name}"
            Image.fromarray(pred_mask * 255).save(str(output_path))

            message = f"  ✓ 已保存: {output_path.name}"

            # 有真值就顺便算指标
            if args.mask_path:
                mask_file = os.path.join(args.mask_path, base_name)
                if os.path.exists(mask_file):
                    gt = np.array(Image.open(mask_file).convert("L")) > 127
                    if gt.shape != pred_mask.shape:
                        gt = np.array(
                            Image.open(mask_file).convert("L").resize(
                                (args.img_size, args.img_size), Image.NEAREST)
                        ) > 127
                    dice, iou = evaluate_pair(pred_mask, gt)
                    dice_scores.append(dice)
                    iou_scores.append(iou)
                    message += f"  Dice={dice:.4f} IoU={iou:.4f}"
                else:
                    message += f"  (找不到真值掩码 {base_name}，跳过指标)"

            logging.info(message)
            success_count += 1

        except Exception as e:
            logging.error(f"  ✗ 预测失败: {str(e)}")
            fail_count += 1

    # 总结
    logging.info("=" * 80)
    logging.info("预测完成!")
    logging.info(f"成功: {success_count} 张")
    logging.info(f"失败: {fail_count} 张")
    logging.info(f"总计: {len(image_paths)} 张")

    if dice_scores:
        # 整批累计计算，与训练时的评估口径一致
        logging.info(f"平均 Dice: {np.mean(dice_scores):.4f} ({len(dice_scores)} 张)")
        logging.info(f"平均 IoU : {np.mean(iou_scores):.4f}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
