import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import os
import logging
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image
from unet_model import Unet
from data_loader import PROJECT_ROOTS, COVID_TRAIN_IMG, COVID_UNHEALTHY_LUNGS

# 配置日志系统
def setup_logging():
    """配置预测日志系统"""
    log_dir = PROJECT_ROOTS / "log"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'prediction_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(str(log_file), mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def predict_single_image(model, image_path, device, img_size=512):

    # 加载图像
    image = Image.open(image_path).convert("L")

    # 调整尺寸
    if image.size != (img_size, img_size):
        image = image.resize((img_size, img_size), Image.BILINEAR)

    # 图像预处理（与训练时相同）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],std=[0.5]),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)  # 增加批次维度(?这里升维)
    
    # 预测
    model.eval()
    
    with torch.no_grad():
        output = model(image_tensor)
        #这里使用sigmoid函数（到时会看看再改，sigmoid感觉不是很好）
        prediction = (output > 0.5).float()  # 可更改项: 阈值
    
    prediction = prediction.squeeze().cpu().numpy()
        
    return prediction

# 主预测函数
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
                       help='训练好的模型路径')
    parser.add_argument('--image_path', type=str, required=True,
                       help='待预测的图像或目录路径')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='输出目录（默认: results）')
    parser.add_argument('--img_size', type=int, default=512,
                       help='图像尺寸（默认: 512）')

    args = parser.parse_args()

    # 检查模型文件
    if not os.path.exists(args.model_path):
        logging.error(f"模型文件不存在: {args.model_path}")
        sys.exit(1)

    logging.info(f"模型路径: {args.model_path}")
    logging.info(f"图像尺寸: {args.img_size}x{args.img_size}")

    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"使用设备: {device} ({gpu_name})")
    else:
        logging.info(f"使用设备: {device} (CPU)")

    # 加载模型
    try:
        logging.info("正在加载模型...")
        model = Unet(in_channels=1, out_channels=1)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()

        # 计算参数量
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

            image_paths = [os.path.join(args.image_path, f)
                        for f in os.listdir(args.image_path)
                        if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

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

    for idx, image_path in enumerate(image_paths, 1):
        try:
            logging.info(f"[{idx}/{len(image_paths)}] 处理: {os.path.basename(image_path)}")

            # 预测
            prediction = predict_single_image(model, image_path, device, args.img_size)

            # 保存结果
            prediction_img = Image.fromarray((prediction * 255).astype(np.uint8))

            base_name = os.path.basename(image_path)
            output_path = output_dir / f"pred_{base_name}"

            prediction_img.save(str(output_path))
            logging.info(f"  ✓ 已保存: {output_path.name}")
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
    logging.info("=" * 80)
        
if __name__ == "__main__":
    main()
    
                