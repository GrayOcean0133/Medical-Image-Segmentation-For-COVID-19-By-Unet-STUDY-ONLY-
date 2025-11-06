# 第三方库导入
import os
import torch
import torch.distributed as dist
import torch.optim as optim
import argparse
import logging
import sys
import csv
from datetime import datetime
from pathlib import Path

# torch导入
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# 项目导入
PROJECT_ROOTS = Path(__file__).parent.parent
from data_loader import COVID_UNHEALTHY_LUNGS, COVID_UNHEALTHY_LUNGS_MASK, COVIDCTDataset, get_data_transforms
from unet_model import Unet, combined_loss

# 路径常量
LOG_OUTPUT = PROJECT_ROOTS / "log"
METRICS_OUTPUT = PROJECT_ROOTS / "metrics"

# 配置日志（只配置一次，在主进程启动前）
def setup_logging():
    """配置全局日志系统"""
    # 确保日志目录存在
    LOG_OUTPUT.mkdir(exist_ok=True)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_OUTPUT / f'training_{timestamp}.log'

    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - [Rank %(rank)s] - %(message)s' if hasattr(logging, 'rank') else '%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(str(log_file), mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # 同时输出到控制台
        ]
    )
    return log_file

# 训练指标记录器
class MetricsLogger:
    """
    训练指标记录器 - 记录详细的训练数据用于复盘分析
    """
    def __init__(self, log_dir, timestamp, enable_batch_logging=False):
        """
        Args:
            log_dir: 日志目录
            timestamp: 时间戳
            enable_batch_logging: 是否记录每个batch的详细信息
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.enable_batch_logging = enable_batch_logging

        # Epoch级别的metrics文件
        self.epoch_metrics_file = self.log_dir / f"epoch_metrics_{timestamp}.csv"
        self.epoch_writer = None
        self.epoch_file = None

        # Batch级别的metrics文件（可选）
        if enable_batch_logging:
            self.batch_metrics_file = self.log_dir / f"batch_metrics_{timestamp}.csv"
            self.batch_writer = None
            self.batch_file = None

        self._init_epoch_logger()
        if enable_batch_logging:
            self._init_batch_logger()

    def _init_epoch_logger(self):
        """初始化epoch级别的CSV记录器"""
        self.epoch_file = open(self.epoch_metrics_file, 'w', newline='', encoding='utf-8')
        self.epoch_writer = csv.writer(self.epoch_file)

        # 写入表头
        headers = [
            'Epoch',
            'Avg_Loss',
            'Min_Batch_Loss',
            'Max_Batch_Loss',
            'Learning_Rate',
            'Epoch_Time_Seconds',
            'Best_Loss_So_Far',
            'Is_Best_Model',
            'Timestamp'
        ]
        self.epoch_writer.writerow(headers)
        self.epoch_file.flush()
        logging.info(f"Epoch指标记录文件: {self.epoch_metrics_file}")

    def _init_batch_logger(self):
        """初始化batch级别的CSV记录器"""
        self.batch_file = open(self.batch_metrics_file, 'w', newline='', encoding='utf-8')
        self.batch_writer = csv.writer(self.batch_file)

        # 写入表头
        headers = [
            'Epoch',
            'Batch',
            'Loss',
            'Learning_Rate',
            'Timestamp'
        ]
        self.batch_writer.writerow(headers)
        self.batch_file.flush()
        logging.info(f"Batch指标记录文件: {self.batch_metrics_file}")

    def log_epoch(self, epoch, avg_loss, min_loss, max_loss, lr, epoch_time, best_loss, is_best):
        """记录epoch级别的指标"""
        row = [
            epoch,
            f"{avg_loss:.6f}",
            f"{min_loss:.6f}",
            f"{max_loss:.6f}",
            f"{lr:.8f}",
            f"{epoch_time:.2f}",
            f"{best_loss:.6f}",
            is_best,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
        self.epoch_writer.writerow(row)
        self.epoch_file.flush()

    def log_batch(self, epoch, batch, loss, lr):
        """记录batch级别的指标"""
        if not self.enable_batch_logging:
            return

        row = [
            epoch,
            batch,
            f"{loss:.6f}",
            f"{lr:.8f}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
        self.batch_writer.writerow(row)
        # 每10个batch刷新一次文件
        if batch % 10 == 0:
            self.batch_file.flush()

    def close(self):
        """关闭文件"""
        if self.epoch_file:
            self.epoch_file.close()
        if self.batch_file:
            self.batch_file.close()

# DDP初始化
def setup(rank, world_size):
    """初始化分布式训练环境"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        if rank == 0:
            logging.info(f"成功初始化分布式进程组，world_size={world_size}")
    except Exception as e:
        logging.error(f"初始化分布式进程组失败: {str(e)}")
        raise


def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()
    logging.info("分布式进程组已清理")

def train_ddp(rank, world_size, args):
    """
    DDP分布式训练主函数
    Args:
        rank: 进程排名
        world_size: 总进程数
        args: 命令行参数
    """
    try:
        # 设置当前设备
        torch.cuda.set_device(rank)
        device_name = torch.cuda.get_device_name(rank)

        # 初始化分布式环境
        setup(rank, world_size)

        if rank == 0:
            logging.info("=" * 80)
            logging.info(f"开始训练 - 进程 {rank}")
            logging.info(f"GPU {rank}: {device_name}")
            logging.info("=" * 80)

        # 创建模型并移动到GPU
        if rank == 0:
            logging.info("正在创建U-Net模型...")

        model = Unet(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            features=args.features
        ).to(rank)

        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if rank == 0:
            logging.info(f"模型参数: 输入通道={args.in_channels}, 输出通道={args.out_channels}")
            logging.info(f"特征通道数: {args.features}")
            logging.info(f"总参数量: {total_params:,}")
            logging.info(f"可训练参数量: {trainable_params:,}")

        # 使用DDP包装模型
        model = DDP(model, device_ids=[rank])
        if rank == 0:
            logging.info("模型已用DDP包装")

        # 定义优化器
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        if rank == 0:
            logging.info(f"优化器: Adam, 学习率={args.lr}")

        # 加载数据
        if rank == 0:
            logging.info("正在加载数据集...")

        train_transform, val_transform = get_data_transforms()

        try:
            # 训练集
            train_dataset = COVIDCTDataset(
                images_dir=args.train_images_dir,
                masks_dir=args.train_masks_dir,
                transform=train_transform,
                img_size=args.img_size
            )

            if rank == 0:
                logging.info(f"训练集加载成功: {len(train_dataset)} 张图像")
                logging.info(f"图像目录: {args.train_images_dir}")
                logging.info(f"掩码目录: {args.train_masks_dir}")

        except Exception as e:
            logging.error(f"加载数据集失败: {str(e)}")
            raise

        # 分布式采样器
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

        # 数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        if rank == 0:
            logging.info(f"DataLoader配置: batch_size={args.batch_size}, num_workers={args.num_workers}")
            logging.info(f"每个epoch的batch数量: {len(train_loader)}")
            logging.info(f"每个GPU的有效batch_size: {args.batch_size}")
            logging.info(f"全局有效batch_size: {args.batch_size * world_size}")

        # 初始化最佳损失
        best_loss = float('inf')

        # 初始化metrics logger（只在主进程）
        metrics_logger = None
        if rank == 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_logger = MetricsLogger(
                log_dir=METRICS_OUTPUT,
                timestamp=timestamp,
                enable_batch_logging=args.log_batch_metrics  # 通过参数控制是否记录batch级别
            )
            logging.info("=" * 80)
            logging.info(f"开始训练: {args.epochs} 个epochs")
            logging.info("=" * 80)

        # 主训练循环
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)  # 确保每个epoch的数据打乱顺序不同

            model.train()
            epoch_loss = 0.0
            epoch_start_time = datetime.now()

            # 记录epoch内的最小和最大loss
            min_batch_loss = float('inf')
            max_batch_loss = float('-inf')

            for batch_idx, (images, masks) in enumerate(train_loader):
                # 移入GPU
                images = images.to(rank, non_blocking=True)
                masks = masks.to(rank, non_blocking=True)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                outputs = model(images)

                # 计算损失
                loss = combined_loss(outputs, masks)

                # 反向传播
                loss.backward()

                # 更新参数
                optimizer.step()

                # 记录loss
                loss_value = loss.item()
                epoch_loss += loss_value
                min_batch_loss = min(min_batch_loss, loss_value)
                max_batch_loss = max(max_batch_loss, loss_value)

                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']

                # 记录batch级别的metrics
                if rank == 0 and metrics_logger:
                    metrics_logger.log_batch(
                        epoch=epoch + 1,
                        batch=batch_idx,
                        loss=loss_value,
                        lr=current_lr
                    )

                # 每100个batch输出一次进度（或更频繁）
                if batch_idx % args.log_interval == 0 and rank == 0:
                    progress = (batch_idx / len(train_loader)) * 100
                    logging.info(
                        f'Epoch [{epoch+1}/{args.epochs}] - '
                        f'进度 {progress:.1f}% ({batch_idx}/{len(train_loader)}) - '
                        f'Batch Loss: {loss_value:.6f} - '
                        f'LR: {current_lr:.8f}'
                    )
        
            avg_loss = epoch_loss / len(train_loader)
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()

            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']

            # Epoch结束时的日志输出
            if rank == 0:
                # 计算剩余时间估计
                remaining_epochs = args.epochs - (epoch + 1)
                eta_seconds = remaining_epochs * epoch_time
                eta_hours = eta_seconds / 3600

                # 判断是否为最佳模型
                is_best = avg_loss < best_loss

                logging.info("=" * 80)
                logging.info(f'Epoch [{epoch+1}/{args.epochs}] 完成')
                logging.info(f'平均损失: {avg_loss:.6f}')
                logging.info(f'最小Batch损失: {min_batch_loss:.6f}')
                logging.info(f'最大Batch损失: {max_batch_loss:.6f}')
                logging.info(f'学习率: {current_lr:.8f}')
                logging.info(f'本epoch用时: {epoch_time:.2f}秒')
                logging.info(f'预计剩余时间: {eta_hours:.2f}小时')

                # 记录epoch级别的metrics
                if metrics_logger:
                    metrics_logger.log_epoch(
                        epoch=epoch + 1,
                        avg_loss=avg_loss,
                        min_loss=min_batch_loss,
                        max_loss=max_batch_loss,
                        lr=current_lr,
                        epoch_time=epoch_time,
                        best_loss=best_loss,
                        is_best=is_best
                    )

                # 使用绝对路径保存检查点
                checkpoint_dir = PROJECT_ROOTS / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)

                # 保存检查点（每10个epoch或最佳模型）
                if (epoch + 1) % 10 == 0 or avg_loss < best_loss:
                    try:
                        checkpoint = {
                            'epoch': epoch + 1,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': avg_loss
                        }
                        checkpoint_path = checkpoint_dir / f'model_epoch_{epoch+1}.pth'
                        torch.save(checkpoint, str(checkpoint_path))
                        logging.info(f'Success: 已保存检查点: {checkpoint_path.name}')

                        if avg_loss < best_loss:
                            improvement = best_loss - avg_loss
                            best_loss = avg_loss
                            # 保存最佳模型
                            best_model_path = checkpoint_dir / 'best_model.pth'
                            torch.save(model.module.state_dict(), str(best_model_path))
                            logging.info(f'Success: 新的最佳模型! Loss: {avg_loss:.4f} (提升: {improvement:.4f})')
                            logging.info(f'Success: 已保存最佳模型: {best_model_path.name}')
                    except Exception as e:
                        logging.error(f'ERROR: 保存模型失败: {str(e)}', exc_info=True)

                logging.info("=" * 80)

        # 训练完成
        if rank == 0:
            logging.info("=" * 80)
            logging.info("训练完成!")
            logging.info(f"最佳损失: {best_loss:.6f}")
            logging.info("=" * 80)

            # 关闭metrics logger
            if metrics_logger:
                metrics_logger.close()
                logging.info(f"训练指标已保存到: {metrics_logger.log_dir}")

    except Exception as e:
        if rank == 0:
            logging.error(f"训练过程中发生错误: {str(e)}", exc_info=True)
            # 确保metrics logger被关闭
            if 'metrics_logger' in locals() and metrics_logger:
                metrics_logger.close()
        raise
    finally:
        # 清理
        cleanup()
    
def main():
    """主函数"""
    # 配置日志系统（在所有操作之前）
    log_file = setup_logging()
    logging.info("=" * 80)
    logging.info("COVID-19肺部CT分割 - U-Net DDP训练")
    logging.info("=" * 80)
    logging.info(f"日志文件: {log_file}")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="COVID-19肺部CT分割 - U-Net DDP训练")

    # 数据参数
    parser.add_argument('--train_images_dir', type=str, required=False, default=COVID_UNHEALTHY_LUNGS,
                       help='训练图像目录路径')
    parser.add_argument('--train_masks_dir', type=str, required=False, default=COVID_UNHEALTHY_LUNGS_MASK,
                       help='训练掩码目录路径')
    parser.add_argument('--img_size', type=int, default=512,
                       help='输入图像尺寸（默认: 512）')

    # 模型参数
    parser.add_argument('--in_channels', type=int, default=1,
                       help='输入图像通道数（默认: 1，灰度图）')
    parser.add_argument('--out_channels', type=int, default=1,
                       help='输出通道数（默认: 1，二分类）')
    parser.add_argument('--features', type=list, default=[64, 128, 256, 512],
                       help='U-Net各层特征通道数（默认: [64, 128, 256, 512]）')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数（默认: 200）')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='每个GPU的批次大小（默认: 4）')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率（默认: 0.001）')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载工作进程数（默认: 4）')

    # 日志和监控参数
    parser.add_argument('--log_batch_metrics', action='store_true',
                       help='是否记录每个batch的详细指标到CSV文件（会生成大量数据）')
    parser.add_argument('--log_interval', type=int, default=1,
                       help='日志输出间隔（每N个batch输出一次，默认: 1，即每个batch都输出）')

    args = parser.parse_args()

    # 记录训练参数
    logging.info("训练配置:")
    logging.info(f"  Epochs: {args.epochs}")
    logging.info(f"  Batch Size (per GPU): {args.batch_size}")
    logging.info(f"  Learning Rate: {args.lr}")
    logging.info(f"  Num Workers: {args.num_workers}")
    logging.info(f"  Image Size: {args.img_size}x{args.img_size}")
    logging.info(f"模型配置:")
    logging.info(f"  Input Channels: {args.in_channels}")
    logging.info(f"  Output Channels: {args.out_channels}")
    logging.info(f"  Features: {args.features}")
    logging.info(f"日志配置:")
    logging.info(f"  日志输出间隔: 每{args.log_interval}个batch")
    logging.info(f"  记录Batch级别指标: {'是' if args.log_batch_metrics else '否'}")

    # 检测GPU
    if not torch.cuda.is_available():
        logging.error("CUDA不可用! 请检查GPU配置。")
        sys.exit(1)

    world_size = torch.cuda.device_count()
    logging.info(f"检测到 {world_size} 个GPU:")
    for i in range(world_size):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        logging.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")

    # 启动分布式训练
    logging.info("=" * 80)
    logging.info("启动分布式训练...")
    logging.info("=" * 80)

    try:
        torch.multiprocessing.spawn(
            train_ddp,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
        logging.info("训练成功完成!")
    except Exception as e:
        logging.error(f"训练失败: {str(e)}", exc_info=True)
        sys.exit(1)
    
if __name__ == "__main__":
    # 创建必要的目录
    try:
        checkpoint_dir = PROJECT_ROOTS / "checkpoints"
        results_dir = PROJECT_ROOTS / "results"
        log_dir = PROJECT_ROOTS / "log"
        metrics_dir = PROJECT_ROOTS / "metrics"

        checkpoint_dir.mkdir(exist_ok=True)
        results_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
        metrics_dir.mkdir(exist_ok=True)

        print(f"工作目录: {PROJECT_ROOTS}")
        print(f"检查点保存目录: {checkpoint_dir}")
        print(f"结果保存目录: {results_dir}")
        print(f"日志保存目录: {log_dir}")
        print(f"训练指标保存目录: {metrics_dir}")
        print("=" * 80)

        main()

    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"启动失败: {str(e)}")
        sys.exit(1)
    