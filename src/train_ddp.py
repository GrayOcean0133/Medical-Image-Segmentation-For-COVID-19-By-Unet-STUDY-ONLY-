# 第三方库导入
import os
import sys
import csv
import random
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim

# torch导入
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# 项目导入
PROJECT_ROOTS = Path(__file__).parent.parent
from data_loader import (COVID_UNHEALTHY_LUNGS, COVID_UNHEALTHY_LUNGS_MASK,
                         COVIDCTDataset, get_data_transforms, split_train_val)
from unet_model import (DEFAULT_FEATURES, Unet, combined_loss,
                        compute_pos_weight)
from checkpoint import save_checkpoint

# 路径常量
LOG_OUTPUT = PROJECT_ROOTS / "log"
METRICS_OUTPUT = PROJECT_ROOTS / "metrics"
CHECKPOINT_OUTPUT = PROJECT_ROOTS / "checkpoints"


def set_seed(seed):
    """固定随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 当前进程的 rank，仅用于日志前缀
_CURRENT_RANK = ['main']


class _RankFormatter(logging.Formatter):
    """在日志行里带上 rank，方便区分 DDP 各进程的输出。"""

    def format(self, record):
        if not hasattr(record, 'rank'):
            record.rank = _CURRENT_RANK[0]
        return super().format(record)


def set_log_rank(rank):
    _CURRENT_RANK[0] = rank


def setup_logging(log_file=None):
    """配置全局日志系统。

    spawn 出来的子进程是全新解释器，不会继承父进程的 logging 配置，
    因此每个 rank 进入训练函数后都要再调用一次（传入父进程已确定的 log_file，
    这样所有 rank 都往同一个文件里追加）。
    """
    LOG_OUTPUT.mkdir(exist_ok=True, parents=True)

    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = LOG_OUTPUT / f'training_{timestamp}.log'
    log_file = Path(log_file)

    root = logging.getLogger()
    # 反复调用本函数不能累积 handler，否则每条日志会重复 world_size 次
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.INFO)

    formatter = _RankFormatter(
        '%(asctime)s - [%(levelname)s] - [rank %(rank)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    file_handler = logging.FileHandler(str(log_file), mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    return log_file


# 训练指标记录器
class MetricsLogger:
    """
    训练指标记录器 - 记录详细的训练数据用于复盘分析
    """

    EPOCH_HEADERS = [
        'Epoch', 'Train_Loss', 'Min_Batch_Loss', 'Max_Batch_Loss',
        'Val_Loss', 'Val_Dice', 'Val_IoU',
        'Learning_Rate', 'Epoch_Time_Seconds', 'Best_Dice_So_Far',
        'Is_Best_Model', 'Timestamp',
    ]
    BATCH_HEADERS = ['Epoch', 'Batch', 'Loss', 'Learning_Rate', 'Timestamp']

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
        self.epoch_writer.writerow(self.EPOCH_HEADERS)
        self.epoch_file.flush()
        logging.info(f"Epoch指标记录文件: {self.epoch_metrics_file}")

    def _init_batch_logger(self):
        """初始化batch级别的CSV记录器"""
        self.batch_file = open(self.batch_metrics_file, 'w', newline='', encoding='utf-8')
        self.batch_writer = csv.writer(self.batch_file)
        self.batch_writer.writerow(self.BATCH_HEADERS)
        self.batch_file.flush()
        logging.info(f"Batch指标记录文件: {self.batch_metrics_file}")

    def log_epoch(self, epoch, train_loss, min_loss, max_loss, lr, epoch_time,
                  best_dice, is_best, val_loss=None, val_dice=None, val_iou=None):
        """记录epoch级别的指标。验证集未启用时 val_* 记为 -1。"""
        row = [
            epoch,
            f"{train_loss:.6f}",
            f"{min_loss:.6f}",
            f"{max_loss:.6f}",
            f"{val_loss:.6f}" if val_loss is not None else "-1",
            f"{val_dice:.6f}" if val_dice is not None else "-1",
            f"{val_iou:.6f}" if val_iou is not None else "-1",
            f"{lr:.8f}",
            f"{epoch_time:.2f}",
            f"{best_dice:.6f}",
            is_best,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        ]
        self.epoch_writer.writerow(row)
        self.epoch_file.flush()

    def log_batch(self, epoch, batch, loss, lr):
        """记录batch级别的指标"""
        if not self.enable_batch_logging:
            return

        row = [epoch, batch, f"{loss:.6f}", f"{lr:.8f}",
               datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
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
    if dist.is_initialized():
        dist.destroy_process_group()
        logging.info("分布式进程组已清理")


def barrier(world_size):
    """非单进程时同步所有 rank。val 只在 rank0 跑，因此必须在前后加栅栏，
    否则其余 rank 会提前进入下一个 epoch 的梯度同步，造成挂起。"""
    if world_size > 1 and dist.is_initialized():
        dist.barrier()


def build_datasets(args):
    """构造训练/验证数据集。返回 (train_dataset, val_dataset, val_is_split)"""
    train_transform, val_transform = get_data_transforms()

    # 显式给了验证集目录，就直接用
    if args.val_images_dir and args.val_masks_dir:
        train_dataset = COVIDCTDataset(
            images_dir=args.train_images_dir, masks_dir=args.train_masks_dir,
            transform=train_transform, img_size=args.img_size)
        val_dataset = COVIDCTDataset(
            images_dir=args.val_images_dir, masks_dir=args.val_masks_dir,
            transform=val_transform, img_size=args.img_size)
        return train_dataset, val_dataset, False

    # 否则从训练集里切一部分出来做验证
    full_train = COVIDCTDataset(
        images_dir=args.train_images_dir, masks_dir=args.train_masks_dir,
        transform=train_transform, img_size=args.img_size)
    if args.val_split <= 0:
        return full_train, None, False

    # 先切分（Subset 共享底层数据），再单独用无增强的副本包一层验证集
    train_subset, val_subset = split_train_val(full_train, args.val_split, seed=args.seed)

    val_source = COVIDCTDataset(
        images_dir=args.train_images_dir, masks_dir=args.train_masks_dir,
        transform=val_transform, img_size=args.img_size)
    val_dataset = torch.utils.data.Subset(val_source, val_subset.indices)
    return train_subset, val_dataset, True


@torch.no_grad()
def evaluate(model, loader, device, alpha, class_balance):
    """在验证集上评估，返回 (val_loss, dice, iou)。

    Dice / IoU 在整个验证集上累计一次性计算，而不是逐图平均，
    避免小病灶样本被大病灶样本稀释。
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    smooth = 1e-6
    intersection = pred_sum = target_sum = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        pos_weight = compute_pos_weight(masks) if class_balance else None
        loss = combined_loss(logits, masks, alpha=alpha, pos_weight=pos_weight)
        total_loss += loss.item()
        n_batches += 1

        pred = (torch.sigmoid(logits) > 0.5).float().contiguous().view(-1)
        target = masks.contiguous().view(-1)
        intersection += (pred * target).sum().item()
        pred_sum += pred.sum().item()
        target_sum += target.sum().item()

    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    union = pred_sum + target_sum - intersection
    iou = (intersection + smooth) / (union + smooth)

    model.train()
    return total_loss / max(n_batches, 1), dice, iou


def train_ddp(rank, world_size, args):
    """
    DDP分布式训练主函数
    """
    metrics_logger = None
    try:
        # 子进程要自己重新配置日志（spawn 不继承父进程的 logging 配置）
        setup_logging(args.log_file)
        set_log_rank(rank)

        set_seed(args.seed + rank)

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

        config = {
            "in_channels": args.in_channels,
            "out_channels": args.out_channels,
            "features": list(args.features),
        }
        model = Unet(**config).to(rank)

        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if rank == 0:
            logging.info(f"模型参数: 输入通道={args.in_channels}, 输出通道={args.out_channels}")
            logging.info(f"特征通道数: {config['features']}")
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

        # 学习率调度器
        scheduler = None
        if args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            if rank == 0:
                logging.info(f"学习率调度器: CosineAnnealingLR, T_max={args.epochs}")

        # AMP 混合精度
        scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
        if rank == 0:
            logging.info(f"混合精度(AMP): {'开启' if args.amp else '关闭'}")

        # 加载数据
        if rank == 0:
            logging.info("正在加载数据集...")

        train_dataset, val_dataset, val_is_split = build_datasets(args)

        if rank == 0:
            logging.info(f"训练集: {len(train_dataset)} 张图像")
            logging.info(f"图像目录: {args.train_images_dir}")
            logging.info(f"掩码目录: {args.train_masks_dir}")
            if val_dataset is not None:
                src = f"由训练集切分 (val_split={args.val_split})" if val_is_split else "独立目录"
                logging.info(f"验证集: {len(val_dataset)} 张图像 ({src})")

        # 分布式采样器
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
        )

        # 数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
        )

        # 验证集只在 rank0 上跑，不需要分布式采样器
        val_loader = None
        if val_dataset is not None and rank == 0:
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )

        if rank == 0:
            logging.info(f"DataLoader配置: batch_size={args.batch_size}, num_workers={args.num_workers}")
            logging.info(f"每个epoch的batch数量: {len(train_loader)}")
            logging.info(f"每个GPU的有效batch_size: {args.batch_size}")
            logging.info(f"全局有效batch_size: {args.batch_size * world_size}")

        # 初始化最佳指标
        best_dice = -1.0
        best_val_loss = float('inf')

        # 初始化metrics logger（只在主进程）
        if rank == 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_logger = MetricsLogger(
                log_dir=METRICS_OUTPUT, timestamp=timestamp,
                enable_batch_logging=args.log_batch_metrics,
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
                optimizer.zero_grad(set_to_none=True)

                # 前向传播 + 损失（AMP 下 autocast 负责降精度）
                with torch.amp.autocast('cuda', enabled=args.amp):
                    outputs = model(images)
                    pos_weight = compute_pos_weight(masks) if args.class_balance else None
                    loss = combined_loss(outputs, masks, alpha=args.alpha, pos_weight=pos_weight)

                # 反向传播（scaler 在未启用 AMP 时是空操作）
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

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
                        epoch=epoch + 1, batch=batch_idx,
                        loss=loss_value, lr=current_lr,
                    )

                # 按间隔输出进度
                if rank == 0 and args.log_interval > 0 and batch_idx % args.log_interval == 0:
                    logging.info(
                        f"Epoch [{epoch+1}/{args.epochs}] "
                        f"Batch [{batch_idx}/{len(train_loader)}] Loss: {loss_value:.6f}"
                    )

            # 学习率调度
            if scheduler is not None:
                scheduler.step()

            avg_loss = epoch_loss / max(len(train_loader), 1)
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()

            # 验证（只在 rank0），前后加栅栏同步其它 rank
            barrier(world_size)
            val_loss = val_dice = val_iou = None
            if val_loader is not None:
                val_loss, val_dice, val_iou = evaluate(
                    model, val_loader, rank, args.alpha, args.class_balance
                )
            barrier(world_size)

            # Epoch结束时的日志输出
            if rank == 0:
                # 判断并立即更新最优指标，保证日志/CSV 里记的是含本 epoch 的最优值
                if val_dice is not None:
                    is_best = val_dice > best_dice
                    if is_best:
                        best_dice = val_dice
                    score_desc = f"验证Dice {val_dice:.4f}"
                    best_desc = f"最佳验证Dice {best_dice:.4f}"
                else:
                    is_best = avg_loss < best_val_loss
                    if is_best:
                        best_val_loss = avg_loss
                    score_desc = f"训练损失 {avg_loss:.4f}"
                    best_desc = f"最佳训练损失 {best_val_loss:.6f}"

                # 计算剩余时间估计
                remaining_epochs = args.epochs - (epoch + 1)
                eta_hours = remaining_epochs * epoch_time / 3600

                logging.info("=" * 80)
                logging.info(f'Epoch [{epoch+1}/{args.epochs}] 完成')
                logging.info(f'平均损失: {avg_loss:.6f}')
                logging.info(f'最小Batch损失: {min_batch_loss:.6f}')
                logging.info(f'最大Batch损失: {max_batch_loss:.6f}')
                if val_dice is not None:
                    logging.info(f'验证损失: {val_loss:.6f}')
                    logging.info(f'验证Dice: {val_dice:.4f}  验证IoU: {val_iou:.4f}')
                logging.info(f'学习率: {current_lr:.8f}')
                logging.info(f'本epoch用时: {epoch_time:.2f}秒')
                logging.info(f'预计剩余时间: {eta_hours:.2f}小时')
                logging.info(f'{best_desc}')

                # 记录epoch级别的metrics
                if metrics_logger:
                    metrics_logger.log_epoch(
                        epoch=epoch + 1, train_loss=avg_loss,
                        min_loss=min_batch_loss, max_loss=max_batch_loss,
                        lr=current_lr, epoch_time=epoch_time,
                        best_dice=max(best_dice, 0.0), is_best=is_best,
                        val_loss=val_loss, val_dice=val_dice, val_iou=val_iou,
                    )

                checkpoint_metrics = {
                    'train_loss': avg_loss, 'val_loss': val_loss,
                    'val_dice': val_dice, 'val_iou': val_iou,
                }

                try:
                    # 保存最佳模型（含结构配置，推理端才能正确重建网络）
                    if is_best:
                        save_checkpoint(
                            CHECKPOINT_OUTPUT / 'best_model.pth',
                            model.module, config=config, optimizer=optimizer,
                            epoch=epoch + 1, metrics=checkpoint_metrics,
                        )
                        logging.info(f'Success: 新的最佳模型! {score_desc}')

                    # 定期保存检查点
                    if (epoch + 1) % args.save_every == 0:
                        save_checkpoint(
                            CHECKPOINT_OUTPUT / f'model_epoch_{epoch+1}.pth',
                            model.module, config=config, optimizer=optimizer,
                            epoch=epoch + 1, metrics=checkpoint_metrics,
                        )
                        logging.info(f'Success: 已保存检查点: model_epoch_{epoch+1}.pth')
                except Exception as e:
                    logging.error(f'ERROR: 保存模型失败: {str(e)}', exc_info=True)

                logging.info("=" * 80)

        # 训练完成
        if rank == 0:
            logging.info("=" * 80)
            logging.info("训练完成!")
            if best_dice >= 0:
                logging.info(f"最佳验证Dice: {best_dice:.6f}")
            else:
                logging.info(f"最佳训练损失: {best_val_loss:.6f}")
            if metrics_logger:
                metrics_logger.close()
                logging.info(f"训练指标已保存到: {metrics_logger.log_dir}")
            logging.info("=" * 80)

    except Exception as e:
        if rank == 0:
            logging.error(f"训练过程中发生错误: {str(e)}", exc_info=True)
            if metrics_logger:
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
    parser.add_argument('--train_images_dir', type=str, default=str(COVID_UNHEALTHY_LUNGS),
                        help='训练图像目录路径')
    parser.add_argument('--train_masks_dir', type=str, default=str(COVID_UNHEALTHY_LUNGS_MASK),
                        help='训练掩码目录路径')
    parser.add_argument('--val_images_dir', type=str, default=None,
                        help='验证图像目录（不填则从训练集按 --val_split 切分）')
    parser.add_argument('--val_masks_dir', type=str, default=None,
                        help='验证掩码目录（不填则从训练集按 --val_split 切分）')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='从训练集划出多少比例做验证（默认: 0.15，设 0 关闭验证）')
    parser.add_argument('--img_size', type=int, default=512,
                        help='输入图像尺寸（默认: 512）')

    # 模型参数
    parser.add_argument('--in_channels', type=int, default=1,
                        help='输入图像通道数（默认: 1，灰度图）')
    parser.add_argument('--out_channels', type=int, default=1,
                        help='输出通道数（默认: 1，二分类）')
    parser.add_argument('--features', type=int, nargs='+', default=list(DEFAULT_FEATURES),
                        help=f'U-Net各层特征通道数（默认: {" ".join(map(str, DEFAULT_FEATURES))}）')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数（默认: 200）')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='每个GPU的批次大小（默认: 4）')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率（默认: 0.001）')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='组合损失中 Dice 的权重（默认: 0.5）')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作进程数（默认: 4）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    parser.add_argument('--scheduler', choices=['none', 'cosine'], default='none',
                        help='学习率调度器（默认: none）')
    parser.add_argument('--amp', action='store_true',
                        help='启用混合精度训练（需要支持 FP16 的 GPU）')
    parser.add_argument('--class_balance', action=argparse.BooleanOptionalAction, default=True,
                        help='按负/正样本比例动态设置 BCE pos_weight（默认: 开启）')
    parser.add_argument('--save_every', type=int, default=10,
                        help='每N个epoch保存一次检查点（默认: 10）')

    # 日志和监控参数
    parser.add_argument('--log_batch_metrics', action='store_true',
                        help='是否记录每个batch的详细指标到CSV文件（会生成大量数据）')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='日志输出间隔（每N个batch输出一次，设为0关闭，默认: 50）')
    # 内部使用：由 setup_logging() 填写，传递给子进程，保证所有 rank 写同一个日志文件
    parser.add_argument('--log_file', type=str, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()
    args.log_file = str(log_file)

    # 记录训练参数
    logging.info("训练配置:")
    logging.info(f"  Epochs: {args.epochs}")
    logging.info(f"  Batch Size (per GPU): {args.batch_size}")
    logging.info(f"  Learning Rate: {args.lr}")
    logging.info(f"  Num Workers: {args.num_workers}")
    logging.info(f"  Image Size: {args.img_size}x{args.img_size}")
    logging.info(f"  Seed: {args.seed}")
    logging.info(f"  验证集比例: {args.val_split}")
    logging.info(f"  混合精度(AMP): {args.amp}")
    logging.info(f"  类别平衡(pos_weight): {args.class_balance}")
    logging.info(f"模型配置:")
    logging.info(f"  Input Channels: {args.in_channels}")
    logging.info(f"  Output Channels: {args.out_channels}")
    logging.info(f"  Features: {args.features}")
    logging.info(f"日志配置:")
    logging.info(f"  日志输出间隔: 每{args.log_interval}个batch")
    logging.info(f"  记录Batch级别指标: {'是' if args.log_batch_metrics else '否'}")

    # 校验参数
    if bool(args.val_images_dir) != bool(args.val_masks_dir):
        logging.error("--val_images_dir 与 --val_masks_dir 必须同时提供。")
        sys.exit(1)

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
            train_ddp, args=(world_size, args), nprocs=world_size, join=True,
        )
        logging.info("训练成功完成!")
    except Exception as e:
        logging.error(f"训练失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # 创建必要的目录
    try:
        for directory in (CHECKPOINT_OUTPUT, PROJECT_ROOTS / "results", LOG_OUTPUT, METRICS_OUTPUT):
            directory.mkdir(exist_ok=True, parents=True)

        print(f"工作目录: {PROJECT_ROOTS}")
        print(f"检查点保存目录: {CHECKPOINT_OUTPUT}")
        print(f"结果保存目录: {PROJECT_ROOTS / 'results'}")
        print(f"日志保存目录: {LOG_OUTPUT}")
        print(f"训练指标保存目录: {METRICS_OUTPUT}")
        print("=" * 80)

        main()

    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"启动失败: {str(e)}")
        sys.exit(1)
