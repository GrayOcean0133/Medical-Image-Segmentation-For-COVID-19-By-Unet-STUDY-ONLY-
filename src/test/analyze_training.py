#!/usr/bin/env python
"""
训练指标分析和可视化脚本

功能：
1. 读取训练过程中记录的CSV指标文件
2. 生成Loss曲线图
3. 分析训练趋势
4. 导出分析报告

使用方法：
    python analyze_training.py
    python analyze_training.py --metrics_file metrics/epoch_metrics_20251106_103015.csv
    python analyze_training.py --batch_metrics  # 分析batch级别的数据
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# 项目根目录 (test目录的父目录的父目录)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def find_latest_metrics_file(metrics_type='epoch'):
    """查找最新的metrics文件"""
    metrics_dir = PROJECT_ROOT / "metrics"
    if not metrics_dir.exists():
        print(f"错误: metrics目录不存在: {metrics_dir}")
        return None

    pattern = f"{metrics_type}_metrics_*.csv"
    files = list(metrics_dir.glob(pattern))

    if not files:
        print(f"错误: 未找到{metrics_type}级别的metrics文件")
        return None

    # 返回最新的文件
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return latest


def analyze_epoch_metrics(csv_file):
    """分析epoch级别的训练指标"""
    print("=" * 80)
    print(f"分析Epoch级别训练指标")
    print("=" * 80)
    print(f"数据文件: {csv_file}")

    # 读取数据
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"错误: 读取CSV文件失败: {e}")
        return

    print(f"\n数据概览:")
    print(f"  总Epoch数: {len(df)}")
    print(f"  最佳Loss: {df['Avg_Loss'].min():.6f} (Epoch {df.loc[df['Avg_Loss'].idxmin(), 'Epoch']})")
    print(f"  最终Loss: {df['Avg_Loss'].iloc[-1]:.6f}")
    print(f"  总训练时间: {df['Epoch_Time_Seconds'].sum() / 3600:.2f}小时")

    # 创建输出目录
    output_dir = PROJECT_ROOT / "analysis"
    output_dir.mkdir(exist_ok=True)

    # 1. Loss曲线图
    plt.figure(figsize=(14, 10))

    # 子图1: 平均Loss曲线
    plt.subplot(2, 2, 1)
    plt.plot(df['Epoch'], df['Avg_Loss'], 'b-', linewidth=2, label='Average Loss')
    plt.plot(df['Epoch'], df['Best_Loss_So_Far'], 'r--', linewidth=1.5, label='Best Loss So Far', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2: Loss范围图 (Min, Avg, Max)
    plt.subplot(2, 2, 2)
    plt.fill_between(df['Epoch'], df['Min_Batch_Loss'], df['Max_Batch_Loss'], alpha=0.3, label='Loss Range')
    plt.plot(df['Epoch'], df['Avg_Loss'], 'r-', linewidth=2, label='Avg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Range (Min/Avg/Max per Epoch)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图3: 学习率曲线
    plt.subplot(2, 2, 3)
    plt.plot(df['Epoch'], df['Learning_Rate'], 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 子图4: 每个Epoch的训练时间
    plt.subplot(2, 2, 4)
    plt.bar(df['Epoch'], df['Epoch_Time_Seconds'], alpha=0.7, color='orange')
    plt.axhline(y=df['Epoch_Time_Seconds'].mean(), color='r', linestyle='--',
                label=f'平均: {df["Epoch_Time_Seconds"].mean():.2f}秒')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Time per Epoch', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f'training_analysis_{csv_file.stem}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Loss曲线图已保存: {output_path}")

    # 2. 详细统计分析
    plt.figure(figsize=(12, 6))

    # Loss改善分析
    plt.subplot(1, 2, 1)
    loss_improvement = df['Avg_Loss'].diff() * -1  # 负值表示改善
    plt.plot(df['Epoch'][1:], loss_improvement[1:], 'b-', linewidth=1.5)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Improvement')
    plt.title('Loss Improvement per Epoch', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Loss标准差分析（batch间的波动）
    plt.subplot(1, 2, 2)
    batch_std = df['Max_Batch_Loss'] - df['Min_Batch_Loss']
    plt.plot(df['Epoch'], batch_std, 'purple', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Std (Max - Min)')
    plt.title('Batch Loss Stability', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f'training_stats_{csv_file.stem}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 统计分析图已保存: {output_path}")

    # 3. 生成文本报告
    report_path = output_dir / f'training_report_{csv_file.stem}.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("训练分析报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"数据文件: {csv_file}\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n\n")

        f.write("训练概览:\n")
        f.write(f"  总Epoch数: {len(df)}\n")
        f.write(f"  总训练时间: {df['Epoch_Time_Seconds'].sum() / 3600:.2f}小时\n")
        f.write(f"  平均每Epoch时间: {df['Epoch_Time_Seconds'].mean():.2f}秒\n\n")

        f.write("Loss统计:\n")
        f.write(f"  初始Loss: {df['Avg_Loss'].iloc[0]:.6f}\n")
        f.write(f"  最终Loss: {df['Avg_Loss'].iloc[-1]:.6f}\n")
        f.write(f"  最佳Loss: {df['Avg_Loss'].min():.6f} (Epoch {df.loc[df['Avg_Loss'].idxmin(), 'Epoch']})\n")
        f.write(f"  Loss改善: {(df['Avg_Loss'].iloc[0] - df['Avg_Loss'].iloc[-1]):.6f} ")
        f.write(f"({((df['Avg_Loss'].iloc[0] - df['Avg_Loss'].iloc[-1]) / df['Avg_Loss'].iloc[0] * 100):.2f}%)\n\n")

        f.write("学习率:\n")
        f.write(f"  初始学习率: {df['Learning_Rate'].iloc[0]:.8f}\n")
        f.write(f"  最终学习率: {df['Learning_Rate'].iloc[-1]:.8f}\n\n")

        f.write("Top 5最佳Epochs:\n")
        top5 = df.nsmallest(5, 'Avg_Loss')[['Epoch', 'Avg_Loss', 'Learning_Rate']]
        for idx, row in top5.iterrows():
            f.write(f"  Epoch {row['Epoch']:3d}: Loss={row['Avg_Loss']:.6f}, LR={row['Learning_Rate']:.8f}\n")

    print(f"✓ 文本报告已保存: {report_path}")
    print("\n" + "=" * 80)


def analyze_batch_metrics(csv_file):
    """分析batch级别的训练指标"""
    print("=" * 80)
    print(f"分析Batch级别训练指标")
    print("=" * 80)
    print(f"数据文件: {csv_file}")

    # 读取数据
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"错误: 读取CSV文件失败: {e}")
        return

    print(f"\n数据概览:")
    print(f"  总Batch数: {len(df)}")
    print(f"  Epoch范围: {df['Epoch'].min()} - {df['Epoch'].max()}")

    # 创建输出目录
    output_dir = PROJECT_ROOT / "analysis"
    output_dir.mkdir(exist_ok=True)

    # 绘制batch级别的loss曲线
    plt.figure(figsize=(16, 6))

    # 全部batch的loss曲线（可能很密集）
    plt.subplot(1, 2, 1)
    plt.plot(range(len(df)), df['Loss'], 'b-', linewidth=0.5, alpha=0.6)
    # 添加移动平均线
    window = 100
    if len(df) > window:
        rolling_mean = df['Loss'].rolling(window=window).mean()
        plt.plot(range(len(df)), rolling_mean, 'r-', linewidth=2,
                label=f'{window}-batch Moving Avg')
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.title('Batch-level Loss (All Training)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 按Epoch分组的box plot
    plt.subplot(1, 2, 2)
    epochs = sorted(df['Epoch'].unique())
    batch_data = [df[df['Epoch'] == epoch]['Loss'].values for epoch in epochs]
    plt.boxplot(batch_data, labels=epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Batch Loss')
    plt.title('Batch Loss Distribution per Epoch', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    output_path = output_dir / f'batch_analysis_{csv_file.stem}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Batch分析图已保存: {output_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='训练指标分析工具')
    parser.add_argument('--metrics_file', type=str, default=None,
                       help='指定要分析的metrics CSV文件路径')
    parser.add_argument('--batch_metrics', action='store_true',
                       help='分析batch级别的指标（需要启用--log_batch_metrics训练）')

    args = parser.parse_args()

    # 确定要分析的文件
    if args.metrics_file:
        metrics_file = Path(args.metrics_file)
        if not metrics_file.exists():
            print(f"错误: 文件不存在: {metrics_file}")
            sys.exit(1)
    else:
        # 自动查找最新的文件
        metrics_type = 'batch' if args.batch_metrics else 'epoch'
        metrics_file = find_latest_metrics_file(metrics_type)
        if not metrics_file:
            sys.exit(1)

    # 执行分析
    if 'batch_metrics' in metrics_file.name:
        analyze_batch_metrics(metrics_file)
    else:
        analyze_epoch_metrics(metrics_file)

    print("\n分析完成！")
    print(f"查看分析结果: {PROJECT_ROOT / 'analysis'}")


if __name__ == "__main__":
    main()
