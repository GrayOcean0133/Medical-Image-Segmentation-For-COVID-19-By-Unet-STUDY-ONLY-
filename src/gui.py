import torch
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import logging
from datetime import datetime
from pathlib import Path
from unet_model import Unet
from predict import predict_single_image

# tkinter标准库导入
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 配置GUI日志系统
def setup_gui_logging():
    """配置GUI日志系统"""
    from data_loader import PROJECT_ROOTS
    log_dir = PROJECT_ROOTS / "log"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'gui_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(str(log_file), mode='a', encoding='utf-8')
        ]
    )
    logging.info("=" * 80)
    logging.info("COVID-19肺部CT分割 - GUI界面启动")
    logging.info("=" * 80)
    return log_file


class COVIDSegmentationGUI:
    """
    COVID-19肺部CT分割图形界面
    使用tkinter实现（Python标准库，无需额外安装）
    """

    def __init__(self, root, model_path, img_size=512):
        self.root = root
        self.model_path = model_path
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_image_path = None
        self.current_prediction = None

        # 加载模型
        self.model = self.load_model()

        # 设置窗口
        self.root.title('COVID-19肺部CT分割系统')
        self.root.geometry('900x700')
        self.root.resizable(True, True)

        # 初始化UI
        self.init_ui()

    def load_model(self):
        """加载训练好的模型"""
        try:
            logging.info(f"开始加载模型: {self.model_path}")
            model = Unet(in_channels=1, out_channels=1)
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()

            total_params = sum(p.numel() for p in model.parameters())
            logging.info(f"模型加载成功，参数量: {total_params:,}")
            logging.info(f"使用设备: {self.device}")
            return model
        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            logging.error(error_msg, exc_info=True)
            messagebox.showerror("错误", error_msg)
            return None

    def init_ui(self):
        """初始化用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # 1. 文件选择区域
        self.create_file_selection_area(main_frame)

        # 2. 图像显示区域
        self.create_image_display_area(main_frame)

        # 3. 参数设置区域
        self.create_settings_area(main_frame)

        # 4. 状态栏
        self.create_status_bar(main_frame)

    def create_file_selection_area(self, parent):
        """创建文件选择区域"""
        file_frame = ttk.LabelFrame(parent, text="文件操作", padding="10")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        file_frame.columnconfigure(1, weight=1)

        # 文件路径显示
        ttk.Label(file_frame, text="当前文件:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.file_path_var = tk.StringVar(value="未选择文件")
        file_label = ttk.Label(file_frame, textvariable=self.file_path_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        file_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)

        # 按钮区域
        button_frame = ttk.Frame(file_frame)
        button_frame.grid(row=0, column=2, padx=5)

        self.load_btn = ttk.Button(button_frame, text="加载图像", command=self.load_image_dialog)
        self.load_btn.pack(side=tk.LEFT, padx=2)

        self.segment_btn = ttk.Button(button_frame, text="分割", command=self.segment_image, state=tk.DISABLED)
        self.segment_btn.pack(side=tk.LEFT, padx=2)

        self.save_btn = ttk.Button(button_frame, text="保存结果", command=self.save_result, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=2)

        self.clear_btn = ttk.Button(button_frame, text="清除", command=self.clear_display)
        self.clear_btn.pack(side=tk.LEFT, padx=2)

    def create_image_display_area(self, parent):
        """创建图像显示区域"""
        image_frame = ttk.LabelFrame(parent, text="图像显示", padding="10")
        image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        # 原始图像
        original_frame = ttk.Frame(image_frame)
        original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(1, weight=1)

        ttk.Label(original_frame, text="原始CT图像", font=('Arial', 10, 'bold')).grid(row=0, column=0)

        self.original_canvas = tk.Canvas(original_frame, width=350, height=350,
                                        bg='gray85', relief=tk.SUNKEN, borderwidth=2)
        self.original_canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.original_canvas.create_text(175, 175, text="原始图像将显示在这里",
                                        fill="gray50", font=('Arial', 10))

        # 分割结果
        segmented_frame = ttk.Frame(image_frame)
        segmented_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        segmented_frame.columnconfigure(0, weight=1)
        segmented_frame.rowconfigure(1, weight=1)

        ttk.Label(segmented_frame, text="分割结果", font=('Arial', 10, 'bold')).grid(row=0, column=0)

        self.segmented_canvas = tk.Canvas(segmented_frame, width=350, height=350,
                                         bg='gray85', relief=tk.SUNKEN, borderwidth=2)
        self.segmented_canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.segmented_canvas.create_text(175, 175, text="分割结果将显示在这里",
                                         fill="gray50", font=('Arial', 10))

    def create_settings_area(self, parent):
        """创建参数设置区域"""
        settings_frame = ttk.LabelFrame(parent, text="模型参数设置", padding="10")
        settings_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        settings_frame.columnconfigure(1, weight=1)

        # 设备信息
        info_text = f"计算设备: {self.device}  |  图像尺寸: {self.img_size}×{self.img_size}"
        ttk.Label(settings_frame, text=info_text).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)

        # 阈值滑块
        ttk.Label(settings_frame, text="置信度阈值:").grid(row=1, column=0, sticky=tk.W, padx=5)

        self.threshold_var = tk.DoubleVar(value=0.5)
        self.threshold_slider = ttk.Scale(settings_frame, from_=0.1, to=0.9,
                                         variable=self.threshold_var,
                                         orient=tk.HORIZONTAL,
                                         command=self.on_threshold_changed)
        self.threshold_slider.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)

        self.threshold_label = ttk.Label(settings_frame, text="0.50")
        self.threshold_label.grid(row=1, column=2, sticky=tk.W, padx=5)

    def create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)

        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X)

    def on_threshold_changed(self, value):
        """阈值滑块变化事件"""
        threshold = float(value)
        self.threshold_label.config(text=f"{threshold:.2f}")

    def load_image_dialog(self):
        """打开文件对话框加载图像"""
        file_path = filedialog.askopenfilename(
            title="选择CT图像",
            filetypes=[
                ("PNG文件", "*.png"),
                ("JPEG文件", "*.jpg *.jpeg"),
                ("所有图像文件", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        """加载并显示图像"""
        if not file_path or not os.path.exists(file_path):
            logging.warning(f"图像文件不存在: {file_path}")
            messagebox.showwarning("警告", "文件不存在!")
            return

        try:
            logging.info(f"加载图像: {os.path.basename(file_path)}")

            # 加载图像
            image = Image.open(file_path).convert('L')
            logging.info(f"  原始尺寸: {image.size}")

            # 调整显示尺寸
            display_image = image.copy()
            display_image.thumbnail((350, 350), Image.Resampling.LANCZOS)

            # 转换为PhotoImage
            self.original_photo = ImageTk.PhotoImage(display_image)

            # 更新画布
            self.original_canvas.delete("all")
            self.original_canvas.create_image(175, 175, image=self.original_photo)

            # 更新文件路径显示
            self.file_path_var.set(os.path.basename(file_path))
            self.current_image_path = file_path

            # 启用分割按钮
            self.segment_btn.config(state=tk.NORMAL)

            self.status_var.set(f"已加载图像: {os.path.basename(file_path)}")
            logging.info(f"图像加载成功")

        except Exception as e:
            error_msg = f"加载图像失败: {str(e)}"
            logging.error(error_msg, exc_info=True)
            messagebox.showerror("错误", error_msg)

    def segment_image(self):
        """对当前图像进行分割"""
        if not self.current_image_path or not self.model:
            messagebox.showwarning("警告", "请先加载图像!")
            return

        # 获取当前阈值
        threshold = self.threshold_var.get()
        logging.info(f"开始分割图像: {os.path.basename(self.current_image_path)}")
        logging.info(f"  置信度阈值: {threshold:.2f}")

        # 禁用按钮，防止重复操作
        self.segment_btn.config(state=tk.DISABLED)
        self.status_var.set("正在进行分割...")
        self.root.update()

        # 在新线程中执行分割，避免界面卡顿
        def segment_thread():
            try:
                start_time = datetime.now()

                # 进行预测
                prediction = predict_single_image(
                    self.model, self.current_image_path,
                    self.device, self.img_size
                )

                # 应用阈值
                prediction_binary = (prediction > threshold).astype(np.uint8) * 255
                self.current_prediction = prediction_binary

                elapsed = (datetime.now() - start_time).total_seconds()
                logging.info(f"分割完成，用时: {elapsed:.2f}秒")

                # 在主线程中更新UI
                self.root.after(0, self.on_segmentation_finished, prediction_binary)

            except Exception as e:
                logging.error(f"分割失败: {str(e)}", exc_info=True)
                self.root.after(0, self.on_segmentation_error, str(e))

        threading.Thread(target=segment_thread, daemon=True).start()

    def on_segmentation_finished(self, prediction_binary):
        """分割完成回调"""
        try:
            # 创建分割结果图像
            result_image = Image.fromarray(prediction_binary)
            result_image.thumbnail((350, 350), Image.Resampling.NEAREST)

            # 转换为PhotoImage
            self.segmented_photo = ImageTk.PhotoImage(result_image)

            # 更新画布
            self.segmented_canvas.delete("all")
            self.segmented_canvas.create_image(175, 175, image=self.segmented_photo)

            # 重新启用按钮
            self.segment_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            self.status_var.set("分割完成!")

        except Exception as e:
            self.on_segmentation_error(str(e))

    def on_segmentation_error(self, error_msg):
        """分割错误回调"""
        messagebox.showerror("分割错误", f"分割过程中出错: {error_msg}")
        self.segment_btn.config(state=tk.NORMAL)
        self.status_var.set("分割失败")

    def save_result(self):
        """保存分割结果"""
        if self.current_prediction is None:
            logging.warning("尝试保存结果，但没有可用的分割结果")
            messagebox.showwarning("警告", "没有可保存的分割结果!")
            return

        # 打开保存对话框
        file_path = filedialog.asksaveasfilename(
            title="保存分割结果",
            defaultextension=".png",
            filetypes=[
                ("PNG文件", "*.png"),
                ("JPEG文件", "*.jpg"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            try:
                logging.info(f"保存分割结果到: {file_path}")
                result_image = Image.fromarray(self.current_prediction)
                result_image.save(file_path)

                file_size = os.path.getsize(file_path) / 1024  # KB
                logging.info(f"  保存成功，文件大小: {file_size:.2f} KB")

                self.status_var.set(f"结果已保存: {os.path.basename(file_path)}")
                messagebox.showinfo("成功", f"分割结果已保存到:\n{file_path}")
            except Exception as e:
                error_msg = f"保存失败: {str(e)}"
                logging.error(error_msg, exc_info=True)
                messagebox.showerror("错误", error_msg)

    def clear_display(self):
        """清除显示"""
        # 清除画布
        self.original_canvas.delete("all")
        self.original_canvas.create_text(175, 175, text="原始图像将显示在这里",
                                        fill="gray50", font=('Arial', 10))

        self.segmented_canvas.delete("all")
        self.segmented_canvas.create_text(175, 175, text="分割结果将显示在这里",
                                         fill="gray50", font=('Arial', 10))

        # 重置变量
        self.file_path_var.set("未选择文件")
        self.current_image_path = None
        self.current_prediction = None

        # 重置按钮状态
        self.segment_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)

        self.status_var.set("已清除显示")


def main():
    """主函数"""
    # 配置日志系统
    log_file = setup_gui_logging()

    # 模型路径（使用绝对路径）
    project_root = Path(__file__).parent.parent
    model_path = project_root / 'checkpoints' / 'best_model.pth'

    logging.info(f"项目根目录: {project_root}")
    logging.info(f"模型路径: {model_path}")
    logging.info(f"日志文件: {log_file}")

    # 检查模型是否存在
    if not model_path.exists():
        logging.error(f"模型文件不存在: {model_path}")
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        messagebox.showerror("错误", f'模型文件不存在: {model_path}\n请先训练模型!')
        logging.info("程序退出: 模型文件不存在")
        return

    try:
        # 创建主窗口
        logging.info("创建主窗口...")
        root = tk.Tk()

        # 创建GUI
        logging.info("初始化GUI...")
        app = COVIDSegmentationGUI(root, str(model_path))

        logging.info("GUI启动成功，进入主循环")
        logging.info("=" * 80)

        # 运行应用
        root.mainloop()

        logging.info("GUI主循环结束")

    except Exception as e:
        logging.error(f"GUI运行错误: {str(e)}", exc_info=True)
        raise
    finally:
        logging.info("程序退出")
        logging.info("=" * 80)


if __name__ == "__main__":
    main()
