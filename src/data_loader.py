import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from torchvision.transforms import v2 as T
from pathlib import Path


# 路径集合
PROJECT_ROOTS = Path(__file__).parent.parent
COVID_TRAIN_IMG = PROJECT_ROOTS / "data_set/CT_COVID"
COVID_UNHEALTHY_LUNGS = COVID_TRAIN_IMG / "frames"
COVID_UNHEALTHY_LUNGS_MASK = COVID_TRAIN_IMG / "masks" 
COVID_HEALTHY_IMG =  PROJECT_ROOTS / "data_set/Healthy"

# 定义一个Dataset类作为输入
class COVIDCTDataset(Dataset):
    # CT.png数据类，包括图像和对应mask(蒙版？这里叫"掩码"我感觉不对)
    
    def __init__(self,images_dir,masks_dir,transform=None,img_size=512,normalize=True):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.transform  = transform
        self.img_size   = img_size
        self.normalize  = normalize

        # 排序保证多次运行顺序一致（os.listdir 的顺序依赖文件系统）
        self.image_files = sorted(f for f in os.listdir(images_dir) if f.endswith(".png"))

        # 提前检查掩码是否齐全，而不是训练跑到一半才抛 FileNotFoundError
        missing = [f for f in self.image_files
                   if not os.path.exists(os.path.join(masks_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} 张图像在 {masks_dir} 中找不到同名掩码，"
                f"例如: {missing[:5]}"
            )

        self.to_tensor = transforms.ToTensor()
        self.normalize_t = transforms.Normalize(mean=[0.5],std=[0.5])

        print(f"找到 {len(self.image_files)} 张图像在目录 {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 获取单个样本名字 (图像-掩码对)
        img_name = self.image_files[idx]
        mask_name = self.image_files[idx]
        # 找到图像-样本对
        img_path = os.path.join(self.images_dir,img_name)
        mask_path = os.path.join(self.masks_dir,mask_name)
        # 加载图像-样本对
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # 为了保证一下还是规范一下512*512的情况
        if image.size != (self.img_size,self.img_size):
            image = image.resize((self.img_size,self.img_size),Image.BILINEAR)
            mask = mask.resize((self.img_size,self.img_size),Image.NEAREST)

        if self.transform:
            # 关键：image 和 mask 必须作为一组同时传入，
            # 由 transforms.v2 采样一次随机参数并同步作用到两者。
            # 旧写法 torch.manual_seed(torch.initial_seed()) 每个样本都重置成
            # 同一个随机状态，等于完全没有做数据增强，而且会污染全局 RNG。
            image, mask = self.transform(image, mask)
            image = image.float()
            mask = mask.float()
        else:
            # 转为张量
            image = self.to_tensor(image)
            mask  = self.to_tensor(mask)
        # 归一化
        if self.normalize:
            image = self.normalize_t(image)

        # 掩码（蒙版MASK）二值化（白色->1显露/黑色0->遮罩）
        mask = (mask > 0.5).float()

        #成对出现处理完的image-mask
        return image,mask
    
#实例化一个变焕(矫正原始的img-mask的方向用)
def get_data_transforms():
    """返回 (train_transform, val_transform)。

    使用 torchvision.transforms.v2：它支持一次调用同时接收 image 与 mask，
    并把同一组随机参数应用到两者，几何变换不会错位。
    旋转对图像和掩码都使用 NEAREST，避免插值把二值掩码边缘糊掉。
    """
    train_transform = T.Compose([
        T.ToImage(),                    # PIL -> uint8 张量，之后的几何变换才能同步作用于 image/mask
        T.RandomHorizontalFlip(p=0.5),  #0.5的概率水平翻转
        T.RandomVerticalFlip(p=0.5),    #0.5的概率垂直翻转
        T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.NEAREST, fill=0),
        T.ToDtype(torch.float32, scale=True),
    ])

    val_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    return train_transform,val_transform


def split_train_val(dataset, val_split=0.0, seed=42):
    """按比例把数据集切成 (train_subset, val_subset)。

    val_split=0 时返回 (dataset, None)。
    """
    if val_split <= 0:
        return dataset, None

    n_total = len(dataset)
    n_val = max(1, int(round(n_total * val_split)))
    if n_val >= n_total:
        raise ValueError(f"val_split={val_split} 过大：{n_total} 个样本中留出 {n_val} 个")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    val_indices, train_indices = indices[:n_val], indices[n_val:]

    print(f"数据划分: 训练 {len(train_indices)} / 验证 {len(val_indices)}")
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

