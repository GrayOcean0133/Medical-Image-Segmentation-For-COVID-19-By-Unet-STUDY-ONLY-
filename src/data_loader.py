import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
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
    
    def __init__(self,images_dir,masks_dir,transform=None,img_size=512):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.transform  = transform
        self.img_size   = img_size
    
        #获得所有的文件名，512*512大小，.png文件格式
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
        
        #灰度化（本来就是黑白还要灰度吗？存疑）
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5],std=[0.5])
        
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
            # 随机种子生成（确保image和mask使用相同的变换）
            seed = torch.initial_seed()
            # 实例化原始CT图用卷积核
            torch.manual_seed(seed)
            image = self.transform(image)
            # 实例化MASK CT图用卷积核
            torch.manual_seed(seed)
            mask = self.transform(mask)
        else:
            # 转为张量
            image = self.to_tensor(image)
            mask  = self.to_tensor(mask)
        # 归一化
        image = self.normalize(image)
        
        # 掩码（蒙版MASK）二值化（白色->1显露/黑色0->遮罩）
        mask = (mask > 0.5).float()
        
        #成对出现处理完的image-mask
        return image,mask 
    
#实例化一个变焕(矫正原始的img-mask的方向用)
def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), #0.5的概率水平翻转
        transforms.RandomVerticalFlip(p=0.5),   #0.5的概率垂直翻转
        transforms.RandomRotation(degrees=10),  #10°的随机旋转角
        transforms.ToTensor()  # 添加ToTensor转换
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor()  # 验证集也需要ToTensor
    ])

    return train_transform,val_transform
