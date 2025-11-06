import torch
import torch.nn as nn
import torch.nn.functional as F


#定义双卷积块类：卷积 -> 批归一化 -> ReLU -> 卷积 -> 批归一化 -> ReLU//卷两次
class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        # 上卷下，提取特征值 ---->降分辨率,3*3->1
        self.double_conv = nn.Sequential(
            # 第一次卷积(二位图像卷积3*3卷积核)
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=False),
            # kernel_size 卷积核大小3*3
            # padding 步进到时候看效果调整1或2(3行不行？没有交叉重叠我感觉有点不行。)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #第二次卷积-重复第一次的步骤。
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x): #前向传播
        return self.double_conv(x)
    
class Unet(nn.Module):
    def __init__(self,in_channels=1,out_channels=1,features=[64,128,128,256]):
        # 初始化U-Net
        # -in_channels: 输入图像通道数（CT图像为1，灰度图）
        # -out_channels: 输出通道数（二分类为1）
        # -features: 编码器各层的基础通道数，控制模型容量
        super(Unet,self).__init__()
        
        
        # 编码器
        self.encoder = nn.ModuleList()
        # 池化
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        # 编码器实例化
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            # 下一层的输入通道数等于当前层的输出通道数
            in_channels = feature
            
        # 第一层实现：瓶颈层(输入层)
        self.bottleneck = DoubleConv(features[-1],features[-1]*2)
        
        # 解码器路径构建
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # 构建解码器(下往上超采样-转置卷积-定位) ----->升分辨率 1->2*2
        for feature in reversed(features):
            # 上采样：将通道数从 feature*2 减半到 feature
            self.upsamples.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2,stride=2)
            )
            # 解码器卷积块（输入通道数要×2，因为要拼接跳跃连接）
            # 拼接后: feature (上采样) + feature (skip) = feature*2
            self.decoder.append(DoubleConv(feature * 2,feature))
        # 最后输出层，1*1卷积，映射到所需的输出通道数    
        self.final_conv = nn.Conv2d(features[0],out_channels,kernel_size=1)

# 前向传播
    def forward(self,x):
        # 存储编码器各层的输出，用于跳跃连接
        skip_connections = []
        
        # 编码器路径
        for down in self.encoder:
            x = down(x) # 双卷积后的X
            skip_connections.append(x)  # 留存特征图备份 
            x = self.pool(x) #下采样（最大池化）
            
        # 第一册-瓶颈层
        x = self.bottleneck(x)
        
        # 跳转连接实现
        skip_connections = skip_connections[::-1]
        
        # 解码器路径实现
        for idx in range(len(self.decoder)):
            # 上采样实现
            x = self.upsamples[idx](x)
            
            # 条约连接实现
            skip_connection = skip_connections[idx]
            
            # 确保尺寸匹配
            if x.shape != skip_connection.shape:
                x = F.interpolate(x,size=skip_connection.shape[2:],mode="bilinear",align_corners=True)
            
            x = torch.cat((skip_connection,x), dim=1)
            
            x = self.decoder[idx](x)
            
        return torch.sigmoid(self.final_conv(x)) # 用sigmoid函数作为输出函数滤波输出

def dice_loss(pred, target, smooth = 1e-6):
# 用Dice()函数作为损失函数

    # 展平预测张量和目标张量
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    # 计算交集
    intersection = (pred * target).sum()
    
    # 计算Dice函数
    dice = (2. * intersection +smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def combined_loss(pred,target,alpha=0.5):
    dice_loss_val = dice_loss(pred, target)
    bce_loss = F.binary_cross_entropy(pred,target)
    return alpha * dice_loss_val + (1- alpha) * bce_loss
    
