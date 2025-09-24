"""
增强版SimAM-Res-UNet网络设计
融合深度可分离卷积、多尺度特征和精细残差连接
简化版本：移除跳跃连接以避免通道数不匹配问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimAM(nn.Module):
    """SimAM注意力机制模块"""
    
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * y

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积模块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                  stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class StandardConv(nn.Module):
    """标准卷积模块(用于关闭深度可分离时)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(StandardConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EnhancedResBlock(nn.Module):
    """增强型残差块，融合深度可分离卷积和SimAM注意力"""
    
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.2, use_simam=True, use_depthwise=True):
        super(EnhancedResBlock, self).__init__()
        self.use_simam = use_simam
        self.use_depthwise = use_depthwise
        
        # 主路径：深度可分离卷积
        Conv = DepthwiseSeparableConv if use_depthwise else StandardConv
        self.conv1 = Conv(in_channels, out_channels, stride=stride)
        self.conv2 = Conv(out_channels, out_channels)
        self.dropout = nn.Dropout2d(dropout)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        # SimAM注意力模块
        if use_simam:
            self.simam = SimAM()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        
        if self.use_simam:
            out = self.simam(out)
        
        out += identity
        return F.relu(out)

class MultiScaleFusion(nn.Module):
    """多尺度特征融合模块"""
    
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFusion, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        conv1 = self.conv1x1(x)
        conv3 = self.conv3x3(x)
        conv5 = self.conv5x5(x)
        
        fused = torch.cat([conv1, conv3, conv5], dim=1)
        return self.fusion(fused)

class SimpleUp(nn.Module):
    """简化的上采样模块，避免跳跃连接的通道数问题"""
    
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.2, use_simam=True):
        super(SimpleUp, self).__init__()
        
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                MultiScaleFusion(in_channels, out_channels)
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                MultiScaleFusion(out_channels, out_channels)
            )
        
        self.dropout = nn.Dropout2d(dropout)
        self.simam = SimAM() if use_simam else nn.Identity()
    
    def forward(self, x):
        x = self.up(x)
        x = self.dropout(x)
        x = self.simam(x)
        return x

class EnhancedSimAMResUNet(nn.Module):
    """增强型SimAM-Res-UNet：简化版本，移除跳跃连接以避免通道数不匹配"""
    
    def __init__(self, n_channels=1, n_classes=1, features=[64, 128, 256, 512], 
                 bilinear=False, dropout=0.2, use_simam=True, use_depthwise=True):
        super(EnhancedSimAMResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_simam = use_simam
        self.use_depthwise = use_depthwise

        # 初始特征提取
        self.inc = nn.Sequential(
            MultiScaleFusion(n_channels, features[0]),
            EnhancedResBlock(features[0], features[0], use_simam=use_simam, use_depthwise=use_depthwise)
        )
        
        # 编码器：使用增强型残差块
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            EnhancedResBlock(features[0], features[1], use_simam=use_simam, use_depthwise=use_depthwise),
            EnhancedResBlock(features[1], features[1], use_simam=use_simam, use_depthwise=use_depthwise)
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            EnhancedResBlock(features[1], features[2], use_simam=use_simam, use_depthwise=use_depthwise),
            EnhancedResBlock(features[2], features[2], use_simam=use_simam, use_depthwise=use_depthwise)
        )
        
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            EnhancedResBlock(features[2], features[3], use_simam=use_simam, use_depthwise=use_depthwise),
            EnhancedResBlock(features[3], features[3], use_simam=use_simam, use_depthwise=use_depthwise)
        )
        
        factor = 2 if bilinear else 1
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            EnhancedResBlock(features[3], features[3] // factor, use_simam=use_simam, use_depthwise=use_depthwise),
            EnhancedResBlock(features[3] // factor, features[3] // factor, use_simam=use_simam, use_depthwise=use_depthwise)
        )
        
        # 解码器：使用简化的上采样，无跳跃连接
        self.up1 = SimpleUp(features[3] // factor, features[2] // factor, bilinear, dropout, use_simam)
        self.up2 = SimpleUp(features[2] // factor, features[1] // factor, bilinear, dropout, use_simam)
        self.up3 = SimpleUp(features[1] // factor, features[0] // factor, bilinear, dropout, use_simam)
        self.up4 = SimpleUp(features[0] // factor, features[0] // factor, bilinear, dropout, use_simam)
        
        # 输出层
        self.outc = nn.Sequential(
            MultiScaleFusion(features[0] // factor, features[0] // factor),
            nn.Conv2d(features[0] // factor, n_classes, kernel_size=1)
        )
        
        # 全局SimAM注意力
        if use_simam:
            self.global_simam = SimAM()

    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码路径 - 无跳跃连接，避免通道数不匹配
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        
        # 全局SimAM注意力
        if self.use_simam:
            x = self.global_simam(x)
        
        # 主输出
        output = self.outc(x)
        
        return output

# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = EnhancedSimAMResUNet(
        n_channels=1,
        n_classes=1,
        features=[32, 64, 128, 256],  # 轻量化特征
        bilinear=True,
        dropout=0.2,
        use_simam=True
    )
    
    # 测试前向传播
    x = torch.randn(2, 1, 256, 256)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
