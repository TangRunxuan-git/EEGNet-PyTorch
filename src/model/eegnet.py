"""
EEGNet: Compact Convolutional Neural Network for EEG-based BCIs
论文: https://iopscience.iop.org/article/10.1088/1741-2552/aace8c
官方 TensorFlow 实现参考: https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py

PyTorch 实现，完全遵循论文架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet 模型
    
    参数:
        n_channels: EEG 通道数 (BCI IV 2a: 22)
        n_classes: 分类数 (4)
        F1: 时间滤波器数量 (默认 8)
        D: 深度乘数 (默认 2)
        F2: 逐点滤波器数量 (默认 F1 * D = 16)
        dropout_rate: Dropout 概率 (默认 0.25)
        kern_length: 时间卷积核长度 (默认 64, 对应 128Hz 采样率)
    """
    
    def __init__(self, n_channels=22, n_classes=4, F1=8, D=2, F2=None, 
                 dropout_rate=0.25, kern_length=64):
        super(EEGNet, self).__init__()
        
        # 如果未指定 F2，则设为 F1 * D（论文推荐）
        if F2 is None:
            F2 = F1 * D
        
        # ========== Block 1 ==========
        # 时间卷积: 学习频率滤波器
        # 输入: (batch, 1, n_channels, time)
        # 输出: (batch, F1, n_channels, time)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kern_length),
            padding='same',
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)
        
        # 深度卷积: 学习空间滤波器 (每个时间滤波器独立)
        # 输出: (batch, F1 * D, 1, time)
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,  # 深度卷积的关键: 分组数等于输入通道数
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # ========== Block 2 ==========
        # 可分离卷积 = 深度卷积 + 逐点卷积
        # 深度卷积部分: 每个特征图独立学习时间摘要
        self.separable_depth = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding='same',
            groups=F1 * D,  # 深度卷积
            bias=False
        )
        # 逐点卷积部分: 混合特征图
        self.separable_point = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # ========== 分类器 ==========
        # 论文: 直接接 Softmax，不添加额外全连接层
        self.flatten = nn.Flatten()
        
        # 动态计算全连接层输入维度
        self._feature_dim = None
        self.fc = None
        self.n_classes = n_classes
        
    def _get_feature_dim(self, dummy_input):
        """计算展平后的特征维度"""
        with torch.no_grad():
            out = self._forward_features(dummy_input)
        return out.shape[1]
    
    def _forward_features(self, x):
        """只运行特征提取部分，不包含分类器"""
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)  # 论文使用 ELU 激活
        
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        
        return x
    
    def forward(self, x):
        """
        前向传播
        
        输入: x (batch, 1, n_channels, time)
        输出: logits (batch, n_classes)
        """
        x = self._forward_features(x)
        x = self.flatten(x)
        
        # 动态创建全连接层（第一次 forward 时）
        if self.fc is None:
            self._feature_dim = x.shape[1]
            self.fc = nn.Linear(self._feature_dim, self.n_classes)
            # 移动到与输入相同的设备
            self.fc = self.fc.to(x.device)
        
        x = self.fc(x)
        return x


def create_eegnet(n_channels=22, n_classes=4, dropout_rate=0.25, sampling_rate=250):
    """
    创建 EEGNet 模型的工厂函数
    
    参数:
        n_channels: EEG 通道数
        n_classes: 分类数
        dropout_rate: Dropout 概率 (论文: within-subject 用 0.5, cross-subject 用 0.25)
        sampling_rate: 采样率 (Hz)
    
    返回:
        EEGNet 模型实例
    """
    # 论文 Table 2: kernLength = 采样率的一半 (128Hz 时 = 64)
    # 对于 250Hz 原始数据，使用 kern_length = 125 或保持 64 都可以
    # 保持 64 会更关注高频信息，125 会覆盖更宽频带
    kern_length = 64  # 保持与论文一致（论文实际重采样到 128Hz）
    
    # 论文配置: F1=8, D=2, F2=16
    model = EEGNet(
        n_channels=n_channels,
        n_classes=n_classes,
        F1=8,
        D=2,
        F2=16,
        dropout_rate=dropout_rate,
        kern_length=kern_length
    )
    
    return model


# 测试代码
if __name__ == "__main__":
    print("=" * 50)
    print("测试 EEGNet 模型")
    print("=" * 50)
    
    # 模拟 BCI IV 2a 数据批次
    batch_size = 32
    n_channels = 22
    time_points = 501  # 250Hz * 2s ≈ 500
    
    dummy_input = torch.randn(batch_size, 1, n_channels, time_points)
    
    # 创建模型
    model = create_eegnet(n_channels=n_channels, n_classes=4, dropout_rate=0.25)
    
    # 前向传播
    output = model(dummy_input)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 打印网络结构
    print("\n网络结构:")
    print(model)
    
    print("\n✅ EEGNet 模型测试通过!")