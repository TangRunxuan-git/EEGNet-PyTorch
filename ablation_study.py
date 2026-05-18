"""
EEGNet 消融研究
验证深度卷积和可分离卷积对性能的贡献
论文 Section 3.3, Table 4
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from src.data.loader import BCIC2aDataset
from src.model.eegnet import EEGNet
import argparse
from tqdm import tqdm


class AblatedEEGNet(nn.Module):
    """支持消融的 EEGNet 变体"""
    def __init__(self, n_channels, n_classes=4, dropout_rate=0.5,
                 use_depthwise=True, use_separable=True):
        super().__init__()
        
        F1, D, F2 = 8, 2, 16
        kern_length = 64
        
        # Block 1: 时间卷积（始终保留）
        self.conv1 = nn.Conv2d(1, F1, (1, kern_length), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Block 1: 空间卷积（可消融）
        if use_depthwise:
            self.spatial_conv = nn.Conv2d(F1, F1*D, (n_channels, 1), groups=F1, bias=False)
        else:
            self.spatial_conv = nn.Conv2d(F1, F1*D, (n_channels, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)
        
        # Block 2: 可分离卷积（可消融）
        if use_separable:
            self.sep_depth = nn.Conv2d(F1*D, F1*D, (1, 16), groups=F1*D, padding='same', bias=False)
            self.sep_point = nn.Conv2d(F1*D, F2, (1, 1), bias=False)
        else:
            self.sep_depth = nn.Conv2d(F1*D, F2, (1, 16), padding='same', bias=False)
            self.sep_point = nn.Identity()
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.flatten = nn.Flatten()
        # 动态计算全连接层输入维度
        self._fc_in_features = None
        self.fc = None
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.F2 = F2
    
    def _get_fc_in_features(self, x):
        """动态计算展平后的特征维度"""
        with torch.no_grad():
            # 先通过所有卷积层但不展平
            x = self.conv1(x)
            x = self.bn1(x)
            x = nn.ELU()(x)
            
            x = self.spatial_conv(x)
            x = self.bn2(x)
            x = nn.ELU()(x)
            x = self.pool1(x)
            x = self.drop1(x)
            
            x = self.sep_depth(x)
            x = self.sep_point(x)
            x = self.bn3(x)
            x = nn.ELU()(x)
            x = self.pool2(x)
            x = self.drop2(x)
            
            return x.shape[1] * x.shape[2] * x.shape[3]
    
    def forward(self, x):
        # 前向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ELU()(x)
        
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = nn.ELU()(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = self.bn3(x)
        x = nn.ELU()(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = self.flatten(x)
        
        # 第一次 forward 时创建全连接层
        if self.fc is None:
            self._fc_in_features = x.shape[1]
            self.fc = nn.Linear(self._fc_in_features, self.n_classes).to(x.device)
        
        return self.fc(x)


def train_and_evaluate(model, train_loader, val_loader, device, epochs=100, lr=0.01):
    """训练并返回最佳验证准确率"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        
        # 每 10 个 epoch 评估一次
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x).argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            acc = correct / total
            if acc > best_acc:
                best_acc = acc
            print(f"    Epoch {epoch+1}/{epochs}, Val Acc: {acc:.4f}")
    
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/workspace/bci_eegnet_repro/data')
    parser.add_argument('--subject', type=int, default=9, help='使用最好的受试者做消融')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("EEGNet 消融研究")
    print(f"受试者: {args.subject}")
    print(f"设备: {device}")
    print("=" * 60)
    
    # 加载数据
    dataset = BCIC2aDataset(args.data_dir, args.subject)
    n_channels = dataset[0][0].shape[1]
    print(f"通道数: {n_channels}")
    
    # 划分训练/验证
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 四种配置
    configs = [
        ("完整 EEGNet", True, True),
        ("移除深度卷积", False, True),
        ("移除可分离卷积", True, False),
        ("移除两者", False, False),
    ]
    
    results = {}
    for name, use_dw, use_sw in configs:
        print(f"\n训练: {name}")
        model = AblatedEEGNet(n_channels, dropout_rate=args.dropout,
                              use_depthwise=use_dw, use_separable=use_sw).to(device)
        
        # 先用一个 dummy 输入初始化全连接层
        dummy = torch.randn(1, 1, n_channels, 501).to(device)
        _ = model(dummy)
        
        # 统计参数量
        params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {params:,}")
        
        acc = train_and_evaluate(model, train_loader, val_loader, device,
                                 epochs=args.epochs, lr=args.lr)
        results[name] = acc
        print(f"  最佳验证准确率: {acc:.4f}")
    
    # 输出结果
    print("\n" + "=" * 60)
    print("消融实验结果汇总")
    print("=" * 60)
    baseline = results["完整 EEGNet"]
    for name, acc in results.items():
        drop = baseline - acc
        print(f"{name:20s}: {acc:.4f} (相对完整模型: {drop:+.4f})")
    
    if baseline > 0:
        print(f"\n结论: 深度卷积贡献了 {baseline - results['移除深度卷积']:.4f}，"
              f"可分离卷积贡献了 {baseline - results['移除可分离卷积']:.4f}")


if __name__ == "__main__":
    main()
