"""
EEGNet 对比实验
与 DeepConvNet、ShallowConvNet、FBCSP 对比
论文 Section 2.2.2 和 Section 2.2.3
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.signal import butter, filtfilt
from src.data.loader import BCIC2aDataset
from torch.utils.data import DataLoader, Subset
import argparse
import warnings
warnings.filterwarnings('ignore')


# ============ 1. DeepConvNet (PyTorch 实现) ============
class DeepConvNet(nn.Module):
    """论文 Section 5.1, Table 5"""
    def __init__(self, n_channels, n_classes=4, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 25, (1, 5), bias=False)
        self.conv2 = nn.Conv2d(25, 25, (n_channels, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(25)
        self.pool1 = nn.MaxPool2d((1, 2))
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.conv3 = nn.Conv2d(25, 50, (1, 5), bias=False)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d((1, 2))
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.conv4 = nn.Conv2d(50, 100, (1, 5), bias=False)
        self.bn3 = nn.BatchNorm2d(100)
        self.pool3 = nn.MaxPool2d((1, 2))
        self.drop3 = nn.Dropout(dropout_rate)
        
        self.conv5 = nn.Conv2d(100, 200, (1, 5), bias=False)
        self.bn4 = nn.BatchNorm2d(200)
        self.pool4 = nn.MaxPool2d((1, 2))
        self.drop4 = nn.Dropout(dropout_rate)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(200 * 32, n_classes)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv2(torch.relu(self.conv1(x)))))
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = torch.relu(self.bn2(self.conv3(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = torch.relu(self.bn3(self.conv4(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        
        x = torch.relu(self.bn4(self.conv5(x)))
        x = self.pool4(x)
        x = self.drop4(x)
        
        x = self.flatten(x)
        return self.fc(x)


# ============ 2. ShallowConvNet (PyTorch 实现) ============
class ShallowConvNet(nn.Module):
    """论文 Section 5.1, Table 6"""
    def __init__(self, n_channels, n_classes=4, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 40, (1, 13), bias=False)
        self.conv2 = nn.Conv2d(40, 40, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.drop = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(40 * 62, n_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = x ** 2
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-7, max=10000))
        x = self.drop(x)
        x = self.flatten(x)
        return self.fc(x)


# ============ 3. FBCSP (传统方法) ============
class FBCSP:
    def __init__(self):
        self.classifiers = []
    
    def _bandpass_filter(self, data, fs, low, high):
        nyq = fs / 2
        b, a = butter(4, [low/nyq, high/nyq], btype='band')
        return filtfilt(b, a, data, axis=-1)
    
    def fit(self, X, y):
        fs = 250
        freq_bands = [(4,8), (8,12), (12,16), (16,20), (20,24), (24,28), (28,32), (32,36), (36,40)]
        
        for band in freq_bands:
            filtered = np.array([self._bandpass_filter(x, fs, band[0], band[1]) for x in X])
            clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
            X_flat = filtered.reshape(filtered.shape[0], -1)
            clf.fit(X_flat, y)
            self.classifiers.append(clf)
        return self
    
    def predict(self, X):
        fs = 250
        freq_bands = [(4,8), (8,12), (12,16), (16,20), (20,24), (24,28), (28,32), (32,36), (36,40)]
        all_preds = []
        for clf, band in zip(self.classifiers, freq_bands):
            filtered = np.array([self._bandpass_filter(x, fs, band[0], band[1]) for x in X])
            X_flat = filtered.reshape(filtered.shape[0], -1)
            all_preds.append(clf.predict_proba(X_flat))
        avg_proba = np.mean(all_preds, axis=0)
        return np.argmax(avg_proba, axis=1)


def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.01):
    model = model.to(device)
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
        
        if (epoch + 1) % 20 == 0:
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


def run_comparison(subject_id, data_dir, device, epochs=100):
    """运行对比实验"""
    print(f"\n加载受试者 {subject_id} 数据...")
    dataset = BCIC2aDataset(data_dir, subject_id)
    n_channels = dataset[0][0].shape[1]
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    results = {}
    
    # EEGNet
    print("\n训练 EEGNet...")
    from src.model.eegnet import EEGNet
    eegnet = EEGNet(n_channels=n_channels, n_classes=4, dropout_rate=0.5)
    eegnet_params = sum(p.numel() for p in eegnet.parameters())
    results['EEGNet'] = train_model(eegnet, train_loader, val_loader, device, epochs=epochs, lr=0.01)
    print(f"  EEGNet 参数量: {eegnet_params:,}, 准确率: {results['EEGNet']:.4f}")
    
    # DeepConvNet
    print("\n训练 DeepConvNet...")
    deepconv = DeepConvNet(n_channels=n_channels)
    deepconv_params = sum(p.numel() for p in deepconv.parameters())
    results['DeepConvNet'] = train_model(deepconv, train_loader, val_loader, device, epochs=epochs, lr=0.01)
    print(f"  DeepConvNet 参数量: {deepconv_params:,}, 准确率: {results['DeepConvNet']:.4f}")
    
    # ShallowConvNet
    print("\n训练 ShallowConvNet...")
    shallowconv = ShallowConvNet(n_channels=n_channels)
    shallowconv_params = sum(p.numel() for p in shallowconv.parameters())
    results['ShallowConvNet'] = train_model(shallowconv, train_loader, val_loader, device, epochs=epochs, lr=0.01)
    print(f"  ShallowConvNet 参数量: {shallowconv_params:,}, 准确率: {results['ShallowConvNet']:.4f}")
    
    # FBCSP
    print("\n训练 FBCSP...")
    X_train = torch.cat([x for x, _ in train_ds]).numpy().squeeze(1)
    y_train = torch.cat([y for _, y in train_ds]).numpy()
    X_val = torch.cat([x for x, _ in val_ds]).numpy().squeeze(1)
    y_val = torch.cat([y for _, y in val_ds]).numpy()
    
    fbcsp = FBCSP()
    fbcsp.fit(X_train, y_train)
    val_pred = fbcsp.predict(X_val)
    results['FBCSP'] = np.mean(val_pred == y_val)
    print(f"  FBCSP 准确率: {results['FBCSP']:.4f}")
    
    return results, eegnet_params, deepconv_params, shallowconv_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=int, default=9, help='受试者编号')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--data_dir', type=str, default='/mnt/workspace/bci_eegnet_repro/data')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("EEGNet 对比实验 (与 DeepConvNet, ShallowConvNet, FBCSP)")
    print(f"受试者: {args.subject}, 设备: {device}, 轮数: {args.epochs}")
    print("=" * 60)
    
    results, eegnet_params, deepconv_params, shallowconv_params = run_comparison(
        args.subject, args.data_dir, device, args.epochs
    )
    
    print("\n" + "=" * 60)
    print("对比实验结果汇总")
    print("=" * 60)
    print(f"{'模型':<18} {'参数量':<12} {'准确率':<10}")
    print("-" * 40)
    print(f"{'EEGNet':<18} {eegnet_params:<12,} {results['EEGNet']:<10.4f}")
    print(f"{'DeepConvNet':<18} {deepconv_params:<12,} {results['DeepConvNet']:<10.4f}")
    print(f"{'ShallowConvNet':<18} {shallowconv_params:<12,} {results['ShallowConvNet']:<10.4f}")
    print(f"{'FBCSP (传统)':<18} {'N/A':<12} {results['FBCSP']:<10.4f}")
    
    print(f"\n结论:")
    print(f"  EEGNet 比 DeepConvNet 参数量减少: {deepconv_params / eegnet_params:.1f}x")
    print(f"  EEGNet 比 ShallowConvNet 参数量减少: {shallowconv_params / eegnet_params:.1f}x")
    print(f"  EEGNet vs DeepConvNet 准确率: {results['EEGNet'] - results['DeepConvNet']:+.4f}")
    print(f"  EEGNet vs FBCSP 准确率: {results['EEGNet'] - results['FBCSP']:+.4f}")


if __name__ == "__main__":
    main()
