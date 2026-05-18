"""
EEGNet Cross-subject 实验 - 使用官方测试集标签
训练: 8个受试者的 *T.gdf
测试: 留一受试者的 *E.gdf (使用官方标签顺序)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import mne
import scipy.io


class BCIC2aTrainDataset(Dataset):
    """训练集：使用 *T.gdf"""
    def __init__(self, data_dir, subject_id):
        file_path = os.path.join(data_dir, f"A0{subject_id}T.gdf")
        raw = mne.io.read_raw_gdf(file_path, preload=True, verbose='ERROR')
        raw.resample(128)
        raw.filter(4.0, None, fir_design='firwin', verbose='ERROR')
        
        events, event_dict = mne.events_from_annotations(raw, verbose='ERROR')
        target_stims = ['769', '770', '771', '772']
        target_codes = [event_dict[s] for s in target_stims]
        
        picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')
        epochs = mne.Epochs(raw, events, event_id=target_codes,
                            tmin=0.5, tmax=2.5, picks=picks,
                            baseline=None, preload=True, verbose='ERROR')
        
        X = epochs.get_data() * 1e6
        orig_codes = epochs.events[:, -1]
        label_map = {code: i for i, code in enumerate(target_codes)}
        y = np.array([label_map[code] for code in orig_codes])
        
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][np.newaxis, :, :]), torch.tensor(self.y[idx])


class BCIC2aTestDataset(Dataset):
    """测试集：使用 *E.gdf，按官方标签顺序提取 trials"""
    def __init__(self, data_dir, labels_dir, subject_id):
        # 加载官方标签
        label_file = os.path.join(labels_dir, f"A0{subject_id}E.mat")
        label_data = scipy.io.loadmat(label_file)
        self.y = label_data['classlabel'].flatten().astype(np.int64) - 1  # 1-4 → 0-3
        
        # 加载EEG数据
        file_path = os.path.join(data_dir, f"A0{subject_id}E.gdf")
        raw = mne.io.read_raw_gdf(file_path, preload=True, verbose='ERROR')
        raw.resample(128)
        raw.filter(4.0, None, fir_design='firwin', verbose='ERROR')
        
        # 获取所有事件的时间点
        events, event_dict = mne.events_from_annotations(raw, verbose='ERROR')
        
        # 只保留 769-772 运动想象事件的起始时间
        target_codes = [event_dict[s] for s in ['769', '770', '771', '772'] if s in event_dict]
        mask = np.isin(events[:, 2], target_codes)
        cue_events = events[mask]
        
        # 提取每个 cue 时刻的 trial (tmin=0.5, tmax=2.5)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')
        
        X_list = []
        for event in cue_events:
            onset = event[0] / raw.info['sfreq']  # 转换为秒
            tmin = 0.5
            tmax = 2.5
            # 提取数据
            data, times = raw[:, int((onset + tmin) * raw.info['sfreq']):int((onset + tmax) * raw.info['sfreq'])]
            if data.shape[1] == int((tmax - tmin) * raw.info['sfreq']):
                X_list.append(data)
        
        X = np.stack(X_list, axis=0) * 1e6  # (n_trials, n_channels, n_times)
        
        # 确保数据与标签数量匹配
        if len(X) != len(self.y):
            print(f"警告: 提取的 trials={len(X)}, 标签={len(self.y)}，使用最小公倍数")
            min_len = min(len(X), len(self.y))
            X = X[:min_len]
            self.y = self.y[:min_len]
        
        self.X = X.astype(np.float32)
        print(f"Test subject {subject_id}: {len(self.y)} samples, X shape={self.X.shape}, labels: {np.bincount(self.y)}")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][np.newaxis, :, :]), torch.tensor(self.y[idx])


def create_eegnet(n_channels, n_classes=4, dropout_rate=0.25):
    import sys
    sys.path.append('/mnt/workspace/bci_eegnet_repro')
    from src.model.eegnet import EEGNet
    return EEGNet(
        n_channels=n_channels,
        n_classes=n_classes,
        F1=8, D=2, F2=16,
        dropout_rate=dropout_rate,
        kern_length=64
    )


def train_model(model, train_loader, val_loader, device, epochs=200, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
        
        if (epoch + 1) % 50 == 0:
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
            print(f"  Epoch {epoch+1}/{epochs}, Val Acc: {acc:.4f}")
    return best_acc


def main():
    data_dir = "/mnt/workspace/bci_eegnet_repro/data"
    labels_dir = "/mnt/workspace/bci_eegnet_repro/true_labels"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("EEGNet Cross-subject (官方测试集)")
    print(f"设备: {device}")
    print("=" * 60)
    
    results = {}
    
    for test_subj in range(1, 10):
        print(f"\n--- 测试受试者 {test_subj} ---")
        
        # 训练数据：其他8人的 *T.gdf
        train_datasets = []
        for subj in range(1, 10):
            if subj != test_subj:
                ds = BCIC2aTrainDataset(data_dir, subj)
                train_datasets.append(ds)
        
        train_dataset = ConcatDataset(train_datasets)
        n_channels = train_datasets[0][0][0].shape[0]
        
        # 划分验证集
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)
        
        # 测试数据
        test_dataset = BCIC2aTestDataset(data_dir, labels_dir, test_subj)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        model = create_eegnet(n_channels).to(device)
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        best_val_acc = train_model(model, train_loader, val_loader, device, epochs=200, lr=0.001)
        
        # 测试
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        test_acc = correct / total
        
        results[test_subj] = test_acc
        print(f"  验证准确率: {best_val_acc:.4f}")
        print(f"  测试准确率: {test_acc:.4f}")
    
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    for subj, acc in results.items():
        print(f"受试者 {subj}: {acc:.4f}")
    print(f"\n平均: {np.mean(list(results.values())):.4f} ± {np.std(list(results.values())):.4f}")


if __name__ == "__main__":
    main()
