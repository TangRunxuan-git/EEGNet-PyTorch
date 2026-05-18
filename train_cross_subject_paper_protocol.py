"""
EEGNet Cross-subject 实验 - 严格遵循论文 Section 2.3 协议
训练集: 5 个其他受试者 (随机)
验证集: 3 个其他受试者
测试集: 留一受试者 (使用 *E.gdf + 官方标签)
论文: "select the training data from 5 other subjects at random to be the training set 
       and the training data from the remaining 3 subjects to be the validation set"
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import mne
import scipy.io
import argparse
from datetime import datetime


class BCIC2aTrainDataset(Dataset):
    """训练/验证集：使用 *T.gdf，同时返回受试者ID以便分层划分"""
    def __init__(self, data_dir, subject_id):
        self.subject_id = subject_id
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
        print(f"  Subject {subject_id}: {len(self.y)} samples")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][np.newaxis, :, :]), torch.tensor(self.y[idx])


class BCIC2aTestDataset(Dataset):
    """测试集：使用 *E.gdf + 官方标签"""
    def __init__(self, data_dir, labels_dir, subject_id):
        # 加载官方标签
        label_file = os.path.join(labels_dir, f"A0{subject_id}E.mat")
        label_data = scipy.io.loadmat(label_file)
        self.y = label_data['classlabel'].flatten().astype(np.int64) - 1
        
        # 加载EEG数据
        file_path = os.path.join(data_dir, f"A0{subject_id}E.gdf")
        raw = mne.io.read_raw_gdf(file_path, preload=True, verbose='ERROR')
        raw.resample(128)
        raw.filter(4.0, None, fir_design='firwin', verbose='ERROR')
        
        # 用 768 事件定位 trial 起始
        events, event_dict = mne.events_from_annotations(raw, verbose='ERROR')
        target_code = None
        for stim, code in event_dict.items():
            if stim == '768':
                target_code = code
                break
        if target_code is None:
            raise ValueError("未找到 768 事件")
        
        cue_events = events[events[:, 2] == target_code]
        picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')
        sfreq = raw.info['sfreq']
        
        X_list = []
        for event in cue_events:
            onset = event[0] / sfreq
            start = int((onset + 0.5) * sfreq)
            end   = int((onset + 2.5) * sfreq)
            if end <= raw.n_times:
                data = raw.get_data(picks=picks, start=start, stop=end)
                if data.shape[1] == int(2.0 * sfreq):
                    X_list.append(data)
        
        X = np.stack(X_list, axis=0) * 1e6
        if len(X) != len(self.y):
            min_len = min(len(X), len(self.y))
            X = X[:min_len]
            self.y = self.y[:min_len]
        
        self.X = X.astype(np.float32)
        print(f"  Test subject {subject_id}: {len(self.y)} samples, shape={self.X.shape}")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][np.newaxis, :, :]), torch.tensor(self.y[idx])


def create_eegnet(n_channels, dropout_rate=0.25):
    import sys
    sys.path.append('/mnt/workspace/bci_eegnet_repro')
    from src.model.eegnet import EEGNet
    return EEGNet(
        n_channels=n_channels,
        n_classes=4,
        F1=8, D=2, F2=16,
        dropout_rate=dropout_rate,
        kern_length=64
    )


def train_model(model, train_loader, val_loader, device, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        
        # 每 50 个 epoch 评估验证集
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
            if acc > best_val_acc:
                best_val_acc = acc
            print(f"    Epoch {epoch+1}/{epochs}, Val Acc: {acc:.4f}")
    return best_val_acc


def run_cross_subject_experiment(data_dir, labels_dir, test_subj, args, device, random_seed=None):
    """单次留一实验，严格按论文划分训练/验证集"""
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # 所有其他受试者
    other_subjs = [s for s in range(1, 10) if s != test_subj]
    # 随机选 5 个作为训练集，剩余 3 个作为验证集
    train_subjs = np.random.choice(other_subjs, size=5, replace=False).tolist()
    val_subjs = [s for s in other_subjs if s not in train_subjs]
    
    print(f"\n  训练受试者: {train_subjs}")
    print(f"  验证受试者: {val_subjs}")
    
    # 加载训练集 (5 人)
    train_datasets = [BCIC2aTrainDataset(data_dir, s) for s in train_subjs]
    train_dataset = ConcatDataset(train_datasets)
    n_channels = train_datasets[0][0][0].shape[0]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # 加载验证集 (3 人)
    val_datasets = [BCIC2aTrainDataset(data_dir, s) for s in val_subjs]
    val_dataset = ConcatDataset(val_datasets)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 加载测试集 (留一受试者的 *E.gdf)
    test_dataset = BCIC2aTestDataset(data_dir, labels_dir, test_subj)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 创建模型
    model = create_eegnet(n_channels, dropout_rate=args.dropout_rate).to(device)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    best_val_acc = train_model(model, train_loader, val_loader, device,
                               epochs=args.epochs, lr=args.lr)
    
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
    
    return test_acc, best_val_acc


def main():
    parser = argparse.ArgumentParser(description="EEGNet Cross-subject (严格论文协议)")
    parser.add_argument('--data_dir', type=str, default='/mnt/workspace/bci_eegnet_repro/data')
    parser.add_argument('--labels_dir', type=str, default='/mnt/workspace/bci_eegnet_repro/true_labels')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮数 (论文用500)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001, help='学习率 (论文Adam默认)')
    parser.add_argument('--dropout_rate', type=float, default=0.25, help='Cross-subject dropout (论文0.25)')
    parser.add_argument('--seed', type=int, default=2024, help='随机种子 (保证复现)')
    parser.add_argument('--repeat', type=int, default=1, help='重复次数 (论文中重复30次取平均，可设>1)')
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(1,10)), help='测试受试者列表')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("EEGNet Cross-subject 实验 (严格遵循论文 Section 2.3)")
    print(f"设备: {device}")
    print(f"训练轮数: {args.epochs}, 学习率: {args.lr}, Dropout: {args.dropout_rate}")
    print(f"重复次数: {args.repeat} (每次重新随机划分训练/验证集)")
    print("=" * 70)
    
    all_test_accs = {subj: [] for subj in args.subjects}
    
    for rep in range(args.repeat):
        print(f"\n========== Repeat {rep+1}/{args.repeat} ==========")
        # 为每次重复设置不同种子，若重复1次则固定种子
        if args.repeat > 1:
            current_seed = args.seed + rep
        else:
            current_seed = args.seed
        
        for test_subj in args.subjects:
            print(f"\n--- 测试受试者 {test_subj} (种子 {current_seed}) ---")
            test_acc, _ = run_cross_subject_experiment(
                args.data_dir, args.labels_dir, test_subj, args, device, random_seed=current_seed
            )
            all_test_accs[test_subj].append(test_acc)
            print(f"  测试准确率: {test_acc:.4f}")
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("最终结果 (每个受试者多次重复的平均 ± 标准差)")
    print("=" * 70)
    mean_accs = []
    for subj in args.subjects:
        accs = all_test_accs[subj]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs) if len(accs) > 1 else 0.0
        mean_accs.append(mean_acc)
        print(f"受试者 {subj:2d}: {mean_acc:.4f} ± {std_acc:.4f}")
    
    overall_mean = np.mean(mean_accs)
    overall_std = np.std(mean_accs)
    print(f"\n总体平均: {overall_mean:.4f} ± {overall_std:.4f}")
    print(f"随机基线 (4类): 0.2500")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'cross_subject_paper_{timestamp}.txt'
    with open(log_file, 'w') as f:
        f.write(f"Parameters: {vars(args)}\n")
        f.write(f"Overall mean: {overall_mean:.4f} ± {overall_std:.4f}\n")
        for subj in args.subjects:
            f.write(f"Subject {subj}: {all_test_accs[subj]}\n")
    print(f"\n结果保存至: {log_file}")


if __name__ == "__main__":
    main()
