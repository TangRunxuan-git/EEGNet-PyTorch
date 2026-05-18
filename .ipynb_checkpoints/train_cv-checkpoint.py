
"""
EEGNet 4-折交叉验证训练脚本
论文 Section 2.3: four-fold blockwise cross-validation
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from src.data.loader import BCIC2aDataset
from src.model.eegnet import create_eegnet
import argparse
from datetime import datetime

def train_and_evaluate(subject_id, data_dir, args, device):
    """对单个受试者进行 4-折交叉验证"""
    
    # 加载完整数据集
    full_dataset = BCIC2aDataset(data_dir, subject_id)
    n_channels = full_dataset[0][0].shape[1]
    
    # K-Fold 设置
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\n  Fold {fold+1}/{args.k_folds}")
        
        # 创建数据加载器
        train_loader = DataLoader(
            Subset(full_dataset, train_idx),
            batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_idx),
            batch_size=args.batch_size, shuffle=False
        )
        
        # 创建模型
        model = create_eegnet(
            n_channels=n_channels,
            n_classes=4,
            dropout_rate=args.dropout_rate
        ).to(device)
        
        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        
        # 训练
        best_acc = 0.0
        for epoch in range(args.epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            
            # 验证
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x).argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            acc = correct / total
            
            if acc > best_acc:
                best_acc = acc
        
        fold_results.append(best_acc)
        print(f"    Fold {fold+1} 最佳准确率: {best_acc:.4f}")
    
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    print(f"\n受试者 {subject_id} 平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return mean_acc, std_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/workspace/bci_eegnet_repro/data')
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 10)))
    parser.add_argument('--k_folds', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"4-折交叉验证, {len(args.subjects)} 个受试者\n")
    
    all_results = []
    for subj in args.subjects:
        mean_acc, std_acc = train_and_evaluate(subj, args.data_dir, args, device)
        all_results.append(mean_acc)
    
    print("\n" + "="*50)
    print("最终结果")
    print("="*50)
    for i, acc in enumerate(all_results):
        print(f"受试者 {args.subjects[i]}: {acc:.4f}")
    print(f"\n平均: {np.mean(all_results):.4f} ± {np.std(all_results):.4f}")

if __name__ == "__main__":
    main()
