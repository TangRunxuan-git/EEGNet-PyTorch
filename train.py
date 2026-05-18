
"""
EEGNet 完整训练脚本
数据集: BCI Competition IV 2a
论文: EEGNet (JNE 2018)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from datetime import datetime

# 导入自定义模块
import sys
sys.path.append('/mnt/workspace/bci_eegnet_repro')
from src.data.loader import get_dataloaders
from src.model.eegnet import create_eegnet


def set_seed(seed=2024):
    """设置随机种子确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / total, correct / total


def validate_epoch(model, loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validating', leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return total_loss / total, correct / total


def train_subject(subject_id, data_dir, args, device):
    """训练单个受试者"""
    print(f"\n{'='*60}")
    print(f"训练受试者 {subject_id}")
    print(f"{'='*60}")
    
    # 创建数据加载器
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        subject_id=subject_id,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers
    )
    
    # 获取输入维度
    sample_x, _ = next(iter(train_loader))
    n_channels = sample_x.shape[2]
    print(f"通道数: {n_channels}")
    
    # 创建模型
    model = create_eegnet(
        n_channels=n_channels,
        n_classes=4,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    
    # 训练记录
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # 模型保存路径
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'subject_{subject_id}_best.pth')
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # 验证
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 早停和保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, save_path)
            print(f"Epoch {epoch}: 保存最佳模型 (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Epoch {epoch}: 早停触发")
                break
        
        # 打印进度
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\n受试者 {subject_id} 最佳验证准确率: {best_val_acc:.4f}")
    return best_val_acc, history


def main():
    parser = argparse.ArgumentParser(description='训练 EEGNet 模型')
    parser.add_argument('--data_dir', type=str, default='/mnt/workspace/bci_eegnet_repro/data',
                        help='数据目录')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='TensorBoard 日志目录')
    parser.add_argument('--subjects', type=int, nargs='+', default=[1,2,3,4,5,6,7,8,9],
                        help='要训练的受试者列表')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=300, help='最大迭代轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减')
    parser.add_argument('--dropout_rate', type=float, default=0.25, help='Dropout 概率')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--patience', type=int, default=50, help='早停耐心值')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=2024, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 记录参数
    print(f"\n训练参数:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
        writer.add_text('args', f'{key}: {value}')
    
    # 训练每个受试者
    all_accuracies = []
    all_histories = {}
    
    for subject_id in args.subjects:
        acc, history = train_subject(subject_id, args.data_dir, args, device)
        all_accuracies.append(acc)
        all_histories[subject_id] = history
        
        # 记录到 TensorBoard
        for epoch, (train_loss, train_acc, val_loss, val_acc) in enumerate(
            zip(history['train_loss'], history['train_acc'], 
                history['val_loss'], history['val_acc'])
        ):
            writer.add_scalar(f'Subject_{subject_id}/train_loss', train_loss, epoch)
            writer.add_scalar(f'Subject_{subject_id}/train_acc', train_acc, epoch)
            writer.add_scalar(f'Subject_{subject_id}/val_loss', val_loss, epoch)
            writer.add_scalar(f'Subject_{subject_id}/val_acc', val_acc, epoch)
    
    # 统计结果
    accuracies = np.array(all_accuracies)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n{'='*60}")
    print(f"最终结果 (9 个受试者)")
    print(f"{'='*60}")
    for i, acc in enumerate(all_accuracies):
        print(f"受试者 {args.subjects[i]}: {acc:.4f}")
    print(f"\n平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # 保存结果
    with open(os.path.join(log_dir, 'results.txt'), 'w') as f:
        f.write(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
        f.write(f"Per-subject accuracies: {all_accuracies}\n")
        f.write(f"\nArgs: {vars(args)}\n")
    
    writer.add_text('results', f'Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}')
    writer.close()
    
    print(f"\n日志保存至: {log_dir}")
    print(f"模型检查点保存至: {args.save_dir}")


if __name__ == "__main__":
    main()
