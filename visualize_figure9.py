"""
论文 Figure 9 复现 (修复版)
Single-trial EEG feature relevance using gradient (DeepLIFT approximation)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import torch
import torch.nn as nn
from src.model.eegnet import EEGNet
from src.data.loader import BCIC2aDataset
import glob


def get_channel_positions(n_ch=22):
    """获取电极位置 (固定 22 个)"""
    ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
                'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_pos = []
    for ch in ch_names[:n_ch]:
        if ch in montage.ch_names:
            ch_pos.append(montage.get_positions()['ch_pos'][ch][:2])
        else:
            ch_pos.append([0, 0])
    return np.array(ch_pos), ch_names[:n_ch]


def plot_topoplot_safe(weights, ch_pos, ax, title, vmax=None):
    """安全绘制拓扑图，处理维度不匹配"""
    # 确保权重长度与电极位置一致 (截断到 22)
    if len(weights) > len(ch_pos):
        weights = weights[:len(ch_pos)]
    elif len(weights) < len(ch_pos):
        weights = np.pad(weights, (0, len(ch_pos) - len(weights)))
    
    if vmax is None:
        vmax = max(abs(weights.max()), abs(weights.min()))
    if vmax == 0:
        vmax = 1
    
    try:
        # 新版本 MNE
        im, _ = mne.viz.plot_topomap(weights, ch_pos, axes=ax, show=False,
                                     vmin=-vmax, vmax=vmax)
    except TypeError:
        try:
            # 旧版本 MNE
            im = mne.viz.plot_topomap(weights, ch_pos, axes=ax, show=False,
                                      vmin=-vmax, vmax=vmax)
        except:
            # 最旧版本
            im = mne.viz.plot_topomap(weights, ch_pos, axes=ax, show=False)
    ax.set_title(title, fontsize=9)
    return im


def compute_relevance_gradient(model, x, target_class, device):
    """使用梯度计算 relevance (DeepLIFT 近似)"""
    x = x.to(device).requires_grad_(True)
    
    logits = model(x)
    prob = torch.softmax(logits, dim=1)
    
    model.zero_grad()
    prob[0, target_class].backward()
    
    relevance = x.grad.detach().cpu().numpy()[0, 0]
    return relevance, prob[0, target_class].item()


def load_model(subject_id=3):
    """加载 EEGNet 模型"""
    checkpoint_dirs = ['./checkpoints', './outputs/checkpoints']
    for ckpt_dir in checkpoint_dirs:
        if os.path.exists(ckpt_dir):
            checkpoints = glob.glob(os.path.join(ckpt_dir, f'*subject_{subject_id}*.pth'))
            if not checkpoints:
                checkpoints = glob.glob(os.path.join(ckpt_dir, '*.pth'))
            if checkpoints:
                latest = max(checkpoints, key=os.path.getctime)
                checkpoint = torch.load(latest, map_location='cpu')
                
                data_dir = "/mnt/workspace/bci_eegnet_repro/data"
                dataset = BCIC2aDataset(data_dir, subject_id=subject_id)
                n_channels = dataset[0][0].shape[1]
                
                model = EEGNet(n_channels=n_channels, n_classes=4, dropout_rate=0.5)
                dummy = torch.randn(1, 1, n_channels, 501)
                with torch.no_grad():
                    _ = model(dummy)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                model.eval()
                return model, dataset
    return None, None


def main():
    print("=" * 60)
    print("复现论文 Figure 9: Single-trial feature relevance")
    print("受试者 3 (SMR dataset)")
    print("=" * 60)
    
    os.makedirs('./outputs/figure9', exist_ok=True)
    
    # 获取电极位置 (22 个)
    ch_pos, ch_names = get_channel_positions(22)
    print(f"电极位置: {len(ch_pos)} 个")
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, dataset = load_model(3)
    if model is None:
        print("❌ 未找到模型")
        return
    model = model.to(device)
    print("✅ 模型加载成功")
    
    X = dataset.X
    y = dataset.y
    print(f"数据: {X.shape}")
    
    # 计算预测
    print("计算预测...")
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for i in range(min(len(X), 100)):  # 只取前100个加速
            x_tensor = torch.from_numpy(X[i][np.newaxis, np.newaxis, :, :]).float().to(device)
            logits = model(x_tensor)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = np.argmax(prob)
            all_probs.append(prob)
            all_preds.append(pred)
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    
    # 找示例
    examples = []
    
    # 示例 A: Left (0), 正确, 高置信度
    left_correct = np.where((y[:100] == 0) & (all_preds == 0) & (all_probs[:, 0] > 0.8))[0]
    if len(left_correct) > 0:
        idx = left_correct[0]
        examples.append(('A', idx, 'Left', all_probs[idx, 0], True))
    
    # 示例 B: Right (1), 正确, 高置信度
    right_correct = np.where((y[:100] == 1) & (all_preds == 1) & (all_probs[:, 1] > 0.8))[0]
    if len(right_correct) > 0:
        idx = right_correct[0]
        examples.append(('B', idx, 'Right', all_probs[idx, 1], True))
    
    # 示例 C: 低置信度或错误
    low_conf = np.where((all_probs.max(axis=1) < 0.6))[0]
    if len(low_conf) > 0:
        idx = low_conf[0]
        true_label = ['Left', 'Right', 'Foot', 'Tongue'][y[idx]]
        examples.append(('C', idx, true_label, all_probs[idx, y[idx]], False))
    
    print(f"找到 {len(examples)} 个示例")
    
    # 生成图表
    fs = 250
    tmin, tmax = -0.5, 1.0
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for row, (ex_id, idx, true_label, prob, correct) in enumerate(examples[:3]):
        print(f"处理示例 {ex_id}: True={true_label}, P={prob:.3f}")
        
        x = X[idx]
        x_tensor = torch.from_numpy(x[np.newaxis, np.newaxis, :, :]).float().to(device)
        target_class = y[idx]
        
        relevance, pred_prob = compute_relevance_gradient(model, x_tensor, target_class, device)
        
        # 截取时间窗口
        start_sample = int((tmin + 0.5) * fs)
        end_sample = int((tmax + 0.5) * fs)
        start_sample = max(0, start_sample)
        end_sample = min(x.shape[1], end_sample)
        
        rel_window = relevance[:, start_sample:end_sample]
        n_times = rel_window.shape[1]
        time_axis = np.linspace(tmin, tmax, n_times)
        
        # 热力图
        vmax_rel = np.abs(rel_window).max()
        im = axes[row, 0].imshow(rel_window, aspect='auto', cmap='RdBu_r',
                                extent=[time_axis[0], time_axis[-1], 0, rel_window.shape[0]],
                                vmin=-vmax_rel, vmax=vmax_rel)
        axes[row, 0].set_ylabel('Channels')
        if row == 2:
            axes[row, 0].set_xlabel('Time (seconds)')
        axes[row, 0].set_title(f'P = {pred_prob:.2f}')
        
        # ~50ms 拓扑图
        idx_50ms = np.argmin(np.abs(time_axis - 0.05))
        plot_topoplot_safe(rel_window[:, idx_50ms], ch_pos, axes[row, 1],
                          f't ≈ {time_axis[idx_50ms]*1000:.0f}ms')
        
        # ~150ms 拓扑图
        idx_150ms = np.argmin(np.abs(time_axis - 0.15))
        plot_topoplot_safe(rel_window[:, idx_150ms], ch_pos, axes[row, 2],
                          f't ≈ {time_axis[idx_150ms]*1000:.0f}ms')
    
    # 添加列标签
    axes[0, 0].set_title('Relevance Heatmap', fontsize=11)
    axes[0, 1].set_title('t ≈ 50ms', fontsize=11)
    axes[0, 2].set_title('t ≈ 150ms', fontsize=11)
    
    # 添加行标签
    for row, (ex_id, idx, true_label, prob, correct) in enumerate(examples[:3]):
        axes[row, 0].set_ylabel(f'{ex_id}: {true_label}', fontsize=10, rotation=0, labelpad=40)
    
    plt.suptitle('Figure 9: Single-trial EEG feature relevance (DeepLIFT approximation)', fontsize=14)
    plt.tight_layout()
    plt.savefig('./outputs/figure9/figure9_combined.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 60)
    print("Figure 9 复现完成！")
    print(f"输出: ./outputs/figure9/figure9_combined.png")
    print("=" * 60)


if __name__ == "__main__":
    main()