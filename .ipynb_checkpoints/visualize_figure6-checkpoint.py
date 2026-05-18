"""
论文 Figure 6 正确复现
- (A) Spatial topoplots for each spatial filter
- (B) Mean wavelet time-frequency difference between target and non-target trials
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_topomap
from scipy import signal
from src.model.eegnet import EEGNet
from src.data.loader import BCIC2aDataset
import glob


def get_channel_positions(n_ch=22):
    """获取电极位置"""
    ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
                'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_pos = np.zeros((len(ch_names), 2))
    for i, ch in enumerate(ch_names[:n_ch]):
        if ch in montage.ch_names:
            ch_pos[i] = montage.get_positions()['ch_pos'][ch][:2]
    return ch_pos, ch_names[:n_ch]


def plot_topoplot_single(weights, ch_pos, ch_names, ax, title):
    """空间拓扑图 - 兼容不同 MNE 版本"""
    try:
        # 新版本 MNE (1.x)
        im, _ = plot_topomap(weights, ch_pos, axes=ax, show=False, names=ch_names)
    except TypeError:
        try:
            # 旧版本 MNE
            im, _ = plot_topomap(weights, ch_pos, axes=ax, show=False)
        except:
            # 更旧版本
            im = plot_topomap(weights, ch_pos, axes=ax, show=False)
    ax.set_title(title, fontsize=10)
    return im


def plot_topoplot_save(weights, ch_pos, ch_names, save_path, title):
    """保存单个空间拓扑图"""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    try:
        im, _ = plot_topomap(weights, ch_pos, axes=ax, show=False, names=ch_names)
    except TypeError:
        try:
            im, _ = plot_topomap(weights, ch_pos, axes=ax, show=False)
        except:
            im = plot_topomap(weights, ch_pos, axes=ax, show=False)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_time_frequency(data, fs=128):
    """计算时频表示"""
    if len(data.shape) == 3:
        avg_data = np.mean(data, axis=(0, 1))
    else:
        avg_data = np.mean(data, axis=0)
    
    nperseg = int(fs * 0.5)
    noverlap = nperseg // 2
    
    f, t, Sxx = signal.spectrogram(avg_data, fs=fs, nperseg=nperseg,
                                   noverlap=noverlap, mode='psd', window='hann')
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # 只保留 4-40Hz
    freq_mask = (f >= 4) & (f <= 40)
    f = f[freq_mask]
    Sxx_db = Sxx_db[freq_mask, :]
    
    return f, t, Sxx_db


def plot_time_frequency_diff(tf_target, tf_nontarget, f, t, save_path, title):
    """时频差异图"""
    diff = tf_target - tf_nontarget
    
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.pcolormesh(t, f, diff, shading='gouraud', cmap='RdBu_r')
    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(4, 30)
    plt.colorbar(im, ax=ax, label='Power difference (dB)', shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def load_latest_model():
    """加载训练好的模型"""
    checkpoint_dirs = ['./checkpoints', './outputs/checkpoints']
    for ckpt_dir in checkpoint_dirs:
        if os.path.exists(ckpt_dir):
            checkpoints = glob.glob(os.path.join(ckpt_dir, '*.pth'))
            if checkpoints:
                latest = max(checkpoints, key=os.path.getctime)
                checkpoint = torch.load(latest, map_location='cpu')
                
                data_dir = "/mnt/workspace/bci_eegnet_repro/data"
                dataset = BCIC2aDataset(data_dir, subject_id=9)
                n_channels = dataset[0][0].shape[1]
                
                model = EEGNet(n_channels=n_channels, n_classes=4, dropout_rate=0.5)
                dummy = torch.randn(1, 1, n_channels, 501)
                with torch.no_grad():
                    _ = model(dummy)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                return model, latest
    return None, None


def main():
    print("=" * 60)
    print("复现论文 Figure 6: Target vs Non-target 时频差异")
    print("=" * 60)
    
    os.makedirs('./outputs/figure6', exist_ok=True)
    
    # 1. 电极位置
    ch_pos, ch_names = get_channel_positions(22)
    print(f"电极位置: {len(ch_names)} 个")
    
    # 2. 加载模型
    model, ckpt_path = load_latest_model()
    if model is None:
        print("❌ 未找到模型")
        return
    print(f"✅ 加载模型: {ckpt_path}")
    model.eval()
    
    # 3. 提取空间滤波器 (Figure 6A)
    spatial_weights = model.depthwise_conv.weight.detach().cpu().numpy()
    n_filters = min(spatial_weights.shape[0], 4)
    
    print(f"\n生成 Figure 6A: {n_filters} 个空间滤波器拓扑图")
    for i in range(n_filters):
        w = spatial_weights[i, 0, :22, 0]
        save_path = f'./outputs/figure6/spatial_filter_{i+1}.png'
        plot_topoplot_save(w, ch_pos, ch_names, save_path, f'Spatial Filter {i+1}')
        print(f"  ✅ {save_path}")
    
    # 4. 加载数据 (用于 Figure 6B)
    data_dir = "/mnt/workspace/bci_eegnet_repro/data"
    dataset = BCIC2aDataset(data_dir, subject_id=9)
    X = dataset.X
    y = dataset.y
    
    # 二分类: Target = Left (0), Non-target = others
    target_class = 0
    y_binary = (y == target_class).astype(np.int64)
    class_names = ['Left', 'Right', 'Foot', 'Tongue']
    print(f"\n生成 Figure 6B: Target={class_names[target_class]} vs Non-target")
    
    X_target = X[y_binary == 1]
    X_nontarget = X[y_binary == 0]
    
    fs = 128
    f, t, tf_target = compute_time_frequency(X_target, fs)
    _, _, tf_nontarget = compute_time_frequency(X_nontarget, fs)
    
    print(f"  时频图维度: f={len(f)}, t={len(t)}")
    
    # 生成 4 个时频图
    for i in range(n_filters):
        save_path = f'./outputs/figure6/time_frequency_filter_{i+1}.png'
        plot_time_frequency_diff(tf_target, tf_nontarget, f, t, save_path,
                                  title=f'Filter {i+1}: {class_names[target_class]} vs Others')
        print(f"  ✅ {save_path}")
    
    # 5. 生成组合图
    print("\n生成组合图 (Figure 6 排版)")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 第一行: 空间拓扑图
    for i in range(n_filters):
        w = spatial_weights[i, 0, :22, 0]
        plot_topoplot_single(w, ch_pos, ch_names, axes[0, i], f'Spatial Filter {i+1}')
    
    # 第二行: 时频差异图
    diff = tf_target - tf_nontarget
    for i in range(n_filters):
        im = axes[1, i].pcolormesh(t, f, diff, shading='gouraud', cmap='RdBu_r')
        axes[1, i].set_xlabel('Time (seconds)')
        axes[1, i].set_ylabel('Frequency (Hz)')
        axes[1, i].set_title(f'Filter {i+1}: Target - Non-target')
        axes[1, i].set_ylim(4, 30)
        plt.colorbar(im, ax=axes[1, i], label='dB', shrink=0.6)
    
    plt.suptitle('Figure 6: EEGNet Feature Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig('./outputs/figure6/figure6_combined.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ ./outputs/figure6/figure6_combined.png")
    
    print("\n" + "=" * 60)
    print("Figure 6 复现完成！")
    print("输出目录: ./outputs/figure6/")
    print("=" * 60)


if __name__ == "__main__":
    main()