"""
论文 Figure 7 复现
- Top: 8 个时间滤波器的波形 (0.25秒窗口)
- Bottom: 每个时间滤波器对应的两个空间滤波器 (脑地形图)
使用 within-subject 训练的 EEGNet-8,2 模型
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_topomap
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


def plot_topoplot_on_ax(weights, ch_pos, ax, title):
    """在指定轴上绘制空间拓扑图"""
    try:
        im, _ = mne.viz.plot_topomap(weights, ch_pos, axes=ax, show=False)
    except TypeError:
        im = mne.viz.plot_topomap(weights, ch_pos, axes=ax, show=False)
    ax.set_title(title, fontsize=8)
    return im


def load_model_for_subject(subject_id=3):
    """加载指定受试者的模型"""
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
                return model, latest
    return None, None


def estimate_frequency(temporal_kernel, fs=128):
    """
    估计时间滤波器的中心频率
    通过计算零交叉点数量
    """
    # 去除 DC 分量
    kernel = temporal_kernel - np.mean(temporal_kernel)
    # 找零交叉点
    zero_crossings = np.where(np.diff(np.signbit(kernel)))[0]
    if len(zero_crossings) >= 2:
        # 周期数 ≈ 零交叉点数 / 2
        n_cycles = len(zero_crossings) / 2
        duration = len(kernel) / fs  # 秒
        freq = n_cycles / duration
        return freq
    return 0


def main():
    print("=" * 60)
    print("复现论文 Figure 7: EEGNet-8,2 特征可视化")
    print("受试者 3 (SMR dataset)")
    print("=" * 60)
    
    os.makedirs('./outputs/figure7', exist_ok=True)
    
    # 1. 获取电极位置
    ch_pos, ch_names = get_channel_positions(22)
    
    # 2. 加载受试者 3 的模型
    subject_id = 3
    model, ckpt_path = load_model_for_subject(subject_id)
    if model is None:
        print(f"❌ 未找到受试者 {subject_id} 的模型，尝试创建新模型")
        data_dir = "/mnt/workspace/bci_eegnet_repro/data"
        dataset = BCIC2aDataset(data_dir, subject_id=subject_id)
        n_channels = dataset[0][0].shape[1]
        model = EEGNet(n_channels=n_channels, n_classes=4, dropout_rate=0.5)
    else:
        print(f"✅ 加载模型: {ckpt_path}")
    
    model.eval()
    
    # 3. 提取参数
    temporal_weights = model.conv1.weight.detach().cpu().numpy()  # (F1, 1, 1, kern_len)
    depthwise_weights = model.depthwise_conv.weight.detach().cpu().numpy()  # (F1*D, F1, n_ch, 1)
    
    F1 = temporal_weights.shape[0]  # 应该是 8
    D = depthwise_weights.shape[0] // F1  # 应该是 2
    
    print(f"F1={F1}, D={D}")
    
    # 4. 创建 Figure 7: 3 行 x 8 列
    fig = plt.figure(figsize=(16, 8))
    
    # 定义网格: 3 行，每行 8 列，高度比例 [1, 1, 1]
    gs = fig.add_gridspec(3, 8, hspace=0.4, wspace=0.3,
                          height_ratios=[1, 1, 1])
    
    fs = 128  # 重采样后的采样率
    window_duration = 0.25  # 0.25 秒窗口
    n_samples_window = int(fs * window_duration)  # 32 个采样点
    
    # 论文 Figure 7 显示的是 0.25 秒窗口
    # 我们的时间核是 64 个采样点 (0.5 秒)，取前 32 个点显示
    
    for i in range(F1):
        # === 第一行: 时间滤波器 (0.25 秒窗口) ===
        ax1 = fig.add_subplot(gs[0, i])
        temporal_kernel = temporal_weights[i, 0, 0, :n_samples_window]
        time_axis = np.linspace(0, window_duration, n_samples_window)
        ax1.plot(time_axis, temporal_kernel, 'b-', linewidth=1.5)
        ax1.set_xlim(0, window_duration)
        ax1.set_ylim(-0.3, 0.3)
        ax1.set_xticks([0, 0.125, 0.25])
        ax1.set_yticks([-0.2, 0, 0.2])
        ax1.tick_params(axis='both', labelsize=7)
        ax1.set_title(f'Temp. Filter {i+1}', fontsize=9)
        if i == 0:
            ax1.set_ylabel('Amplitude', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 计算并打印估计频率
        freq = estimate_frequency(temporal_kernel, fs)
        if freq > 0:
            print(f"  Filter {i+1}: {freq:.1f} Hz")
        
        # === 第二行: 第一个空间滤波器 ===
        ax2 = fig.add_subplot(gs[1, i])
        spatial1 = depthwise_weights[i * D, 0, :22, 0]  # 第一个空间滤波器
        try:
            im1, _ = mne.viz.plot_topomap(spatial1, ch_pos, axes=ax2, show=False)
        except:
            im1 = mne.viz.plot_topomap(spatial1, ch_pos, axes=ax2, show=False)
        ax2.set_title(f'Space Filter {i+1}.1', fontsize=8)
        ax2.tick_params(axis='both', labelsize=6)
        
        # === 第三行: 第二个空间滤波器 ===
        ax3 = fig.add_subplot(gs[2, i])
        spatial2 = depthwise_weights[i * D + 1, 0, :22, 0]  # 第二个空间滤波器
        try:
            im2, _ = mne.viz.plot_topomap(spatial2, ch_pos, axes=ax3, show=False)
        except:
            im2 = mne.viz.plot_topomap(spatial2, ch_pos, axes=ax3, show=False)
        ax3.set_title(f'Space Filter {i+1}.2', fontsize=8)
        ax3.tick_params(axis='both', labelsize=6)
    
    plt.suptitle('Figure 7: EEGNet-8,2 Feature Visualization (Subject 3, SMR dataset)', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig('./outputs/figure7/figure7_combined.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 60)
    print("Figure 7 复现完成！")
    print(f"输出: ./outputs/figure7/figure7_combined.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
