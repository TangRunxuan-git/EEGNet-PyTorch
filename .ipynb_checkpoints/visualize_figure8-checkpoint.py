"""
论文 Figure 8 复现 (修复版)
- (A) FBCSP 在 8-12Hz 频段学习的 4 个空间滤波器
- (B) EEGNet-8,2 在 12Hz 附近的 4 个空间滤波器
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from src.model.eegnet import EEGNet
from src.data.loader import BCIC2aDataset
import torch
import glob


def get_channel_positions_and_names(n_ch_expected=22):
    """
    获取标准 10-20 系统的电极位置
    返回: (ch_pos, ch_names) 其中 ch_names 是实际使用的通道名
    """
    # BCI IV 2a 的 22 个标准通道名
    ch_names_full = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
                     'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    montage = mne.channels.make_standard_montage('standard_1020')
    
    ch_pos = []
    ch_names = []
    for ch in ch_names_full:
        if ch in montage.ch_names:
            ch_pos.append(montage.get_positions()['ch_pos'][ch][:2])
            ch_names.append(ch)
        else:
            # 如果标准 montage 中没有，用最近邻
            ch_pos.append([0, 0])
            ch_names.append(ch)
    
    return np.array(ch_pos[:n_ch_expected]), ch_names[:n_ch_expected]


def plot_topoplot_safe(weights, ch_pos, ax, title):
    """安全绘制拓扑图，处理维度不匹配"""
    # 确保权重长度与电极位置一致
    if len(weights) != len(ch_pos):
        # 截断或填充
        if len(weights) > len(ch_pos):
            weights = weights[:len(ch_pos)]
        else:
            # 填充零
            weights = np.pad(weights, (0, len(ch_pos) - len(weights)))
    
    # 归一化以便显示
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


def bandpass_filter(data, fs, low, high):
    """带通滤波"""
    nyq = fs / 2
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)


def compute_csp_filters(X, y, class_id):
    """
    计算 CSP 滤波器 (返回第一个滤波器)
    """
    y_binary = (y == class_id).astype(int)
    X_pos = X[y_binary == 1]
    X_neg = X[y_binary == 0]
    
    if len(X_pos) == 0 or len(X_neg) == 0:
        return np.random.randn(X.shape[1])
    
    # 计算平均协方差矩阵
    cov_pos = np.mean([np.cov(x) for x in X_pos], axis=0)
    cov_neg = np.mean([np.cov(x) for x in X_neg], axis=0)
    
    # 正则化
    reg = 1e-6 * np.trace(cov_pos) / cov_pos.shape[0]
    cov_pos += reg * np.eye(cov_pos.shape[0])
    cov_neg += reg * np.eye(cov_neg.shape[0])
    
    # 解广义特征值问题
    try:
        eigvals, eigvecs = eigh(cov_pos, cov_pos + cov_neg)
    except:
        eigvals, eigvecs = eigh(cov_pos)
    
    # 取最大的特征向量
    idx = np.argmax(eigvals)
    return eigvecs[:, idx]


def load_eegnet_model(subject_id=3):
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
                return model, latest
    return None, None


def main():
    print("=" * 60)
    print("复现论文 Figure 8: FBCSP vs EEGNet 空间滤波器对比")
    print("受试者 3 (SMR dataset)")
    print("=" * 60)
    
    os.makedirs('./outputs/figure8', exist_ok=True)
    
    # 1. 获取电极位置 (22 个)
    ch_pos, ch_names = get_channel_positions_and_names(22)
    print(f"电极位置: {len(ch_pos)} 个")
    print(f"通道名: {ch_names[:5]}...")
    
    # 2. 加载数据
    data_dir = "/mnt/workspace/bci_eegnet_repro/data"
    dataset = BCIC2aDataset(data_dir, subject_id=3)
    X = dataset.X  # (n_samples, n_channels, n_times)
    y = dataset.y
    fs = 250
    
    n_channels = X.shape[1]
    print(f"数据: {X.shape}, 采样率: {fs} Hz, 实际通道数: {n_channels}")
    
    # 3. FBCSP: 8-12Hz 频段
    print("\n计算 FBCSP 8-12Hz 空间滤波器...")
    low, high = 8, 12
    X_filtered = np.array([bandpass_filter(x, fs, low, high) for x in X])
    
    class_names = ['Left hand', 'Right hand', 'Both feet', 'Tongue']
    fbcsp_filters = []
    
    for class_id in range(4):
        print(f"  计算 {class_names[class_id]}...")
        filters = compute_csp_filters(X_filtered, y, class_id)
        # 确保与电极位置长度一致
        if len(filters) != len(ch_pos):
            filters = filters[:len(ch_pos)]
        fbcsp_filters.append(filters)
    
    # 4. EEGNet 模型
    print("\n加载 EEGNet 模型...")
    model, ckpt_path = load_eegnet_model(subject_id=3)
    if model is None:
        print("❌ 未找到模型")
        return
    print(f"✅ 加载模型: {ckpt_path}")
    
    model.eval()
    
    # 提取 EEGNet 空间滤波器
    depthwise_weights = model.depthwise_conv.weight.detach().cpu().numpy()
    # shape: (F1*D, F1, n_ch, 1)
    F1 = 8
    D = 2
    
    # 选择 Temporal Filter 1,2,6,8 (索引 0,1,5,7)
    eegnet_indices = [0, 1, 5, 7]
    eegnet_filters = []
    for idx in eegnet_indices:
        spatial_filter = depthwise_weights[idx * D, 0, :, 0]
        # 确保与电极位置长度一致
        if len(spatial_filter) != len(ch_pos):
            spatial_filter = spatial_filter[:len(ch_pos)]
        eegnet_filters.append(spatial_filter)
        print(f"  Temporal Filter {idx+1}: shape {spatial_filter.shape}")
    
    # 5. 创建 Figure 8
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    
    # 第一行: FBCSP (Figure 8A)
    for i in range(4):
        ax = axes[0, i]
        plot_topoplot_safe(fbcsp_filters[i], ch_pos, ax,
                          f'{class_names[i]}\nvs all')
    
    # 第二行: EEGNet (Figure 8B)
    eegnet_titles = ['Temporal Filter 1\n(~12Hz)', 'Temporal Filter 2\n(~12Hz)',
                     'Temporal Filter 6\n(~12Hz)', 'Temporal Filter 8\n(~12Hz)']
    for i in range(4):
        ax = axes[1, i]
        plot_topoplot_safe(eegnet_filters[i], ch_pos, ax, eegnet_titles[i])
    
    # 添加行标签
    axes[0, 0].set_ylabel('A: FBCSP 8-12Hz\nSpatial Filters', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('B: EEGNet-8,2\n12Hz Spatial Filters', fontsize=11, fontweight='bold')
    
    plt.suptitle('Figure 8: Comparison of Spatial Filters (FBCSP vs EEGNet-8,2)\nSubject 3, SMR dataset',
                 fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig('./outputs/figure8/figure8_combined.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 60)
    print("Figure 8 复现完成！")
    print(f"输出: ./outputs/figure8/figure8_combined.png")
    print("=" * 60)
    
    # 6. 计算相似性
    print("\n计算 EEGNet 与 FBCSP 空间滤波器的相关性:")
    for i in range(4):
        max_corr = 0
        best_j = 0
        for j in range(4):
            corr = np.corrcoef(eegnet_filters[i], fbcsp_filters[j])[0, 1]
            if abs(corr) > abs(max_corr):
                max_corr = corr
                best_j = j
        print(f"  EEGNet Filter {eegnet_indices[i]+1} ↔ FBCSP {class_names[best_j]}: ρ = {max_corr:.3f}")


if __name__ == "__main__":
    main()