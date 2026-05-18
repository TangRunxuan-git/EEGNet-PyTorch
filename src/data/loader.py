"""
BCI Competition IV 2a 数据集加载器
从训练集中划分训练集和验证集（因为官方测试集无标签）
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import mne


class BCIC2aDataset(Dataset):
    def __init__(self, data_dir, subject_id, tmin=0.5, tmax=2.5, l_freq=4.0, h_freq=None):
        self.data_dir = data_dir
        self.subject_id = subject_id
        self.tmin = tmin
        self.tmax = tmax
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.file_path = os.path.join(data_dir, f"A0{subject_id}T.gdf")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        self._load_and_process()
    
    def _load_and_process(self):
        raw = mne.io.read_raw_gdf(self.file_path, preload=True, verbose='ERROR')
        if self.h_freq is None:
            raw.filter(self.l_freq, None, fir_design='firwin', verbose='ERROR')
        else:
            raw.filter(self.l_freq, self.h_freq, fir_design='firwin', verbose='ERROR')
        
        events, event_dict = mne.events_from_annotations(raw, verbose='ERROR')
        
        target_stims = ['769', '770', '771', '772']   # 不是 768！
        target_codes = [event_dict[s] for s in target_stims if s in event_dict]
        
        if len(target_codes) != 4:
            raise ValueError(f"文件中只找到 {len(target_codes)} 类运动想象事件")
        
        # 筛选事件
        mask = np.isin(events[:, 2], target_codes)
        filtered_events = events[mask]
        
        # 检查每个类别都有样本
        for code in target_codes:
            if np.sum(filtered_events[:, 2] == code) == 0:
                raise ValueError(f"事件代码 {code} 没有样本")
        
        picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')
        epochs = mne.Epochs(raw, filtered_events, event_id=target_codes,
                            tmin=self.tmin, tmax=self.tmax, picks=picks,
                            baseline=None, preload=True, verbose='ERROR')
        
        X = epochs.get_data() * 1e6
        orig_codes = epochs.events[:, 2]
        label_map = {code: i for i, code in enumerate(target_codes)}
        y = np.array([label_map[code] for code in orig_codes])
        
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        print(f"Subject {self.subject_id}: {len(self.y)} trials, "
              f"{X.shape[1]} channels, {X.shape[2]} time points, "
              f"labels: {np.bincount(self.y)}")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx][np.newaxis, :, :]
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def get_dataloaders(data_dir, subject_id, batch_size=64, val_ratio=0.2, num_workers=0):
    """
    从训练集中划分训练集和验证集
    """
    full_dataset = BCIC2aDataset(data_dir, subject_id)
    
    # 划分
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader


if __name__ == "__main__":
    data_dir = "/mnt/workspace/bci_eegnet_repro/data"
    print("=" * 50)
    print("测试数据加载器 (从训练集划分验证集)")
    print("=" * 50)
    
    train_loader, val_loader = get_dataloaders(data_dir, subject_id=1, batch_size=32)
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    for batch_x, batch_y in train_loader:
        print(f"Batch输入形状: {batch_x.shape}")
        break
    
    print("\n✅ 数据加载器测试通过!")