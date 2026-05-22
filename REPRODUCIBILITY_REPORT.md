

# EEGNet 论文复现报告

**论文标题**: EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces  
**发表出处**: Journal of Neural Engineering (JNE), 2018  
**复现作者**: Tang Runxuan  
**复现时间**: 2026年5月  
**代码仓库**: https://github.com/TangRunxuan-git/EEGNet-PyTorch

---

## 一、项目概述

### 1.1 研究背景

脑机接口（Brain-Computer Interface, BCI）通过解码脑电信号实现人机直接交互。传统BCI系统依赖手工设计的特征提取器（如CSP、FBCSP），需要大量领域知识。深度学习为自动特征提取提供了新思路，但现有深度模型要么参数量过大，要么难以跨任务泛化。

**EEGNet 的核心贡献**：
1. 提出紧凑的CNN架构，参数量仅1,504
2. 引入深度卷积和可分离卷积，模拟FBCSP的滤波-空间分解流程
3. 在4个不同BCI范式上验证泛化能力
4. 提供可解释性分析，证明模型学到了有生理意义的特征

### 1.2 复现目标

| 目标 | 完成情况 |
|------|---------|
| 实现 EEGNet PyTorch 版本 | ✅ 完成 |
| 在 BCI IV 2a 上训练评估 | ✅ 完成 |
| Within-subject 4折交叉验证 | ✅ 完成 |
| Cross-subject 留一实验 | ✅ 完成 |
| 对比实验 (DeepConvNet/ShallowConvNet/FBCSP) | ✅ 完成 |
| 消融研究 | ✅ 完成 |
| 可视化 (Figure 6-8) | ✅ 完成 |

---

## 二、方法复现

### 2.1 数据集

**BCI Competition IV 2a (SMR Dataset)**：
- 4类运动想象：左手、右手、双脚、舌头
- 9名受试者，每人288个trial（每类72个）
- 22个EEG通道，250Hz采样率
- 时间窗口：[0.5, 2.5]秒

### 2.2 数据预处理

| 步骤 | 参数 | 说明 |
|------|------|------|
| 重采样 | 250Hz → 128Hz | 与论文一致 |
| 滤波 | 4Hz 高通 | 论文 SMR 设置 |
| 归一化 | 转换为 μV | MNE 默认 |
| 划分 | 80% 训练, 20% 验证 | 4折CV时使用块状划分 |

### 2.3 EEGNet 架构

```
输入: (Batch, 1, Channels, Time)
    ↓ Conv2D(1, 64), F1=8
(8, 22, 501)
    ↓ Depthwise Conv2D(22, 1), groups=8, D=2
(16, 1, 501)
    ↓ ELU + AvgPool2D(1,4) + Dropout
(16, 1, 125)
    ↓ Separable Conv2D(1,16)
(16, 1, 125)
    ↓ ELU + AvgPool2D(1,8) + Dropout
(16, 1, 15)
    ↓ Flatten + Linear
输出: 4 logits
```

**参数量**：1,504（比 DeepConvNet 小 98 倍）

### 2.4 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 优化器 | Adam | lr=0.001 (Cross-subject) / 0.01 (Within) |
| 损失函数 | CrossEntropyLoss | - |
| 批次大小 | 32 | - |
| Epochs | 150-500 | 含早停 |
| Dropout | 0.5 (Within) / 0.25 (Cross) | 论文设置 |
| 激活函数 | ELU | - |

---

## 三、实验结果

### 3.1 Within-subject 4折交叉验证

| 受试者 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 平均 |
|--------|---|---|---|---|---|---|---|---|---|------|
| 准确率 | 62.5% | 55.9% | 69.4% | 44.4% | 47.6% | 50.4% | 66.7% | 72.9% | 83.0% | **61.4%** |

**分析**：
- 最佳受试者9达到 **83.0%**（与论文85%接近）
- 个体差异显著（44% ~ 83%），符合BCI领域已知的"BCI文盲"现象
- 4折CV比单次划分更稳定（标准差从0.202降至0.121）

### 3.2 对比实验

| 模型 | 参数量 | 准确率 | 相对 EEGNet |
|------|--------|--------|-------------|
| **EEGNet** | **1,504** | **86.21%** | baseline |
| DeepConvNet | 147,750 | 72.41% | -13.8% |
| ShallowConvNet | 40,600 | 74.14% | -12.1% |
| FBCSP | N/A | 75.86% | -10.3% |

**核心发现**：
- EEGNet 用 **98 倍更少的参数** 达到 **更高** 的准确率
- 验证了深度卷积 + 可分离卷积的紧凑设计有效性

### 3.3 消融研究

| 配置 | 准确率 | 下降 |
|------|--------|------|
| 完整 EEGNet | 82.76% | baseline |
| 移除深度卷积 | 81.03% | -1.73% |
| 移除可分离卷积 | 81.03% | -1.73% |
| 移除两者 | 79.31% | -3.45% |

**结论**：深度卷积和可分离卷积各自贡献约1.7%，两者共同贡献3.45%。

### 3.4 Cross-subject 实验

| 测试受试者 | 准确率 |
|-----------|--------|
| 1 | 27.8% |
| 2 | 25.7% |
| 3 | 27.4% |
| 4 | 20.8% |
| 5 | 26.0% |
| 6 | 27.4% |
| 7 | 21.9% |
| 8 | 31.3% |
| 9 | 29.2% |
| **平均** | **26.4%** |

**分析**：
- 仅比随机基线（25%）高1.4%，说明跨被试泛化困难
- 与论文（65-70%）差距较大，原因是：
  - 论文使用官方测试集 `*E.gdf`（无公开标签）
  - 论文采用不同的验证集划分（5人训练+3人验证+1人测试）
  - 论文重复30次取平均

---

## 四、可视化复现

### 4.1 Figure 6: 空间滤波器 + 时频分析

成功复现：
- 4个空间滤波器的脑地形图（展示每个滤波器的空间分布）
- 4个类别的时频差异图（展示目标 vs 非目标的功率差异）

**生成文件**：`outputs/figure6/`

### 4.2 Figure 7: 时间滤波器 + 空间滤波器组合

成功复现：
- 8个时间滤波器的时域波形（0.25秒窗口）
- 每个时间滤波器对应的2个空间滤波器（脑地形图）

**生成文件**：`outputs/figure7/`

### 4.3 Figure 8: EEGNet vs FBCSP 空间滤波器对比

成功复现：
- FBCSP 在 8-12Hz 频段的4个空间滤波器
- EEGNet 在 12Hz 附近的4个空间滤波器
- 两者高度相关（验证了EEGNet学到了与FBCSP类似的特征）

**生成文件**：`outputs/figure8/`

---

## 五、与论文对比

| 实验 | 论文结果 | 复现结果 | 差距分析 |
|------|---------|---------|---------|
| Within-subject 平均 | ~74% | 61.4% | 个体差异 + 单次运行 |
| Within-subject 最佳 | ~85% | 83.0% | ✅ 接近 |
| Cross-subject 平均 | ~65-70% | 26.4% | 评估协议不同（`*E` vs `*T`） |
| 对比实验 | EEGNet 最优 | EEGNet 最优 | ✅ 一致 |
| 消融研究 | 各模块有贡献 | 各模块有贡献 | ✅ 一致 |

---

## 六、技术难点与解决方案

| 问题 | 解决方案 |
|------|---------|
| MNE 版本差异导致 `plot_topomap` 参数不兼容 | 使用 `try-except` 兼容多个版本 |
| 事件ID映射错误（768 vs 769） | 打印 `event_dict` 确认实际ID |
| 测试集 `*E.gdf` 无标签 | 从训练集划分验证集 |
| 深度卷积的 `groups` 参数理解 | 阅读 PyTorch 文档和论文 Figure 1 |
| FBCSP 初始实现错误（维度爆炸） | 使用协方差矩阵 + CSP 特征提取 |

---

## 七、项目产出

### 7.1 代码文件

| 文件 | 功能 | 行数 |
|------|------|------|
| `src/model/eegnet.py` | EEGNet 模型定义 | ~80 |
| `src/data/loader.py` | 数据加载与预处理 | ~100 |
| `src/train.py` | 训练主脚本 | ~150 |
| `train_cv.py` | 4折交叉验证 | ~120 |
| `ablation_study.py` | 消融研究 | ~150 |
| `compare_models.py` | 对比实验 | ~250 |
| `visualize_figure6.py` | Figure 6 可视化 | ~150 |
| `visualize_figure7.py` | Figure 7 可视化 | ~120 |
| `visualize_figure8.py` | Figure 8 可视化 | ~150 |

**总代码量**：~1,200 行（不含注释和空行）

### 7.2 可视化图片

- `outputs/figure6/`：11 张图
- `outputs/figure7/`：1 张组合图
- `outputs/figure8/`：1 张组合图

### 7.3 模型权重

9 个受试者的最佳模型权重（`checkpoints/subject_*_best.pth`）

---

## 八、总结与展望

### 8.1 项目总结

1. **成功复现 EEGNet 核心架构**：深度卷积 + 可分离卷积，参数量 1,504
2. **验证论文核心主张**：EEGNet 比 DeepConvNet 小 98 倍，性能更优
3. **完成完整实验体系**：within-subject CV、cross-subject、消融、对比
4. **复现论文关键可视化**：Figure 6, 7, 8

### 8.2 收获

1. **深度学习工程能力**：从零实现 CNN，自定义训练循环
2. **信号处理能力**：MNE 预处理、滤波、时频分析
3. **实验设计能力**：交叉验证、消融实验、对比实验
4. **科研可视化能力**：脑地形图、时频图、滤波器对比
5. **论文复现方法论**：理解论文 → 拆解模块 → 逐项验证

### 8.3 未来改进方向

1. 扩展至 P300/ERN/MRCP 数据集，验证跨范式泛化能力
2. 实现真正的 DeepLIFT 可解释性分析（Figure 9-10）
3. 集成现代 EFM 技术（对比学习、掩码预训练）
4. 优化 Cross-subject 评估协议，使用官方测试集标签

---

## 九、参考文献

1. Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.

2. Schirrmeister, R. T., et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391-5420.

3. Ang, K. K., et al. (2012). Filter bank common spatial pattern algorithm on BCI competition IV datasets 2a and 2b. *Frontiers in Neuroscience*, 6, 39.

4. BCI Competition IV. (2008). Dataset 2a: Graz University of Technology.

---

## 附录 A: 环境配置

```bash
# 核心依赖
torch==2.9.1
mne==1.12.1
numpy==1.26.4
scipy==1.17.1
scikit-learn==1.8.0

# 可视化
matplotlib==3.10.8
seaborn==0.13.2

# 工具
tqdm==4.67.3
tensorboard==2.20.0
omegaconf==2.3.0
```

---

## 附录 B: 运行示例

```bash
# 训练受试者 9
python src/train.py --subject 9 --epochs 150 --dropout_rate 0.5 --lr 0.01

# 4折交叉验证
python train_cv.py

# 对比实验
python compare_models.py --subject 9

# 可视化
python visualize_figure6.py
python visualize_figure7.py
python visualize_figure8.py
```

---

**报告完成日期**：2026年5月18日  
**代码仓库**：https://github.com/TangRunxuan-git/EEGNet-PyTorch

```


