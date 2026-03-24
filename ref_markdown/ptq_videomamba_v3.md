# 技术路线：基于时空门控熵的 VideoMamba 算子级真实权重量化 (PTQ 范式)

本技术路线针对视频动作识别中状态空间模型（SSM）对低精度极度敏感导致“状态崩溃（State Collapse）”的问题，提出一种适配 VideoMamba 架构的**训练后真实权重量化（Post-Training Real Weight-Only Quantization）**解决方案。

核心创新在于：利用 Mamba 模块的选择性门控参数进行隐式的时空动态解耦；摒弃传统的 Block 级（块级）“一刀切”与缺乏显存收益的模拟量化（Fake Quantization），转而实施**算子级（Component-Level）的精准解耦与真实混合精度压缩（W8A16 / W4A16）**，在无损调用官方高效 CUDA 算子的同时，实现显存的真实大幅缩减。

---

## 第一阶段：扫描序列提取与时空网格重构 (Sequence Extraction & Spatiotemporal Reconstruction)

**目标**：严格处理 VideoMamba 的 1D 扫描序列，提取中心化 CLS Token，并将序列重构为时空维度的物理网格以便进行特征分组与统计。

1. **中心 CLS Token 剥离与重排**：
   VideoMamba Block 的输入激活值张量展开为一维序列 $X_{in} \in \mathbb{R}^{B \times L \times C}$，其中 $L = T \cdot S + 1$。识别并剥离位于**序列中间**的 CLS Token，提取纯时空特征序列 $X_{st} \in \mathbb{R}^{B \times (T \cdot S) \times C}$。

2. **时空组与扫描窗划分**：
   将 1D 扫描序列根据物理意义 Reshape 回 3D 时空张量 $\mathbb{R}^{B \times T \times S \times C}$。将总帧数 $T$ 划分为 $G$ 个��空组（Groups），每组包含 $K = T/G$ 帧，第 $g$ 组特征张量记为 $X_g \in \mathbb{R}^{B \times K \times S \times C}$。

---

## 第二阶段：校准阶段 (Calibration) —— 组件级解耦与门控时空熵评估

**目标**：利用少量无标签校准数据集，提取 Mamba 模块的核心控制参数，量化各层算子的时空信息冗余度，拟合决策阈值。

### 1. 算子级敏感度划分与绝对保护策略
VideoMamba Block 内部组件对量化敏感度差异巨大：
- **绝对保护区（维持 FP16/BF16 精度）**：步长与状态生成投影层（`dt_proj`, `x_proj`）、一维卷积（`Conv1D`）以及底层的 `selective_scan_cuda` 算子。这些控制着 SSM 隐状态 $h_t$ 的递推，参数量不足 20%，但一旦量化极易引发状态崩溃。
- **显存压缩区（目标量化层）**：输入/输出特征投影层（`in_proj`, `out_proj`）。占据 80%+ 参数量与显存，具有较高的通道冗余度。

### 2. 组内残差幅度 ($R_g$) 与 门控时空解耦 ($E_{spa, g}, E_{temp, g}$)
提取校准集上 Mamba 离散化门控张量 $\Delta_g \in \mathbb{R}^{B \times K \times S \times D}$ 进行解耦分析：
- **残差幅度**：计算组内特征相对组平均锚点 $\bar{x}_g$ 的残差 L2 范数期望 $R_g$。
- **空间门控熵 (Spatial Gating Entropy)**：固定时间步 $k$，对同一帧内的空间块计算归一化响应的香农熵（低熵表示高度聚焦特定空间目标）：
  $$ E_{spa, g} = \mathbb{E}_{k} \left[ - \sum_{s=1}^{S} \tilde{\Delta}_{k,s} \log_2 (\tilde{\Delta}_{k,s} + \epsilon) \right], \quad \tilde{\Delta} = \text{Softmax}_S(\Delta_g) $$
- **时间门控熵 (Temporal Gating Entropy)**：固定空间位置 $s$，对时间维度响应计算香农熵（低熵表示局部发生剧烈运动）：
  $$ E_{temp, g} = \mathbb{E}_{s} \left[ - \sum_{k=1}^{K} \tilde{\Delta}_{k,s} \log_2 (\tilde{\Delta}_{k,s} + \epsilon) \right], \quad \tilde{\Delta} = \text{Softmax}_K(\Delta_g) $$

### 3. 归一化统计与决策打分 ($S_g$)
对上述指标归一化后，构建综合冗余度得分 $S_g$：
$$ S_g = \alpha \widetilde{R}_g + \beta \left( 1 - \widetilde{E}_{spa, g} \right) + \gamma \left( 1 - \widetilde{E}_{temp, g} \right) $$

---

## 第三阶段：真实权重压缩与持久化 (Real Weight Compression & Serialization)

**目标**：依据冗余度得分，仅对压缩区的投影层（`in_proj`, `out_proj`）执行真实的 Weight-Only 量化（截断为整型），彻底抛弃全精度浮点权重，实现显存的大幅缩减。

1. **混合位宽静态分配**：
   - 提取校准集分布的前 $p\%$ 分位数作为阈值 $\tau$。
   - 若某 Block 的高得分比例超标（动态突变多，冗余度低），为该 Block 的投影层分配 **INT8 (W8A16)**。
   - 若某 Block 得分普遍较低（背景冗余高），为投影层分配极低精度 **INT4 (W4A16)**。

2. **真实量化与物理打包 (Packing)**：
   对目标权重 $W$ 提取 FP16 缩放因子 $s$，执行严格的物理截断：
   $$ W_{int} = \text{Clamp}(\text{Round}(W / s), q_{min}, q_{max}) $$
   - **INT8 转换**：直接保存为 `torch.int8` 数据类型。
   - **INT4 打包**：通过位运算（Bit-shifting）将两个 INT4 数值合并为一个 `torch.uint8` 保存。

3. **量化权重持久化**：
   生成仅包含 `q_weight` (整型张量)、`scale` (FP16 缩放因子) 以及被保护核心参数 (FP16) 的全新 `state_dict`，导出 `.pth` 模型文件，其体积相比原全精度模型缩小至 1/4 到 1/2。

---

## 第四阶段：构建量化推理引擎与全量评测 (Quantized Inference & Evaluation)

**目标**：重写 VideoMamba 网络，在完整测试集上进行混合精度推理，实现真实常驻显存的降低。

### 1. 自定义量化算子替换 (Custom QuantizedLinear)
在推理代码中，将原有的 `nn.Linear` 替换为自定义的 `QuantizedLinear` 层：
- **常驻显存控制**：层内通过 `register_buffer` 加载预先保存的 INT8 / UINT8 权重，大幅降低显存占用（Static Memory Footprint）。
- **动态反量化 (On-the-fly Dequantization)**：在前向传播时：
  $$ W_{fp16} = W_{int} \cdot s $$
  随后通过 `F.linear(X, W_{fp16})` 完成计算。
- **无缝对接官方 CUDA 核心**：由于激活值 $X$ 在整个传递过程中始终保持为 FP16，当数据流转到 SSM 核心算子时，依然可以直接、无损地调用 VideoMamba 官方的 `selective_scan_cuda` 加速引擎。

### 2. 全量评测指标
通过此真实量化部署流程，在避免由于隐状态量化引起 State Collapse 的前提下，验证以下指标：
- **任务精度**：确保 Top-1 / Top-5 精度在 W8A16/W4A16 混合精度下不发生断崖式下跌（基线 83.4%，目标保持在 82.0%+）。
- **显存收益**：通过 `torch.cuda.memory_allocated()` 验证常驻显存占用的真实下降。