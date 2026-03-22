# 技术路线：基于选择性门控时空解耦的残差感知混合精度量化 (PTQ 范式) —— 针对 VideoMamba

本技术路线针对视频动作识别中状态空间模型（SSM）在时序扫描中的量化误差累积与状态崩溃问题，提出一种适配 VideoMamba 架构的**训练后量化（Post-Training Quantization, PTQ）**解决方案。核心创新在于利用 Mamba 模块的选择性门控参数进行隐式的时空动态解耦。

---

## 第一阶段：扫描序列提取与时空网格重构 (Sequence Extraction & Spatiotemporal Reconstruction)

**目标**：严格处理 VideoMamba 的 1D 扫描序列，提取中心化 CLS Token，并将序列重构为时空维度的物理网格以便进行特征分组。

1. **中心 CLS Token 剥离与重排**：
   VideoMamba Block 的输入激活值张量展开为一维序列 $X_{in} \in \mathbb{R}^{B \times L \times C}$，其中 $L = T \cdot S + 1$。识别并剥离位于**序列中间**的 CLS Token，提取纯时空特征序列 $X_{st} \in \mathbb{R}^{B \times (T \cdot S) \times C}$。

2. **时空组与扫描窗划分**：
   将 1D 扫描序列根据物理意义 Reshape 回 3D 时空张量 $\mathbb{R}^{B \times T \times S \times C}$。将总帧数 $T$ 划分为 $G$ 个时空组（Groups），每组包含 $K = T/G$ 帧，第 $g$ 组特征张量记为 $X_g \in \mathbb{R}^{B \times K \times S \times C}$。

---

## 第二阶段：校准阶段 (Calibration) —— 门控分布统计与阈值标定

**目标**：利用少量无标签校准数据集，提取 FP32 Mamba 模块的组内残差与**选择性步长（Step Size $\Delta$）**的时空分布指标，拟合得出归一化统计区间与决策阈值 $\tau$。

### 1. 组内残差幅度 ($R_g$) 与 门控时空解耦 ($E_{spa, g}, E_{temp, g}$)
衡量第 $g$ 组内 SSM 递推过程中的特征变化剧烈程度与信息冗余度：
- **残差幅度**：计算组内特征相对组平均锚点 $\bar{x}_g$ 的残差 L2 范数期望：
  $$ R_g = \mathbb{E}_{b \sim B} \left[ \frac{1}{M} \sum_{i=1}^{M} \| x_{g, i}^{(b)} - \bar{x}_g^{(b)} \|_2 \right] $$
- **空间门控熵 (Spatial Gating Entropy)**：在 Mamba 中，步长 $\Delta$ 决定了对当前 Token 信息的吸收率。提取该组内离散化门控张量 $\Delta_g \in \mathbb{R}^{B \times K \times S \times D}$。固定时间步 $k$，对同一帧内的空间块计算归一化响应（Softmax）的香农熵（低熵表示 SSM 在空间上高度聚焦特定目标）：
  $$ E_{spa, g} = \mathbb{E}_{k} \left[ - \sum_{s=1}^{S} \tilde{\Delta}_{k,s} \log_2 (\tilde{\Delta}_{k,s} + \epsilon) \right], \quad \text{其中} \; \tilde{\Delta} = \text{Softmax}_S(\Delta_g) $$
- **时间门控熵 (Temporal Gating Entropy)**：同理，固定空间位置 $s$，对时间维度上的 $\Delta_g$ 响应计算香农熵（低熵表示局部发生了剧烈运动，SSM 状态更新频繁）：
  $$ E_{temp, g} = \mathbb{E}_{s} \left[ - \sum_{k=1}^{K} \tilde{\Delta}_{k,s} \log_2 (\tilde{\Delta}_{k,s} + \epsilon) \right], \quad \text{其中} \; \tilde{\Delta} = \text{Softmax}_K(\Delta_g) $$

### 2. 归一化统计与联合阈值标定 ($\tau$)
在校准集上获取上述三项指标的极值区间，并进行归一化（$\widetilde{R}_g, \widetilde{E}_{spa, g}, \widetilde{E}_{temp, g}$）。构建 VideoMamba 的综合敏感度得分 $S_g$：
$$ S_g = \alpha \widetilde{R}_g + \beta \left( 1 - \widetilde{E}_{spa, g} \right) + \gamma \left( 1 - \widetilde{E}_{temp, g} \right) $$
提取所有校准样本分布的前 $p\%$ 分位数作为该 Mamba Block 的截断阈值 $\tau$，并预估 1D 扫描序列的误差衰减系数 $\lambda$。

---

## 第三阶段：快速评估阶段 (Quick Eval) —— 动态分配与块级位宽

**目标**：在迷你验证集（Quick Eval Set）上运行推理，统计各 Mamba Block 的高敏感激活比例，进而静态分配全局量化位宽。

1. **动态敏感度得分统计**：
   在前向传播中，基于 Phase 2 的归一化区间，计算当前组得分 $\hat{S}_g$ 并执行二值阶跃函数：
   $$ Z_g = \mathbb{I}(\hat{S}_g > \tau) $$

2. **生成位宽分配表 (`block_bits`)**：
   统计每个 Mamba Block（含输入/输出投影、Conv1D 及 SSM 核心参数）内触发 $Z_g=1$ 的平均比率 `group_high_ratio`。
   静态路由策略：若某 Block 高敏感触发率 $\ge 30\%$，则判定为**状态密集型算子**，分配较高位宽（如 $b_{block} = 8$）；否则分配低位宽（如 $b_{block} = 4$）。产出配置字典 `block_bits`。

---

## 第四阶段：全量评估阶段 (Full Eval) —— 模拟量化与状态误差补偿

**目标**：在完整测试集上，根据 `block_bits` 对权重和激活执行 Fake Quantization，并应用适配 1D 扫描轨迹的误差前馈机制以减轻精度损失。

### 1. 权重参数的模拟量化 (Weight Fake Quantization)
针对 Mamba 层特有权重（$W_{in}, W_{out}, \Delta_{proj}, B_{proj}, C_{proj}$），按 `block_bits` 执行模拟量化：
$$ W_{sim} = \text{Clamp}(\text{Round}(W / s), q_{min}, q_{max}) \cdot s $$

### 2. 混合精度状态与 1D 扫描误差前馈 (1D-Scan Error Forwarding)
不同于 Attention 的层间跳跃，SSM 存在深度的层间残差与序列维度的隐状态 ($h_t$) 递推。对此施加双向误差补偿：

- 当 Block 为 **INT8 ($b_{block}=8$)**：计算当前特征的量化截断误差 $e_g = X_g - \tilde{X}_{INT8}$。将误差 Carry 不仅沿网络深度传递，更沿着 **1D 扫描轨迹**叠加至下一个序列组的输入中：
  $$ X_{g+1} \leftarrow X_{g+1} + \lambda \cdot e_g $$
- 当 Block 为 **INT4 ($b_{block}=4$)**：由于 INT4 极低精度容易导致 SSM 隐状态崩溃（State Collapse），引入**隐状态残差复用（Hidden-State Residual Reuse）**。利用上一时刻的高精度反量化输出作为稳定锚点：
  $$ Y_g = \tilde{Y}_{g-1} + \text{Dequant}(\text{Quant}(\Delta X_g, b_{block}=4)) $$

通过此流程，无需重训练（Zero Retraining）即可在保留时空解耦物理意义的前提下，实现 VideoMamba 在目标混合精度下的高精度量化推理。