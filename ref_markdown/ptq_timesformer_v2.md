# 技术路线：基于熵解耦注意力的残差感知混合精度量化 (PTQ 范式)

本技术路线针对视频动作识别中量化误差在时间维度上跨帧传播与累积的问题，提出一种基于**训练后量化（Post-Training Quantization, PTQ）**的解决方案。依托 TimeSformer 的分离时空注意力（Divided Space-Time Attention）架构，整体方案分为四个核心阶段：时空分组、校准标定、快速评估（位宽分配）以及全量评估（模拟量化与补偿）。

---

## 第一阶段：视频序列定义与时空分组 (Sequence Definition & Grouping)

**目标**：严格处理 TimeSformer 原生 Token 序列，剥离 CLS Token 并完成时空维度重排与分组。

1. **输入特征重排**：
   Transformer Block 的输入激活值张量为 $X_{in} \in \mathbb{R}^{B \times (1 + T \cdot S) \times C}$。剥离首位的 CLS Token，提取空间序列 $X_{spa} \in \mathbb{R}^{B \times (T \cdot S) \times C}$。由于 TimeSformer 原生排列顺序为 $(H, W, T)$ 交错，需将其重排（Rearrange）为时空独立格式 $X \in \mathbb{R}^{B \times T \times S \times C}$。

2. **时空分组**：
   将总帧数 $T$ 沿时间维度均匀划分为 $G$ 个连续的时空组（Groups），每组包含 $K = T/G$ 帧。第 $g$ 组特征张量记为 $X_g \in \mathbb{R}^{B \times K \times S \times C}$，用于后续的敏感度评估。

---

## 第二阶段：校准阶段 (Calibration) —— 分布统计与阈值标定

**目标**：利用少量无标签校准数据集，提取全精度（FP32）模型的组级度量指标，拟合得出归一化统计区间与决策阈值 $\tau$。

### 1. 组内残差幅度 ($R_g$) 与 注意力熵解耦 ($E_{spa, g}, E_{temp, g}$)
衡量第 $g$ 组内特征变化剧烈程度与信息冗余度：
- **残差幅度**：计算组内所有 Token 相对于组平均特征锚点 $\bar{x}_g$ 的残差 L2 范数期望：
  $$ R_g = \mathbb{E}_{b \sim B} \left[ \frac{1}{M} \sum_{i=1}^{M} \| x_{g, i}^{(b)} - \bar{x}_g^{(b)} \|_2 \right] $$
- **空间注意力熵**：提取空间 Attention QKV 概率矩阵 $A_{spa}$ 计算香农熵（低熵表示高度关注特定空间目标）：
  $$ E_{spa, g} = \mathbb{E} \left[ - \sum_{j=1}^{S} A_{spa} \log_2 (A_{spa} + \epsilon) \right] $$
- **时间注意力熵**：同理，提取时间 Attention 矩阵 $A_{temp}$ 计算香农熵：
  $$ E_{temp, g} = \mathbb{E} \left[ - \sum_{k=1}^{K} A_{temp} \log_2 (A_{temp} + \epsilon) \right] $$

### 2. 归一化统计与联合阈值标定 ($\tau$)
在校准集上获取上述三项指标的 Min-Max 极值区间，并对特征进行归一化（记为 $\widetilde{R}_g, \widetilde{E}_{spa, g}, \widetilde{E}_{temp, g}$）。构建综合敏感度得分（熵越低越敏感，取反）：
$$ S_g = \alpha \widetilde{R}_g + \beta \left( 1 - \widetilde{E}_{spa, g} \right) + \gamma \left( 1 - \widetilde{E}_{temp, g} \right) $$
提取所有校准样本分布的前 $p\%$ 分位数作为该 Block 的截断阈值 $\tau$。同时在校准集上预估误差前馈的衰减系数 $\lambda$。

---

## 第三阶段：快速评估阶段 (Quick Eval) —— 动态门控与块级位宽分配

**目标**：在迷你验证集（Quick Eval Set）上运行推理，统计各网络层的高敏感激活比例，进而为每个 Block 静态分配全局量化位宽（Block-Level Bit-width Allocation）。

1. **动态门控得分统计**：
   在 Quick Eval 前向传播中，基于 Phase 2 得到的归一化区间，计算当前组得分 $\hat{S}_g$ 并执行二值阶跃函数：
   $$ Z_g = \mathbb{I}(\hat{S}_g > \tau) $$
   若 $Z_g=1$ 则代表高敏感，否则为低敏感。

2. **生成位宽分配表 (`block_bits`)**：
   统计每个 Block 内所有组触发 $Z_g=1$ 的平均比率，得到 `group_high_ratio`。
   设置静态路由策略：若某 Block 的高敏感触发率达到阈值（如 $\ge 30\%$），则判定该 Block 为**信息密集型算子**，分配较高位宽（如 $b_{block} = 8$ 即 INT8）；反之，分配极低位宽（如 $b_{block} = 4$ 即 INT4）。
   最终产出一个包含所有网络层量化位宽的配置字典 `block_bits`。

---

## 第四阶段：全量评估阶段 (Full Eval) —— 模拟量化与误差补偿

**目标**：在完整的测试集上，根据分配好的 `block_bits` 对权重和激活执行模拟量化（Fake Quantization），并应用误差前馈以减轻量化带来的精度损失。

### 1. 权重参数的模拟量化 (Weight Fake Quantization)
在全量评估时，不直接更改底层内核，而是利用 Fake Quantization 模拟量化截断误差。按 `block_bits` 字典加载的指定位宽 $b = b_{block}$，对当前 Block 的全精度权重 $W$ 执行量化并立即反量化：
$$ W_{sim} = \text{Dequant}(\text{Quant}(W, b_{block})) = \text{Clamp}(\text{Round}(W / s), q_{min}, q_{max}) \cdot s $$
以此带有舍入损失的 $W_{sim}$ 代替原权重进行后续张量乘法。

### 2. 混合精度特征与误差前馈 (Mixed-Precision Activation & Forwarding)
同理，特征激活也依据 $b_{block}$ 承受模拟量化噪声。为了抑制误差随 Transformer 深度的指数级放大，我们对组间特征施加显式的误差补偿：

- 当 Block 为 **INT8 ($b_{block}=8$)**：计算当前特征的量化截断误差 $e_g = X_g - \tilde{X}_{INT8}$，利用衰减系数 $\lambda$ 计算补偿 Carry 并叠加至下一组特征输入：
  $$ X_{g+1} \leftarrow X_{g+1} + \lambda \cdot e_g $$
- 当 Block 为 **INT4 ($b_{block}=4$)**：由于 INT4 量化区间极窄导致失真剧烈，引入**残差复用（Residual Reuse）**。利用上一组反量化输出作为基准锚点，当前组仅对极其微小的帧间增量 $\Delta X_g = X_g - X_{g-1}$ 施加 4-bit Fake Quantization：
  $$ Y_g = \tilde{Y}_{g-1} + \text{Dequant}(\text{Quant}(\Delta X_g, b_{block}=4)) $$

通过此三段式（校准区间 -> 快速统计分配 -> 全量伪量化评测）的 PTQ 流程，在无需重训练（Zero Retraining）的情况下，准确测量出模型在目标混合精度方案下的真实压缩表现与精度持留率。

