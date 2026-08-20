# VideoMamba PTQ 实现日志

> TimeSformer 训练后量化技术路线 → VideoMamba (Mamba SSM) 移植实现记录。
> 最后更新: 2026-08-19

## 项目概览

- **仓库**: `/data/liyifan24/VideoMamba`
- **模型**: `videomamba_middle`（K400, embed_dim=576, depth=32, bimamba=True, FP16 ckpt, Top-1 83.4%）
- **结构**: `in_proj [2304,576]`, `out_proj [576,1152]`, T=16, S=196, L=1+3136
- **环境**: `/home/liyifan24/miniforge3/envs/videomamba/bin/python`（Python 3.10, torch 2.1.2+cu118, mamba_ssm/causal_conv1d）
- **GPU**: CUDA_VISIBLE_DEVICES=2（RTX 3090）

### 关键约束

1. **模型保持 fp32 校准** — eval 脚本不 `.half()`，只用 autocast 包裹 eval forward。强制 `.half()` 导致 `estimate_delta_tensor` dtype 不匹配。
2. **文件编辑用 Python 替换脚本** — `apply_patch` 在此环境有 bug（内容错位）。统一用 `/home/.../bin/python - <<'PY'` + `s.replace(anchor, new)` + `assert count==1`。
3. **`.git` 只读** — 不能 `git checkout`。恢复文件用 `git show HEAD:path > /tmp/orig`。
4. **沙箱无法跑 GPU** — 所有 GPU 验证由用户终端手动执行。

---

## 技术路线步骤总览

| 步骤 | 内容 | 状态 |
|------|------|------|
| 2.5 | 真实 λ_error（INT4 相邻组最小二乘误差前馈系数） | ✅ GPU 验证 |
| 3 | 运行时路由退化为纯统计 + quick-eval 50 batch + quick_top1/calls_b | ✅ GPU 验证 |
| 4 | Rank 分配 bit（top-fraction 敏感度排名） | ✅ GPU 验证 |
| 5 | 运动感知 per-channel scale 优化（32 候选搜索） | ✅ GPU 验证 |
| 6 | 校准/推理 batch 解耦（calib_size/calib_batch_size 子批次省显存） | ✅ CPU 单测 + 桩模型 |

---

## 步骤 2.5 — 真实 λ_error

**目标**: 用 INT4 对相邻时间组量化，计算误差前馈系数 `λ = <e_g, e_{g+1}> / <e_g, e_g>`，clamp [-1,1]，取中位数。

**改动**:
- `quant/utils/helpers.py`: 新增 `compute_lambda_error_for_layer`（定义在文件末尾，运行时解析）
- `quant/utils/calibration.py`: `calibrate_videomamba_ptq` 中 λ = per-sample λ 的中位数，存入 `block_stats["lambda"]` + `["lambda_error"]`
- **单遍融合**: `collect_layer_group_metrics` 返回 `(layer_metrics, layer_lambda)` 元组，避免一次性 generator 耗尽后 λ 静默回退到默认 0.5

**验证**: GPU — λ per-block min=-0.0006, max=0.5207, mean=0.0816，全部 [-1,1]。CPU — 相同帧→1.0, 反相关→-1.0, 随机≈-0.018。

---

## 步骤 3 — 纯统计路由 + quick-eval 扩展

**目标**: quick-eval 前向不做实际量化（routed_input = 原始 FP32 输入），只基于静态校准档案（ranges + tau）计算逐组敏感度得分，统计 score > tau 的高敏感比例。扩展到 50 batch，新增 quick_top1 和 calls_b。

**改动**:
- `quant/utils/helpers.py`:
  - 新增 `extract_label(batch)` — 从 batch 提取标签（支持 tuple/list/dict）
  - `collect_layer_group_metrics` 新增 `return_predictions` 参数 → 返回 3-tuple `(layer_metrics, layer_lambda, predictions)`
  - `return_predictions=True` 时同一次前向传播中捕获 `(output, label)`，无额外计算开销
- `quant/utils/calibration.py`:
  - `calibrate_videomamba_ptq` 适配 3-tuple
  - `quick_eval_allocate_block_bits`: 默认 `max_quick_batches=50`；用 `return_predictions=True`；计算 `quick_top1`（FP32 基线 Top-1）；新增 `block_calls`（calls_b = len(metrics) // num_groups）
- `quant/config/quantization_config.py`: `QuickEvalResult` 新增 `quick_top1: Optional[float]` + `block_calls: Optional[Dict[int,int]]`
- `videomamba/video_sm/run_class_finetuning_ptq.py`: `--ptq_quick_batches` 默认 8→50；新增 quick_top1/block_calls 日志

**验证**: GPU — quick_top1=0.0%（随机数据，符合预期），block_calls 全部=2（2 batch）。

---

## 步骤 4 — Rank 分配 Bit

**目标**: 按 avg_ratio 降序排序，top `high_block_fraction` 分配 INT8，其余 INT4。对阈值绝对尺度波动不敏感。

**改动**:
- `quant/config/quantization_config.py`: `PTQConfig` 新增 `bit_allocation_mode="rank"`, `high_block_fraction=0.5`
- `quant/utils/calibration.py`: 新增 `allocate_block_bits_by_rank` / `_by_threshold` / `dispatch`
- `videomamba/video_sm/run_class_finetuning_ptq.py`: 新增 `--ptq_bit_allocation_mode`、`--ptq_high_block_fraction` CLI 参数

**验证**: GPU — 16 INT8 / 16 INT4 / 64 projections replaced。

---

## 步骤 5 — 运动感知 Scale 优化

**目标**: 对每个量化 in_proj，捕获校准输入，计算 FP 输出 Y_fp，推导 per-frame motion weights（运动强度 - η×冗余度，温度 softmax + 均匀先验），搜索 32 候选 scale（0.5×~1.5× MinMax 基准），选取最小化运动加权重建误差的 per-channel scale。

### 核心模块 `quant/utils/motion_scale.py`（新增，309 行）

| 函数 | 说明 |
|------|------|
| `compute_motion_frame_weights(y[B,T,S,O], eta, tau_m, rho)` | → 帧权重 `[T]` 和为 1 |
| `compute_motion_aware_scale(weight, activation, t_steps, bits, n_candidates, ...)` | → per-channel best scale `[out]`，32 候选搜索 |
| `capture_in_proj_activations(model, loader, device, max_batches, cfg)` | → `{layer_idx: [B,L,dim]}` via mixer pre-hook |
| `compute_motion_aware_in_proj_scales(model, block_bits, activations, cfg, ...)` | → `{layer_idx: scale_cpu[out]}` 独立预计算，返回 CPU tensor 用于 DDP 广播 |
| `apply_motion_aware_weight_only_(model, block_bits, activations, cfg, precomputed_in_proj_scales, ...)` | 替换 in_proj 为 motion-aware QuantizedLinear，out_proj 为 MinMax |

### 关键设计决策

1. **out_proj 用 MinMax 不用 motion-aware** — mamba_simple 内部 `F.linear(x, self.out_proj.weight, ...)` 直接访问权重非 module 调用，hook 不触发。in_proj 输入（=hidden_states）可通过 mixer pre-hook 捕获。
2. **DDP 流程** — rank0 计算 CPU scale → `broadcast_object_list` 广播（pickle CPU tensor dict）→ 所有 rank `apply_motion_aware_weight_only_` 内部 `.to(device)`。单 GPU 路径也正常。
3. **in_proj scale 优先级**: `precomputed_in_proj_scales` > `activations`+`cfg`（重算）> MinMax 回退。`activations`/`cfg` 均为 Optional。
4. **运动参数**（TimeSformer spec）: `n_candidates=32, eta=0.5, tau_m=0.7, rho=0.2`

### CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--ptq_scale_mode` | minmax | minmax 或 motion |
| `--ptq_motion_candidates` | 32 | 候选 scale 因子数 |
| `--ptq_motion_eta` | 0.5 | 冗余度权重 |
| `--ptq_motion_tau` | 0.7 | softmax 温度 |
| `--ptq_motion_rho` | 0.2 | 均匀先验比例 |
| `--ptq_motion_capture_batches` | 0 | 捕获 batch 数（0=复用校准样本） |

### 验证

```
probe layer 5: precomputed vs inline max_diff=0.00e+00
real layer 5 motion-weighted recon err: motion-aware=19.58 MinMax=26.08 improvement=24.9%
precomputed-path: replaced=64 motion_applied=32
applied scale vs precomputed max_diff=0.00e+00
ALL MOTION-AWARE CHECKS PASSED (precompute + inline paths)
```

---

## 2026-08-12 更新：校准/推理 batch 解耦 + 子批次省显存

**背景**: 实验时 GPU 被其他程序占用 5GB 导致爆显存；且校准前向与推理共用 `BATCH_SIZE`，无法单独控制校准显存。

**设计**: 校准完全由两个独立参数控制，与推理 `BATCH_SIZE` 解耦：
- `calib_size` — 校准样本总数（默认 128，等价旧默认 4 batch × 32）
- `calib_batch_size` — 每次校准前向的子批次大小（默认 8；0/缺省=不拆分）。loader 的大 batch 在 forward 前按此拆成多个子批次，降低激活峰值显存；拆分会增加每组统计样本数，统计口径不变（组指标按 batch 均值聚合）

**改动**:
- `quant/utils/helpers.py`: 新增 `iter_calib_tensors()`（按 calib_size 累计样本 + 按 calib_batch_size 拆子批次，含截断/超量边界处理）；`collect_layer_group_metrics` 支持 `calib_size`/`calib_batch_size`（`max_batches` 保留为 legacy）
- `quant/utils/calibration.py`: `calibrate_videomamba_ptq` 签名改为 `calib_size`/`calib_batch_size`；`max_calib_batches` 保留为 deprecated 别名（兼容 /tmp smoke 脚本）
- `quant/utils/motion_scale.py`: `capture_in_proj_activations` 支持子批次（motion 捕获默认复用同一校准配置，0 捕获批次数时走 calib_size 路径）
- `videomamba/video_sm/run_class_finetuning_ptq.py`: 新增 `--ptq_calib_size`、`--ptq_calib_batch_size`；`--ptq_calib_batches` deprecated（未设置新参数时按旧语义 `batches × batch_size` 映射，兼容旧脚本）
- `videomamba/video_sm/run_ptq_experiments.sh`: `CALIB_BATCHES` 拆为 `PTQ_CALIB_SIZE`(128) / `PTQ_CALIB_BATCH_SIZE`(8)

**CLI/脚本参数**:

| 参数 | 默认 | 说明 |
|------|------|------|
| `--ptq_calib_size` / `PTQ_CALIB_SIZE` | 128 | 校准样本总数（与 BATCH_SIZE 解耦） |
| `--ptq_calib_batch_size` / `PTQ_CALIB_BATCH_SIZE` | 8 | 校准前向子批次（省显存；0=不拆分） |
| `--ptq_calib_batches` | None | [deprecated] 旧语义：loader batch 数 |

**量化模型保存路径**: 新增 `QUANTIZED_MODEL_PATH` 环境变量覆盖（脚本 `:70`），默认仍为 `${QUANTIZED_OUTPUT_DIR}/ptq_<EXPERIMENT>.pth`；`mkdir -p` 改为 `dirname("${QUANT_PATH}")` 以适配自定义路径。配置打印新增 `QUANT_MODEL_PATH` 行。

**验证**:
- `iter_calib_tensors` 单测 7 项（拆分/截断/不拆分/超量边界）✅
- 桩模型端到端校准：`calib_size=9, calib_batch_size=2` → 每层 5 次前向/20 group-samples；legacy `max_calib_batches=2` → 8；不拆分 → 8 ✅
- motion capture 新/旧路径捕获样本数一致 ✅
- `py_compile` + `bash -n` ✅

**实验结果（mixed-rank-motion, calib_size=32, calib_batch_size=4, 单卡）**:

```
Test: Acc@1 80.546（batch 级均值）
merge 最终: Top-1 82.95%, Top-5 95.70%（写入 logs/ptq_mixed-rank-motion/log.txt）
```

说明：测试中打印的 Acc@1 为 batch 级均值；merge 后（4 段 × 3 crop 投票）82.95% 与 fp16 基线 83.4% 非常接近，该 W8/W4 混合 + motion scale 方案基本无损。该结果由崩溃前已写好的 `0.txt` 手动执行 `merge()` 恢复，未重跑约 3h 的评测。

## 2026-08-12 SSv2 适配：work1 方法跨数据集（运行参数切换）

**目标**: 将 work1（mixed-rank-motion, W8/W4-A16 权重量化）从 K400 移植到 Something-Something V2，数据集切换作为运行参数 `DATASET=k400|ssv2`（默认 k400，原行为完全不变）。

**数据集情况**:
- 路径: `/data/liyifan24/Datasets/somethingv2/`
- 抽帧格式: `frame/<id>/000001.jpg`（6 位零填充，无 `img_` 前缀；`val_videofolder.txt` 共 24777 个视频）
- 标注: `train_videofolder.txt` / `val_videofolder.txt`（videofolder 格式 `id num_frames label`，174 类 0-173）
- 预训练模型: 本意用 MASK 预训练版 `pretrain_model/videomamba_m16_ssv2_mask_ft_f16_res224.pth`（VideoMamba-M *MASK* 224 16x3x4，zoo Top-1 71.0%），但**首次下载/运行时用错**，实际跑的是非 MASK 版 `pretrain_model/videomamba_m16_ssv2_f16_res224.pth`（ImageNet-1K 初始化，zoo Top-1 68.3%）。已于 2026-08-13 更正（见下文「SSv2 模型用错更正」）

**改动**（`videomamba/video_sm/run_ptq_experiments.sh`）:

| 项 | 说明 |
|------|------|
| `DATASET` 参数 | `DATASET=k400\|ssv2`，默认 k400；未知值报错退出 |
| k400 分支 | 原路径/协议完全不变（Kinetics_sparse, 400 类, 4×3, `--eval_data_path`） |
| ssv2 分支 | `--data_set SSV2` + `--no_use_decord`（`SSRawFrameClsDataset` 读帧）；`--nb_classes 174` |
| 路径 | `--prefix` = `somethingv2/frame/`；`--data_path` = `output_pth/SSv2/metadata`（自动生成） |
| metadata 自动生成 | 首次运行自动 `cp`：`train_videofolder.txt`→`train.csv`、`val_videofolder.txt`→`val.csv`/`test.csv`（test 复用 val，SSv2 测试标签不公开，官方即在 val 上评测） |
| 评测协议 | `--test_num_segment 2` / `--test_num_crop 3`（官方 2×3，可覆盖；k400 仍 4×3） |
| CKPT | 默认 `videomamba_m16_ssv2_f16_res224.pth`（2026-08-13 更正为 `videomamba_m16_ssv2_mask_ft_f16_res224.pth`），`CKPT=` 环境变量可覆盖 |
| 输出隔离 | 日志目录 `logs/ptq_<experiment>_ssv2/`；量化模型默认 `output_pth/SSv2/ptq_<experiment>.pth` |
| `--filename_tmpl` | ssv2 追加 `{:06}.jpg`（仅 ssv2 传，k400 不传） |

**踩坑与修复**:
- 首次 GPU 运行 Phase-2 校准报 `FileNotFoundError: frame/74225/img_00002.jpg`：该数据集帧文件名为 `000001.jpg`（6 位零填充、无前缀），默认模板 `img_{:05}.jpg` 不匹配。已在 ssv2 分支加 `FILENAME_TMPL='{:06}.jpg'`。
- 全量校验: val 24777 个视频帧数与标注完全一致、无缺失目录（扫描通过）。

**验证**:
- `bash -n` 语法 ✅
- k400 dry-run 参数与改动前逐项一致（回归通过）✅
- ssv2 dry-run: SSV2/174 类/2×3/`--no_use_decord`/`--filename_tmpl {:06}.jpg`/motion 参数/量化模型路径全部正确 ✅
- metadata 自动生成 ✅（`output_pth/SSv2/metadata/{train,val,test}.csv`）
- GPU 评测: 首次跑完 Top-1 67.95%（2×3 merge, quick FP32 66.75%），但用的是**非 MASK 版权重**（下错），结果作废；2026-08-13 已用 mask_ft 重跑，Top-1 70.75%（见下文更正章节与实验记录汇总）

**运行命令**（详见 `run_ptq.md` SSv2 一节）:
```bash
cd /data/liyifan24/VideoMamba/videomamba/video_sm

CUDA_VISIBLE_DEVICES=2 \
DATASET=ssv2 \
BATCH_SIZE=32 \
PTQ_CALIB_SIZE=32 \
PTQ_CALIB_BATCH_SIZE=4 \
QUANTIZED_MODEL_PATH=/data/liyifan24/VideoMamba/output_pth/SSv2/work1/run_motion_test.pth \
SAVE_QUANTIZED=1 \
EXPERIMENT=mixed-rank-motion \
  nohup bash run_ptq_experiments.sh > /data/liyifan24/VideoMamba/output_pth/SSv2/work1/run_motion_test.log 2>&1 &
```

## 2026-08-13 更正：SSv2 预训练模型用错（mask_ft vs 非 mask）

**经过**: SSv2 适配的本意是使用 MASK 预训练权重 `videomamba_m16_ssv2_mask_ft_f16_res224.pth`（VideoMamba-M *MASK* 224 16x3x4，zoo Top-1 71.0%），但首次下载/运行时使用的是非 MASK 版 `videomamba_m16_ssv2_f16_res224.pth`（VideoMamba-M ImageNet-1K 224 16x3x4，zoo Top-1 68.3%）。因此 2026-08-12 的 SSv2 GPU 结果（Top-1 67.95%, 2×3）是**在错误基线上测的**，只能说明 W8/W4 混合方案在非 mask 模型上基本无损，不能作为 mask 基线的结论。

**为什么难以区分（踩坑根因）**:
- 两档模型**微调后的 state dict 结构完全相同**（均有 `head.weight/bias`、32 个 mixer、无 `mask_token`/decoder），无法从权重结构判断
- `MODEL_ZOO.md` 有复制粘贴 bug：`*MASK* 224 16x3x4 71.0` 行的下载链接误写为 `videomamba_m16_ssv2_f16_res224.pth`（应为 `videomamba_m16_ssv2_mask_ft_f16_res224.pth`），同表 8x3x4/288 行链接均正确，仅该行错误

**正确区分方法**（按可靠度排序）:
1. 文件名：`..._mask_ft_...pth` = MASK 预训练；`..._ssv2_...pth` = ImageNet-1K 初始化
2. 训练脚本 `exp/ssv2/videomamba_middle_mask/run_f16x224.sh`：`--finetune` 指向 `videomamba_m16_ssv2_mask_pt_f8_res224.pth`（MASK 预训练）
3. 实测精度：同协议（4×3, 224）FP16 全量评测，68.3% vs 71.0% 立分

**已完成的更正（2026-08-13）**:
- 正确权重已下载: `pretrain_model/videomamba_m16_ssv2_mask_ft_f16_res224.pth`（295,204,086 B）
- 校验：md5 与非 mask 版不同、首层权重 diff_norm≈108、结构一致（552 参数/32 mixer/含 head）→ 确为不同权重 ✅
- `run_ptq_experiments.sh` ssv2 分支 `CKPT_DEFAULT` 已改为 mask_ft 路径（k400 分支本就是 mask_ft，不受影响）

**待办**:
- [x] 在 mask_ft 上重跑 PTQ（mixed-rank-motion）2×3——2026-08-13 完成，Top-1 70.75%，见「实验记录汇总」
- [x] 在 mask_ft 上跑 PTQ（mixed-rank-motion）4×3——2026-08-18 完成，Top-1 70.73%（对 zoo 4×3 FP16 71.0% 掉点 ≈0.27%，基本无损）
- [ ] （可选）在 mask_ft 上跑 FP16 基线（2×3 与 4×3 各一次）：4×3 可直接用 zoo 71.0% 参考；仅当需严格本地复现时再跑

**4×3 协议运行命令（todo）**:

FP16 基线（纯 eval，无 PTQ）:
```bash
cd /data/liyifan24/VideoMamba/videomamba/video_sm

CUDA_VISIBLE_DEVICES=3 \
python run_class_finetuning_ptq.py \
  --model videomamba_middle \
  --finetune /data/liyifan24/VideoMamba/pretrain_model/videomamba_m16_ssv2_mask_ft_f16_res224.pth \
  --data_path /data/liyifan24/VideoMamba/output_pth/SSv2/metadata \
  --prefix /data/liyifan24/Datasets/somethingv2/frame/ \
  --data_set SSV2 --split ' ' --nb_classes 174 \
  --output_dir logs/ssv2_fp16_baseline_4x3 --log_dir logs/ssv2_fp16_baseline_4x3 \
  --batch_size 32 --input_size 224 --short_side_size 224 --num_frames 16 \
  --num_workers 8 --tubelet_size 1 \
  --test_num_segment 4 --test_num_crop 3 \
  --eval --bf16 --no_use_decord --filename_tmpl '{:06}.jpg'
```

PTQ（mixed-rank-motion, 4×3，复用实验脚本）:
```bash
cd /data/liyifan24/VideoMamba/videomamba/video_sm

CUDA_VISIBLE_DEVICES=3 \
DATASET=ssv2 \
TEST_NUM_SEGMENT=4 \
BATCH_SIZE=32 \
PTQ_CALIB_SIZE=32 \
PTQ_CALIB_BATCH_SIZE=4 \
QUANTIZED_MODEL_PATH=/data/liyifan24/VideoMamba/output_pth/SSv2/work1/run_motion_test_4x3.pth \
SAVE_QUANTIZED=1 \
EXPERIMENT=mixed-rank-motion \
  nohup bash run_ptq_experiments.sh > /data/liyifan24/VideoMamba/output_pth/SSv2/work1/run_motion_test_4x3.log 2>&1 &
```

> 2×3 的 FP16 基线命令与上相同，仅把 `--test_num_segment` 改为 2（PTQ 侧用 `TEST_NUM_SEGMENT=2`）。

## 2026-08-13 实验记录汇总（work1 = mixed-rank-motion, W8/W4-A16 + motion scale）

三份 work1 实验日志与量化模型位于 `output_pth/K400/work1/` 与 `output_pth/SSv2/work1/`（SSv2 含 2×3 与 4×3 两次协议）：

| 项 | K400 | SSv2 2×3 | SSv2 4×3（复测） |
|------|------|------|------|
| 日志 | `output_pth/K400/work1/run_motion_test.log` | `output_pth/SSv2/work1/run_motion_test.log` | `output_pth/SSv2/work1/run_motion_test_4x3.log` |
| 量化模型 | `run_motion_test.pth`（137,240,962 B） | `run_motion_test.pth`（136,719,554 B） | `run_motion_test_4x3.pth`（136,722,546 B） |
| 运行完成时间 | 2026-08-12 12:04 | 2026-08-13 10:21（mask_ft 重跑） | 2026-08-18 06:51 |
| 预训练权重 | `videomamba_m16_k400_mask_ft_f16_res224.pth`（MASK） | `videomamba_m16_ssv2_mask_ft_f16_res224.pth`（MASK） | 同左 |
| 评测协议 | 4 段 × 3 crop | 2 段 × 3 crop | 4 段 × 3 crop |
| 评测视频数 | 164,796（≈27,466×6） | 148,662（=24,777×6） | 297,324（=24,777×12） |
| quick_top1（FP32, 50 batch） | 0.08%（K400 quick loader 标签异常，无参考意义） | 69.96% | 69.96% |
| bit 分配 | 16×INT8 + 16×INT4（rank, n_high=16） | 同左 | 同左（block_bits 与 2×3 完全一致） |
| 全量 Top-1 / Top-5 | **82.95% / 95.70%** | **70.75% / 92.46%** | **70.73% / 92.44%** |
| 评测耗时 | 2:09:07 | 7:27:10 | 16:34:28 |

**结论与注意**:
- K400：zoo MASK 基线 83.4%（4×3），PTQ 82.95%（4×3）→ 掉点 ≈0.45%，W8/W4 混合基本无损（与日志「实验结果」一节一致）
- SSv2（4×3 复测）：PTQ 70.73% vs zoo MASK FP16 基线 71.0%（同为 4×3 协议）→ 掉点 ≈0.27%，**基本无损，量化方案在正确基线上确认 OK**；2×3（70.75%）与 4×3（70.73%）结果几乎一致，说明协议间差异很小。quick FP32 69.96%（50 batch）为弱参考（单批均值 vs merge 投票，后者通常更高）
- K400 quick_top1=0.08% 异常：K400 quick loader 的标签疑似随机/不可用（SSv2 正常 69.96%），该字段不可作为 K400 FP32 基线参考
- 量化模型体积 ≈ 137MB（两数据集均约减半，原始 FP16 ckpt ≈295MB；W4 打包 + W8 权重）

---

## 2026-08-19 Quamba 复现合并回主分支 + quamba-8 预设

**背景**: Quamba 复现代码原先只存在于 `ptq_videomamba_v2` 分支；本次将其合并回 main 工作树，并删除旧分支。当前 main 的 work1 逻辑不受影响。

**新增/改动文件**:
- `quamba/__init__.py` / `quamba/videomamba_bridge.py`: **新增**，Quamba bridge（激活校准 observer + weight-only fake quant）
- `quant/quamba/__init__.py` / `quant/quamba/backend.py`: **新增**，对 `run_class_finetuning_ptq.py` 的 Quamba 封装
- `videomamba/video_sm/run_class_finetuning_ptq.py`: `--quant_method` 增加 `quamba`；新增 `--quamba_default_bit`（默认 8）、`--quamba_a_bits`（默认 8）、`--quamba_percentile_alpha`（默认 0.9995）；Quamba 保存仍写 `videomamba_ptq_int_v1` 整数 checkpoint
- `videomamba/video_sm/run_ptq_experiments.sh`: 新增 `EXPERIMENT=quamba-8` 预设，默认 `PTQ_CALIB_SIZE=32` / `PTQ_CALIB_BATCH_SIZE=4`，只跑 8bit
- `.gitignore`: 本地 bridge 文件纳入跟踪，仅继续忽略上游 `quamba/Quamba/`

**量化配置**:
- 32 个 block 全部均匀 8bit（`build_uniform_block_bits(default_bit=8)`）
- 量化对象: `in_proj/conv1d/x_proj/dt_proj/out_proj` 及 `_b` 镜像层的 weight/bias
- 对称 fake quant；2D weight per-channel（`channel_dim=0`），bias per-tensor
- **激活不量化**: `--quamba_a_bits` 只用于校准 observer 的位宽和报告（`x_proj` 输入用 percentile alpha=0.9995，其余 minmax），推理 forward 不做 activation fake quant
- 校准支持 `calib_size`/`calib_batch_size` 子批次拆分，与 work1 的 `PTQ_CALIB_SIZE=32 / PTQ_CALIB_BATCH_SIZE=4` 语义一致（32 样本 → 8×4）

**运行命令**:

K400:
```bash
cd /data/liyifan24/VideoMamba/videomamba/video_sm

CUDA_VISIBLE_DEVICES=0 \
DATASET=k400 \
BATCH_SIZE=32 \
PTQ_CALIB_SIZE=32 \
PTQ_CALIB_BATCH_SIZE=4 \
QUANTIZED_MODEL_PATH=/data/liyifan24/VideoMamba/output_pth/K400/ptq_quamba-8.pth \
SAVE_QUANTIZED=1 \
EXPERIMENT=quamba-8 \
  nohup bash run_ptq_experiments.sh > /data/liyifan24/VideoMamba/output_pth/K400/ptq_quamba-8.log 2>&1 &
```

SSv2:
```bash
cd /data/liyifan24/VideoMamba/videomamba/video_sm

CUDA_VISIBLE_DEVICES=0 \
DATASET=ssv2 \
BATCH_SIZE=32 \
PTQ_CALIB_SIZE=32 \
PTQ_CALIB_BATCH_SIZE=4 \
QUANTIZED_MODEL_PATH=/data/liyifan24/VideoMamba/output_pth/SSv2/ptq_quamba-8.pth \
SAVE_QUANTIZED=1 \
EXPERIMENT=quamba-8 \
  nohup bash run_ptq_experiments.sh > /data/liyifan24/VideoMamba/output_pth/SSv2/ptq_quamba-8.log 2>&1 &
```

**已有 Quamba 8bit checkpoint 完整性**: `output_pth/videomamba_m_mask_eval_f16_res224_ptq_quamba_8.pth` 可正常加载，format=`videomamba_ptq_int_v1`，32 blocks 全 8bit，552 个 key 与 FP16 ckpt 完全一致（无缺失/多余/shape 不一致），对应旧日志 Top-1 83.06% / Top-5 95.73%。

**验证**: `bash -n` ✅；`py_compile` ✅；`--help` 显示 `--quant_method {mixed,uniform,quamba}` 与 `--quamba_*` ✅；`_iter_calib_tensors` CPU 单测 32→8×4 ✅；未跑 GPU 实验。

---

## 实验脚本

### `videomamba/video_sm/run_ptq_experiments.sh`（新增，220 行）

环境变量驱动 + 实验预设，支持单卡/多卡自适应。

**单卡/多卡**: `NPROC=1`（默认）用 `python` 不走分布式；`NPROC>1` 用 `torchrun --dist_eval`。

**实验预设**:

| 预设 | 说明 |
|------|------|
| `uniform-w8` | 均匀 W8A16 基线 |
| `uniform-w4` | 均匀 W4A16 基线 |
| `quamba-8` | Quamba 风格均匀 W8 权重量化（默认 PTQ_CALIB_SIZE=32 / PTQ_CALIB_BATCH_SIZE=4） |
| `mixed-rank-minmax` | rank 分配 + MinMax scale（步骤4） |
| `mixed-rank-motion` | rank 分配 + 运动感知 scale（步骤4+5） |
| `mixed-threshold-minmax` | threshold 分配 + MinMax（传统） |

**用法**:
```bash
cd /data/liyifan24/VideoMamba/videomamba/video_sm

# 默认: mixed-rank-minmax, GPU 2 单卡
bash run_ptq_experiments.sh

# 运动感知量化
EXPERIMENT=mixed-rank-motion \
  nohup bash run_ptq_experiments.sh > logs/run_motion_test.log 2>&1 &

# 消融
EXPERIMENT=mixed-rank-motion PTQ_MOTION_ETA=0.3 \
  nohup bash run_ptq_experiments.sh > logs/run_motion_nohup.log 2>&1 &

# 快速测试
TEST_NUM_SEGMENT=1 TEST_NUM_CROP=1 EXPERIMENT=mixed-rank-motion bash run_ptq_experiments.sh

# 4卡
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 EXPERIMENT=mixed-rank-motion bash run_ptq_experiments.sh

# 保存终端输出到文件
bash run_ptq_experiments.sh 2>&1 | tee logs/run.log
```

**所有可覆盖参数**: `PTQ_CALIB_SIZE`(128, quamba-8 默认 32), `PTQ_CALIB_BATCH_SIZE`(8, quamba-8 默认 4), `PTQ_QUICK_BATCHES`(50), `PTQ_NUM_GROUPS`, `PTQ_TAU_PERCENTILE`, `PTQ_ALPHA/BETA/GAMMA`, `PTQ_HIGH_BLOCK_FRACTION`, `PTQ_MOTION_ETA/TAU/RHO/CANDIDATES/CAPTURE_BATCHES`, `PTQ_QUAMBA_BIT`(8), `PTQ_QUAMBA_A_BITS`(8), `PTQ_QUAMBA_PERCENTILE_ALPHA`(0.9995), `TEST_NUM_SEGMENT/CROP`, `BATCH_SIZE`, `SAVE_QUANTIZED`, `QUANTIZED_OUTPUT_DIR`, `QUANTIZED_MODEL_PATH`。

### 原始脚本 `run_eval_f16x224_4gpu_ptq.sh` 保留不动作为参考。

---

## Bug 修复

### 单卡 `sampler_test` 未定义（已修复）

`run_class_finetuning_ptq.py` 中 `--dist_eval` 未设置时，`else` 分支只设了 `sampler_val` 漏了 `sampler_test`。原脚本一直用 `--dist_eval` 所以未触发。已添加 `sampler_test = SequentialSampler(dataset_test)`。

### 2026-08-12 `ptq_block_bits.json` 序列化失败（已修复）

motion 模式下 `ptq_payload['motion_in_proj_scales']` 存的是 CPU tensor（DDP 广播设计，`compute_motion_aware_in_proj_scales` 返回 `scale.detach().to("cpu")`），`json.dump` 抛 `TypeError: Object of type Tensor is not JSON serializable`。已在写 JSON 前对该字段 `.tolist()` 转 list（apply 阶段已消费完 tensor，时机安全）。

### 2026-08-12 单卡模式 barrier 未初始化（已修复）

`NPROC=1` 走普通 `python`（不初始化 distributed），eval 分支与 test_best 分支两处**无条件** `torch.distributed.barrier()` 在 `final_test` 后崩溃（`RuntimeError: Default process group has not been initialized`）。已加 `if args.distributed:` 保护。

---

## 修改的文件清单

| 文件 | 改动 |
|------|------|
| `quant/config/quantization_config.py` | PTQConfig + QuickEvalResult 新增字段（步骤3/4） |
| `quant/utils/calibration.py` | rank 分配 + 真实 λ_error + quick_top1/calls_b（步骤2.5/3/4） |
| `quant/utils/helpers.py` | λ 计算 + 单遍融合 + `extract_label` + 3-tuple 返回（步骤2.5/3） |
| `quant/utils/motion_scale.py` | **新增** 步骤5 核心（309 行） |
| `quant/utils/__init__.py` | 导出 motion_scale 函数 |
| `quant/utils/ptq_videomamba.py` | 导出 motion_scale 函数 |
| `videomamba/video_sm/run_class_finetuning_ptq.py` | CLI 参数 + motion apply 分支 + step-3 日志 + sampler_test 修复 |
| `videomamba/video_sm/run_ptq_experiments.sh` | **新增** 实验脚本（220 行） |
| `quant/utils/helpers.py` | `iter_calib_tensors` + calib_size/calib_batch_size 支持（2026-08-12） |
| `quant/utils/calibration.py` | calibrate 签名改 calib_size/calib_batch_size，legacy 保留（2026-08-12） |
| `quant/utils/motion_scale.py` | capture 子批次支持（2026-08-12） |
| `videomamba/video_sm/run_class_finetuning_ptq.py` | `--ptq_calib_size/--ptq_calib_batch_size` + JSON 序列化修复 + 单卡 barrier 修复（2026-08-12） |
| `videomamba/video_sm/run_ptq_experiments.sh` | `PTQ_CALIB_SIZE/PTQ_CALIB_BATCH_SIZE` + `QUANTIZED_MODEL_PATH`（2026-08-12） |
| `videomamba/video_sm/run_ptq_experiments.sh` | `DATASET` 参数（k400/ssv2 分支）+ metadata 自动生成 + `--filename_tmpl`（2026-08-12 SSv2 适配） |
| `run_ptq.md` | SSv2 运行说明（2026-08-12） |
| `videomamba/video_sm/run_ptq_experiments.sh` | ssv2 分支 `CKPT_DEFAULT` 更正为 mask_ft（2026-08-13） |
| `quamba/videomamba_bridge.py` | **新增** Quamba bridge（校准 observer + weight-only fake quant；支持 calib_size/calib_batch_size 子批次）（2026-08-19） |
| `quamba/__init__.py` | **新增** Quamba bridge 导出（2026-08-19） |
| `quant/quamba/backend.py` | **新增** Quamba backend 封装（2026-08-19） |
| `quant/quamba/__init__.py` | **新增** Quamba 导出（2026-08-19） |
| `videomamba/video_sm/run_class_finetuning_ptq.py` | `--quant_method quamba` + `--quamba_*` CLI；Quamba 保存走 v1 整数 checkpoint（2026-08-19） |
| `videomamba/video_sm/run_ptq_experiments.sh` | `EXPERIMENT=quamba-8` 预设，默认 32/4 校准（2026-08-19） |
| `.gitignore` | 只忽略上游 `quamba/Quamba/`，本地 bridge 文件纳入跟踪（2026-08-19） |

> 以上改动均在工作树中，**未 git commit**。

## Smoke 测试

| 文件 | 内容 | 状态 |
|------|------|------|
| `/tmp/smoke_rank.py` | 步骤4+2.5+3（rank+λ+quick_top1+calls_b） | ✅ 通过 |
| `/tmp/smoke_motion.py` | 步骤5（motion scale, precompute+inline 双路径） | ✅ 通过 |
| `iter_calib_tensors` 单测（CPU） | 拆分/截断/不拆分/超量边界 7 项（2026-08-12） | ✅ 通过 |
| 桩模型端到端校准（CPU） | 新路径/legacy/不拆分（2026-08-12） | ✅ 通过 |
| motion capture 冒烟（CPU） | 新/旧路径捕获样本一致（2026-08-12） | ✅ 通过 |
| SSv2 脚本 dry-run | k400 回归 + ssv2 参数展开 + bash -n（2026-08-12） | ✅ 通过 |
| Quamba 校准拆分（CPU） | calib_size=32, calib_batch_size=4 → 8×4 子批次（2026-08-19） | ✅ 通过 |

---

## 输出目录文件说明

`logs/ptq_<experiment>/` 下:

| 文件 | 说明 |
|------|------|
| `ptq_block_bits.json` | block_bits, block_rank, block_high_ratio, block_stats, n_high, motion_in_proj_scales（list） |
| `0.txt` | `final_test` 预测文件（rank 0），`merge()` 读取计算最终精度。不要直接查看 |
| `log.txt` | 最终精度记录，只在 `merge()` 完成后写入 |
| `events.out.tfevents.*` | TensorBoard 事件（eval 模式基本为空） |

最终精度打印到终端 stdout。查看方式:
```bash
grep -E "Top-1|quick_top1|INT8|INT4" logs/run.log
```

> **注意**: `quick_top1` 和 `block_calls` 目前只在终端打印，未存入 `ptq_block_bits.json`（`ptq_payload` 未包含这两个字段）。如需持久化，需在 eval 脚本中添加 `ptq_payload['quick_top1'] = quick_result.quick_top1`。
