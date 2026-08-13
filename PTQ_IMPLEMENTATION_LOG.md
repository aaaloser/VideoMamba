# VideoMamba PTQ 实现日志

> TimeSformer 训练后量化技术路线 → VideoMamba (Mamba SSM) 移植实现记录。
> 最后更新: 2026-08-12

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
- 预训练模型: `pretrain_model/videomamba_m16_ssv2_f16_res224.pth`（用户上传，fp16）

**改动**（`videomamba/video_sm/run_ptq_experiments.sh`）:

| 项 | 说明 |
|------|------|
| `DATASET` 参数 | `DATASET=k400\|ssv2`，默认 k400；未知值报错退出 |
| k400 分支 | 原路径/协议完全不变（Kinetics_sparse, 400 类, 4×3, `--eval_data_path`） |
| ssv2 分支 | `--data_set SSV2` + `--no_use_decord`（`SSRawFrameClsDataset` 读帧）；`--nb_classes 174` |
| 路径 | `--prefix` = `somethingv2/frame/`；`--data_path` = `output_pth/SSv2/metadata`（自动生成） |
| metadata 自动生成 | 首次运行自动 `cp`：`train_videofolder.txt`→`train.csv`、`val_videofolder.txt`→`val.csv`/`test.csv`（test 复用 val，SSv2 测试标签不公开，官方即在 val 上评测） |
| 评测协议 | `--test_num_segment 2` / `--test_num_crop 3`（官方 2×3，可覆盖；k400 仍 4×3） |
| CKPT | 默认 `videomamba_m16_ssv2_f16_res224.pth`，`CKPT=` 环境变量可覆盖 |
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
- GPU 评测: **待重跑**（filename_tmpl 修复后尚未跑完）

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

**所有可覆盖参数**: `PTQ_CALIB_SIZE`(128), `PTQ_CALIB_BATCH_SIZE`(8), `PTQ_QUICK_BATCHES`(50), `PTQ_NUM_GROUPS`, `PTQ_TAU_PERCENTILE`, `PTQ_ALPHA/BETA/GAMMA`, `PTQ_HIGH_BLOCK_FRACTION`, `PTQ_MOTION_ETA/TAU/RHO/CANDIDATES/CAPTURE_BATCHES`, `TEST_NUM_SEGMENT/CROP`, `BATCH_SIZE`, `SAVE_QUANTIZED`, `QUANTIZED_OUTPUT_DIR`, `QUANTIZED_MODEL_PATH`。

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
