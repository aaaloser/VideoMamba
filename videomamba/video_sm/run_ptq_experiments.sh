#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# VideoMamba PTQ 实验脚本 (步骤 3/4/5 全量支持)
# ------------------------------------------------------------
# 用法:
#   bash run_ptq_experiments.sh                      # 默认: mixed-rank-minmax, GPU 2
#   EXPERIMENT=mixed-rank-motion bash run_ptq_experiments.sh
#   EXPERIMENT=uniform-w4 bash run_ptq_experiments.sh
#
# 消融 (在实验预设基础上覆盖单个参数):
#   EXPERIMENT=mixed-rank-motion PTQ_MOTION_ETA=0.3 bash run_ptq_experiments.sh
#   EXPERIMENT=mixed-rank-minmax PTQ_HIGH_BLOCK_FRACTION=0.75 bash run_ptq_experiments.sh
#
# 切换数据集 (默认 k400):
#   DATASET=ssv2 bash run_ptq_experiments.sh            # Something-Something V2 (抽帧数据)
#   DATASET=ssv2 CKPT=/path/to/ssv2_ckpt.pth bash run_ptq_experiments.sh
#
# 切换 GPU/多卡:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 bash run_ptq_experiments.sh
#
# 可用实验预设:
#   uniform-w8             均匀 W8A16 基线
#   uniform-w4             均匀 W4A16 基线
#   mixed-rank-minmax      混合精度 + rank 分配 + MinMax scale (步骤4)
#   mixed-rank-motion      混合精度 + rank 分配 + 运动感知 scale (步骤4+5)
#   mixed-threshold-minmax 混合精度 + threshold 分配 + MinMax scale (传统)
# ============================================================

# ---- GPU 配置 ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
NPROC=${NPROC:-1}
export MASTER_PORT=$((12000 + RANDOM % 20000))
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# ---- 实验模式 ----
EXPERIMENT=${EXPERIMENT:-mixed-rank-minmax}

# ---- 数据集切换 (k400 | ssv2) ----
DATASET=${DATASET:-k400}

# ---- 数据/模型路径 (按数据集) ----
OUTPUT_BASE=${OUTPUT_BASE:-./logs}

case "${DATASET}" in
  k400)
    DATA_SET='Kinetics_sparse'
    NB_CLASSES=400
    PREFIX='/data/liyifan24/Datasets/Kinetics-400/'
    DATA_PATH='/data/liyifan24/Datasets/Kinetics-400/'
    EVAL_DATA_PATH='/data/liyifan24/Datasets/Kinetics-400/val_model_label.csv'
    CKPT_DEFAULT='/data/liyifan24/VideoMamba/pretrain_model/videomamba_m16_k400_mask_ft_f16_res224.pth'
    USE_DECORD_FLAG=()            # 默认 use_decord=True (视频文件)
    TEST_SEG_DEFAULT=4
    JOB_SUFFIX_DEFAULT=''
    QUANT_SUBDIR=''
    FILENAME_TMPL=''
    ;;
  ssv2)
    DATA_SET='SSV2'
    NB_CLASSES=174
    PREFIX='/data/liyifan24/Datasets/somethingv2/frame/'
    DATA_PATH='/data/liyifan24/VideoMamba/output_pth/SSv2/metadata'
    EVAL_DATA_PATH=''
    CKPT_DEFAULT='/data/liyifan24/VideoMamba/pretrain_model/videomamba_m16_ssv2_mask_ft_f16_res224.pth'
    USE_DECORD_FLAG=(--no_use_decord)   # 抽帧数据, 走 SSRawFrameClsDataset
    TEST_SEG_DEFAULT=2
    JOB_SUFFIX_DEFAULT='_ssv2'
    QUANT_SUBDIR='SSv2'
    FILENAME_TMPL='{:06}.jpg'   # frame/ 下文件名为 000001.jpg (6位零填充; 默认 img_{:05}.jpg 不匹配)
    ;;
  *)
    echo "ERROR: Unknown DATASET='${DATASET}'"
    echo "Available: k400, ssv2"
    exit 1
    ;;
esac

CKPT=${CKPT:-${CKPT_DEFAULT}}

# ---- SSv2 metadata 自动生成 (videofolder 格式 -> train/val/test.csv) ----
if [ "${DATASET}" = "ssv2" ]; then
  SSV2_META_SRC='/data/liyifan24/Datasets/somethingv2'
  mkdir -p "${DATA_PATH}"
  for entry in train:train_videofolder.txt val:val_videofolder.txt test:val_videofolder.txt; do
    csv_name="${entry%%:*}"; src_name="${entry##*:}"
    if [ ! -f "${DATA_PATH}/${csv_name}.csv" ]; then
      cp "${SSV2_META_SRC}/${src_name}" "${DATA_PATH}/${csv_name}.csv"
      echo "[run] generated ${DATA_PATH}/${csv_name}.csv (from ${SSV2_META_SRC}/${src_name})"
    fi
  done
fi

# ---- 通用参数 (均可通过环境变量覆盖) ----
BATCH_SIZE=${BATCH_SIZE:-32}
CALIB_SIZE=${PTQ_CALIB_SIZE:-128}
CALIB_BATCH_SIZE=${PTQ_CALIB_BATCH_SIZE:-8}
QUICK_BATCHES=${PTQ_QUICK_BATCHES:-50}
NUM_GROUPS=${PTQ_NUM_GROUPS:-4}
TAU_PERCENTILE=${PTQ_TAU_PERCENTILE:-85}
HIGH_BIT=${PTQ_HIGH_BIT:-8}
LOW_BIT=${PTQ_LOW_BIT:-4}
HIGH_RATIO_THRESHOLD=${PTQ_HIGH_RATIO_THRESHOLD:-0.2}
ALPHA=${PTQ_ALPHA:-0.5}
BETA=${PTQ_BETA:-0.25}
GAMMA=${PTQ_GAMMA:-0.25}

# ---- 运动感知参数 (步骤5) ----
MOTION_CANDIDATES=${PTQ_MOTION_CANDIDATES:-32}
MOTION_ETA=${PTQ_MOTION_ETA:-0.5}
MOTION_TAU=${PTQ_MOTION_TAU:-0.7}
MOTION_RHO=${PTQ_MOTION_RHO:-0.2}
MOTION_CAPTURE_BATCHES=${PTQ_MOTION_CAPTURE_BATCHES:-0}

# ---- 评测参数 ----
TEST_NUM_SEGMENT=${TEST_NUM_SEGMENT:-${TEST_SEG_DEFAULT}}
TEST_NUM_CROP=${TEST_NUM_CROP:-3}

# ---- 保存量化模型 ----
SAVE_QUANTIZED=${SAVE_QUANTIZED:-0}
QUANTIZED_OUTPUT_DIR=${QUANTIZED_OUTPUT_DIR:-/data/liyifan24/VideoMamba/output_pth}
QUANTIZED_MODEL_PATH=${QUANTIZED_MODEL_PATH:-}

# ============================================================
# 实验预设: 设置 E_ 前缀默认值
# ============================================================
case "${EXPERIMENT}" in
  uniform-w8)
    E_QUANT='uniform';   E_UNIFORM_BIT=8; E_BIT_ALLOC='rank';      E_SCALE='minmax'; E_FRAC=0.5
    ;;
  uniform-w4)
    E_QUANT='uniform';   E_UNIFORM_BIT=4; E_BIT_ALLOC='rank';      E_SCALE='minmax'; E_FRAC=0.5
    ;;
  mixed-rank-minmax)
    E_QUANT='mixed';    E_UNIFORM_BIT=4; E_BIT_ALLOC='rank';      E_SCALE='minmax'; E_FRAC=0.5
    ;;
  mixed-rank-motion)
    E_QUANT='mixed';    E_UNIFORM_BIT=4; E_BIT_ALLOC='rank';      E_SCALE='motion'; E_FRAC=0.5
    ;;
  mixed-threshold-minmax)
    E_QUANT='mixed';    E_UNIFORM_BIT=4; E_BIT_ALLOC='threshold';  E_SCALE='minmax'; E_FRAC=0.5
    ;;
  *)
    echo "ERROR: Unknown EXPERIMENT='${EXPERIMENT}'"
    echo "Available: uniform-w8, uniform-w4, mixed-rank-minmax, mixed-rank-motion, mixed-threshold-minmax"
    exit 1
    ;;
esac

# ---- 应用环境变量覆盖 (env > 预设) ----
QUANT_METHOD=${PTQ_QUANT_METHOD:-$E_QUANT}
UNIFORM_BIT=${PTQ_UNIFORM_BIT:-$E_UNIFORM_BIT}
BIT_ALLOCATION_MODE=${PTQ_BIT_ALLOCATION_MODE:-$E_BIT_ALLOC}
SCALE_MODE=${PTQ_SCALE_MODE:-$E_SCALE}
HIGH_BLOCK_FRACTION=${PTQ_HIGH_BLOCK_FRACTION:-$E_FRAC}

# ---- 输出目录 ----
JOB_NAME="ptq_${EXPERIMENT}${JOB_SUFFIX_DEFAULT}"
OUTPUT_DIR="${OUTPUT_BASE}/${JOB_NAME}"
mkdir -p "${OUTPUT_DIR}"
if [ -n "${QUANTIZED_MODEL_PATH}" ]; then
  QUANT_PATH="${QUANTIZED_MODEL_PATH}"
else
  QUANT_PATH="${QUANTIZED_OUTPUT_DIR}/${QUANT_SUBDIR:+${QUANT_SUBDIR}/}${JOB_NAME}.pth"
fi

# ============================================================
# 打印实验配置
# ============================================================
echo "============================================================"
echo " VideoMamba PTQ Experiment"
echo "============================================================"
echo "  EXPERIMENT        : ${EXPERIMENT}"
echo "  DATASET           : ${DATASET}  (data_set=${DATA_SET}, nb_classes=${NB_CLASSES})"
echo "  GPU               : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, nproc=${NPROC}"
echo "  CKPT              : ${CKPT}"
echo "  QUANT_METHOD      : ${QUANT_METHOD}"
echo "  SCALE_MODE        : ${SCALE_MODE}"
echo "  BIT_ALLOC_MODE    : ${BIT_ALLOCATION_MODE}"
echo "  HIGH_BLOCK_FRAC   : ${HIGH_BLOCK_FRACTION}"
if [ "${QUANT_METHOD}" = "uniform" ]; then
  echo "  UNIFORM_BIT       : ${UNIFORM_BIT}"
fi
echo "  CALIB_SIZE        : ${CALIB_SIZE} samples"
echo "  CALIB_SUB_BATCH   : ${CALIB_BATCH_SIZE} (0=no split)"
echo "  QUICK_BATCHES     : ${QUICK_BATCHES}"
echo "  NUM_GROUPS        : ${NUM_GROUPS}"
echo "  TAU_PERCENTILE    : ${TAU_PERCENTILE}"
echo "  ALPHA/BETA/GAMMA  : ${ALPHA} / ${BETA} / ${GAMMA}"
if [ "${SCALE_MODE}" = "motion" ]; then
  echo "  MOTION candidates : ${MOTION_CANDIDATES}"
  echo "  MOTION eta/tau/rho: ${MOTION_ETA} / ${MOTION_TAU} / ${MOTION_RHO}"
  echo "  MOTION cap_batches: ${MOTION_CAPTURE_BATCHES} (0=reuse calib)"
fi
echo "  TEST_SEG/CROP      : ${TEST_NUM_SEGMENT} / ${TEST_NUM_CROP}"
echo "  OUTPUT_DIR        : ${OUTPUT_DIR}"
echo "  SAVE_QUANTIZED    : ${SAVE_QUANTIZED}"
if [ "${SAVE_QUANTIZED}" = "1" ]; then
  echo "  QUANT_MODEL_PATH  : ${QUANT_PATH}"
fi
echo "============================================================"

# ============================================================
# 构建 PTQ 参数
# ============================================================
PTQ_ARGS=(
  --ptq_enable
  --quant_method "${QUANT_METHOD}"
  --ptq_calib_size "${CALIB_SIZE}"
  --ptq_calib_batch_size "${CALIB_BATCH_SIZE}"
  --ptq_quick_batches "${QUICK_BATCHES}"
  --ptq_num_groups "${NUM_GROUPS}"
  --ptq_tau_percentile "${TAU_PERCENTILE}"
  --ptq_high_ratio_threshold "${HIGH_RATIO_THRESHOLD}"
  --ptq_high_bit "${HIGH_BIT}"
  --ptq_low_bit "${LOW_BIT}"
  --ptq_alpha "${ALPHA}"
  --ptq_beta "${BETA}"
  --ptq_gamma "${GAMMA}"
  --ptq_cls_token_position auto
  --ptq_bit_allocation_mode "${BIT_ALLOCATION_MODE}"
  --ptq_high_block_fraction "${HIGH_BLOCK_FRACTION}"
  --ptq_scale_mode "${SCALE_MODE}"
)

if [ "${QUANT_METHOD}" = "uniform" ]; then
  PTQ_ARGS+=(--ptq_uniform_bit "${UNIFORM_BIT}")
fi

if [ "${SCALE_MODE}" = "motion" ]; then
  PTQ_ARGS+=(
    --ptq_motion_candidates "${MOTION_CANDIDATES}"
    --ptq_motion_eta "${MOTION_ETA}"
    --ptq_motion_tau "${MOTION_TAU}"
    --ptq_motion_rho "${MOTION_RHO}"
    --ptq_motion_capture_batches "${MOTION_CAPTURE_BATCHES}"
  )
fi

if [ "${SAVE_QUANTIZED}" = "1" ]; then
  mkdir -p "$(dirname "${QUANT_PATH}")"
  PTQ_ARGS+=(--ptq_save_quantized_model --ptq_quantized_model_path "${QUANT_PATH}")
fi

# ============================================================
# 运行评测 (单卡用 python, 多卡用 torchrun --dist_eval)
# ============================================================
COMMON_ARGS=(
    --model videomamba_middle
    --finetune "${CKPT}"
    --data_path "${DATA_PATH}"
    --prefix "${PREFIX}"
    --data_set "${DATA_SET}"
    --split ' '
    --nb_classes "${NB_CLASSES}"
    --log_dir "${OUTPUT_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --batch_size "${BATCH_SIZE}"
    --input_size 224
    --short_side_size 224
    --num_frames 16
    --num_workers 8
    --tubelet_size 1
    --test_num_segment "${TEST_NUM_SEGMENT}"
    --test_num_crop "${TEST_NUM_CROP}"
    --eval
    --bf16
    "${USE_DECORD_FLAG[@]}"
    "${PTQ_ARGS[@]}"
)

# --eval_data_path 仅 k400 (Kinetics_sparse test 分支使用); SSV2 分支固定读 data_path/test.csv
if [ -n "${EVAL_DATA_PATH}" ]; then
  COMMON_ARGS+=(--eval_data_path "${EVAL_DATA_PATH}")
fi

# --filename_tmpl 仅 ssv2 (抽帧文件名模板, 默认 img_{:05}.jpg 与本数据集不匹配)
if [ -n "${FILENAME_TMPL}" ]; then
  COMMON_ARGS+=(--filename_tmpl "${FILENAME_TMPL}")
fi

if [ "${NPROC}" -le 1 ]; then
    echo "[run] Single GPU mode (python, no dist_eval)"
    python run_class_finetuning_ptq.py "${COMMON_ARGS[@]}"
else
    echo "[run] Multi-GPU mode (torchrun --nproc_per_node=${NPROC} --dist_eval)"
    torchrun \
      --nproc_per_node="${NPROC}" \
      --master_port="${MASTER_PORT}" \
      run_class_finetuning_ptq.py \
        "${COMMON_ARGS[@]}" \
        --dist_eval
fi
