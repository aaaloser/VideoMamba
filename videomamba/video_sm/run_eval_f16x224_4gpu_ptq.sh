#!/usr/bin/env bash
set -euo pipefail

export MASTER_PORT=$((12000 + RANDOM % 20000))
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

JOB_NAME='videomamba_middle_mask_eval_f16_res224_ptq_uniform4'
OUTPUT_DIR="./logs/${JOB_NAME}"
BATCH_SIZE=32

# Quantization mode setup:
# - QUANT_METHOD='mixed'   : current method (calibration + quick-eval routing)
# - QUANT_METHOD='uniform' : global uniform baseline (W8A16 or W4A16)
#   - set UNIFORM_BIT=8 for Uniform W8A16
#   - set UNIFORM_BIT=4 for Uniform W4A16
QUANT_METHOD='uniform'
UNIFORM_BIT=4

# ===== 按你的实际路径填写 =====
PREFIX='/data/liyifan24/Datasets/Kinetics-400/'
DATA_PATH='/data/liyifan24/Datasets/Kinetics-400/'
EVAL_DATA_PATH='/data/liyifan24/Datasets/Kinetics-400/val_model_label.csv'
CKPT='/data/liyifan24/VideoMamba/pretrain_model/videomamba_m16_k400_mask_ft_f16_res224.pth'
# ====================

torchrun \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  run_class_finetuning_ptq.py \
    --model videomamba_middle \
    --finetune "${CKPT}" \
    --data_path "${DATA_PATH}" \
    --eval_data_path "${EVAL_DATA_PATH}" \
    --prefix "${PREFIX}" \
    --data_set 'Kinetics_sparse' \
    --split ' ' \
    --nb_classes 400 \
    --log_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --input_size 224 \
    --short_side_size 224 \
    --num_frames 16 \
    --num_workers 8 \
    --tubelet_size 1 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --dist_eval \
    --eval \
    --bf16 \
    --ptq_enable \
    --quant_method "${QUANT_METHOD}" \
    --ptq_uniform_bit "${UNIFORM_BIT}" \
    --ptq_calib_batches 4 \
    --ptq_quick_batches 8 \
    --ptq_num_groups 4 \
    --ptq_tau_percentile 85 \
    --ptq_high_ratio_threshold 0.2 \
    --ptq_high_bit 8 \
    --ptq_low_bit 4 \
    --ptq_alpha 0.5 \
    --ptq_beta 0.25 \
    --ptq_gamma 0.25 \
    --ptq_cls_token_position auto \
    --ptq_save_quantized_model \
    --ptq_quantized_model_path /data/liyifan24/VideoMamba/output_pth/${JOB_NAME}.pth
