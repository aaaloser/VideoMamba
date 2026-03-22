#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

JOB_NAME='videomamba_middle_mask_eval_f8_res224_1gpu'
OUTPUT_DIR="./logs/${JOB_NAME}"
BATCH_SIZE=8

# ===== 路径配置 =====
PREFIX='/data/liyifan24/Datasets/Kinetics-400/'
DATA_PATH='/data/liyifan24/Datasets/Kinetics-400/'
EVAL_DATA_PATH='/data/liyifan24/Datasets/Kinetics-400/val.csv'
CKPT='/data/liyifan24/VideoMamba/pretrain_model/videomamba_m16_k400_mask_ft_f16_res224.pth'
# ====================

python run_class_finetuning.py \
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
  --num_frames 8 \
  --input_size 224 \
  --short_side_size 224 \
  --num_workers 8 \
  --tubelet_size 1 \
  --test_num_segment 1 \
  --test_num_crop 1 \
  --eval \
  --bf16