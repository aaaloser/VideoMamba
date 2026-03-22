#!/usr/bin/env bash
set -euo pipefail

export MASTER_PORT=$((12000 + RANDOM % 20000))
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

JOB_NAME='videomamba_middle_mask_eval_f8_res224_heavy'
OUTPUT_DIR="./logs/${JOB_NAME}"
BATCH_SIZE=64

# ===== 路径配置 =====
PREFIX='/data/liyifan24/Datasets/Kinetics-400/'
DATA_PATH='/data/liyifan24/Datasets/Kinetics-400/'
EVAL_DATA_PATH='/data/liyifan24/Datasets/Kinetics-400/val.csv'
CKPT='/data/liyifan24/VideoMamba/pretrain_model/videomamba_m16_k400_mask_ft_f16_res224.pth'
# ====================

torchrun \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  run_class_finetuning.py \
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
    --num_workers 24 \
    --tubelet_size 1 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --dist_eval \
    --eval \
    --bf16