

# work1
## K400
```bash
cd /data/liyifan24/VideoMamba/videomamba/video_sm

CUDA_VISIBLE_DEVICES=2 \
DATASET=k400 \
BATCH_SIZE=32 \
PTQ_CALIB_SIZE=32 \
PTQ_CALIB_BATCH_SIZE=4 \
QUANTIZED_MODEL_PATH=/data/liyifan24/VideoMamba/output_pth/K400/work1/run_motion_test.pth \
SAVE_QUANTIZED=1 \
EXPERIMENT=mixed-rank-motion \
  nohup bash run_ptq_experiments.sh > /data/liyifan24/VideoMamba/output_pth/K400/work1/run_motion_test.log 2>&1 &
```

## SSv2
数据集为抽帧格式（`/data/liyifan24/Datasets/somethingv2/frame/<id>/000001.jpg`，6 位零填充；脚本已传 `--filename_tmpl '{:06}.jpg'`），
脚本自动从 `train_videofolder.txt`/`val_videofolder.txt` 生成 metadata
（`output_pth/SSv2/metadata/{train,val,test}.csv`，`test.csv` 复用 val，因 SSv2 测试标签不公开）。
评测协议为官方 2 段 × 3 crop（可用 `TEST_NUM_SEGMENT`/`TEST_NUM_CROP` 覆盖）。

```bash
cd /data/liyifan24/VideoMamba/videomamba/video_sm
# 默认2×3 crop评测协议
CUDA_VISIBLE_DEVICES=1 \
DATASET=ssv2 \
BATCH_SIZE=32 \
PTQ_CALIB_SIZE=32 \
PTQ_CALIB_BATCH_SIZE=4 \
QUANTIZED_MODEL_PATH=/data/liyifan24/VideoMamba/output_pth/SSv2/work1/run_motion_test.pth \
SAVE_QUANTIZED=1 \
EXPERIMENT=mixed-rank-motion \
  nohup bash run_ptq_experiments.sh > /data/liyifan24/VideoMamba/output_pth/SSv2/work1/run_motion_test.log 2>&1 &

# 4×3 crop评测协议
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

# Quamba
## K400
```bash
cd /data/liyifan24/VideoMamba/videomamba/video_sm

CUDA_VISIBLE_DEVICES=2 \
DATASET=k400 \
BATCH_SIZE=32 \
PTQ_CALIB_SIZE=32 \
PTQ_CALIB_BATCH_SIZE=4 \
QUANTIZED_MODEL_PATH=/data/liyifan24/VideoMamba/output_pth/K400/ptq_quamba-8.pth \
SAVE_QUANTIZED=1 \
EXPERIMENT=quamba-8 \
  nohup bash run_ptq_experiments.sh > /data/liyifan24/VideoMamba/output_pth/K400/ptq_quamba-8.log 2>&1 &
```

## SSv2
```bash
cd /data/liyifan24/VideoMamba/videomamba/video_sm

CUDA_VISIBLE_DEVICES=3 \
DATASET=ssv2 \
BATCH_SIZE=32 \
PTQ_CALIB_SIZE=32 \
PTQ_CALIB_BATCH_SIZE=4 \
QUANTIZED_MODEL_PATH=/data/liyifan24/VideoMamba/output_pth/SSv2/ptq_quamba-8.pth \
SAVE_QUANTIZED=1 \
EXPERIMENT=quamba-8 \
  nohup bash run_ptq_experiments.sh > /data/liyifan24/VideoMamba/output_pth/SSv2/ptq_quamba-8.log 2>&1 &
```

# 4×3 crop评测协议
```bash
cd /data/liyifan24/VideoMamba/videomamba/video_sm

CUDA_VISIBLE_DEVICES=0 \
DATASET=ssv2 \
TEST_NUM_SEGMENT=4 \
TEST_NUM_CROP=3 \
BATCH_SIZE=32 \
PTQ_CALIB_SIZE=32 \
PTQ_CALIB_BATCH_SIZE=4 \
QUANTIZED_MODEL_PATH=/data/liyifan24/VideoMamba/output_pth/SSv2/ptq_quamba-8_4x3.pth \
SAVE_QUANTIZED=1 \
EXPERIMENT=quamba-8 \
  nohup bash run_ptq_experiments.sh > /data/liyifan24/VideoMamba/output_pth/SSv2/ptq_quamba-8_4x3.log 2>&1 &
```
