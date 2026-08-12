

# work1
## K400
```bash
cd /data/liyifan24/VideoMamba/videomamba/video_sm

CUDA_VISIBLE_DEVICES=2 \
BATCH_SIZE=32 \
PTQ_CALIB_SIZE=32 \
PTQ_CALIB_BATCH_SIZE=4 \
QUANTIZED_MODEL_PATH=/data/liyifan24/VideoMamba/output_pth/K400/work1/run_motion_test.pth \
SAVE_QUANTIZED=1 \
EXPERIMENT=mixed-rank-motion \
  nohup bash run_ptq_experiments.sh > /data/liyifan24/VideoMamba/output_pth/K400/work1/run_motion_test.log 2>&1 &
# 用 GPU 2（nvidia-smi 里的编号）
```
