#!/bin/bash
# IDU Training with FlowEdit - NYC_004
# Date: 2026-03-01
# GPU: 11.73 GiB VRAM
# Based on successful JAX_068 configuration

# Environment variable for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training parameters
DATASET_PATH="./data/datasets_NYC/NYC_004/"
OUTPUT_PATH="./outputs/NYC_idu/NYC_004"
CHECKPOINT="./outputs/NYC/NYC_004/chkpnt30000.pth"

# Execute training
uv run train.py \
  -s "$DATASET_PATH" \
  -m "$OUTPUT_PATH" \
  --start_checkpoint "$CHECKPOINT" \
  --iterative_datasets_update \
  --eval \
  --port 6209 \
  --kernel_size 0.1 \
  --resolution 1 \
  --sh_degree 1 \
  --appearance_enabled \
  --lambda_depth 0 \
  --lambda_opacity 0 \
  --idu_opacity_reset_interval 5000 \
  --idu_refine \
  --idu_num_samples_per_view 2 \
  --densify_grad_threshold 0.002 \
  --idu_num_cams 1 \
  --idu_use_flow_edit \
  --idu_render_size 256 \
  --idu_flow_edit_n_min 4 \
  --idu_flow_edit_n_max 10 \
  --idu_grid_size 1 \
  --idu_grid_width 512 \
  --idu_grid_height 512 \
  --idu_episode_iterations 10000 \
  --idu_iter_full_train 0 \
  --idu_opacity_cooling_iterations 500 \
  --lambda_pseudo_depth 0.5 \
  --idu_densify_until_iter 9000 \
  --idu_train_ratio 0.75
