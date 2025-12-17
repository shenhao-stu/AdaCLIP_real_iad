#!/bin/bash

# Train AdaCLIP on Real-IAD 256, then run zero-shot prediction on
# Real-IAD variety subset using the best checkpoint.
#
# Usage (from project root):
#   conda activate AdaCLIP
#   bash train_and_predict_realiad.sh
#
# Notes:
# - Batch size is fixed at 1 (model code limitation).
# - Checkpoint is saved to ${SAVE_ROOT}/models/${MODEL_NAME}_best.pth.
# - Prediction outputs go to ./data/Real-IAD/realiad_variety_subset_predict_zero_shot.

set -e

cd "$(dirname "$0")"

# -------------------------
# Training configuration
# -------------------------
SAVE_ROOT="./workspaces/realiad_256"
MODEL="ViT-L-14-336"
EPOCH=10
LR=0.001
BATCH=1
IMG_SIZE=518
PROMPT_DEPTH=4
PROMPT_LEN=5
PROMPT_TYPE="SD"
PROMPT_BRANCH="VL"
USE_HSF="True"
K_CLUSTERS=20

# Build model name consistent with tools/training_tools.py
MODEL_NAME="0s-pretrained-realiad_256-${MODEL}-${PROMPT_TYPE}-${PROMPT_BRANCH}-D${PROMPT_DEPTH}-L${PROMPT_LEN}-HSF-K${K_CLUSTERS}"
CKPT_BEST="${SAVE_ROOT}/models/${MODEL_NAME}_best.pth"

echo "==== Training AdaCLIP on Real-IAD 256 ===="
python train.py \
  --training_data realiad_256 \
  --testing_data realiad_256 \
  --save_path "${SAVE_ROOT}" \
  --model "${MODEL}" \
  --epoch "${EPOCH}" \
  --learning_rate "${LR}" \
  --batch_size "${BATCH}" \
  --image_size "${IMG_SIZE}" \
  --print_freq 1 \
  --valid_freq 1 \
  --prompting_depth "${PROMPT_DEPTH}" \
  --prompting_length "${PROMPT_LEN}" \
  --prompting_type "${PROMPT_TYPE}" \
  --prompting_branch "${PROMPT_BRANCH}" \
  --use_hsf "${USE_HSF}" \
  --k_clusters "${K_CLUSTERS}"

echo "Training finished. Expect best checkpoint at: ${CKPT_BEST}"

# -------------------------
# Prediction configuration
# -------------------------
DATA_ROOT="./data/Real-IAD/realiad_variety_subset"
PRED_ROOT="./data/Real-IAD/realiad_variety_subset_predict_zero_shot"
THRESH=0.5

if [ ! -f "${CKPT_BEST}" ]; then
  echo "Checkpoint not found: ${CKPT_BEST}"
  echo "Please verify training completed successfully."
  exit 1
fi

echo "==== Predicting Real-IAD variety subset (zero-shot) ===="
python predict_realiad_variety_zero_shot.py \
  --data_root "${DATA_ROOT}" \
  --pred_root "${PRED_ROOT}" \
  --ckt_path "${CKPT_BEST}" \
  --model "${MODEL}" \
  --batch_size 1 \
  --threshold "${THRESH}"

echo "All done."


