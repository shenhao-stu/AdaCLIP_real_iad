#!/bin/bash

# Simple helper script to train AdaCLIP on the Real-IAD 256 dataset,
# then use the trained checkpoint later for zero-shot prediction on
# realiad_variety_subset.
#
# Usage (from project root, environment already created):
#   conda activate AdaCLIP
#   bash train_realiad_256.sh

set -e

cd "$(dirname "$0")"

python train.py \
  --training_data realiad_256 \
  --testing_data realiad_256 \
  --save_path ./workspaces/realiad_256 \
  --model ViT-L-14-336 \
  --epoch 10 \
  --learning_rate 0.001 \
  --batch_size 1 \
  --image_size 518 \
  --print_freq 1 \
  --valid_freq 1 \
  --prompting_depth 4 \
  --prompting_length 5 \
  --prompting_type SD \
  --prompting_branch VL \
  --use_hsf True \
  --k_clusters 20


