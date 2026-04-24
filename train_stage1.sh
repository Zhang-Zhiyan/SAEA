#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

TRAIN_DATA_PATH="<path/to/artEmisX_train.json>"
VAL_DATA_PATH="<path/to/artEmisX_val.json>"
IMAGE_ROOT="<path/to/artwork/images>"

MODEL_CHECKPOINT="<path/to/llava-onevision-qwen2-0.5b-ov>"

OUTPUT_DIR="./checkpoints_stage1"
mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo "  Stage 1: Emotion Expert Training"
echo "  GPU : $CUDA_VISIBLE_DEVICES"
echo "  Data: $TRAIN_DATA_PATH"
echo "  Out : $OUTPUT_DIR"
echo "========================================================"

python train_emotion_expert.py \
    --train_data_path "$TRAIN_DATA_PATH" \
    --val_data_path   "$VAL_DATA_PATH" \
    --image_root      "$IMAGE_ROOT" \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --output_dir      "$OUTPUT_DIR" \
    --batch_size      32 \
    --num_epochs      15 \
    --learning_rate   1e-4 \
    --weight_decay    0.01 \
    --num_workers     8 \
    --patience        5 \
    --save_every      1 \
    --max_val_samples 20000

echo ""
echo "Stage 1 complete. Best model: $OUTPUT_DIR/emotion_expert_stage1_best.pth"
echo "Run train_stage2.sh to proceed with Stage 2 training."
