#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0,1
GPU_MODE="ddp"
GPU_IDS="0,1"
GPU_ID=0

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

TRAIN_JSON="<path/to/artEmisX_train.json>"
VAL_JSON="<path/to/artEmisX_test.json>"
VAL_COCO_ANN="<path/to/artEmisX_test_annot_exp.json>"
IMAGE_ROOT="<path/to/artwork/images>"

LLAVA_CKPT="<path/to/llava-onevision-qwen2-0.5b-ov>"
EMOTION_EXPERT_CKPT="./checkpoints_stage1/emotion_expert_stage1_best.pth"
GPT2_PATH="<path/to/gpt2/pretrain_model>"
GPT2_TOKENIZER_PATH="<path/to/gpt2/pretrain_tokenizer>"

SAVE_DIR="./checkpoints_stage2"
CAPTION_SAVE_DIR="./captions_stage2"
mkdir -p "$SAVE_DIR" "$CAPTION_SAVE_DIR"

echo "========================================================"
echo "  Stage 2: LLaVA-GPT2 Hybrid Training"
echo "  GPU mode  : $GPU_MODE  ($CUDA_VISIBLE_DEVICES)"
echo "  Stage1 ckpt: $EMOTION_EXPERT_CKPT"
echo "  Out       : $SAVE_DIR"
echo "========================================================"

python train_hybrid_gpt2.py \
    --llava_ckpt           "$LLAVA_CKPT" \
    --emotion_expert_ckpt  "$EMOTION_EXPERT_CKPT" \
    --gpt2_path            "$GPT2_PATH" \
    --gpt2_tokenizer_path  "$GPT2_TOKENIZER_PATH" \
    --train_json           "$TRAIN_JSON" \
    --val_json             "$VAL_JSON" \
    --val_coco_ann         "$VAL_COCO_ANN" \
    --image_root           "$IMAGE_ROOT" \
    --save_dir             "$SAVE_DIR" \
    --caption_save_dir     "$CAPTION_SAVE_DIR" \
    --llava_size           "0.5b" \
    --gpu_mode             "$GPU_MODE" \
    --gpu_id               "$GPU_ID" \
    --gpu_ids              "$GPU_IDS" \
    --epochs               15 \
    --bs                   32 \
    --lr                   2e-5 \
    --eval_top_p           0.9 \
    --eval_temperature     1.0

echo ""
echo "Stage 2 complete. Best checkpoint saved to: $SAVE_DIR"
