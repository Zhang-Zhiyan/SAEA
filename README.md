# Stimuli-Aware Emotion Adaptor for Enhancing LLM in Affective Explanation Captioning

<div align="center">

**[Stimuli-Aware Emotion Adaptor for Enhancing LLM in Affective Explanation Captioning]()**

[Zhiyan Zhang]()<sup>1</sup>, [Peipei Song]()<sup>1</sup>†, [Jinpeng Hu]()<sup>2</sup>, [Weidong Chen]()<sup>1</sup>, [Lin Ni]()<sup>1</sup>, [Xun Yang]()<sup>1</sup>

<sup>1</sup> University of Science and Technology of China &nbsp;&nbsp; <sup>2</sup> Hefei University of Technology

*ICASSP 2026*

†Corresponding author

</div>

---

## Overview

Affective Explanation Captioning (AEC) requires not only recognizing viewer-perceived emotions evoked by an image, but also generating natural-language explanations for those emotions. We propose **SAEA (Stimuli-Aware Emotion Adaptor)**, which explicitly enhances LLMs with emotion-aware visual priors by:

1. **Emotional Stimuli Estimation** — learning positive/negative emotion prototypes to score each image patch by its emotional relevance.
2. **Emotional Token Selection** — reweighting and compressing tokens via spatial pooling and Q-Former aggregation.
3. **Emotion Distribution Learning** — using KL-divergence against soft label distributions to capture diverse affective cues beyond a single dominant label.

The extracted emotion features are injected into an LLM (GPT-2) as cross-attention memory, guiding coherent and emotion-grounded explanation generation.

<div align="center">
<a href="images/image2.jpg">Framework figure</a>
<br>
<em>Two-stage SAEA framework: (left) emotion adaptor stage; (right) LLM explanation generation stage.</em>
</div>


---

## Repository Structure

```
SAEA/
├── train_emotion_expert.py    # Stage 1: train SAEA emotion adaptor
├── train_hybrid_gpt2.py       # Stage 2: train LLaVA-GPT2 hybrid model
├── train_stage1.sh            # Stage 1 launch script
├── train_stage2.sh            # Stage 2 launch script
│
├── models/
│   ├── hybrid_model.py        # LlavaGpt2Hybrid (full two-stage model)
│   ├── feature_adapters.py    # Feature adapters: 896→768 / 4096→768
│   └── gpt2_inputs.py         # GPT-2 input packing & special tokens
│
├── datasets/
│   └── emotion_dataset.py     # EmotionDataset (Stage 1) + ArtEmisXDataset (Stage 2)
│
├── utils/
│   └── eval_utils.py          # Generation sampling helpers
│
├── llava/                     # Modified LLaVA-OneVision (our emotion-aware fork)
│   └── model/
│       ├── emotion_expert.py  # EmotionOV — core SAEA module
│       ├── llava_arch.py      # LLaVA architecture with emotion feature injection
│       └── experts/
│           └── emotion_expert_v1.py
│
├── reproduce/
│   └── models/gpt.py          # GPT-2 with cross-attention support
│
└── cococaption/               # COCO caption evaluation toolkit
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Zhang-Zhiyan/SAEA
cd SAEA
```

### 2. Create the conda environment

```bash
# Option A: from environment.yml (recommended)
conda env create -f environment.yml
conda activate saea

# Option B: manual install
conda create -n saea python=3.10 -y
conda activate saea
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Install the local LLaVA fork

```bash
pip install -e .
```

### 4. Download Stanford CoreNLP (for SPICE metric, optional)

```bash
bash cococaption/get_stanford_models.sh
```

---

## Pretrained Models

| Component                  | Description                   | Source                                                       |
| -------------------------- | ----------------------------- | ------------------------------------------------------------ |
| LLaVA-OneVision-Qwen2-0.5B | Visual backbone               | [HuggingFace](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov) |
| SigLIP-so400m-patch14-384  | Visual encoder (inside LLaVA) | [HuggingFace](https://huggingface.co/google/siglip-so400m-patch14-384) |
| GPT-2                      | Language decoder              | [HuggingFace](https://huggingface.co/openai-community/gpt2)  |


---
## Datasets

The 9 emotion categories (in order): `amusement`, `awe`, `contentment`, `excitement`, `anger`, `disgust`, `fear`, `sadness`, `something else`.

| Dataset | Images | Source | Split (train/val/test) |
|---------|--------|--------|------------------------|
| [Affection](https://affective-explanations.org/) | real-world | Flickr | 85% / 5% / 10% |
| [ArtEmis v1.0](https://www.artemisdataset.org/) | WikiArt | WikiArt | 85% / 5% / 10% |
| [ArtEmis v2.0](https://www.artemisdataset-v2.org/) | WikiArt | WikiArt | 85% / 5% / 10% |

After downloading, organise as:

```
data/
├── images/
│   ├── affection/         # real-world images
│   └── wikiart/           # WikiArt paintings
└── annotations/
    ├── artEmisX_train.json
    ├── artEmisX_val.json
    ├── artEmisX_test.json
    └── artEmisX_test_annot_exp.json   # COCO-format ground truth for eval
```
The processing steps of the dataset can be referred to at [SEVLM](https://github.com/BetterZH/SEVLM-code)
---

## Training

### Stage 1 — Emotion Expert (SAEA)

Edit `train_stage1.sh` to set your data paths, then run:

```bash
bash train_stage1.sh
```

Or run directly:

```bash
python train_emotion_expert.py \
    --train_data_path data/annotations/artEmisX_train.json \
    --val_data_path   data/annotations/artEmisX_val.json \
    --image_root      data/images/wikiart \
    --model_checkpoint pretrained/llava-onevision-qwen2-0.5b-ov \
    --output_dir      checkpoints_stage1 \
    --batch_size 64 \
    --num_epochs 4 \
    --learning_rate 1e-4 \
    --patience 5
```

The best checkpoint (selected by validation accuracy) is saved to `checkpoints_stage1/emotion_expert_stage1_best.pth`.

**Key hyperparameters:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 32 | Batch size (paper uses 64 on A100) |
| `--num_epochs` | 15 | Max epochs (paper uses 4) |
| `--learning_rate` | 1e-4 | Initial LR |
| `--patience` | 5 | Early-stop patience |
| `--max_val_samples` | 20000 | Val samples cap (speed) |

---

### Stage 2 — Hybrid Model (LLaVA + GPT-2)

Edit `train_stage2.sh` and set `EMOTION_EXPERT_CKPT` to the Stage 1 best checkpoint, then:

```bash
bash train_stage2.sh
```

Or run directly (single GPU):

```bash
python train_hybrid_gpt2.py \
    --llava_ckpt          pretrained/llava-onevision-qwen2-0.5b-ov \
    --emotion_expert_ckpt checkpoints_stage1/emotion_expert_stage1_best.pth \
    --gpt2_path           pretrained/gpt2/pretrain_model \
    --gpt2_tokenizer_path pretrained/gpt2/pretrain_tokenizer \
    --train_json          data/annotations/artEmisX_train.json \
    --val_json            data/annotations/artEmisX_test.json \
    --val_coco_ann        data/annotations/artEmisX_test_annot_exp.json \
    --image_root          data/images/wikiart \
    --save_dir            checkpoints_stage2 \
    --caption_save_dir    captions_stage2 \
    --llava_size          0.5b \
    --gpu_mode single --gpu_id 0 \
    --epochs 4 --bs 64 --lr 2e-5
```

Multi-GPU DDP (recommended):

```bash
python train_hybrid_gpt2.py \
    ... \
    --gpu_mode ddp --gpu_ids "0,1"
```

**Key hyperparameters:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--llava_size` | `0.5b` | Backbone: `0.5b` (hidden=896) or `7b` (hidden=4096) |
| `--epochs` | 15 | Training epochs (paper uses 4) |
| `--bs` | 32 | Batch size (paper uses 64 on A100) |
| `--lr` | 2e-5 | Learning rate |
| `--eval_top_p` | 0.9 | Nucleus sampling p |

COCO metrics (BLEU-1/2/3/4, METEOR, ROUGE-L) are computed automatically after each epoch and saved to `captions_stage2/scores_epoch_N.json`.

---

## Evaluation

BLEU, METEOR, and ROUGE-L are computed automatically during Stage 2 training using the bundled `cococaption` toolkit. To run evaluation standalone on a saved checkpoint, pass `--resume_from_checkpoint` and set `--epochs` to 0 (or any epoch ≤ the loaded epoch).

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhang2026saea,
  title     = {Stimuli-Aware Emotion Adaptor for Enhancing LLM in Affective Explanation Captioning},
  author    = {Zhang, Zhiyan and Song, Peipei and Hu, Jinpeng and Chen, Weidong and Ni, Lin and Yang, Xun},
  booktitle = {ICASSP},
  year      = {2026}
}
```

---

## Acknowledgement

This work was supported by the National Natural Science Foundation of China (62402471, U22A2094, 62302474, 62402158, 62272435). This research was also supported by the advanced computing resources provided by the Supercomputing Center of the USTC.

This codebase builds upon [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT), [GPT-2](https://github.com/openai/gpt-2), and the [SEVLM](https://github.com/YuanJianhao508/SEVLM) codebase. We thank the authors for their open-source contributions.

---

## License

This project is released under the [Apache 2.0 License](LICENSE).
