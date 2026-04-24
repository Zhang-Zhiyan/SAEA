"""
Two-stage hybrid model: LLaVA vision encoder + EmotionOV emotion expert + GPT-2 decoder.

Stage 1 trains only the EmotionOV emotion expert (see train_emotion_expert.py).
Stage 2 trains the full hybrid model end-to-end (see train_hybrid_gpt2.py).
"""

import torch
import torch.nn as nn
import re
import math
from typing import Dict, List, Tuple

from llava.model.llava_arch import unpad_image
from llava.mm_utils import get_anyres_image_grid_shape
from llava.model.emotion_expert import EmotionOV
from models.feature_adapters import SmartFeatureAdapter, SmartFeatureAdapter896
from models.gpt2_inputs import (
    get_predicted_emotion_names,
    build_dynamic_questions,
    pack_gpt2_inputs,
    ensure_special_tokens,
)


class LlavaGpt2Hybrid(nn.Module):
    """LLaVA vision + EmotionOV emotion expert + GPT-2 decoder with cross-attention.

    The model follows a two-stage training strategy:
    - Stage 1 (separate): trains EmotionOV emotion expert to classify emotions.
    - Stage 2 (this class): trains feature adapters + GPT-2 decoder for
      emotion-guided affective explanation generation.

    Forward flow:
        images -> LLaVA encoder -> mm_projector -> img_adapter [B,L,768]
               -> EmotionOV  -> emo_adapter  [B,768] global, [B,L',768] local
        GPT-2 cross-attends on (global, local) memory to generate explanation.

    Args:
        llava_backbone:    Loaded LLaVA model (frozen during Stage 2, except mm_projector).
        emotion_expert:    EmotionOV instance (loaded from Stage 1 checkpoint).
        gpt2_decoder:      GPT2LMHeadModel with cross-attention support.
        gpt2_tokenizer:    GPT-2 tokenizer (will receive special tokens).
        keep_mm_projector: If True, mm_projector is kept trainable in Stage 2.
        prefix_len:        Number of learnable prefix tokens (0 = disabled).
        max_length:        Maximum sequence length for GPT-2 inputs.
    """

    def __init__(
        self,
        llava_backbone,
        emotion_expert: EmotionOV,
        gpt2_decoder,
        gpt2_tokenizer,
        keep_mm_projector: bool = True,
        prefix_len: int = 0,
        max_length: int = 256,
    ):
        super().__init__()
        self.llava = llava_backbone
        self.emotion_expert = emotion_expert
        self.gpt2 = gpt2_decoder
        self.tokenizer = gpt2_tokenizer
        self.prefix_len = prefix_len
        self.max_length = max_length
        self.keep_mm_projector = keep_mm_projector

        if hasattr(self.gpt2.config, 'len_prefix'):
            self.gpt2.config.len_prefix = prefix_len

                                                                                                    
        self.img_adapter = SmartFeatureAdapter(896, 768)
        self.emo_adapter = SmartFeatureAdapter(896, 768)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        ensure_special_tokens(self.tokenizer)
        self.gpt2.resize_token_embeddings(len(self.tokenizer))

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if images.dtype != torch.float32:
            images = images.to(torch.float32)
        return self.llava.encode_images(images)

    @torch.no_grad()
    def project_images(self, image_features: torch.Tensor) -> torch.Tensor:
        if self.keep_mm_projector:
            return self.llava.project_images(image_features)
        return image_features

    def get_emotion_features(
        self,
        ov_image_features: torch.Tensor,
        sig_multi_level_features: List[torch.Tensor],
        split_sizes: List[int],
    ):
        local_emo, global_emo, emotion_preds = self.llava.get_emotion_features(
            ov_image_features, sig_multi_level_features, split_sizes
        )
        return local_emo, global_emo, emotion_preds

    def _build_encoder_states(
        self,
        proj_img_feats: torch.Tensor,
        local_emo_feats: torch.Tensor,
        global_emo_feats: torch.Tensor,
        split_sizes: List[int],
        image_sizes: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build GPT-2 cross-attention memory following LLaVA anyres/unpad logic.

        Returns:
            global_768: [B, 768]  – compact global emotion memory
            local_768:  [B, L, 768] – spatial image+emotion memory
        """
        B = len(split_sizes)
        D = proj_img_feats.shape[-1]
        cfg = self.llava.get_model().config
        vt = self.llava.get_model().get_vision_tower()
        height = width = vt.num_patches_per_side
        mm_patch_merge_type = getattr(cfg, "mm_patch_merge_type", "flat")
        image_aspect_ratio = getattr(cfg, "image_aspect_ratio", "square")

        img_split = torch.split(proj_img_feats, split_sizes)
        emo_split = torch.split(local_emo_feats, split_sizes, dim=0)

        per_image_seqs = []
        for image_idx, image_feature in enumerate(img_split):
            per_emo = emo_split[image_idx]
            if image_feature.shape[0] > 1:
                major_emotion_feat = per_emo
                base_major_emotion_feat = major_emotion_feat[0]
                major_emotion_feat = major_emotion_feat[1:]
                global_emotion_feat = global_emo_feats[image_idx]
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]

                matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                max_num_patches = int(matched_anyres_max_num_patches.group(1)) if matched_anyres_max_num_patches else 1000000

                vision_tower_image_size = vt.image_size
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    image_sizes[image_idx], cfg.image_grid_pinpoints, vision_tower_image_size
                )

                mef = major_emotion_feat.reshape(num_patch_height, num_patch_width, 2, 2, -1)
                mef = mef.permute(4, 0, 2, 1, 3).contiguous()
                mef = mef.flatten(1, 2).flatten(2, 3)
                mef = torch.cat(
                    (mef, self.llava.get_model().image_newline[:, None, None].expand(*mef.shape[:-1], 1).to(mef.device)),
                    dim=-1,
                )
                mef = mef.flatten(1, 2).transpose(0, 1)

                imf = image_feature.reshape(num_patch_height, num_patch_width, height, width, -1)
                unit = imf.shape[2]
                imf = imf.permute(4, 0, 2, 1, 3).contiguous()
                imf = imf.flatten(1, 2).flatten(2, 3)
                imf = unpad_image(imf, image_sizes[image_idx])
                c, h, w = imf.shape
                times = math.sqrt(h * w / (max_num_patches * unit ** 2))
                if times > 1.1:
                    imf = imf[None]
                    imf = nn.functional.interpolate(imf, [int(h // times), int(w // times)], mode="bilinear")[0]
                imf = torch.cat(
                    (imf, self.llava.get_model().image_newline[:, None, None].expand(*imf.shape[:-1], 1).to(imf.device)),
                    dim=-1,
                )
                imf = imf.flatten(1, 2).transpose(0, 1)

                image_feature = torch.cat((base_image_feature, imf), dim=0)
                emotion_feature = torch.cat((global_emotion_feat, base_major_emotion_feat, mef), dim=0)
            else:
                major_emotion_feat = per_emo[0]
                global_emotion_feat = global_emo_feats[image_idx]
                emotion_feature = torch.cat((global_emotion_feat, major_emotion_feat), dim=0)
                image_feature = image_feature[0]
                if "unpad" in mm_patch_merge_type:
                    image_feature = torch.cat((image_feature, self.llava.get_model().image_newline[None]), dim=0)
                    emotion_feature = torch.cat((emotion_feature, self.llava.get_model().image_newline[None]), dim=0)

            local_seq = torch.cat((image_feature, emotion_feature), dim=0)
            per_image_seqs.append(local_seq)

        max_L = max(x.shape[0] for x in per_image_seqs)
        padded = []
        for x in per_image_seqs:
            if x.shape[0] < max_L:
                pad = torch.zeros(max_L - x.shape[0], D, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            padded.append(x)
        local_feats = torch.stack(padded, dim=0)
        local_768 = self.img_adapter(local_feats).contiguous()
        global_768 = self.emo_adapter(global_emo_feats).squeeze(1).contiguous()
        return global_768, local_768

    def forward(
        self,
        images,
        target_explanations: List[str],
        image_sizes: List[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            images:              Batch tensor [B,3,H,W] or list of tensors.
            target_explanations: Ground-truth explanation strings.
            image_sizes:         List of original (width, height) tuples for unpad.

        Returns:
            dict with 'loss' and 'emotion_preds'.
        """
        device = next(self.parameters()).device

        if isinstance(images, torch.Tensor):
            if images.ndim == 4:
                images_cat = images.to(device).to(torch.float32)
                split_sizes = [1] * images_cat.size(0)
            else:
                raise ValueError(f"Unexpected images tensor shape: {images.shape}")
        else:
            images_cat = torch.cat([im.to(device).to(torch.float32) for im in images], dim=0)
            split_sizes = [im.shape[0] if im.ndim == 4 else 1 for im in images]

        ov_feats, sig_multi_feats = self.encode_images(images_cat)
        proj_img_feats = self.project_images(ov_feats)
        local_emo, global_emo, emotion_preds = self.get_emotion_features(ov_feats, sig_multi_feats, split_sizes)

        if image_sizes is None:
            image_sizes = [(384, 384) for _ in range(len(split_sizes))]

        global_mem, local_mem = self._build_encoder_states(
            proj_img_feats, local_emo, global_emo, split_sizes, image_sizes
        )

        emo_names = get_predicted_emotion_names(emotion_preds)
        questions = build_dynamic_questions(emo_names)
        packed = pack_gpt2_inputs(
            self.tokenizer, questions, target_explanations,
            max_length=self.max_length, prefix_len=self.prefix_len,
        )
        input_ids = packed["input_ids"].to(device)
        labels = packed["labels"].to(device)
        token_type_ids = packed["token_type_ids"].to(device)

        outputs = self.gpt2(
            input_ids=input_ids.contiguous(),
            token_type_ids=token_type_ids.contiguous(),
            encoder_hidden_states=(global_mem.contiguous(), local_mem.contiguous()),
            labels=labels,
            return_dict=True,
        )
        return {
            "loss": outputs.loss[0] if isinstance(outputs.loss, tuple) else outputs.loss,
            "emotion_preds": emotion_preds.detach(),
        }
