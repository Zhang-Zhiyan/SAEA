"""
Dataset classes for the two-stage SAEA training framework.

Stage 1 (emotion expert training):
    EmotionDataset  – artEmisX JSON format, returns emotion distributions for KL training.

Stage 2 (hybrid model training):
    ArtEmisXDataset – artEmisX JSON format, returns (image, emotion label, explanation text).
    LlavaDataset    – LLaVA conversation JSON format, returns the same fields.
    EvalDataset     – lightweight dataset for COCO-style captioning evaluation.
"""

import os
import json
import unicodedata
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Optional, Tuple

                                                            
EMOTION_LABELS: List[str] = [
    'amusement', 'awe', 'contentment', 'excitement',
    'anger', 'disgust', 'fear', 'sadness', 'something else',
]


def proc_distribution(distributions: list) -> Tuple[bool, Optional[str]]:
    """Return (has_dominant, dominant_label) if one emotion exceeds 50%."""
    if max(distributions) > 0.5:
        idx = distributions.index(max(distributions))
        return True, EMOTION_LABELS[idx]
    return False, None


                                                                             
                 
                                                                             

class EmotionDataset(Dataset):
    """Dataset for Stage 1 emotion-expert training.

    Reads artEmisX JSON format:
        {sample_id: {image_name, origin_emotion_distribution, emotions, explanations, ...}}

    Each sample returns:
        image:                  pre-processed image tensor (via image_processor)
        emotion_distribution:   soft label tensor [9]
        ground_truth_idx:       index of dominant emotion (-1 if none)
        has_dominant_emotion:   bool flag
        sample_id / image_name: metadata for debugging
    """

    def __init__(self, data_path: str, image_root: str, image_processor, model_config,
                 max_samples: Optional[int] = None):
        from llava.mm_utils import process_images

        self.image_root = image_root
        self.image_processor = image_processor
        self.model_config = model_config
        self._process_images = process_images

        print(f"Loading Stage-1 data from {data_path}")
        with open(data_path, 'r') as f:
            raw = json.load(f)

        self.samples = list(raw.items())
        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id, sample_data = self.samples[idx]
        image_name = sample_data['image_name']
        image_path = os.path.join(self.image_root, f"{image_name}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_root, f"{image_name}.png")
        if not os.path.exists(image_path):
            return None

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self._process_images([image], self.image_processor, self.model_config)[0]

            dist = sample_data['origin_emotion_distribution']
            emotion_distribution = torch.tensor(dist, dtype=torch.float32)
            has_max, dominant = proc_distribution(dist)
            gt_idx = EMOTION_LABELS.index(dominant) if has_max else -1

            return {
                'image': image_tensor,
                'emotion_distribution': emotion_distribution,
                'ground_truth_idx': gt_idx,
                'has_dominant_emotion': has_max,
                'sample_id': sample_id,
                'image_name': image_name,
            }
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None


def collate_fn_stage1(batch):
    """Collate for Stage 1, filtering None samples."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        'images': [b['image'] for b in batch],
        'emotion_distributions': torch.stack([b['emotion_distribution'] for b in batch]),
        'ground_truth_indices': torch.tensor([b['ground_truth_idx'] for b in batch], dtype=torch.long),
        'has_dominant_emotions': [b['has_dominant_emotion'] for b in batch],
        'sample_ids': [b['sample_id'] for b in batch],
        'image_names': [b['image_name'] for b in batch],
    }


                                                                             
                  
                                                                             

class ArtEmisXDataset(Dataset):
    """Dataset for Stage 2 hybrid training – artEmisX JSON format.

    JSON format: {id: {image_name, emotions: [...], explanations: [...], ...}}
    Expands every (emotion, explanation) pair into a separate sample.

    Each sample returns:
        image:       pre-processed image tensor [3,H,W]
        emotion:     emotion label string
        explanation: explanation text
        image_size:  original PIL (width, height)
    """

    def __init__(self, json_path: str, image_root: str, image_processor, max_len: int = 256):
        self.image_root = image_root
        self.image_processor = image_processor
        self.max_len = max_len
        self.samples = []

        with open(json_path, 'r') as f:
            data = json.load(f)

        for data_id, sample in data.items():
            img_name = sample['image_name'] + '.jpg'
            img_path = os.path.join(image_root, img_name)
            for emotion, explanation in zip(sample['emotions'], sample['explanations']):
                self.samples.append({
                    'image': img_path,
                    'emotion': emotion,
                    'explanation': explanation,
                    'data_id': data_id,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = s['image']
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            img = Image.open(unicodedata.normalize('NFC', img_path)).convert('RGB')

        orig_size = img.size
        pix = self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
        return {
            "image": pix,
            "explanation": s['explanation'],
            "emotion": s['emotion'],
            "image_size": orig_size,
        }


class LlavaDataset(Dataset):
    """Dataset for Stage 2 hybrid training – LLaVA conversation JSON format.

    JSON format: [{id, image, conversations: [...], metadata: {ground_truth_emotion, ...}}]

    Returns the same fields as ArtEmisXDataset for drop-in compatibility.
    """

    def __init__(self, json_path: str, image_root: str, image_processor, max_len: int = 256):
        self.image_root = image_root
        self.image_processor = image_processor
        self.max_len = max_len
        self.samples = []

        with open(json_path, 'r') as f:
            data = json.load(f)

        for item in data:
            img_path = os.path.join(image_root, item['image'])
            explanation = item['conversations'][1]['value']
            emotion = item['metadata']['ground_truth_emotion']
            self.samples.append({
                'image': img_path,
                'explanation': explanation,
                'emotion': emotion,
                'id': item['id'],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['image']).convert('RGB')
        orig_size = img.size
        pix = self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
        return {
            "image": pix,
            "explanation": s['explanation'],
            "emotion": s['emotion'],
            "image_size": orig_size,
        }


class EvalDataset(Dataset):
    """Lightweight evaluation dataset for COCO-style captioning metrics.

    JSON format same as ArtEmisXDataset but returns image_id for COCO eval.
    """

    def __init__(self, json_path: str, image_root: str, image_processor):
        self.image_root = image_root
        self.image_processor = image_processor

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.ids = list(data.keys())
        self.data = data

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        did = self.ids[idx]
        sample = self.data[did]
        img_path = os.path.join(self.image_root, sample['image_name'] + '.jpg')
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            img = Image.open(unicodedata.normalize('NFC', img_path)).convert('RGB')

        orig_size = img.size
        pix = self.image_processor.preprocess(img, return_tensors='pt')['pixel_values']
        return {"image": pix, "image_id": int(did), "image_size": orig_size}


def collate_fn_stage2(batch):
    """Collate for Stage 2 training (ArtEmisXDataset / LlavaDataset)."""
    images = torch.stack([b['image'] for b in batch], dim=0)
    exps = [b['explanation'] for b in batch]
    image_sizes = [b['image_size'] for b in batch]
    return {"images": images, "explanations": exps, "image_sizes": image_sizes}
