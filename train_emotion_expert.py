                      
"""
Stage 1: Train the EmotionOV emotion expert module.

Training strategy:
  - Freeze the LLaVA-OneVision backbone (vision encoder + LLM).
  - Only train the EmotionOV emotion expert attached to the SigLIP encoder.
  - Optimise with KL-divergence loss against soft emotion distributions.
  - Select the best checkpoint based on validation set accuracy (not KL loss).
  - Early-stop when validation accuracy stops improving for --patience epochs.

Usage:
    python train_emotion_expert.py \
        --train_data_path <path/to/artEmisX_train.json> \
        --val_data_path   <path/to/artEmisX_val.json> \
        --image_root      <path/to/images/> \
        --model_checkpoint <path/to/llava-onevision-qwen2-0.5b-ov> \
        --output_dir      ./checkpoints_stage1
"""

import os
import sys
import argparse
import time
import warnings
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

                                    
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llava.model.builder import load_pretrained_model
from llava.model.emotion_expert import EmotionOV

from datasets.emotion_dataset import (
    EMOTION_LABELS,
    EmotionDataset,
    collate_fn_stage1,
)

warnings.filterwarnings("ignore")


class EmotionExpertTrainer:
    """Trains the EmotionOV emotion expert for Stage 1."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                        
        self.global_step = 0
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        self.epoch_val_accuracies = []

        print(f"Using device: {self.device}")

        self._load_backbone()
        self._create_emotion_expert()
        self._create_dataloaders()
        self._setup_optimizer()

        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

                                                                        
                            
                                                                        

    def _load_backbone(self):
        """Load and freeze the LLaVA-OneVision backbone."""
        print("Loading LLaVA-OneVision backbone...")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            self.args.model_checkpoint,
            None,
            "llava_qwen",
            device_map=None,
            cache_dir='./cache',
            torch_dtype="float32",
        )
        model = model.to(self.device)

                                             
        model.config.image_aspect_ratio = "anyres_max_9"
        model.config.image_grid_pinpoints = "(1x1),...,(6x6)"
        model.config.mm_patch_merge_type = "spatial_unpad"
        model.config.mm_vision_select_layer = -2
        model.config.mm_projector_type = "mlp2x_gelu"

        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        print("Backbone loaded and frozen.")

    def _create_emotion_expert(self):
        """Instantiate the EmotionOV emotion expert (9 classes, 2 embeddings)."""
        print("Creating emotion expert...")
        self.emotion_expert = EmotionOV(num_emotions=9, num_emotion_embeddings=2)
        self.emotion_expert.to(dtype=torch.float32, device=self.device)
        for p in self.emotion_expert.parameters():
            p.requires_grad = True
        n_params = sum(p.numel() for p in self.emotion_expert.parameters())
        print(f"Emotion expert created — {n_params:,} trainable parameters.")

    def _create_dataloaders(self):
        """Create train and validation DataLoaders."""
        print("Creating dataloaders...")
        train_ds = EmotionDataset(
            self.args.train_data_path,
            self.args.image_root,
            self.image_processor,
            self.model.config,
            max_samples=self.args.max_samples,
        )
        val_ds = EmotionDataset(
            self.args.val_data_path,
            self.args.image_root,
            self.image_processor,
            self.model.config,
            max_samples=self.args.max_val_samples,
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, collate_fn=collate_fn_stage1, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.num_workers, collate_fn=collate_fn_stage1,
        )
        print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    def _setup_optimizer(self):
        """AdamW + cosine-warm-restart LR scheduler."""
        trainable = [p for p in self.emotion_expert.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        restart_steps = max(len(self.train_loader) // 2, 1)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=restart_steps, T_mult=1,
            eta_min=self.args.learning_rate * 0.1,
        )

                                                                        
                                      
                                                                        

    def _compute_kl_loss(self, preds, targets):
        """KL-divergence loss between predicted and target distributions."""
        return self.kl_loss(torch.log(preds + 1e-8), targets)

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch; returns average KL loss."""
        self.emotion_expert.train()
        total_loss = 0.0
        num_batches = 0
        t0 = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            images = batch['images']
            targets = batch['emotion_distributions'].to(self.device)

            with torch.no_grad():
                concat_imgs = torch.cat([img.to(self.device, dtype=torch.float32) for img in images], dim=0)
                split_sizes = [img.shape[0] for img in images]
                enc_feats, enc_lvl_feats = self.model.encode_images(concat_imgs)

            self.optimizer.zero_grad()
            try:
                _, _, preds = self.emotion_expert(enc_feats, enc_lvl_feats, split_sizes)
                loss = self._compute_kl_loss(preds, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.emotion_expert.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss/num_batches:.4f}")
            except Exception as e:
                print(f"  [skip] batch {batch_idx} error: {e}")

        elapsed = time.time() - t0
        avg = total_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1} | loss={avg:.4f} | {len(self.train_loader)*self.args.batch_size/elapsed:.1f} samples/s")
        return avg

    @torch.no_grad()
    def validate_accuracy(self):
        """Compute emotion classification accuracy on the validation split.

        Returns:
            overall_accuracy (float), per_emotion_acc (dict), correct (int), total (int)
        """
        self.emotion_expert.eval()
        correct, total = 0, 0
        emotion_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        for batch in tqdm(self.val_loader, desc="  Validation", leave=False):
            if batch is None:
                continue
            images = batch['images']
            gt_indices = batch['ground_truth_indices'].to(self.device)
            has_dominant = batch['has_dominant_emotions']
            try:
                concat_imgs = torch.cat([img.to(self.device, dtype=torch.float32) for img in images], dim=0)
                split_sizes = [img.shape[0] for img in images]
                enc_feats, enc_lvl_feats = self.model.encode_images(concat_imgs)
                _, _, preds = self.emotion_expert(enc_feats, enc_lvl_feats, split_sizes)
                pred_idx = torch.argmax(preds, dim=-1)

                for has_dom, gt, pred in zip(has_dominant, gt_indices, pred_idx):
                    if has_dom and gt >= 0:
                        total += 1
                        emo = EMOTION_LABELS[gt]
                        emotion_stats[emo]['total'] += 1
                        if pred == gt:
                            correct += 1
                            emotion_stats[emo]['correct'] += 1
            except Exception as e:
                print(f"  [skip] validation batch error: {e}")

        overall = correct / total if total > 0 else 0.0
        per_emo = {
            emo: (emotion_stats[emo]['correct'] / emotion_stats[emo]['total']
                  if emotion_stats[emo]['total'] > 0 else 0.0)
            for emo in EMOTION_LABELS
        }
        self.emotion_expert.train()
        return overall, per_emo, correct, total

                                                                        
                    
                                                                        

    def save_checkpoint(self, epoch: int, loss: float, accuracy: float):
        path = os.path.join(self.args.output_dir, f'emotion_expert_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'emotion_expert_state_dict': self.emotion_expert.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
        }, path)
        print(f"  Checkpoint saved: {path}")

    def save_best_model(self, accuracy: float):
        path = os.path.join(self.args.output_dir, 'emotion_expert_stage1_best.pth')
        torch.save({
            'global_step': self.global_step,
            'emotion_expert_state_dict': self.emotion_expert.state_dict(),
            'best_val_accuracy': accuracy,
        }, path)
        print(f"  Best model saved: {path}  (acc={accuracy:.4f})")

                                                                        
                        
                                                                        

    def train(self):
        print(f"\nStarting Stage 1 training")
        print(f"  Epochs: {self.args.num_epochs} | Batch: {self.args.batch_size} | LR: {self.args.learning_rate}")
        print(f"  Early-stop patience: {self.args.patience} epochs\n")

        for epoch in range(self.args.num_epochs):
            avg_loss = self.train_epoch(epoch)
            acc, per_emo, correct, total = self.validate_accuracy()
            self.epoch_val_accuracies.append(acc)

            print(f"  [Epoch {epoch+1}] train_loss={avg_loss:.4f}  val_acc={acc:.4f} ({correct}/{total})")
            for emo, a in per_emo.items():
                if a > 0:
                    print(f"    {emo}: {a:.4f}")

            if acc > self.best_val_accuracy:
                self.best_val_accuracy = acc
                self.patience_counter = 0
                self.save_best_model(acc)
            else:
                self.patience_counter += 1
                print(f"  No improvement. Patience: {self.patience_counter}/{self.args.patience}")
                if self.patience_counter >= self.args.patience:
                    print(f"  Early stopping after {self.patience_counter} epochs without improvement.")
                    break

            if (epoch + 1) % self.args.save_every == 0:
                self.save_checkpoint(epoch, avg_loss, acc)

        print(f"\nStage 1 training done. Best val accuracy: {self.best_val_accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Train EmotionOV emotion expert")

          
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to artEmisX training JSON")
    parser.add_argument("--val_data_path", type=str, required=True,
                        help="Path to artEmisX validation JSON")
    parser.add_argument("--image_root", type=str, required=True,
                        help="Root directory of artwork images")

           
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to pretrained LLaVA-OneVision checkpoint")

              
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early-stopping patience (epochs)")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap training set size (for debugging)")
    parser.add_argument("--max_val_samples", type=int, default=20000,
                        help="Cap validation set size")

            
    parser.add_argument("--output_dir", type=str, default="./checkpoints_stage1",
                        help="Directory for saved checkpoints")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    trainer = EmotionExpertTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
