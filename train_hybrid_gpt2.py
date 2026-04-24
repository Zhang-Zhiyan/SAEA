                      
"""
Stage 2: Train the LLaVA-GPT2 hybrid model for emotion-guided explanation generation.

Prerequisites:
    A Stage 1 checkpoint (emotion_expert_stage1_best.pth) must exist before
    running this script.

Architecture overview:
    LLaVA-OneVision encoder → EmotionOV emotion expert
          ↓                          ↓
    img_adapter (→ 768)        emo_adapter (→ 768)
          ↓                          ↓
              GPT-2 decoder (cross-attention)
                    ↓
            emotion explanation

Training strategy:
    - Freeze LLaVA backbone (except mm_projector which stays trainable).
    - Train: img_adapter, emo_adapter, GPT-2 decoder, mm_projector.
    - Supports single-GPU and multi-GPU DDP via --gpu_mode ddp --gpu_ids "0,1".

Usage (single GPU):
    python train_hybrid_gpt2.py \
        --llava_ckpt      <path/to/llava-onevision-qwen2-0.5b-ov> \
        --emotion_expert_ckpt checkpoints_stage1/emotion_expert_stage1_best.pth \
        --train_json      <path/to/artEmisX_train.json> \
        --val_json        <path/to/artEmisX_test.json> \
        --val_coco_ann    <path/to/artEmisX_test_annot_exp.json> \
        --image_root      <path/to/images/> \
        --gpt2_path       <path/to/gpt2/pretrain_model> \
        --gpt2_tokenizer_path <path/to/gpt2/pretrain_tokenizer> \
        --save_dir        ./checkpoints_stage2 \
        --caption_save_dir ./captions_stage2 \
        --gpu_mode single --gpu_id 0

Usage (multi-GPU DDP):
    python train_hybrid_gpt2.py ... --gpu_mode ddp --gpu_ids "0,1"
"""

import os
import sys
import argparse
import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import GPT2Tokenizer, GPT2Config
from tqdm import tqdm

                                    
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llava.model.builder import load_pretrained_model
from llava.model.emotion_expert import EmotionOV

from reproduce.models.gpt import GPT2LMHeadModel

from models.hybrid_model import LlavaGpt2Hybrid
from models.feature_adapters import SmartFeatureAdapter, SmartFeatureAdapter896, build_adapters
from models.gpt2_inputs import get_predicted_emotion_names, build_dynamic_questions, ensure_special_tokens

from datasets.emotion_dataset import ArtEmisXDataset, LlavaDataset, EvalDataset, collate_fn_stage2
from utils.eval_utils import top_filtering1


def build_val_dataloader(args, image_processor):
    val_ds = EvalDataset(args.val_json, args.image_root, image_processor)
    return DataLoader(
        val_ds, batch_size=max(1, args.bs * args.eval_batch_ratio),
        shuffle=False, num_workers=6,
    )


def sample_generate_batch(model, gpt2_tokenizer, batch, device,
                           max_len, top_k, top_p, temperature):
    """Greedy / nucleus sampling for one evaluation batch."""
    import torch.nn.functional as F

    real_model = model.module if hasattr(model, 'module') else model

    images = batch['image']
    if isinstance(images, torch.Tensor):
        images_list = [images[i].to(device) for i in range(images.size(0))]
    else:
        images_list = [im.to(device) for im in images]
    image_sizes = batch.get('image_size', [(384, 384)] * len(images_list))

    images_cat = torch.cat(images_list, dim=0)
    split_sizes = [im.shape[0] if im.ndim == 4 else 1 for im in images_list]

    ov_feats, sig_multi_feats = real_model.encode_images(images_cat)
    proj_img_feats = real_model.project_images(ov_feats)
    local_emo, global_emo, emotion_preds = real_model.get_emotion_features(
        ov_feats, sig_multi_feats, split_sizes
    )
    global_mem, local_mem = real_model._build_encoder_states(
        proj_img_feats, local_emo, global_emo, split_sizes, image_sizes=image_sizes
    )

    emo_names = get_predicted_emotion_names(emotion_preds)
    questions = build_dynamic_questions(emo_names)

    tok_ids = ensure_special_tokens(gpt2_tokenizer)
    q_sid = tok_ids['question']
    exp_sid = tok_ids['explanation']
    eos_id = gpt2_tokenizer.eos_token_id
    no_sample = True

    results = []
    for i, q in enumerate(questions):
        q_ids = [q_sid] + gpt2_tokenizer.encode(q, add_special_tokens=False) + [exp_sid]
        input_ids = torch.tensor([q_ids], dtype=torch.long, device=device)
        token_type_ids = torch.tensor(
            [[q_sid] * (len(q_ids) - 1) + [exp_sid]], dtype=torch.long, device=device
        )
        for _ in range(max_len):
            out = real_model.gpt2(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                encoder_hidden_states=(global_mem[i:i+1], local_mem[i:i+1]),
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits[0][:, -1, :] / temperature
            logits = top_filtering1(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            token_type_ids = torch.cat(
                [token_type_ids, torch.tensor([[exp_sid]], device=device)], dim=1
            )
            if next_id.item() == eos_id:
                break
        gen_ids = input_ids[0].tolist()
        start = gen_ids.index(exp_sid) + 1 if exp_sid in gen_ids else 0
        end = start + gen_ids[start:].index(eos_id) if eos_id in gen_ids[start:] else len(gen_ids)
        text = gpt2_tokenizer.decode(gen_ids[start:end], skip_special_tokens=True).strip()
        results.append(text)
    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Train LLaVA-GPT2 Hybrid")

                 
    parser.add_argument('--llava_ckpt', type=str, required=True,
                        help='Path to LLaVA-OneVision checkpoint')
    parser.add_argument('--emotion_expert_ckpt', type=str, default=None,
                        help='Path to Stage 1 emotion expert checkpoint (.pth)')
    parser.add_argument('--gpt2_path', type=str, required=True,
                        help='Path to pre-trained GPT-2 model')
    parser.add_argument('--gpt2_tokenizer_path', type=str, required=True,
                        help='Path to GPT-2 tokenizer')

          
    parser.add_argument('--train_json', type=str, required=True,
                        help='artEmisX training JSON')
    parser.add_argument('--val_json', type=str, required=True,
                        help='artEmisX test/val JSON for COCO eval')
    parser.add_argument('--val_coco_ann', type=str, required=True,
                        help='COCO-format annotation JSON for val set')
    parser.add_argument('--image_root', type=str, required=True,
                        help='Root directory of artwork images')
    parser.add_argument('--data_format', type=str, choices=['artemis', 'llava'], default='artemis',
                        help='Input JSON format: artemis (artEmisX) or llava (conversation)')

            
    parser.add_argument('--save_dir', type=str, default='./checkpoints_stage2',
                        help='Directory for model checkpoints')
    parser.add_argument('--caption_save_dir', type=str, default='./captions_stage2',
                        help='Directory for generated captions & COCO metrics')

                               
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_len', type=int, default=30)

                                                 
    parser.add_argument('--llava_size', type=str, choices=['0.5b', '7b'], default='0.5b',
                        help='LLaVA backbone size: 0.5b (hidden=896) or 7b (hidden=4096)')

               
    parser.add_argument('--gpu_mode', type=str, choices=['single', 'ddp'], default='ddp')
    parser.add_argument('--gpu_id', type=int, default=0, help='Single-GPU id')
    parser.add_argument('--gpu_ids', type=str, default='0,1',
                        help='DDP: comma-separated GPU ids, e.g. "0,1"')

                
    parser.add_argument('--eval_batch_ratio', type=int, default=2)
    parser.add_argument('--eval_max_len', type=int, default=30)
    parser.add_argument('--eval_top_k', type=int, default=0)
    parser.add_argument('--eval_top_p', type=float, default=0.9)
    parser.add_argument('--eval_temperature', type=float, default=1.0)

            
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    args = parser.parse_args()

                                                                     
                                                                      
                                                                      
                                                                     
    use_ddp = (args.gpu_mode == 'ddp')
    if use_ddp and 'LOCAL_RANK' not in os.environ and args.gpu_ids:
        gpu_ids_str = args.gpu_ids.strip()
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
        nprocs = len([x for x in gpu_ids_str.split(',') if x.strip()])
        if nprocs > 1:
            cmd = ['torchrun', '--nproc_per_node', str(nprocs), sys.argv[0]] + sys.argv[1:]
            print(f"[Auto-DDP] Launching: {' '.join(cmd)}")
            os.execvp(cmd[0], cmd)

    os.makedirs(args.save_dir, exist_ok=True)

                                                                     
                                
                                                                     
    distributed = 'LOCAL_RANK' in os.environ
    if distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{local_rank}')
        is_main = (dist.get_rank() == 0)
        if is_main:
            print(f"[DDP] world_size={dist.get_world_size()}, local_rank={local_rank}")
    else:
        gpu_id = args.gpu_id if args.gpu_mode == 'single' else 0
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        is_main = True

                                                                     
                         
                                                                     
    tokenizer_llava, llava_model, image_processor, _ = load_pretrained_model(
        args.llava_ckpt, None, 'llava_qwen', device_map=None, torch_dtype='float16'
    )
    llava_model = llava_model.to(device)
    llava_model.eval()

                                                                     
                                              
                                                                     
    print("Loading emotion expert...")
    emo_expert = EmotionOV(num_emotions=9, num_emotion_embeddings=2).to(device)
    if args.emotion_expert_ckpt:
        print(f"  Loading from: {args.emotion_expert_ckpt}")
        emo_expert.load_zero_shot_weights(args.emotion_expert_ckpt)
    else:
        try:
            if hasattr(llava_model.model, 'emotion_encoder') and llava_model.model.emotion_encoder is not None:
                emo_expert = llava_model.model.emotion_encoder
                print("  Loaded from LLaVA checkpoint.")
        except Exception:
            print("  Warning: no emotion expert checkpoint supplied; using random init.")
    emo_expert.eval()
    llava_model.set_emotion_encoder(emo_expert)

                                                                     
                         
                                                                     
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_tokenizer_path)
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_config = GPT2Config.from_pretrained(args.gpt2_path)
    gpt2_config.len_prefix = 0
    gpt2_config.encoder_dim = 768
    gpt2_config.add_cross_attention = True
    gpt2_model = GPT2LMHeadModel.from_pretrained(args.gpt2_path, config=gpt2_config).to(device)

                                                                     
                                   
                                                                     
    model = LlavaGpt2Hybrid(llava_model, emo_expert, gpt2_model, gpt2_tokenizer).to(device)
    img_adapter, emo_adapter = build_adapters(args.llava_size)
    model.img_adapter = img_adapter.to(device)
    model.emo_adapter = emo_adapter.to(device)
    if is_main:
        print(f"Adapters: LLaVA {args.llava_size} -> in_dim={'896' if args.llava_size=='0.5b' else '4096'} -> 768")

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    real_model = model.module if distributed else model

                                                              
    real_model.llava.to(torch.float32)
    real_model.img_adapter.to(torch.float32)
    real_model.emo_adapter.to(torch.float32)
    real_model.gpt2.to(torch.float32)

                                                        
    for p in real_model.llava.parameters():
        p.requires_grad = False
    try:
        for p in real_model.llava.get_model().mm_projector.parameters():
            p.requires_grad = True
    except Exception:
        pass
    for p in real_model.img_adapter.parameters():
        p.requires_grad = True
    for p in real_model.emo_adapter.parameters():
        p.requires_grad = True
    for p in real_model.gpt2.parameters():
        p.requires_grad = True

                                                                     
                          
                                                                     
    if args.data_format == 'artemis':
        train_ds = ArtEmisXDataset(args.train_json, args.image_root, image_processor, args.max_len)
    else:
        train_ds = LlavaDataset(args.train_json, args.image_root, image_processor, args.max_len)

    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        dl = DataLoader(train_ds, batch_size=args.bs, sampler=train_sampler,
                        num_workers=6, collate_fn=collate_fn_stage2)
    else:
        dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                        num_workers=6, collate_fn=collate_fn_stage2)

                                                                     
                                                
                                                                     
    optim = torch.optim.AdamW(
        list(real_model.img_adapter.parameters()) +
        list(real_model.emo_adapter.parameters()) +
        list(real_model.gpt2.parameters()) +
        list(real_model.llava.get_model().mm_projector.parameters()),
        lr=args.lr,
    )

                                                                     
                                      
                                                                     
    start_epoch = 0
    if args.resume_from_checkpoint:
        if is_main:
            print(f"Resuming from: {args.resume_from_checkpoint}")
        ckpt = torch.load(args.resume_from_checkpoint, map_location=device)
        (model.module if distributed else model).load_state_dict(ckpt['model_state_dict'])
        optim.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        if is_main:
            print(f"  Resumed from epoch {start_epoch}")

                                                                     
                                        
                                                                     
    if is_main:
        os.makedirs(args.caption_save_dir, exist_ok=True)
        val_dl = build_val_dataloader(args, image_processor)

                                                                     
                   
                                                                     
    model.train()
    if is_main:
        print(f"\nStarting Stage 2 training | {len(train_ds)} samples | bs={args.bs} | epochs={args.epochs}")

    best_bleu4 = float('-inf')
    best_epoch = -1

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        pbar = tqdm(enumerate(dl), total=len(dl), desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in pbar:
            fwd_ok = torch.tensor(1, device=device)
            try:
                out = model(
                    images=batch['images'],
                    target_explanations=batch['explanations'],
                    image_sizes=batch['image_sizes'],
                )
                loss = out['loss']
            except Exception as e:
                fwd_ok.zero_()
                if is_main:
                    print(f"  Forward error step {step}: {e}")

            if distributed:
                dist.all_reduce(fwd_ok, op=dist.ReduceOp.MIN)
            if fwd_ok.item() == 0:
                optim.zero_grad(set_to_none=True)
                continue

            bwd_ok = torch.tensor(1, device=device)
            try:
                optim.zero_grad()
                loss.backward()
                optim.step()
            except Exception as e:
                bwd_ok.zero_()
                if is_main:
                    print(f"  Backward error step {step}: {e}")

            if distributed:
                dist.all_reduce(bwd_ok, op=dist.ReduceOp.MIN)
            if bwd_ok.item() == 0:
                optim.zero_grad(set_to_none=True)
                continue

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dl)
        if is_main:
            print(f"  Epoch {epoch+1} | avg_loss={avg_loss:.4f}")

                                      
            ckpt_path = os.path.join(args.save_dir, f'hybrid_model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': (model.module if distributed else model).state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)

                                                                         
                                                   
                                                                         
        if is_main:
            from cococaption.pycocotools.coco import COCO
            from cococaption.pycocoevalcap.eval import COCOEvalCap

            model.eval()
            predictions = []
            for batch in tqdm(val_dl, desc=f"  Val epoch {epoch+1}"):
                caps = sample_generate_batch(
                    model, gpt2_tokenizer, batch, device,
                    max_len=args.eval_max_len,
                    top_k=args.eval_top_k,
                    top_p=args.eval_top_p,
                    temperature=args.eval_temperature,
                )
                for cap, img_id in zip(caps, batch['image_id']):
                    predictions.append({"image_id": int(img_id), "caption": cap})
            model.train()

            pred_path = os.path.join(args.caption_save_dir, f'captions_epoch_{epoch+1}.json')
            with open(pred_path, 'w') as f:
                json.dump(predictions, f)

            coco = COCO(args.val_coco_ann)
            coco_res = coco.loadRes(pred_path)
            coco_eval = COCOEvalCap(coco, coco_res)
            coco_eval.evaluate()
            print(f"  COCO metrics @ epoch {epoch+1}: {coco_eval.eval}")

            metrics_path = os.path.join(args.caption_save_dir, f'scores_epoch_{epoch+1}.json')
            with open(metrics_path, 'w') as f:
                json.dump(coco_eval.eval, f)

            bleu4 = coco_eval.eval.get('Bleu_4', 0.0)
            if bleu4 > best_bleu4:
                best_bleu4 = bleu4
                best_epoch = epoch + 1
                best_path = os.path.join(args.save_dir, f'hybrid_model_best_bleu4_epoch_{best_epoch}.pt')
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': (model.module if distributed else model).state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'bleu4': best_bleu4,
                }, best_path)
                print(f"  New best Bleu_4={best_bleu4:.4f} saved to {best_path}")

    if is_main:
        print(f"\nStage 2 training done. Best Bleu_4={best_bleu4:.4f} at epoch {best_epoch}.")


if __name__ == '__main__':
    main()
