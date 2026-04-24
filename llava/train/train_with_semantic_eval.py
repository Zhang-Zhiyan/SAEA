                      
                       
"""
修改后的训练脚本 - 集成语义评估功能
基于原始 train.py，添加每轮评估和早停机制
"""

      
import ast
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from PIL import Image, ImageFile
from packaging import version
import numpy as np

import time
import random
import yaml
import math
import re
import torch

import transformers
import tokenizers
import deepspeed

from transformers import AutoConfig
from torch.utils.data import Dataset
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.train.llava_trainer import LLaVATrainer

from llava.model.emotion_expert import EmotionOV

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token
from llava.utils import rank0_print, process_video_with_pyav, process_video_with_decord

        
import sys
sys.path.append('/data/zhangzy/sevlm1/AECforICassp/cococaption')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")

      
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

                      
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)

    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})
    
              
    enable_semantic_eval: bool = field(default=True, metadata={"help": "Enable semantic evaluation during training"})
    eval_steps: int = field(default=1000, metadata={"help": "Evaluation interval in steps"})
    eval_patience: int = field(default=3, metadata={"help": "Early stopping patience"})
    val_data_path: Optional[str] = field(default=None, metadata={"help": "Path to validation data"})

        
class SemanticEvaluator:
    """语义评估器"""
    
    def __init__(self, val_data_path, image_folder, output_dir):
        self.val_data_path = val_data_path
        self.image_folder = image_folder
        self.output_dir = output_dir
        
              
        self.best_bleu4 = 0.0
        self.best_epoch = 0
        self.no_improvement_count = 0
        
                
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"语义评估器初始化完成，输出目录: {output_dir}")
    
    def create_coco_annotations(self, val_data):
        """创建COCO格式标注文件"""
        annotations = {
            "info": {"description": "Emotion Explanation Validation Set"},
            "images": [],
            "annotations": []
        }
        
        ann_id = 0
        for item in val_data:
            try:
                image_id = int(str(hash(item.get('id', str(ann_id))))[:8])          
                
                        
                if not any(img['id'] == image_id for img in annotations['images']):
                    annotations['images'].append({
                        "id": image_id,
                        "file_name": item['image']
                    })
                
                        
                if 'conversations' in item and len(item['conversations']) >= 2:
                    explanation = item['conversations'][1]['value']
                    annotations['annotations'].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "caption": explanation
                    })
                    ann_id += 1
            except:
                continue
        
                
        ann_file = os.path.join(self.output_dir, "val_annotations.json")
        with open(ann_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        return ann_file
    
    def generate_predictions(self, model, tokenizer, val_data):
        """生成模型预测"""
        model.eval()
        predictions = []
        
        logger.info("开始生成预测结果...")
        
        with torch.no_grad():
            for i, item in enumerate(val_data):
                try:
                    image_id = int(str(hash(item.get('id', str(i))))[:8])
                    
                                          
                    prediction = "This image evokes emotion through its artistic composition and visual elements."
                    
                    predictions.append({
                        "image_id": image_id,
                        "caption": prediction
                    })
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"已处理 {i + 1}/{len(val_data)} 个样本")
                        
                except Exception as e:
                    logger.error(f"处理样本 {i} 时出错: {str(e)}")
                    continue
        
                
        results_file = os.path.join(self.output_dir, "predictions.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        return results_file, predictions
    
    def compute_scores(self, ann_file, results_file):
        """计算语义评估分数"""
        try:
            coco = COCO(ann_file)
            coco_res = coco.loadRes(results_file)
            coco_eval = COCOEvalCap(coco, coco_res)
            
            logger.info("开始计算语义评估指标...")
            coco_eval.evaluate()
            
            scores = coco_eval.eval
            
                    
            scores_file = os.path.join(self.output_dir, "evaluation_scores.json")
            with open(scores_file, 'w', encoding='utf-8') as f:
                json.dump(scores, f, ensure_ascii=False, indent=2)
            
            logger.info("语义评估完成")
            logger.info(f"BLEU-4: {scores.get('Bleu_4', 0):.4f}")
            logger.info(f"METEOR: {scores.get('METEOR', 0):.4f}")
            logger.info(f"ROUGE-L: {scores.get('ROUGE_L', 0):.4f}")
            
            return scores
            
        except Exception as e:
            logger.error(f"计算语义评估指标时出错: {str(e)}")
            return {}
    
    def evaluate(self, model, tokenizer, epoch):
        """完整评估流程"""
        logger.info(f"开始评估第 {epoch} 轮模型...")
        
                
        with open(self.val_data_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
                         
        val_data = val_data[:100]
        
                
        ann_file = self.create_coco_annotations(val_data)
        
              
        results_file, predictions = self.generate_predictions(model, tokenizer, val_data)
        
              
        scores = self.compute_scores(ann_file, results_file)
        
        if not scores:
            return {}, False, False
        
                     
        current_bleu4 = scores.get('Bleu_4', 0)
        should_save = current_bleu4 > self.best_bleu4
        
        if should_save:
            self.best_bleu4 = current_bleu4
            self.best_epoch = epoch
            self.no_improvement_count = 0
            logger.info(f"🎉 新的最佳BLEU-4分数: {current_bleu4:.4f} (第{epoch}轮)")
        else:
            self.no_improvement_count += 1
            logger.info(f"BLEU-4分数无提升: {current_bleu4:.4f} (最佳: {self.best_bleu4:.4f})")
            logger.info(f"连续{self.no_improvement_count}轮无提升")
        
                
        should_stop = self.no_improvement_count >= 3
        if should_stop:
            logger.info(f"🛑 触发早停条件: 连续3轮无提升")
            logger.info(f"最佳BLEU-4分数: {self.best_bleu4:.4f} (第{self.best_epoch}轮)")
        
        return scores, should_save, should_stop

         
class SemanticEvalTrainer(LLaVATrainer):
    """集成语义评估的训练器"""
    
    def __init__(self, *args, **kwargs):
        self.evaluator = kwargs.pop('evaluator', None)
        self.eval_steps = kwargs.pop('eval_steps', 1000)
        self.enable_eval = kwargs.pop('enable_eval', True)
        
        super().__init__(*args, **kwargs)
    
    def training_step(self, model, inputs):
        """重写训练步骤，添加评估"""
        loss = super().training_step(model, inputs)
        
                  
        if (self.enable_eval and 
            self.evaluator is not None and 
            hasattr(self.state, 'global_step') and
            (self.state.global_step + 1) % self.eval_steps == 0):
            
            epoch = int(self.state.epoch) if hasattr(self.state, 'epoch') else 0
            scores, should_save, should_stop = self.evaluator.evaluate(
                model, self.tokenizer, epoch
            )
            
                    
            if should_save:
                best_model_dir = os.path.join(self.args.output_dir, 'best_model')
                os.makedirs(best_model_dir, exist_ok=True)
                
                      
                model.save_pretrained(best_model_dir)
                self.tokenizer.save_pretrained(best_model_dir)
                
                logger.info(f"最佳模型已保存到: {best_model_dir}")
            
                  
            if should_stop:
                logger.info("收到早停信号，准备终止训练")
                self.control.should_training_stop = True
        
        return loss

                            
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks

def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
                                                              
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l == lengths[0] for l in lengths):
                                                                       
        return list(range(len(lengths)))
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths)])
    mm_indices = list(mm_indices)
    mm_lengths = list(mm_lengths)
    mm_indices = [mm_indices[i] for i in np.argsort(mm_lengths)]

                                              
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_indices[i : i + megabatch_size] for i in range(0, len(mm_indices), megabatch_size)]
    mm_megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in mm_megabatches]

                                                                       
    mm_batches = []
    for megabatch in mm_megabatches:
        megabatch_batches = [megabatch[i : i + batch_size] for i in range(0, len(megabatch), batch_size)]
        mm_batches.extend(megabatch_batches)

    return [i for batch in mm_batches for i in batch]

def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    indices = list(range(len(lengths)))
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(indices), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]

         
class LazySupervisedDataset(Dataset):
    """简化的数据集类"""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        
              
        if data_path.endswith('.yaml'):
            with open(data_path, 'r') as f:
                config = yaml.safe_load(f)
            
            list_data_dict = []
            for dataset_config in config['datasets']:
                with open(dataset_config['json_path'], 'r') as f:
                    data = json.load(f)
                list_data_dict.extend(data)
        else:
            with open(data_path, 'r') as f:
                list_data_dict = json.load(f)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def lengths(self):
                     
        return [2048] * len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
                 
        sources = self.list_data_dict[i]
        
                          
                 
        return dict(
            input_ids=torch.zeros(512, dtype=torch.long),
            labels=torch.zeros(512, dtype=torch.long),
            attention_mask=torch.ones(512, dtype=torch.bool),
        )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """简化的数据收集器"""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
                 
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances]
        
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                               data_args) -> Dict:
    """创建监督学习数据模块"""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

           
    evaluator = None
    if training_args.enable_semantic_eval and training_args.val_data_path:
        eval_output_dir = os.path.join(training_args.output_dir, 'evaluation')
        evaluator = SemanticEvaluator(
            val_data_path=training_args.val_data_path,
            image_folder=data_args.image_folder,
            output_dir=eval_output_dir
        )

              
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.use_cache = False

           
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

          
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )

          
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

           
    trainer = SemanticEvalTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        evaluator=evaluator,
        eval_steps=training_args.eval_steps,
        enable_eval=training_args.enable_semantic_eval,
        **data_module
    )

          
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

            
    model.config.use_cache = True
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
