                      
                       
"""
集成COCO评估的LLaVA训练脚本
基于train_mem.py，添加语义评估和早停机制
"""

import os
import sys
import json
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import transformers
from transformers import Trainer

        
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.append('/data/zhangzy/sevlm1/AECforICassp/cococaption')

          
from llava.train.train import (
    ModelArguments, DataArguments, TrainingArguments,
    make_supervised_data_module, LLaVATrainer, get_model
)
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava import conversation as conversation_lib

       
from evaluation.semantic_evaluator import create_evaluator_for_stage2

      
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationArguments:
    """评估相关参数"""
    enable_semantic_eval: bool = field(default=True, metadata={"help": "是否启用语义评估"})
    semantic_eval_steps: int = field(default=500, metadata={"help": "语义评估间隔步数"})
    eval_samples: int = field(default=100, metadata={"help": "评估使用的样本数量"})
    patience: int = field(default=3, metadata={"help": "早停耐心值"})
    min_bleu4_threshold: float = field(default=0.01, metadata={"help": "最小BLEU-4阈值"})

def make_supervised_data_module_with_eval(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """创建支持训练和验证数据集分离的监督学习数据模块"""
    from llava.train.train import LazySupervisedDataset, DataCollatorForSupervisedDataset

              
    if data_args.data_path.endswith('.yaml') or data_args.data_path.endswith('.yml'):
        with open(data_args.data_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)

        datasets = data_config.get('datasets', [])
        if len(datasets) >= 2:
                               
            train_data_path = datasets[0]['json_path']
            eval_data_path = datasets[1]['json_path']

                     
            train_dataset = LazySupervisedDataset(
                tokenizer=tokenizer,
                data_path=train_data_path,
                data_args=data_args
            )

                     
            eval_dataset = LazySupervisedDataset(
                tokenizer=tokenizer,
                data_path=eval_data_path,
                data_args=data_args
            )
        else:
                              
            train_dataset = LazySupervisedDataset(
                tokenizer=tokenizer,
                data_path=data_args.data_path,
                data_args=data_args
            )
            eval_dataset = None
    else:
                            
        train_dataset = LazySupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            data_args=data_args
        )
        eval_dataset = None

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

class LLavaTrainerWithEval(LLaVATrainer):
    """集成语义评估的LLaVA训练器"""
    
    def __init__(self, eval_args: EvaluationArguments = None, **kwargs):
        super().__init__(**kwargs)
        self.eval_args = eval_args or EvaluationArguments()
        self.evaluator = None
        self.best_bleu4 = 0.0
        self.no_improvement_count = 0
        
                
        if self.eval_args.enable_semantic_eval:
            self._init_evaluator()
    
    def _init_evaluator(self):
        """初始化语义评估器"""
        try:
                            
            if hasattr(self.args, 'data_path'):
                              
                data_dir = os.path.dirname(self.args.data_path)
                val_data_path = os.path.join(data_dir, "llava_emotion_dynamic_val.json")
            else:
                val_data_path = "/data/zhangzy/sevlm1/AECforICassp/Emotion-OneVision-1/data/llava_emotion_dynamic_val.json"
            
            if not os.path.exists(val_data_path):
                logger.warning(f"验证数据文件不存在: {val_data_path}")
                return
            
            image_folder = "/data/zhangzy/sevlm1/AECforICassp/Affection_public_language_data"
            eval_output_dir = os.path.join(self.args.output_dir, "semantic_evaluation")
            
            self.evaluator = create_evaluator_for_stage2(
                val_data_path=val_data_path,
                image_folder=image_folder,
                output_dir=eval_output_dir
            )
            
            logger.info("✅ 语义评估器初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 语义评估器初始化失败: {str(e)}")
            self.evaluator = None
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """重写评估方法，集成语义评估"""
        
                
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
                                            
        if self.evaluator and self.state.global_step > 0 and self.state.global_step % self.eval_args.semantic_eval_steps == 0:
            try:
                current_epoch = int(self.state.epoch)
                logger.info(f"🔍 开始第 {current_epoch} 轮语义评估...")
                
                        
                semantic_scores, should_save, should_stop = self.evaluator.evaluate_model(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    epoch=current_epoch
                )
                
                        
                if semantic_scores:
                    for key, value in semantic_scores.items():
                        eval_results[f"eval_semantic_{key}"] = value
                    
                    current_bleu4 = semantic_scores.get('Bleu_4', 0)
                    logger.info(f"📊 语义评估结果 - BLEU-4: {current_bleu4:.4f}")
                    
                                
                    if should_save:
                        self.best_bleu4 = current_bleu4
                        self.no_improvement_count = 0
                        
                                
                        best_model_dir = os.path.join(self.args.output_dir, "best_model")
                        logger.info(f"💾 保存最佳模型到: {best_model_dir}")
                        self.save_model(best_model_dir)
                        
                                
                        eval_info = {
                            "epoch": current_epoch,
                            "global_step": self.state.global_step,
                            "best_bleu4": self.best_bleu4,
                            "semantic_scores": semantic_scores
                        }
                        
                        eval_info_file = os.path.join(best_model_dir, "evaluation_info.json")
                        with open(eval_info_file, 'w', encoding='utf-8') as f:
                            json.dump(eval_info, f, ensure_ascii=False, indent=2)
                    else:
                        self.no_improvement_count += 1
                        logger.info(f"⚠️ 连续 {self.no_improvement_count} 轮无提升")
                    
                            
                    if should_stop:
                        logger.info(f"🛑 触发早停条件，最佳BLEU-4: {self.best_bleu4:.4f}")
                        self.control.should_training_stop = True
                
            except Exception as e:
                logger.error(f"❌ 语义评估过程中出错: {str(e)}")
        
        return eval_results
    
    def log(self, logs: Dict[str, float]) -> None:
        """重写日志方法，添加语义评估信息"""
        
                     
        if hasattr(self, 'best_bleu4'):
            logs["best_bleu4"] = self.best_bleu4
            logs["no_improvement_count"] = self.no_improvement_count
        
        super().log(logs)

def train():
    """主训练函数"""
    
          
    parser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, EvaluationArguments
    ))
    model_args, data_args, training_args, eval_args = parser.parse_args_into_dataclasses()
    
    logger.info("🚀 开始情感检测第二阶段训练（集成语义评估）")
    logger.info(f"模型路径: {model_args.model_name_or_path}")
    logger.info(f"数据路径: {data_args.data_path}")
    logger.info(f"输出目录: {training_args.output_dir}")
    logger.info(f"语义评估: {'启用' if eval_args.enable_semantic_eval else '禁用'}")
    
                            
    bnb_model_from_pretrained_args = {}
    model = get_model(model_args, training_args, bnb_model_from_pretrained_args)
    model.config.use_cache = False
    
                                
    if "mistral" in model_args.model_name_or_path.lower() or "mixtral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="left")
    elif "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
    elif (
        "wizardlm-2" in model_args.model_name_or_path.lower()
        or "vicuna" in model_args.model_name_or_path.lower()
        or "llama" in model_args.model_name_or_path.lower()
        or "yi" in model_args.model_name_or_path.lower()
        or "nous-hermes" in model_args.model_name_or_path.lower()
        and "wizard-2" in model_args.model_name_or_path.lower()
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_vision_tower().load_model(device_map=training_args.device)
        if training_args.bits in [4, 8]:
            model.get_vision_tower().to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        
        data_args.image_processor = model.get_vision_tower().image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        training_args.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        training_args.mm_use_im_start_end = model_args.mm_use_im_start_end

        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)

        if model_args.mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))

        if training_args.bits in [4, 8]:
            from peft.tuners.lora import LoraLayer

            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    if training_args.bf16:
                        module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if "lm_head" in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        if training_args.bf16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)
    
                    
    print("🔄 Loading trained emotion expert...")
    from llava.model.emotion_expert import EmotionOV
    
                            
    emotion_expert = EmotionOV(num_emotions=9, num_emotion_embeddings=2)
    emotion_expert.to(dtype=torch.float32, device=training_args.device)
    
              
    emotion_expert_path = "/data/zhangzy/sevlm1/AECforICassp/Emotion-OneVision-2/checkpoints_stage1_affection_0.5b_accuracy_rightembedding/emotion_expert_stage1_best.pth"
    if os.path.exists(emotion_expert_path):
        checkpoint = torch.load(emotion_expert_path, map_location='cpu')
        
                  
        if 'emotion_expert_state_dict' in checkpoint:
            state_dict = checkpoint['emotion_expert_state_dict']
            print(f"📊 Best validation accuracy: {checkpoint.get('best_val_accuracy', 'N/A')}")
            emotion_expert.load_state_dict(state_dict)
            print("✅ Emotion expert weights loaded successfully!")
        else:
            print("❌ No emotion_expert_state_dict found in checkpoint")
    else:
        print(f"❌ Emotion expert checkpoint not found at: {emotion_expert_path}")
        
                           
    for param in emotion_expert.parameters():
        param.requires_grad = False
    
                 
    model.set_emotion_encoder(emotion_expert)
    print("✅ Emotion expert set to model successfully!")
    
                          
    data_module = make_supervised_data_module_with_eval(tokenizer=tokenizer, data_args=data_args)
    
           
    trainer = LLavaTrainerWithEval(
        eval_args=eval_args,
        model=model,
        args=training_args,
        **data_module
    )
    
          
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logger.info("📂 发现检查点，从检查点恢复训练")
        trainer.train(resume_from_checkpoint=True)
    else:
        logger.info("🆕 开始新的训练")
        trainer.train()
    
            
    trainer.save_state()
    
                 
    if trainer.evaluator:
        final_info = trainer.evaluator.get_best_score_info()
        final_info_file = os.path.join(training_args.output_dir, "final_evaluation_info.json")
        with open(final_info_file, 'w', encoding='utf-8') as f:
            json.dump(final_info, f, ensure_ascii=False, indent=2)
        
        logger.info("📊 训练完成，最终评估信息:")
        logger.info(f"最佳BLEU-4: {final_info.get('best_bleu4', 0):.4f}")
        logger.info(f"最佳轮次: {final_info.get('best_epoch', 0)}")
    
    logger.info("🎉 训练完成！")

if __name__ == "__main__":
    import pathlib
    train()
