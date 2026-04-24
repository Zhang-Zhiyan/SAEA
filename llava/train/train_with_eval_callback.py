                      
                       
"""
在原始训练基础上添加评估回调
避免transformers导入问题
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path

      
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

          
from llava.train.train import train as original_train
from llava.train.llava_trainer import LLaVATrainer

      
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationCallback:
    """评估回调类"""
    
    def __init__(self, eval_steps=2000, output_dir=None):
        self.eval_steps = eval_steps
        self.output_dir = output_dir
        self.last_evaluated_step = 0
        self.best_bleu4 = 0.0
        self.no_improvement_count = 0
        self.patience = 3
        
        logger.info(f"🔍 评估回调初始化完成，每{eval_steps}步评估一次")
    
    def should_evaluate(self, global_step):
        """判断是否应该进行评估"""
        return (global_step > 0 and 
                global_step % self.eval_steps == 0 and 
                global_step != self.last_evaluated_step)
    
    def run_evaluation(self, global_step, checkpoint_path=None):
        """运行外部评估脚本"""
        try:
            logger.info(f"🔍 开始第{global_step}步的语义评估...")
            
                    
            env = os.environ.copy()
            env['PYTHONPATH'] = '/data/zhangzy/sevlm1/AECforICassp/cococaption:' + env.get('PYTHONPATH', '')
            
                    
            eval_script = str(project_root / "demo_evaluation.py")
            cmd = [sys.executable, eval_script]
            
                  
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=str(project_root),
                env=env,
                timeout=300         
            )
            
            if result.returncode == 0:
                logger.info("✅ 语义评估成功完成")
                            
                self._parse_evaluation_results(result.stdout, global_step)
            else:
                logger.error(f"❌ 语义评估失败: {result.stderr}")
            
            self.last_evaluated_step = global_step
            
        except Exception as e:
            logger.error(f"❌ 评估过程中出错: {str(e)}")
    
    def _parse_evaluation_results(self, output, global_step):
        """解析评估结果"""
        try:
                            
            lines = output.split('\n')
            bleu4 = 0.0
            
            for line in lines:
                if 'BLEU-4:' in line:
                    bleu4 = float(line.split('BLEU-4:')[1].strip())
                    break
            
            logger.info(f"📊 第{global_step}步 BLEU-4: {bleu4:.4f}")
            
                       
            if bleu4 > self.best_bleu4:
                self.best_bleu4 = bleu4
                self.no_improvement_count = 0
                logger.info(f"🎉 新的最佳BLEU-4分数: {bleu4:.4f}")
                
                          
                if self.output_dir:
                    best_info = {
                        'best_bleu4': self.best_bleu4,
                        'best_step': global_step,
                        'timestamp': str(Path().resolve())
                    }
                    
                    best_file = Path(self.output_dir) / "best_evaluation_info.json"
                    with open(best_file, 'w', encoding='utf-8') as f:
                        json.dump(best_info, f, ensure_ascii=False, indent=2)
            else:
                self.no_improvement_count += 1
                logger.info(f"⚠️ 连续{self.no_improvement_count}轮无提升")
            
                    
            if self.no_improvement_count >= self.patience:
                logger.info(f"🛑 建议早停：连续{self.patience}轮无提升")
                
        except Exception as e:
            logger.error(f"❌ 解析评估结果失败: {str(e)}")

          
evaluation_callback = None

def init_evaluation_callback(eval_steps=2000, output_dir=None):
    """初始化评估回调"""
    global evaluation_callback
    evaluation_callback = EvaluationCallback(eval_steps, output_dir)

class LLaVATrainerWithCallback(LLaVATrainer):
    """带评估回调的训练器"""
    
    def log(self, logs):
        """重写日志方法，在这里触发评估"""
        super().log(logs)
        
                  
        if evaluation_callback and evaluation_callback.should_evaluate(self.state.global_step):
            evaluation_callback.run_evaluation(
                global_step=self.state.global_step,
                checkpoint_path=self.args.output_dir
            )

def train_with_evaluation():
    """带评估的训练函数"""
    
                        
    import sys
    output_dir = None
    eval_steps = 2000
    
    for i, arg in enumerate(sys.argv):
        if arg == '--output_dir' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
        elif arg == '--eval_steps' and i + 1 < len(sys.argv):
            eval_steps = int(sys.argv[i + 1])
    
             
    init_evaluation_callback(eval_steps=eval_steps, output_dir=output_dir)
    
              
    import llava.train.train as train_module
    original_trainer_class = train_module.LLaVATrainer
    train_module.LLaVATrainer = LLaVATrainerWithCallback
    
    try:
                  
        logger.info("🚀 开始带评估回调的训练")
        original_train()
    finally:
                  
        train_module.LLaVATrainer = original_trainer_class

if __name__ == "__main__":
    train_with_evaluation()
