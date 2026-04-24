"""
情感专家模块
"""

def get_expert_module(expert_name):
    """根据专家名称获取对应的专家类"""
    if expert_name == "emotion_expert_v1":
        from .emotion_expert_v1 import EmotionOV
        return EmotionOV
    elif expert_name == "emotion_expert_emotion8":
        from .emotion_expert_emotion8 import EmotionExpertEmotion8
        return EmotionExpertEmotion8
    elif expert_name == "emotion_expert_del4":
        from .emotion_expert_del4 import EmotionExpertDel4
        return EmotionExpertDel4
    else:
        raise ValueError(f"Unknown expert module: {expert_name}")
