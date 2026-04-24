import torch
from typing import List, Dict, Tuple

EMOTION_LABELS = ['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust',  'fear', 'sadness', 'something else'
]


def get_predicted_emotion_names(emotion_preds: torch.Tensor) -> List[str]:
    """Convert soft predictions [B, 9] -> list of emotion names.
    If emotion_preds is already indices, handle gracefully.
    """
    if emotion_preds.ndim == 2:
        idx = torch.argmax(emotion_preds, dim=-1).tolist()
    else:
        idx = emotion_preds.tolist()
    return [EMOTION_LABELS[i] for i in idx]


def build_dynamic_questions(emotion_names: List[str]) -> List[str]:
    """Create dynamic question strings from emotion names.
    Keep it simple and deterministic to match training/inference.
    """
    questions = []
    for emo in emotion_names:
        q = f"Explain in one sentence the reason why this image evokes {emo}."
        questions.append(q)
    return questions


def ensure_special_tokens(tokenizer) -> Dict[str, int]:
    """Ensure tokenizer has <prefix>, <question>, <explanation> tokens.
    Return their ids.
    """
    add_tokens = []
    for tok in ["<prefix>", "<question>", "<explanation>"]:
        if tokenizer.convert_tokens_to_ids(tok) == tokenizer.unk_token_id:
            add_tokens.append(tok)
    if add_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})
    token_ids = {
        "prefix": tokenizer.convert_tokens_to_ids("<prefix>"),
        "question": tokenizer.convert_tokens_to_ids("<question>"),
        "explanation": tokenizer.convert_tokens_to_ids("<explanation>")
    }
    return token_ids


def pack_gpt2_inputs(
    tokenizer,
    questions: List[str],
    explanations: List[str],
    max_length: int = 256,
    prefix_len: int = 0,
) -> Dict[str, torch.Tensor]:
    """Build GPT2 input_ids, labels, token_type_ids from question+explanation.
    - Prefix tokens are <prefix> repeated prefix_len, labels = -100 (if prefix_len > 0)
    - Question tokens are prefixed by <question>, labels = -100 (we don't learn question)
    - Explanation tokens are prefixed by <explanation>, labels = tokens (we learn only explanation)
    All sequences are right-padded to max_length with pad_token.
    """
    assert len(questions) == len(explanations)
    tok_ids = ensure_special_tokens(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    batch_input_ids: List[List[int]] = []
    batch_labels: List[List[int]] = []
    batch_segments: List[List[int]] = []

    for q_text, exp_text in zip(questions, explanations):
                                     
        if prefix_len > 0:
            prefix_tokens = [tok_ids["prefix"]] * prefix_len
            prefix_labels = [-100] * prefix_len
            prefix_segments = [tok_ids["prefix"]] * prefix_len
        else:
            prefix_tokens = []
            prefix_labels = []
            prefix_segments = []

        q_tokens = [tok_ids["question"]] + tokenizer.encode(q_text, add_special_tokens=False)
        q_labels = [-100] * len(q_tokens)
        q_segments = [tok_ids["question"]] * len(q_tokens)

        exp_tokens = [tok_ids["explanation"]] + tokenizer.encode(exp_text, add_special_tokens=False) + [tokenizer.eos_token_id]
        exp_labels = exp_tokens.copy()
        exp_segments = [tok_ids["explanation"]] * len(exp_tokens)

        full = prefix_tokens + q_tokens + exp_tokens
        full_labels = prefix_labels + q_labels + exp_labels
        full_segments = prefix_segments + q_segments + exp_segments

                  
        if len(full) > max_length:
            full = full[:max_length]
            full_labels = full_labels[:max_length]
            full_segments = full_segments[:max_length]

             
        if len(full) < max_length:
            pad_n = max_length - len(full)
            full = full + [pad_id] * pad_n
            full_labels = full_labels + [-100] * pad_n
            full_segments = full_segments + [tok_ids["explanation"]] * pad_n

        batch_input_ids.append(full)
        batch_labels.append(full_labels)
        batch_segments.append(full_segments)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "token_type_ids": torch.tensor(batch_segments, dtype=torch.long),
    }

