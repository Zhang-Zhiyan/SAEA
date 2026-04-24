"""
Feature adapters for projecting LLaVA visual features into GPT2 cross-attention space.

SmartFeatureAdapter  : generic in_dim -> out_dim adapter (default 4096 -> 768, for 7B backbone)
SmartFeatureAdapter896: specialised 896 -> 768 adapter (for Qwen2-0.5B based LLaVA backbone)
"""

import torch
import torch.nn as nn


class SmartFeatureAdapter(nn.Module):
    """Generic LLaVA feature adapter: in_dim -> out_dim.

    Architecture: residual gating + multi-head self-attention refinement.
    Default maps 4096 -> 768 (suitable for 7B LLaVA backbone).
    Also usable for 896 -> 768 by setting in_dim=896.

    Args:
        in_dim:    input feature dimension (e.g. 4096 for LLaVA-7B, 896 for LLaVA-0.5B)
        out_dim:   output dimension (768 to match GPT-2 cross-attention)
        num_heads: number of attention heads in the refinement layer
        pdrop:     dropout probability
    """

    def __init__(self, in_dim: int = 4096, out_dim: int = 768, num_heads: int = 8, pdrop: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        mid = max(out_dim * 2, min(in_dim, 2048))
        self.main = nn.Sequential(
            nn.Linear(in_dim, mid), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(mid, 1024), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(1024, out_dim),
        )
        self.res = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim)
        )
        self.attn = nn.MultiheadAttention(out_dim, num_heads=num_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, in_dim] -> [B, L, out_dim]"""
        m = self.main(x)
        r = self.res(x)
        g = self.gate(torch.cat([m, r], dim=-1))
        fused = g * m + (1 - g) * r
        refined, _ = self.attn(fused, fused, fused)
        return refined


class SmartFeatureAdapter896(nn.Module):
    """Specialised 896 -> 768 adapter for Qwen2-0.5B based LLaVA backbone.

    Identical structure to SmartFeatureAdapter but with hard-coded in_dim=896
    and an explicit dimension check to catch configuration mistakes early.
    """

    def __init__(self, out_dim: int = 768, num_heads: int = 8, pdrop: float = 0.1):
        super().__init__()
        self.in_dim = 896
        self.out_dim = out_dim

        self.main = nn.Sequential(
            nn.Linear(self.in_dim, 2048), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(2048, 1024), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(1024, self.out_dim),
        )
        self.res = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim), nn.LayerNorm(self.out_dim)
        )
        self.attn = nn.MultiheadAttention(self.out_dim, num_heads=num_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(self.out_dim * 2, self.out_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, 896] -> [B, L, 768]"""
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"SmartFeatureAdapter896 expects last dim {self.in_dim}, got {x.shape[-1]}. "
                "Use SmartFeatureAdapter(in_dim=...) for other backbone sizes."
            )
        m = self.main(x)
        r = self.res(x)
        g = self.gate(torch.cat([m, r], dim=-1))
        fused = g * m + (1 - g) * r
        refined, _ = self.attn(fused, fused, fused)
        return refined


def build_adapters(llava_size: str, out_dim: int = 768) -> tuple:
    """Factory: returns (img_adapter, emo_adapter) matching the given backbone size.

    Args:
        llava_size: '0.5b' for Qwen2-0.5B backbone (hidden 896),
                    '7b' for 7B backbone (hidden 4096)
        out_dim:    GPT-2 cross-attention dimension (default 768)

    Returns:
        Tuple[SmartFeatureAdapter896 | SmartFeatureAdapter, same type]
    """
    if llava_size == '0.5b':
        return SmartFeatureAdapter896(out_dim), SmartFeatureAdapter896(out_dim)
    elif llava_size == '7b':
        return SmartFeatureAdapter(4096, out_dim), SmartFeatureAdapter(4096, out_dim)
    else:
        raise ValueError(f"Unsupported llava_size: '{llava_size}'. Choose '0.5b' or '7b'.")
