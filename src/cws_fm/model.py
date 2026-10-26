"""Neural velocity field for Causal Wavefront Simplex Flow Matching (CWS-FM).
Parameterizes the tangent vector field v_theta(u_t, t) on the positive unit sphere S^{V-1}_+.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from cws_fm.flow import project_tangent_sphere, wavefront_phase


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for continuous time t in [0, 1]."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B, 1) or (B,) tensor in [0, 1]."""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half_dim = self.d_model // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb_scale)
        args = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class TransformerBlock(nn.Module):
    """Bidirectional Transformer Block with adaptive time modulation (adaLN)."""
    def __init__(self, d_model: int, num_heads: int, d_ffn: Optional[int] = None):
        super().__init__()
        d_ffn = d_ffn or 4 * d_model
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )
        # Adaptive time conditioning scale and shift
        self.time_mod = nn.Linear(d_model, 4 * d_model)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # t_emb: (B, D) -> modulate norm layers
        mod = self.time_mod(t_emb).unsqueeze(1)  # (B, 1, 4*D)
        scale_attn, shift_attn, scale_ffn, shift_ffn = mod.chunk(4, dim=-1)

        # Attention sublayer
        norm_x1 = self.attn_norm(x) * (1.0 + scale_attn) + shift_attn
        attn_out, _ = self.attn(norm_x1, norm_x1, norm_x1)
        x = x + attn_out

        # FFN sublayer
        norm_x2 = self.ffn_norm(x) * (1.0 + scale_ffn) + shift_ffn
        x = x + self.ffn(norm_x2)
        return x


class SimplexVelocityField(nn.Module):
    """Bidirectional neural velocity field on the Fisher-Rao probability simplex.
    Computes v_theta(u_t, t) in T_{u_t} S^{V-1}_+ strictly orthogonal to u_t.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 512,
        alpha: float = 0.50,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.alpha = alpha

        # Projections
        self.in_proj = nn.Linear(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.time_embed = SinusoidalTimeEmbedding(d_model)

        # Transformer backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, u_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Args:
            u_t: (B, L, V) point on the unit sphere S^{V-1}_+.
            t: (B, 1) or scalar continuous time in [0, 1].
        Returns:
            v_t: (B, L, V) tangent velocity vector strictly satisfying <u_t, v_t> == 0.
        """
        B, L, V = u_t.shape
        device = u_t.device

        # Continuous time embedding
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([[t]], device=device, dtype=u_t.dtype)
        elif t.dim() == 0:
            t = t.view(1, 1).expand(B, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.size(0) != B:
            t = t.expand(B, 1)

        t_emb = self.time_embed(t)  # (B, D)

        # Token positions and sequence embedding
        positions = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
        h = self.in_proj(u_t) + self.pos_emb(positions)  # (B, L, D)

        # Deep bidirectional propagation
        for block in self.blocks:
            h = block(h, t_emb)

        # Velocity output
        raw_v = self.out_proj(self.final_norm(h))  # (B, L, V)

        # Strict Riemannian orthogonal projection onto T_{u_t} S^{V-1}_+
        tangent_v = project_tangent_sphere(u_t, raw_v)
        return tangent_v

