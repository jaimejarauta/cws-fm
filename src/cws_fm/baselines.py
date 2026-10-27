"""Token-space baselines for the text8 study: masked diffusion (MDLM-style, absorbing mask) and autoregressive."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenTransformer(nn.Module):
    """Pre-norm transformer over token ids; causal for AR, bidirectional for masked diffusion."""

    def __init__(self, vocab: int, d_model: int = 512, layers: int = 8, heads: int = 8, max_len: int = 256,
                 causal: bool = False):
        super().__init__()
        self.causal = causal
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        h = self.embed(x) + self.pos(torch.arange(L, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device) if self.causal else None
        return self.head(self.norm(self.blocks(h, mask=mask, is_causal=self.causal)))


def mdlm_loss(model: TokenTransformer, x: torch.Tensor, mask_id: int) -> torch.Tensor:
    """Mask each token with prob t ~ U(0, 1); CE on masked positions (the 1/t ELBO weight is dropped for stability)."""
    t = torch.rand(x.size(0), 1, device=x.device).clamp_min(1e-3)
    masked = torch.rand(x.shape, device=x.device) < t
    logits = model(torch.where(masked, torch.full_like(x, mask_id), x))
    return F.cross_entropy(logits[masked], x[masked])


@torch.no_grad()
def mdlm_sample(model: TokenTransformer, batch: int, seq_len: int, steps: int, mask_id: int) -> torch.Tensor:
    """Start fully masked; at each step sample all masked tokens and keep the most confident ones (NFE = steps)."""
    device = next(model.parameters()).device
    x = torch.full((batch, seq_len), mask_id, dtype=torch.long, device=device)
    for s in range(steps):
        masked = x == mask_id
        logits = model(x)
        logits[..., mask_id] = -float("inf")
        probs = F.softmax(logits.float(), -1)
        draw = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch, seq_len)
        conf = probs.gather(-1, draw[..., None]).squeeze(-1).masked_fill(~masked, -1.0)
        n_keep = math.ceil(seq_len * (s + 1) / steps) - int((~masked[0]).sum())
        if n_keep > 0:
            idx = conf.topk(n_keep, dim=-1).indices
            x.scatter_(1, idx, draw.gather(1, idx))
    return x


def ar_loss(model: TokenTransformer, x: torch.Tensor) -> torch.Tensor:
    logits = model(x[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))


@torch.no_grad()
def ar_sample(model: TokenTransformer, batch: int, seq_len: int, temperature: float = 1.0) -> torch.Tensor:
    """Ancestral sampling (no KV cache; fine at L = 256). The first token comes from a uniform prompt token."""
    device = next(model.parameters()).device
    vocab = model.head.out_features
    x = torch.randint(0, vocab, (batch, 1), device=device)
    for _ in range(seq_len - 1):
        logits = model(x)[:, -1].float() / temperature
        x = torch.cat([x, torch.multinomial(F.softmax(logits, -1), 1)], 1)
    return x


@torch.no_grad()
def ar_sample_cached(model: TokenTransformer, batch: int, seq_len: int, temperature: float = 1.0) -> torch.Tensor:
    """Same distribution as ar_sample with a per-layer KV cache: one token per forward, the fair AR latency arm."""
    device = next(model.parameters()).device
    vocab = model.head.out_features
    x = torch.randint(0, vocab, (batch, 1), device=device)
    cache = [None] * len(model.blocks.layers)
    for t in range(seq_len - 1):
        h = model.embed(x[:, -1:]) + model.pos.weight[t]
        for i, layer in enumerate(model.blocks.layers):  # norm_first encoder layer, unrolled
            attn = layer.self_attn
            q, k, v = F.linear(layer.norm1(h), attn.in_proj_weight, attn.in_proj_bias).chunk(3, -1)
            if cache[i] is not None:  # ponytail: cat per step is O(L^2) copies; preallocate if L >> 256
                k, v = torch.cat([cache[i][0], k], 1), torch.cat([cache[i][1], v], 1)
            cache[i] = (k, v)
            split = lambda z: z.view(batch, -1, attn.num_heads, attn.head_dim).transpose(1, 2)
            o = F.scaled_dot_product_attention(split(q), split(k), split(v))  # one query sees only the past
            h = h + attn.out_proj(o.transpose(1, 2).reshape(batch, 1, -1))
            h = h + layer.linear2(layer.activation(layer.linear1(layer.norm2(h))))
        logits = model.head(model.norm(h[:, -1])).float() / temperature
        x = torch.cat([x, torch.multinomial(F.softmax(logits, -1), 1)], 1)
    return x


@torch.no_grad()
def judge_bits_per_char(judge: TokenTransformer, x: torch.Tensor, batch: int = 256) -> float:
    """Mean NLL in bits/char of x[:, 1:] under an AR judge (first char excluded)."""
    total, count = 0.0, 0
    for i in range(0, x.size(0), batch):
        xb = x[i:i + batch].to(next(judge.parameters()).device)
        logits = judge(xb[:, :-1]).float()
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), xb[:, 1:].reshape(-1), reduction="sum")
        total += nll.item()
        count += xb[:, 1:].numel()
    return total / count / math.log(2)
