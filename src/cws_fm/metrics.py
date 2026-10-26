"""Evaluation metrics for CWS-FM probability flow matching."""

import torch
import torch.nn.functional as F


def compute_rounding_collapse_rate(probs: torch.Tensor, threshold: float = 0.90) -> float:
    """Computes the percentage of tokens where the maximum categorical probability
    is below threshold (indicating failure to commit to discrete vocabulary tokens).
    Args:
        probs: (B, L, V) probability distributions on Delta^{V-1}.
    Returns:
        collapse_rate: float in [0.0, 1.0]. Zero means 100% sharp token commitment.
    """
    max_p, _ = torch.max(probs, dim=-1)  # (B, L)
    uncommitted = (max_p < threshold).float()
    return float(uncommitted.mean().item())


def compute_simplex_entropy(probs: torch.Tensor, eps: float = 1e-12) -> float:
    """Computes average categorical Shannon entropy across sequence positions.
    H(p) = - sum_v p_v * log(p_v).
    """
    p_clamped = torch.clamp(probs, min=eps)
    entropy = -torch.sum(p_clamped * torch.log(p_clamped), dim=-1)  # (B, L)
    return float(entropy.mean().item())


def compute_tangent_orthogonality_error(u: torch.Tensor, v: torch.Tensor) -> float:
    """Computes maximum absolute inner product |<u, v>| to verify strict tangent space projection."""
    inner = torch.sum(u * v, dim=-1).abs()
    return float(inner.max().item())


def compute_token_accuracy(pred_tokens: torch.Tensor, target_tokens: torch.Tensor) -> float:
    """Computes exact token match accuracy between generated and ground-truth sequences."""
    matches = (pred_tokens == target_tokens).float()
    return float(matches.mean().item())



def compute_palindrome_validity(tokens: torch.Tensor) -> float:
    """Fraction of sequences whose second half is the reverse of the first (the training grammar)."""
    half = tokens.size(1) // 2
    ok = (tokens[:, :half] == torch.flip(tokens[:, -half:], dims=[1])).all(dim=1)
    return float(ok.float().mean().item())


def compute_distinct_ratio(tokens: torch.Tensor) -> float:
    """Unique sequences / total sequences; 1/B means a deterministic sampler."""
    return torch.unique(tokens, dim=0).size(0) / tokens.size(0)


def ngram_entropy(tokens: torch.Tensor, n: int) -> float:
    """Shannon entropy (bits) of the empirical n-gram distribution pooled over all sequences."""
    from collections import Counter
    import math
    counts = Counter(tuple(row[i:i + n].tolist()) for row in tokens for i in range(row.numel() - n + 1))
    total = sum(counts.values())
    return -sum(c / total * math.log2(c / total) for c in counts.values())


def distinct_ngrams(tokens: torch.Tensor, n: int) -> float:
    """Distinct n-grams / total n-grams; low values flag repetitive (collapsed) samples."""
    grams = [tuple(row[i:i + n].tolist()) for row in tokens for i in range(row.numel() - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def word_validity(tokens: torch.Tensor, vocab_words: set) -> float:
    """Fraction of complete space-separated words that occur in the training vocabulary."""
    from cws_fm.text8 import decode
    words = [w for row in tokens for w in decode(row).split(" ")[1:-1] or decode(row).split(" ") if w]
    return sum(w in vocab_words for w in words) / max(1, len(words))
