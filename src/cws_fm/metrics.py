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

