"""Riemannian Optimal Transport Flow Matching on Fisher-Rao Simplex Manifolds.
Provides closed-form geodesics, tangent projections, causal wavefront scheduling,
and fast Riemannian Heun ODE sampling.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def bhattacharyya_sphere_map(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Isometrically maps discrete probability distributions on the simplex Delta^{V-1}
    to the positive unit hypersphere S^{V-1}_+ via the square-root diffeomorphism.
    Under the Fisher-Rao information metric, this map pulls back identically to 4 * Euclidean round metric.
    """
    p_clamped = torch.clamp(p, min=eps)
    p_norm = p_clamped / torch.sum(p_clamped, dim=-1, keepdim=True)
    return torch.sqrt(p_norm)


def inverse_bhattacharyya_sphere_map(u: torch.Tensor) -> torch.Tensor:
    """Inverts the Bhattacharyya sphere map, recovering valid categorical probabilities
    via squaring: p_v = (u_v)^2 / sum (u_v)^2.
    Guarantees sum(p_v) == 1.0 and p_v >= 0 with zero rounding collapse.
    """
    p = torch.square(torch.clamp(u, min=0.0))
    return p / torch.clamp(torch.sum(p, dim=-1, keepdim=True), min=1e-12)


def wavefront_phase(t: torch.Tensor, seq_len: int, alpha: float = 0.50) -> torch.Tensor:
    """Computes the causal wavefront phase tau(t, i) across sequence positions.
    Tokens with smaller indices reach tau=1.0 (locked) earlier, providing causal
    conditioning context to downstream tokens while preserving parallel generation.
    Args:
        t: scalar or (B, 1) global time in [0, 1].
        seq_len: sequence length L.
        alpha: wavefront tilt in [0, 1). alpha=0.50 gives 2x causal wavefront velocity.
    Returns:
        tau: (B, L) tensor with values clamped to [0, 1].
    """
    device = t.device
    pos = torch.linspace(0.0, 1.0, seq_len, device=device).unsqueeze(0)  # (1, L)
    denom = max(1e-5, 1.0 - alpha)
    tau = (t - alpha * pos) / denom
    return torch.clamp(tau, 0.0, 1.0)


def wavefront_delta_tau(t_curr: torch.Tensor, t_next: torch.Tensor, seq_len: int, alpha: float = 0.50) -> torch.Tensor:
    """Exact per-position phase advance tau(t_next) - tau(t_curr); sums to 1 over [0, 1] for every position."""
    return wavefront_phase(t_next, seq_len, alpha) - wavefront_phase(t_curr, seq_len, alpha)


def geodesic_sphere_path(u0: torch.Tensor, u1: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Computes the exact closed-form great-circle geodesic curve on S^{V-1}_+:
    gamma_tau(u0, u1) = (sin((1-tau)*theta) / sin(theta)) * u0 + (sin(tau*theta) / sin(theta)) * u1
    Preserves ||gamma_tau||_2 == 1.0 for all tau in [0, 1].
    """
    inner = torch.sum(u0 * u1, dim=-1, keepdim=True)
    inner_clamped = torch.clamp(inner, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(inner_clamped)

    sin_theta = torch.sin(theta)
    # Numerical stability for nearly identical vectors (theta -> 0)
    safe_sin = torch.where(sin_theta.abs() < 1e-5, torch.ones_like(sin_theta), sin_theta)

    c0 = torch.sin((1.0 - tau) * theta) / safe_sin
    c1 = torch.sin(tau * theta) / safe_sin

    # Linear interpolation fallback when theta -> 0
    c0 = torch.where(sin_theta.abs() < 1e-5, 1.0 - tau, c0)
    c1 = torch.where(sin_theta.abs() < 1e-5, tau, c1)

    gamma = c0 * u0 + c1 * u1
    return gamma / torch.clamp(torch.norm(gamma, p=2, dim=-1, keepdim=True), min=1e-12)


def geodesic_sphere_velocity(u0: torch.Tensor, u1: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Computes the exact analytical tangent velocity vector along the geodesic curve:
    d(gamma_tau)/d(tau) = (-theta * cos((1-tau)*theta) / sin(theta)) * u0 + (theta * cos(tau*theta) / sin(theta)) * u1
    Guarantees <gamma_tau, d(gamma_tau)/d(tau)> == 0 (strictly tangent to the sphere).
    """
    inner = torch.sum(u0 * u1, dim=-1, keepdim=True)
    inner_clamped = torch.clamp(inner, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(inner_clamped)

    sin_theta = torch.sin(theta)
    safe_sin = torch.where(sin_theta.abs() < 1e-5, torch.ones_like(sin_theta), sin_theta)

    v0 = (-theta * torch.cos((1.0 - tau) * theta)) / safe_sin
    v1 = (theta * torch.cos(tau * theta)) / safe_sin

    # Tangent velocity fallback when theta -> 0
    v0 = torch.where(sin_theta.abs() < 1e-5, -torch.ones_like(tau), v0)
    v1 = torch.where(sin_theta.abs() < 1e-5, torch.ones_like(tau), v1)

    vel = v0 * u0 + v1 * u1
    # Orthogonal projection onto the tangent space of gamma_tau for machine precision
    gamma = geodesic_sphere_path(u0, u1, tau)
    return project_tangent_sphere(gamma, vel)


def project_tangent_sphere(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Orthogonally projects vector v onto the tangent space of the unit sphere at u:
    P_u(v) = v - <u, v> * u
    Guarantees <u, P_u(v)> == 0.
    """
    inner = torch.sum(u * v, dim=-1, keepdim=True)
    return v - inner * u


def expected_log_map(u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """sum_k p_k Log_u(e_k) on the unit sphere: the marginal geodesic velocity under posterior p over vertices."""
    theta = torch.acos(u.clamp(0.0, 1.0 - 1e-7))
    w = p * theta / torch.sin(theta).clamp_min(1e-7)  # Log_u(e_k) = theta_k/sin(theta_k) * (e_k - u_k u)
    return w - (w * u).sum(-1, keepdim=True) * u


def riemannian_exp_map(u: torch.Tensor, v: torch.Tensor, step_size: float) -> torch.Tensor:
    """Computes the Riemannian exponential map on the unit sphere along tangent vector v:
    exp_u(h * v) = cos(||h * v||) * u + sin(||h * v||) * (v / ||v||)
    Guarantees that the updated point remains strictly on the unit sphere: ||exp_u(h * v)||_2 == 1.
    """
    delta = v * step_size
    norm_delta = torch.norm(delta, p=2, dim=-1, keepdim=True)
    safe_norm = torch.clamp(norm_delta, min=1e-8)

    direction = delta / safe_norm
    cos_term = torch.cos(norm_delta) * u
    sin_term = torch.sin(norm_delta) * direction

    out = cos_term + sin_term
    return out / torch.clamp(torch.norm(out, p=2, dim=-1, keepdim=True), min=1e-12)


def sample_cws_fm(
    model: torch.nn.Module,
    seq_len: int,
    vocab_size: int,
    num_steps: int = 8,
    alpha: float = 0.50,
    confidence_lock_threshold: float = 0.95,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    heun: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a full sequence in num_steps (4 or 8) parallel ODE steps via Riemannian Euler (optionally Heun) integration.
    Returns:
        tokens: (B, L) discrete token indices
        probs: (B, L, V) probability distributions on the simplex Delta^{V-1}
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    h = 1.0 / num_steps

    # Initialize with stochastic Dirichlet sample on S^{V-1}_+
    p0 = torch.distributions.Dirichlet(torch.ones(vocab_size, device=device)).sample((batch_size, seq_len))
    u_curr = torch.sqrt(p0)
    locked_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    locked_tokens = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    with torch.no_grad():
        for step in range(num_steps):
            t_curr = torch.tensor([[step * h]], device=device)
            t_next = torch.tensor([[(step + 1) * h]], device=device)

            # Step each position by its exact phase advance; sampling the band indicator at grid points
            # (trapezoid) integrated only 75% of each path
            d_tau = wavefront_delta_tau(t_curr, t_next, seq_len, alpha=alpha).unsqueeze(-1)  # (B, L, 1)

            # 1. Predictor step (Euler on sphere); the model predicts d(gamma)/d(tau)
            v_curr = project_tangent_sphere(u_curr, model(u_curr, t_curr))
            u_euler = riemannian_exp_map(u_curr, v_curr * d_tau, step_size=1.0)
            u_euler = torch.clamp(u_euler, min=0.0)
            u_euler = u_euler / torch.norm(u_euler, p=2, dim=-1, keepdim=True)

            # 2. Optional Heun corrector. Off by default: with the posterior parametrization Euler lands on the
            # predicted vertex when d_tau = 1 - tau, while v_next ~ 0 there halves the final step
            if heun:
                v_next = project_tangent_sphere(u_euler, model(u_euler, t_next))
                u_next = riemannian_exp_map(u_curr, 0.5 * (v_curr + v_next) * d_tau, step_size=1.0)
                u_next = torch.clamp(u_next, min=0.0)
                u_next = u_next / torch.norm(u_next, p=2, dim=-1, keepdim=True)
            else:
                u_next = u_euler

            # 3. Adaptive confidence locking based on updated state at t_next
            p_continuous = inverse_bhattacharyya_sphere_map(u_next)
            max_p, argmax_p = torch.max(p_continuous, dim=-1)

            new_locks = (max_p >= confidence_lock_threshold)
            locked_mask = locked_mask | new_locks
            locked_tokens = torch.where(new_locks, argmax_p, locked_tokens)

            # Snap only confirmed high-confidence locked positions
            if locked_mask.any():
                one_hot = F.one_hot(locked_tokens, num_classes=vocab_size).float()
                u_next = torch.where(locked_mask.unsqueeze(-1), one_hot, u_next)

            u_curr = u_next

    # Evaluate probabilities on the continuous spherical trajectory
    final_probs = inverse_bhattacharyya_sphere_map(u_curr)
    final_tokens = torch.argmax(final_probs, dim=-1)
    return final_tokens, final_probs


def flow_matching_loss(model: torch.nn.Module, tokens: torch.Tensor, vocab_size: int, alpha: float = 0.50) -> torch.Tensor:
    """Posterior flow-matching loss: sqrt(Dirichlet(1)) source, geodesic path at the wavefront phase, CE on the clean token."""
    B, L = tokens.shape
    u1 = torch.sqrt(F.one_hot(tokens, num_classes=vocab_size).float())
    t = torch.rand(B, 1, device=tokens.device)
    tau = wavefront_phase(t, L, alpha=alpha).unsqueeze(-1)
    u0 = torch.sqrt(torch.distributions.Dirichlet(torch.ones(vocab_size, device=tokens.device)).sample((B, L)))
    logits, _ = model.logits(geodesic_sphere_path(u0, u1, tau), t)
    return F.cross_entropy(logits.reshape(-1, vocab_size), tokens.reshape(-1))
