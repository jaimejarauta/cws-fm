"""Unit tests for Riemannian flow math on S^{V-1}_+."""

import math
import pytest
import torch
import torch.nn.functional as F

from cws_fm.flow import (
    bhattacharyya_sphere_map,
    inverse_bhattacharyya_sphere_map,
    wavefront_phase,
    geodesic_sphere_path,
    geodesic_sphere_velocity,
    project_tangent_sphere,
    riemannian_exp_map,
)
from cws_fm.metrics import compute_tangent_orthogonality_error


def test_bhattacharyya_sphere_diffeomorphism():
    B, L, V = 2, 8, 32
    probs = F.softmax(torch.randn(B, L, V), dim=-1)

    # Forward map
    u = bhattacharyya_sphere_map(probs)
    # Unit norm test
    norms = torch.norm(u, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Vectors must lie on unit sphere"

    # Inverse map
    p_rec = inverse_bhattacharyya_sphere_map(u)
    assert torch.allclose(probs, p_rec, atol=1e-5), "Inverse map must recover exact probabilities"
    assert torch.allclose(p_rec.sum(dim=-1), torch.ones(B, L), atol=1e-5), "Simplex sum to 1 invariant"


def test_geodesic_properties():
    B, L, V = 2, 4, 16
    u0 = F.normalize(torch.rand(B, L, V) + 0.1, p=2, dim=-1)
    u1 = F.normalize(torch.rand(B, L, V) + 0.1, p=2, dim=-1)

    tau = torch.tensor(0.5)
    gamma_half = geodesic_sphere_path(u0, u1, tau)

    # Norm must remain 1.0 along entire path
    assert torch.allclose(torch.norm(gamma_half, p=2, dim=-1), torch.ones(B, L), atol=1e-5)

    # Tangent velocity must be orthogonal to path position
    vel = geodesic_sphere_velocity(u0, u1, tau)
    ortho_err = compute_tangent_orthogonality_error(gamma_half, vel)
    assert ortho_err < 1e-5, f"Velocity must be orthogonal to path, got max err {ortho_err}"


def test_wavefront_phase_monotonicity():
    seq_len = 16
    t = torch.tensor([[0.5]])
    tau = wavefront_phase(t, seq_len=seq_len, alpha=0.50)  # (1, L)

    # Head tokens (pos 0) must have higher or equal phase than tail tokens (pos 15)
    assert tau[0, 0] >= tau[0, -1], "Wavefront phase must be non-increasing with position index"


def test_riemannian_exp_map():
    B, L, V = 2, 4, 16
    u = F.normalize(torch.rand(B, L, V) + 0.1, p=2, dim=-1)
    raw_v = torch.randn(B, L, V)
    v = project_tangent_sphere(u, raw_v)

    step_size = 0.125
    u_next = riemannian_exp_map(u, v, step_size=step_size)

    # Preserves unit norm
    assert torch.allclose(torch.norm(u_next, p=2, dim=-1), torch.ones(B, L), atol=1e-5)



def test_untrained_sampler_can_fail_collapse_gate():
    from cws_fm.model import SimplexVelocityField
    from cws_fm.flow import sample_cws_fm
    from cws_fm.metrics import compute_rounding_collapse_rate
    torch.manual_seed(0)
    _, probs = sample_cws_fm(SimplexVelocityField(32), seq_len=16, vocab_size=32, num_steps=8, batch_size=8)
    # without forced locking, an untrained field must not look sharp
    assert compute_rounding_collapse_rate(probs, threshold=0.85) > 0.5


def test_wavefront_delta_tau_sums_to_one():
    from cws_fm.flow import wavefront_delta_tau
    for k in (4, 8, 7):
        total = sum(wavefront_delta_tau(torch.tensor([[i / k]]), torch.tensor([[(i + 1) / k]]), 16, 0.5)
                    for i in range(k))
        assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


def test_expected_log_map_matches_geodesic_for_point_posterior():
    from cws_fm.flow import expected_log_map, geodesic_sphere_velocity
    torch.manual_seed(0)
    V = 16
    u = torch.rand(4, V)
    u = u / u.norm(dim=-1, keepdim=True)
    k = torch.tensor([3, 0, 7, 15])
    e = torch.nn.functional.one_hot(k, V).float()
    got = expected_log_map(u, e)
    want = geodesic_sphere_velocity(u, e, torch.zeros(4, 1))  # d/dtau at tau=0 is the log map
    assert torch.allclose(got, want, atol=1e-4)
    assert (got * u).sum(-1).abs().max() < 1e-5


def test_flow_matching_loss_is_finite_and_trains():
    from cws_fm.model import SimplexVelocityField
    from cws_fm.flow import flow_matching_loss
    torch.manual_seed(0)
    model = SimplexVelocityField(8, d_model=32, num_layers=1, num_heads=2, max_seq_len=4)
    tokens = torch.tensor([[1, 2, 2, 1]]).repeat(8, 1)
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    first = flow_matching_loss(model, tokens, 8, alpha=0.5)
    for _ in range(50):
        loss = flow_matching_loss(model, tokens, 8, alpha=0.5)
        opt.zero_grad(); loss.backward(); opt.step()
    assert torch.isfinite(first) and loss < first
