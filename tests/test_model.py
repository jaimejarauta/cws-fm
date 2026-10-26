"""Unit tests for CWS-FM neural velocity field and sampling."""

import pytest
import torch
from cws_fm.model import SimplexVelocityField
from cws_fm.flow import sample_cws_fm
from cws_fm.metrics import compute_tangent_orthogonality_error, compute_rounding_collapse_rate


def test_velocity_field_orthogonality():
    B, L, V = 2, 8, 32
    model = SimplexVelocityField(vocab_size=V, d_model=64, num_layers=2, num_heads=2)

    u = torch.randn(B, L, V).abs()
    u = u / torch.norm(u, p=2, dim=-1, keepdim=True)
    t = torch.tensor([[0.35]])

    v = model(u, t)
    assert v.shape == (B, L, V)

    # Tangent vector must satisfy <u, v> == 0
    err = compute_tangent_orthogonality_error(u, v)
    assert err < 1e-5, f"Velocity output must lie on tangent space, error={err}"


def test_sample_cws_fm_shape_and_range():
    V = 16
    L = 8
    model = SimplexVelocityField(vocab_size=V, d_model=32, num_layers=2, num_heads=2)

    tokens, probs = sample_cws_fm(model, seq_len=L, vocab_size=V, num_steps=4)
    assert tokens.shape == (1, L)
    assert probs.shape == (1, L, V)
    assert (tokens >= 0).all() and (tokens < V).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1, L), atol=1e-5)
