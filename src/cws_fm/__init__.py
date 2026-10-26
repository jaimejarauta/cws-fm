"""Causal Wavefront Simplex Flow Matching (CWS-FM).
Non-Autoregressive Language Modeling via Riemannian Optimal Transport on Simplex Manifolds.
"""

from cws_fm.flow import (
    bhattacharyya_sphere_map,
    inverse_bhattacharyya_sphere_map,
    wavefront_phase,
    geodesic_sphere_path,
    geodesic_sphere_velocity,
    project_tangent_sphere,
    riemannian_exp_map,
    sample_cws_fm,
)
from cws_fm.model import SimplexVelocityField
from cws_fm.metrics import (
    compute_rounding_collapse_rate,
    compute_simplex_entropy,
    compute_tangent_orthogonality_error,
    compute_token_accuracy,
)

__all__ = [
    "bhattacharyya_sphere_map",
    "inverse_bhattacharyya_sphere_map",
    "wavefront_phase",
    "geodesic_sphere_path",
    "geodesic_sphere_velocity",
    "project_tangent_sphere",
    "riemannian_exp_map",
    "sample_cws_fm",
    "SimplexVelocityField",
    "compute_rounding_collapse_rate",
    "compute_simplex_entropy",
    "compute_tangent_orthogonality_error",
    "compute_token_accuracy",
]
