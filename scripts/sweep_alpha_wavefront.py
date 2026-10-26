#!/usr/bin/env python3
"""Stage 2 Wavefront Parameter Ablation: Sweeping wavefront tilt alpha in [0.0, 0.75].
Evaluates how causal phase velocity affects generation accuracy and token entropy.
Saves provenance manifest to results/alpha_sweep_manifest.json.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
import torch
import torch.nn.functional as F

from cws_fm.flow import (
    bhattacharyya_sphere_map,
    geodesic_sphere_path,
    geodesic_sphere_velocity,
    wavefront_phase,
    sample_cws_fm,
)
from cws_fm.model import SimplexVelocityField
from cws_fm.metrics import compute_rounding_collapse_rate, compute_simplex_entropy


def generate_causal_grammar_data(num_samples: int, seq_len: int, vocab_size: int, device: torch.device):
    """Generates grammar where token i strictly depends on all prior tokens 0..i-1."""
    tokens = torch.zeros(num_samples, seq_len, dtype=torch.long, device=device)
    tokens[:, 0] = torch.randint(1, vocab_size, (num_samples,), device=device)
    for pos in range(1, seq_len):
        tokens[:, pos] = (tokens[:, pos - 1] * 3 + 5) % vocab_size
    return tokens


def main():
    parser = argparse.ArgumentParser(description="Sweep wavefront tilt alpha")
    parser.add_argument("--steps", type=int, default=150, help="Training steps per alpha")
    parser.add_argument("--seq-len", type=int, default=24, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=48, help="Vocab size")
    parser.add_argument("--d-model", type=int, default=64, help="Hidden dimension")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 76)
    print("STAGE 2: CAUSAL WAVEFRONT TILT ABLATION SWEEP (alpha in [0.0, 0.75])")
    print(f"Device: {device} | Sequence Length: {args.seq_len} | Vocab: {args.vocab_size}")
    print("=" * 76)

    data = generate_causal_grammar_data(1000, args.seq_len, args.vocab_size, device)
    one_hot = F.one_hot(data, num_classes=args.vocab_size).float()
    u1_all = bhattacharyya_sphere_map(one_hot)
    u0_uniform = torch.ones(args.vocab_size, device=device) / math.sqrt(args.vocab_size)

    alphas = [0.0, 0.25, 0.50, 0.75]
    sweep_results = {}

    print(f"\n{'Alpha Tilt':<12} | {'Description':<26} | {'Final Loss':<12} | {'Collapse %':<12} | {'Entropy':<10}")
    print("-" * 76)

    for alpha in alphas:
        model = SimplexVelocityField(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=3,
            num_heads=4,
            max_seq_len=args.seq_len,
            alpha=alpha,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        model.train()

        for step in range(args.steps):
            idx = torch.randint(0, data.size(0), (16,))
            u1_batch = u1_all[idx]

            t = torch.rand(16, 1, device=device)
            tau = wavefront_phase(t, seq_len=args.seq_len, alpha=alpha).unsqueeze(-1)
            u0_batch = u0_uniform.view(1, 1, -1).expand(16, args.seq_len, -1)

            u_tau = geodesic_sphere_path(u0_batch, u1_batch, tau)
            v_target = geodesic_sphere_velocity(u0_batch, u1_batch, tau)

            v_pred = model(u_tau, t)
            loss = F.mse_loss(v_pred, v_target)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # Evaluate sampling at K=8 steps
        model.eval()
        with torch.no_grad():
            _, probs = sample_cws_fm(
                model,
                seq_len=args.seq_len,
                vocab_size=args.vocab_size,
                num_steps=8,
                alpha=alpha,
                device=device,
            )
            collapse = compute_rounding_collapse_rate(probs, threshold=0.85)
            entropy = compute_simplex_entropy(probs)

        desc = "Pure Parallel (No Wavefront)" if alpha == 0.0 else f"{1.0/(1.0-alpha):.1f}x Phase Velocity"
        print(f"{alpha:<12.2f} | {desc:<26} | {loss.item():<12.5f} | {collapse * 100:<12.1f}% | {entropy:<10.3f}")

        sweep_results[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "final_flow_loss": float(loss.item()),
            "rounding_collapse_rate": collapse,
            "entropy": entropy,
        }

    # Manifest
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    manifest_path = os.path.join(results_dir, "alpha_sweep_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"date": datetime.now(timezone.utc).isoformat(), "sweep": sweep_results}, f, indent=2)

    print(f"\nManifest saved to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
