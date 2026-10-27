#!/usr/bin/env python3
"""Smoke benchmark training and verification runner for CWS-FM.
Evaluates Riemannian probability flow matching with 4-step and 8-step ODE generation.
Saves provenance manifest to results/smoke_manifest.json.
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
    flow_matching_loss,
    bhattacharyya_sphere_map,
    geodesic_sphere_path,
    wavefront_phase,
    sample_cws_fm,
)
from cws_fm.model import SimplexVelocityField
from cws_fm.metrics import (
    compute_palindrome_validity,
    compute_distinct_ratio,
    compute_rounding_collapse_rate,
    compute_simplex_entropy,
    compute_token_accuracy,
    compute_tangent_orthogonality_error,
)


def generate_synthetic_data(num_samples: int, seq_len: int, vocab_size: int, device: torch.device):
    """Generates structured sequence data (palindrome pattern: a, b, c, d, d, c, b, a)."""
    half = seq_len // 2
    prefix = torch.randint(1, vocab_size - 1, (num_samples, half), device=device)
    suffix = torch.flip(prefix, dims=[1])
    data = torch.cat([prefix, suffix], dim=1)
    return data


def main():
    parser = argparse.ArgumentParser(description="Run CWS-FM smoke benchmark")
    parser.add_argument("--steps", type=int, default=200, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=32, help="Vocab size")
    parser.add_argument("--d-model", type=int, default=64, help="Model hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--eval-samples", type=int, default=64, help="Sequences sampled per eval")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--alpha", type=float, default=0.50, help="Wavefront tilt; 0 = all positions in sync")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, mps, cpu)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed)
    print("=" * 70)
    print("CWS-FM: CAUSAL WAVEFRONT SIMPLEX FLOW MATCHING SMOKE BENCHMARK")
    print(f"Device: {device} | Steps: {args.steps} | Vocab: {args.vocab_size} | Seq: {args.seq_len}")
    print("=" * 70)

    # Initialize model
    model = SimplexVelocityField(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
        alpha=args.alpha,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training data
    train_tokens = generate_synthetic_data(1000, args.seq_len, args.vocab_size, device)

    start_time = time.time()
    model.train()

    print("\nTraining Riemannian Velocity Field v_theta on S^{V-1}_+...")
    for step in range(args.steps):
        idx = torch.randint(0, train_tokens.size(0), (args.batch_size,))
        loss = flow_matching_loss(model, train_tokens[idx], args.vocab_size, alpha=args.alpha)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 50 == 0 or step == 0:
            print(f"  Step {step + 1:03d}/{args.steps} | CE Loss: {loss.item():.6f}")

    train_elapsed = time.time() - start_time
    print(f"Training completed in {train_elapsed:.2f}s")

    # Evaluation: 4-Step and 8-Step Fast Sampling
    print("\nEvaluating Fast Non-Autoregressive ODE Sampling:")
    test_results = {}

    for steps in [4, 8]:
        t0 = time.time()
        gen_tokens, gen_probs = sample_cws_fm(
            model,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            num_steps=steps,
            alpha=args.alpha,
            confidence_lock_threshold=0.90,
            batch_size=args.eval_samples,
            device=device,
        )
        sample_time = time.time() - t0

        collapse_rate = compute_rounding_collapse_rate(gen_probs, threshold=0.85)
        entropy = compute_simplex_entropy(gen_probs)
        ortho_err = compute_tangent_orthogonality_error(
            bhattacharyya_sphere_map(gen_probs),
            model(bhattacharyya_sphere_map(gen_probs), torch.tensor([[1.0]], device=device))
        )

        validity = compute_palindrome_validity(gen_tokens)
        distinct = compute_distinct_ratio(gen_tokens)
        test_results[f"steps_{steps}"] = {
            "palindrome_validity": validity,
            "distinct_ratio": distinct,
            "num_steps": steps,
            "sample_time_ms": sample_time * 1000.0,
            "tokens_per_sec": args.seq_len / max(1e-5, sample_time),
            "rounding_collapse_rate": collapse_rate,
            "average_entropy": entropy,
            "orthogonality_error": ortho_err,
        }
        print(f"  [K={steps} Steps] Time: {sample_time * 1000.0:.2f}ms | "
              f"Rate: {args.seq_len / max(1e-5, sample_time):.1f} tok/s | "
              f"Rounding Collapse Rate: {collapse_rate * 100:.1f}% | "
              f"Entropy: {entropy:.3f} | Valid: {validity * 100:.1f}% | Distinct: {distinct * 100:.1f}%")

    # Untrained control: same sampler, random weights; a gate it passes is not evidence
    control = SimplexVelocityField(
        vocab_size=args.vocab_size, d_model=args.d_model, num_layers=args.num_layers,
        num_heads=args.num_heads, max_seq_len=args.seq_len, alpha=args.alpha,
    ).to(device)
    ctrl_tokens, ctrl_probs = sample_cws_fm(control, seq_len=args.seq_len, vocab_size=args.vocab_size,
                                            num_steps=8, alpha=args.alpha, confidence_lock_threshold=0.90,
                                            batch_size=args.eval_samples, device=device)
    ctrl_validity = compute_palindrome_validity(ctrl_tokens)
    ctrl_collapse = compute_rounding_collapse_rate(ctrl_probs, threshold=0.85)
    print(f"  [Untrained control K=8] Collapse: {ctrl_collapse * 100:.1f}% | Valid: {ctrl_validity * 100:.1f}%")

    # Gates (revised 2026-10-26): Gate 1 tests sample quality, not sharpness, since locking makes sharpness cheap.
    # Gate 1: >= 50% of K=8 samples obey the palindrome grammar, >= 90% distinct, untrained control < 5% valid.
    # Gate 2: K=8 sampling of the eval batch within 150 ms.
    k8 = test_results["steps_8"]
    k8_collapse = k8["rounding_collapse_rate"]
    k8_time_ms = k8["sample_time_ms"]
    gate1_pass = k8["palindrome_validity"] >= 0.50 and k8["distinct_ratio"] >= 0.90 and ctrl_validity < 0.05
    gate2_pass = k8_time_ms <= 150.0  # Conservative for CPU/MPS smoke

    print("\nPre-Registered Empirical Gates:")
    print(f"  Gate 1 (Grammar validity >= 50%, distinct >= 90%): {'PASS' if gate1_pass else 'FAIL'} "
          f"({k8['palindrome_validity'] * 100:.1f}%, {k8['distinct_ratio'] * 100:.1f}%; control {ctrl_validity * 100:.1f}%)")
    print(f"  Gate 2 (Fast Inference <= 150ms):     {'PASS' if gate2_pass else 'FAIL'} ({k8_time_ms:.2f}ms)")

    overall_pass = gate1_pass and gate2_pass

    # Save Manifest
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    manifest_path = os.path.join(results_dir, "smoke_manifest.json")

    manifest = {
        "architecture": "cws-fm",
        "description": "Causal Wavefront Simplex Flow Matching",
        "date": datetime.now(timezone.utc).isoformat(),
        "hardware": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "vocab_size": args.vocab_size,
        "train_elapsed_sec": train_elapsed,
        "final_loss": float(loss.item()),
        "eval_k4": test_results["steps_4"],
        "eval_k8": test_results["steps_8"],
        "seed": args.seed,
        "untrained_control": {"palindrome_validity": ctrl_validity, "rounding_collapse_rate": ctrl_collapse},
        "gate1_passed": gate1_pass,
        "gate2_passed": gate2_pass,
        "gate_status": "PASS" if overall_pass else "FAIL",
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {manifest_path}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())

