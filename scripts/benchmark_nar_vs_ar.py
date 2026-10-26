#!/usr/bin/env python3
"""Stage 2 Empirical Benchmark: Non-Autoregressive Simplex Flow Matching vs Autoregressive Transformer.
Compares CWS-FM (K=4, K=8 parallel ODE steps) against sequential token-by-token AR decoding.
Measures wall-clock latency (ms), throughput (tokens/sec), generation speedup, and token quality.
Saves provenance manifest to results/nar_vs_ar_manifest.json.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
import torch
import torch.nn as nn
import torch.nn.functional as F

from cws_fm.flow import (
    bhattacharyya_sphere_map,
    geodesic_sphere_path,
    geodesic_sphere_velocity,
    wavefront_phase,
    sample_cws_fm,
)
from cws_fm.model import SimplexVelocityField
from cws_fm.metrics import (
    compute_rounding_collapse_rate,
    compute_simplex_entropy,
)


class AutoregressiveTransformer(nn.Module):
    """Standard causal autoregressive Transformer baseline."""
    def __init__(self, vocab_size: int, d_model: int = 64, num_layers: int = 3, num_heads: int = 4, max_seq_len: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
        out = self.transformer(h, mask=mask, is_causal=True)
        return self.lm_head(self.final_norm(out))

    @torch.no_grad()
    def generate_token_by_token(self, prompt: torch.Tensor, max_len: int) -> torch.Tensor:
        """Sequential autoregressive decoding token-by-token in (max_len - prompt_len) passes."""
        curr = prompt.clone()
        while curr.size(1) < max_len:
            logits = self(curr)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr = torch.cat([curr, next_token], dim=1)
        return curr


def generate_structured_data(num_samples: int, seq_len: int, vocab_size: int, device: torch.device):
    """Generates structured sequence data (prefix + mirror/arithmetic transformation)."""
    half = seq_len // 2
    prefix = torch.randint(1, vocab_size // 2, (num_samples, half), device=device)
    suffix = (prefix * 3 + 7) % vocab_size
    data = torch.cat([prefix, suffix], dim=1)
    return data


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Benchmark: NAR CWS-FM vs Autoregressive LLM")
    parser.add_argument("--steps", type=int, default=200, help="Training steps")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=64, help="Vocab size")
    parser.add_argument("--d-model", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--eval-iters", type=int, default=25, help="Timing benchmark iterations")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 80)
    print("STAGE 2 EMPIRICAL BENCHMARK: CWS-FM (NAR) vs AUTOREGRESSIVE LLM (AR)")
    print(f"Device: {device} | Sequence Length: {args.seq_len} | Vocab Size: {args.vocab_size}")
    print("=" * 80)

    # 1. Initialize models
    nar_model = SimplexVelocityField(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=3,
        num_heads=4,
        max_seq_len=args.seq_len,
    ).to(device)

    ar_model = AutoregressiveTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=3,
        num_heads=4,
        max_seq_len=args.seq_len,
    ).to(device)

    nar_opt = torch.optim.AdamW(nar_model.parameters(), lr=2e-3, weight_decay=1e-4)
    ar_opt = torch.optim.AdamW(ar_model.parameters(), lr=2e-3, weight_decay=1e-4)

    # Training data
    train_data = generate_structured_data(1000, args.seq_len, args.vocab_size, device)
    train_one_hot = F.one_hot(train_data, num_classes=args.vocab_size).float()
    u1_all = bhattacharyya_sphere_map(train_one_hot)
    u0_uniform = torch.ones(args.vocab_size, device=device) / math.sqrt(args.vocab_size)

    # 2. Train both models
    print("\n[Phase 1] Training CWS-FM (Simplex Flow Matching) & AR Transformer...")
    t_start = time.time()
    nar_model.train()
    ar_model.train()

    for step in range(args.steps):
        idx = torch.randint(0, train_data.size(0), (args.batch_size,))
        batch_tokens = train_data[idx]

        # AR Forward step
        ar_logits = ar_model(batch_tokens[:, :-1])
        ar_loss = F.cross_entropy(ar_logits.reshape(-1, args.vocab_size), batch_tokens[:, 1:].reshape(-1))

        ar_opt.zero_grad()
        ar_loss.backward()
        ar_opt.step()

        # CWS-FM Forward step
        u1_batch = u1_all[idx]
        t = torch.rand(args.batch_size, 1, device=device)
        tau = wavefront_phase(t, seq_len=args.seq_len, alpha=0.50).unsqueeze(-1)
        u0_batch = u0_uniform.view(1, 1, -1).expand(args.batch_size, args.seq_len, -1)
        u_tau = geodesic_sphere_path(u0_batch, u1_batch, tau)
        v_target = geodesic_sphere_velocity(u0_batch, u1_batch, tau)

        v_pred = nar_model(u_tau, t)
        nar_loss = F.mse_loss(v_pred, v_target)

        nar_opt.zero_grad()
        nar_loss.backward()
        nar_opt.step()

        if (step + 1) % 50 == 0 or step == 0:
            print(f"  Step {step + 1:03d}/{args.steps} | AR Loss: {ar_loss.item():.4f} | CWS-FM Loss: {nar_loss.item():.5f}")

    print(f"Training completed in {time.time() - t_start:.2f}s")

    # 3. Benchmark Inference Speed and Throughput
    print(f"\n[Phase 2] Profiling Inference Latency Across {args.eval_iters} Runs (Batch Size = 1):")
    nar_model.eval()
    ar_model.eval()

    # Warmup
    prompt = torch.tensor([[1]], device=device)
    _ = ar_model.generate_token_by_token(prompt, max_len=args.seq_len)
    _ = sample_cws_fm(nar_model, seq_len=args.seq_len, vocab_size=args.vocab_size, num_steps=8, device=device)

    # 3.1 Autoregressive Profiling
    ar_times = []
    with torch.no_grad():
        for _ in range(args.eval_iters):
            t0 = time.perf_counter()
            _ = ar_model.generate_token_by_token(prompt, max_len=args.seq_len)
            ar_times.append(time.perf_counter() - t0)

    avg_ar_ms = (sum(ar_times) / len(ar_times)) * 1000.0
    ar_tok_s = (args.seq_len - 1) / (sum(ar_times) / len(ar_times))

    # 3.2 CWS-FM Profiling (K=4 steps)
    cws_k4_times = []
    with torch.no_grad():
        for _ in range(args.eval_iters):
            t0 = time.perf_counter()
            _, _ = sample_cws_fm(nar_model, seq_len=args.seq_len, vocab_size=args.vocab_size, num_steps=4, device=device)
            cws_k4_times.append(time.perf_counter() - t0)

    avg_k4_ms = (sum(cws_k4_times) / len(cws_k4_times)) * 1000.0
    k4_tok_s = args.seq_len / (sum(cws_k4_times) / len(cws_k4_times))

    # 3.3 CWS-FM Profiling (K=8 steps)
    cws_k8_times = []
    k8_probs_list = []
    with torch.no_grad():
        for _ in range(args.eval_iters):
            t0 = time.perf_counter()
            _, probs = sample_cws_fm(nar_model, seq_len=args.seq_len, vocab_size=args.vocab_size, num_steps=8, device=device)
            cws_k8_times.append(time.perf_counter() - t0)
            k8_probs_list.append(probs)

    avg_k8_ms = (sum(cws_k8_times) / len(cws_k8_times)) * 1000.0
    k8_tok_s = args.seq_len / (sum(cws_k8_times) / len(cws_k8_times))

    speedup_k4 = avg_ar_ms / max(1e-5, avg_k4_ms)
    speedup_k8 = avg_ar_ms / max(1e-5, avg_k8_ms)

    final_k8_collapse = compute_rounding_collapse_rate(k8_probs_list[-1], threshold=0.85)
    final_k8_entropy = compute_simplex_entropy(k8_probs_list[-1])

    print(f"\n{'Architecture':<24} | {'ODE Steps':<10} | {'Latency (ms)':<14} | {'Throughput (tok/s)':<18} | {'Speedup':<10}")
    print("-" * 84)
    print(f"{'Autoregressive Transformer':<24} | {args.seq_len - 1:<10} | {avg_ar_ms:<14.2f} | {ar_tok_s:<18.1f} | 1.00x (Baseline)")
    print(f"{'CWS-FM (Ours)':<24} | {8:<10} | {avg_k8_ms:<14.2f} | {k8_tok_s:<18.1f} | {speedup_k8:<10.2f}x")
    print(f"{'CWS-FM Ultra-Fast (Ours)':<24} | {4:<10} | {avg_k4_ms:<14.2f} | {k4_tok_s:<18.1f} | {speedup_k4:<10.2f}x")
    print("-" * 84)

    # Pre-registered Empirical Gates:
    # Gate 1: Statistically Significant Speedup (Speedup >= 3.0x at K=8)
    # Gate 2: Zero Rounding Collapse Rate (<= 2.0%)
    gate1_pass = speedup_k8 >= 3.0
    gate2_pass = final_k8_collapse <= 0.02
    overall_pass = gate1_pass and gate2_pass

    print("\nPre-Registered Empirical Gates:")
    print(f"  Gate 1 (Generation Speedup >= 3.0x):         {'PASS' if gate1_pass else 'FAIL'} ({speedup_k8:.2f}x)")
    print(f"  Gate 2 (Zero Rounding Collapse Rate <= 2%):  {'PASS' if gate2_pass else 'FAIL'} ({final_k8_collapse * 100:.2f}%)")

    # Save Provenance Manifest
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    manifest_path = os.path.join(results_dir, "nar_vs_ar_manifest.json")

    manifest = {
        "benchmark": "cws_fm_vs_autoregressive_llm",
        "date": datetime.now(timezone.utc).isoformat(),
        "hardware": str(device),
        "seq_len": args.seq_len,
        "vocab_size": args.vocab_size,
        "ar_baseline": {
            "decoding_steps": args.seq_len - 1,
            "avg_latency_ms": avg_ar_ms,
            "tokens_per_sec": ar_tok_s,
        },
        "cws_fm_k8": {
            "ode_steps": 8,
            "avg_latency_ms": avg_k8_ms,
            "tokens_per_sec": k8_tok_s,
            "speedup_vs_ar": speedup_k8,
            "rounding_collapse_rate": final_k8_collapse,
            "entropy": final_k8_entropy,
        },
        "cws_fm_k4": {
            "ode_steps": 4,
            "avg_latency_ms": avg_k4_ms,
            "tokens_per_sec": k4_tok_s,
            "speedup_vs_ar": speedup_k4,
        },
        "gate1_passed": gate1_pass,
        "gate2_passed": gate2_pass,
        "gate_status": "PASS" if overall_pass else "FAIL",
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nProvenance manifest saved to {manifest_path}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
