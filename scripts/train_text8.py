#!/usr/bin/env python3
"""text8 study (Colab run 1): Fisher-Rao flow matching vs masked diffusion vs AR at matched backbone size.
One arm per invocation, resumable. Run the AR arm first: it is also the judge for every other arm.
  --arm ar | fm | mdlm      --compare DIR [DIR ...] applies the gate to finished arms.
Plan and gate: research/COLAB_LONG_RUNS_2026-10-26.md section 1.
"""

import argparse
import json
import math
import os
import time
from collections import Counter

import torch
import torch.nn.functional as F

from cws_fm.baselines import TokenTransformer, ar_loss, ar_sample, judge_bits_per_char, mdlm_loss, mdlm_sample
from cws_fm.flow import flow_matching_loss, geodesic_sphere_path, sample_cws_fm
from cws_fm.metrics import compute_rounding_collapse_rate, distinct_ngrams, ngram_entropy, word_validity
from cws_fm.model import SimplexVelocityField
from cws_fm.runtime import Run
from cws_fm.text8 import decode, load_text8, sample_batch

V = 27
MASK_ID = 27


def build(arm: str, a) -> torch.nn.Module:
    if arm == "fm":
        return SimplexVelocityField(V, d_model=a.d_model, num_layers=a.layers, num_heads=a.heads,
                                    max_seq_len=a.seq_len, alpha=a.alpha)
    return TokenTransformer(V + (arm == "mdlm"), a.d_model, a.layers, a.heads, a.seq_len, causal=(arm == "ar"))


def loss_fn(arm: str, model, x, alpha):
    if arm == "fm":
        return flow_matching_loss(model, x, V, alpha=alpha)
    return ar_loss(model, x) if arm == "ar" else mdlm_loss(model, x, MASK_ID)


def sample(arm, model, a, n, nfe):
    out = []
    for i in range(0, n, a.sample_batch):
        b = min(a.sample_batch, n - i)
        if arm == "fm":
            toks, probs = sample_cws_fm(model, a.seq_len, V, num_steps=nfe, alpha=a.alpha, batch_size=b,
                                        device=a.device, heun=a.heun)
            out.append((toks.cpu(), compute_rounding_collapse_rate(probs, 0.85)))
        elif arm == "mdlm":
            out.append((mdlm_sample(model, b, a.seq_len, nfe, MASK_ID).cpu(), 0.0))
        else:
            out.append((ar_sample(model, b, a.seq_len).cpu(), 0.0))
    return torch.cat([t for t, _ in out]), sum(c for _, c in out) / len(out)


@torch.no_grad()
def posterior_ce(model, data, a, taus=(0.25, 0.5, 0.75)):
    """Held-out CE of the clean-token posterior at fixed phase tau (FM arm only)."""
    g = torch.Generator().manual_seed(1)
    x = sample_batch(data, 256, a.seq_len, g).to(a.device)
    res = {}
    for tau in taus:
        u1 = F.one_hot(x, V).float()
        u0 = torch.sqrt(torch.distributions.Dirichlet(torch.ones(V, device=a.device)).sample(x.shape))
        tau_t = torch.full(x.shape, tau, device=a.device)
        logits, _ = model.logits(geodesic_sphere_path(u0, u1, tau_t.unsqueeze(-1)), tau_t)
        res[f"tau{tau}"] = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1)).item()
    return res


def evaluate(arm, model, a, data, judge):
    train_words = set(w for w, _ in Counter(decode(data["train"][:20_000_000]).split()).most_common(200_000))
    real = sample_batch(data["test"], a.eval_samples, a.seq_len, torch.Generator().manual_seed(2))
    ref = {"unigram_H": ngram_entropy(real, 1), "bigram_H": ngram_entropy(real, 2),
           "distinct4": distinct_ngrams(real, 4), "word_valid": word_validity(real, train_words)}
    if judge is not None:
        ref["judge_bpc"] = judge_bits_per_char(judge, real)
    results = {"real_text": ref}
    nfes = [a.seq_len] if arm == "ar" else a.nfe
    model.eval()
    for nfe in nfes:
        t0 = time.time()
        toks, collapse = sample(arm, model, a, a.eval_samples, nfe)
        r = {"sec_per_sample": (time.time() - t0) / a.eval_samples, "collapse_rate": collapse,
             "unigram_H": ngram_entropy(toks, 1), "bigram_H": ngram_entropy(toks, 2),
             "distinct4": distinct_ngrams(toks, 4), "word_valid": word_validity(toks, train_words),
             "example": decode(toks[0][:120])}
        if judge is not None:
            r["judge_bpc"] = judge_bits_per_char(judge, toks)
        results[f"nfe{nfe}"] = r
        print(f"[eval] nfe={nfe} " + " ".join(f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float)))
        print(f"       sample: {r['example']!r}")
    if arm == "fm":
        results["posterior_ce"] = posterior_ce(model, data["test"], a)
    if arm == "ar":
        results["test_bpc"] = judge_bits_per_char(model, sample_batch(data["test"], 2048, a.seq_len,
                                                                       torch.Generator().manual_seed(3)))
    return results


def load_judge(path, device):
    state = torch.load(os.path.join(path, "checkpoint.pt"), map_location="cpu", weights_only=False)
    c = state["model_cfg"]
    judge = TokenTransformer(V, c["d_model"], c["layers"], c["heads"], c["seq_len"], causal=True)
    judge.load_state_dict(state["model"])
    return judge.to(device).eval()


def compare(dirs, gate_nfe: int = 32):
    """Gate (2026-10-26): best FM within 10% of MDLM judge bpc at NFE 32 with unigram H within 2% of data;
    wavefront kept only if alpha=0.5 beats alpha=0 at >= 2 of the NFEs; KILL if FM > 1.25x MDLM at every NFE."""
    runs = {}
    for d in dirs:
        m = json.load(open(os.path.join(d, "manifest.json")))
        c = m["config"]
        runs[f"{c['arm']}" + (f"-a{c['alpha']}" if c["arm"] == "fm" else "")] = m["results"]
    mdlm = runs.get("mdlm")
    fms = {k: v for k, v in runs.items() if k.startswith("fm")}
    if not mdlm or not fms:
        raise SystemExit("need an mdlm run and at least one fm run")
    ref_h = mdlm["real_text"]["unigram_H"]
    nfes = [k for k in mdlm if k.startswith("nfe")]
    for name, r in fms.items():
        ratios = {n: r[n]["judge_bpc"] / mdlm[n]["judge_bpc"] for n in nfes if n in r}
        print(name, {n: round(v, 3) for n, v in ratios.items()})
    g = f"nfe{gate_nfe}"
    best = min(fms.values(), key=lambda r: r[g]["judge_bpc"])
    pass_ = (best[g]["judge_bpc"] <= 1.10 * mdlm[g]["judge_bpc"]
             and abs(best[g]["unigram_H"] - ref_h) <= 0.02 * ref_h)
    kill = all(r[n]["judge_bpc"] > 1.25 * mdlm[n]["judge_bpc"] for r in fms.values() for n in nfes if n in r)
    keep_wavefront = None
    if "fm-a0.0" in fms and "fm-a0.5" in fms:
        wins = sum(fms["fm-a0.5"][n]["judge_bpc"] < fms["fm-a0.0"][n]["judge_bpc"] for n in nfes)
        keep_wavefront = wins >= 2
    print(f"PASS={pass_} KILL={kill} keep_wavefront={keep_wavefront}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["ar", "fm", "mdlm"])
    p.add_argument("--compare", nargs="+", help="finished arm directories")
    p.add_argument("--gate-nfe", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--heun", action="store_true")
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--nfe", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    p.add_argument("--eval-samples", type=int, default=1000)
    p.add_argument("--sample-batch", type=int, default=250)
    p.add_argument("--judge", default=None, help="directory of the finished AR run")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget-hours", type=float, default=2.0)
    p.add_argument("--eval-only", action="store_true")
    a = p.parse_args()
    if a.compare:
        return compare(a.compare, a.gate_nfe)

    torch.manual_seed(a.seed)
    data = load_text8(a.data_dir)
    run = Run(a.out_dir, vars(a), budget_hours=a.budget_hours)
    run.data_digests["text8_train_sum"] = int(data["train"][:1_000_000].long().sum())
    model = build(a.arm, a).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))
    step, gen = 0, torch.Generator().manual_seed(a.seed)
    state = run.resume()
    if state:
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        step = state["step"]; gen.set_state(state["gen"])
    use_amp = a.device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    model_cfg = {"d_model": a.d_model, "layers": a.layers, "heads": a.heads, "seq_len": a.seq_len}
    save = lambda: run.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                             "gen": gen.get_state(), "model_cfg": model_cfg})

    print(f"[{a.arm}] params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, start step {step}")
    model.train()
    while step < a.steps and not a.eval_only:
        x = sample_batch(data["train"], a.batch, a.seq_len, gen).to(a.device)
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            loss = loss_fn(a.arm, model, x, a.alpha)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        step += 1
        if step % 500 == 0:
            print(f"[{a.arm}] step {step}/{a.steps} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e}")
        if step == 500:
            run.log_throughput(step, a.steps, "steps")
            run.check_budget(step, a.steps)
        if run.due():
            save()
    save()
    judge = load_judge(a.judge, a.device) if a.judge else (model if a.arm == "ar" else None)
    results = evaluate(a.arm, model, a, data, judge)
    run.write_manifest(results)


if __name__ == "__main__":
    main()
