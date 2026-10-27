"""Tests for sample-quality metrics."""

import torch
from cws_fm.metrics import compute_palindrome_validity, compute_distinct_ratio


def test_palindrome_validity():
    good = torch.tensor([[1, 2, 2, 1], [3, 4, 4, 3]])
    bad = torch.tensor([[1, 2, 3, 4]])
    assert compute_palindrome_validity(good) == 1.0
    assert compute_palindrome_validity(bad) == 0.0
    assert abs(compute_palindrome_validity(torch.cat([good, bad])) - 2 / 3) < 1e-6


def test_distinct_ratio():
    x = torch.tensor([[1, 2], [1, 2], [3, 4], [5, 6]])
    assert compute_distinct_ratio(x) == 0.75


def _arm(bpc, h1, ms):
    return {"judge_bpc": bpc, "unigram_H": h1, "latency_ms_b1": ms}


def test_text8_gate():
    from cws_fm.metrics import text8_gate
    ar = {"real_text": {"unigram_H": 4.0}, "nfe256": _arm(1.50, 4.0, 400.0)}
    mdlm = {"nfe8": _arm(1.80, 4.0, 30.0), "nfe32": _arm(1.62, 4.0, 110.0)}
    good = {"nfe8": _arm(1.90, 4.0, 30.0), "nfe32": _arm(1.60, 4.0, 110.0)}  # nfe32 meets every rule
    assert text8_gate(ar, mdlm, {"fm": good})["passed"]
    slow = {"nfe32": _arm(1.60, 4.0, 250.0)}  # > 0.5x AR latency
    collapsed = {"nfe32": _arm(1.40, 3.5, 110.0)}  # low bpc from a collapsed unigram distribution
    worse_than_mdlm = {"nfe32": _arm(1.64, 4.0, 110.0)}
    for fm in (slow, collapsed, worse_than_mdlm):
        assert not text8_gate(ar, mdlm, {"fm": fm})["passed"]
