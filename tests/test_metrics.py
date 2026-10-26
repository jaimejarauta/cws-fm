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
