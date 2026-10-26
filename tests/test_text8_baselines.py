"""Tests for the text8 pipeline, baselines and sample metrics."""

import math

import torch

from cws_fm.text8 import encode, decode, VOCAB, sample_batch
from cws_fm.baselines import TokenTransformer, mdlm_loss, mdlm_sample, ar_loss, ar_sample, judge_bits_per_char
from cws_fm.metrics import ngram_entropy, distinct_ngrams, word_validity


def test_encode_decode_roundtrip():
    s = "the quick brown fox"
    assert decode(encode(s)) == s
    assert len(VOCAB) == 27


def test_sample_batch_shapes():
    data = torch.arange(1000) % 27
    x = sample_batch(data, batch=4, seq_len=16, generator=torch.Generator().manual_seed(0))
    assert x.shape == (4, 16) and x.dtype == torch.long


def test_mdlm_sample_fills_every_position():
    torch.manual_seed(0)
    m = TokenTransformer(vocab=28, d_model=32, layers=1, heads=2, max_len=16, causal=False)
    x = mdlm_sample(m, batch=3, seq_len=16, steps=4, mask_id=27)
    assert x.shape == (3, 16) and (x < 27).all()
    loss = mdlm_loss(m, torch.randint(0, 27, (3, 16)), mask_id=27)
    assert torch.isfinite(loss)


def test_ar_sample_and_judge():
    torch.manual_seed(0)
    m = TokenTransformer(vocab=27, d_model=32, layers=1, heads=2, max_len=16, causal=True)
    x = ar_sample(m, batch=2, seq_len=16)
    assert x.shape == (2, 16)
    assert torch.isfinite(ar_loss(m, x))
    with torch.no_grad():
        m.head.weight.zero_()
        m.head.bias.zero_()
    assert abs(judge_bits_per_char(m, x) - math.log2(27)) < 1e-4  # uniform judge


def test_sample_metrics():
    x = torch.tensor([encode("ab ab ab")])
    assert abs(ngram_entropy(x, 1) - ngram_entropy(torch.tensor([encode("ba ba ba")]), 1)) < 1e-9
    assert distinct_ngrams(x, 2) == 3 / 7  # 'ab','b ',' a' distinct among 7 bigrams
    assert word_validity(x, {"ab"}) == 1.0
    assert word_validity(torch.tensor([encode("ab zz")]), {"ab"}) == 0.5
