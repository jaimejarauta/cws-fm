"""text8 character data (Mahoney, public): 27 symbols, standard 90M / 5M / 5M split."""

import io
import os
import urllib.request
import zipfile

import torch

VOCAB = "abcdefghijklmnopqrstuvwxyz "
_IDX = {c: i for i, c in enumerate(VOCAB)}
URL = "http://mattmahoney.net/dc/text8.zip"


def encode(s: str) -> list:
    return [_IDX[c] for c in s]


def decode(ids) -> str:
    return "".join(VOCAB[int(i)] for i in ids)


def load_text8(cache_dir: str) -> dict:
    """Train / val / test uint8 tensors; downloads once to cache_dir."""
    path = os.path.join(cache_dir, "text8.pt")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        raw = urllib.request.urlopen(URL).read()
        text = zipfile.ZipFile(io.BytesIO(raw)).read("text8").decode()
        data = torch.tensor(encode(text), dtype=torch.uint8)
        torch.save({"train": data[:90_000_000], "val": data[90_000_000:95_000_000], "test": data[95_000_000:]}, path)
    return torch.load(path)


def sample_batch(data: torch.Tensor, batch: int, seq_len: int, generator=None) -> torch.Tensor:
    starts = torch.randint(0, data.numel() - seq_len, (batch,), generator=generator)
    return torch.stack([data[s:s + seq_len] for s in starts]).long()
