import torch

__all__ = ["masked_mean"]


def masked_mean(x: torch.FloatTensor, *, masks: torch.LongTensor) -> torch.FloatTensor:
    """Hitung rata-rata embedding tiap urutan, dengan mengabaikan padding.

Parameter
----------
x:
    Tensor input.
    Shape: [batch, seq_len, model_dim]

masks:
    Tensor mask.
    Shape: [batch, seq_len]

Return
-------
Rata-rata ter-mask.
    Shape: [batch, model_dim]
"""
    masks = masks.unsqueeze(dim=-1)
    return (x * masks).sum(dim=1) / masks.sum(dim=1)
