import torch

__all__ = ["EPS"]


EPS = torch.finfo().eps
"""Bilangan terkecil yang dapat direpresentasikan sehingga `1.0 + eps != 1.0`."""
