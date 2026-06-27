from __future__ import annotations

import math
import typing as t

import torch
from torch import nn
from lightning import LightningModule

from transformer.params import SelfAttentionParams, MultiHeadSelfAttentionParams
from transformer.utils import constants

__all__ = ["MultiHeadSelfAttention", "SelfAttention"]


class MultiHeadSelfAttention(LightningModule):
    def __init__(self: t.Self, params: MultiHeadSelfAttentionParams) -> None:
        super().__init__()
        self.params: MultiHeadSelfAttentionParams = params
        self.model: nn.ModuleDict = nn.ModuleDict(
            {
                "heads": nn.ModuleList(
                    SelfAttention(self.params.attention_params)
                    for _ in range(self.params.num_heads)
                ),
                "proj": nn.Linear(self.params.model_dim, self.params.model_dim),
            }
        )

    def forward(
        self: t.Self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        masks: torch.LongTensor,
    ) -> torch.FloatTensor:
        # gabungkan output tiap attention head
        heads = torch.cat(
            [head(q, k, v, masks=masks) for head in self.model["heads"]], dim=-1
        )
        # shape: [batch, seq_len, value_dim * num_heads (= model_dim)]

        # proyeksikan ke matriks output
        return self.model["proj"](heads)
        # shape: [batch, seq_len, model_dim]


class SelfAttention(LightningModule):
    def __init__(self: t.Self, params: SelfAttentionParams) -> None:
        super().__init__()
        self.params: SelfAttentionParams = params
        self.model: nn.ModuleDict = nn.ModuleDict(
            {
                "query_proj": nn.Linear(
                    self.params.model_dim, self.params.key_dim, bias=False
                ),
                "key_proj": nn.Linear(
                    self.params.model_dim, self.params.key_dim, bias=False
                ),
                "value_proj": nn.Linear(
                    self.params.model_dim, self.params.value_dim, bias=False
                ),
            }
        )

    def forward(
        self: t.Self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        masks: torch.LongTensor,
    ) -> torch.FloatTensor:
        # proyeksikan input ke matriks bobot
        q = self.model["query_proj"](q)
        k = self.model["key_proj"](k)
        # shape: [batch, seq_len, key_dim]
        v = self.model["value_proj"](v)
        # shape: [batch, seq_len, value_dim]

        # hitung score = scaled dot-product (Q·Kᵀ / √dₖ)
        scores = q @ k.mT / math.sqrt(self.params.key_dim)
        # shape: [batch, seq_len, seq_len]

        # attention mask dari tokenizer buat ngabaikan padding (alias key padding mask)
        attn_mask = 1 - masks.unsqueeze(1) * masks.unsqueeze(-1)
        # shape: [batch, seq_len, seq_len]

        # look-ahead mask (segitiga atas) sebelum softmax biar token gak ngintip masa depan
        if self.params.mask:
            attn_mask |= torch.triu(torch.ones_like(scores, dtype=int), diagonal=1)
            # shape: [batch, seq_len, seq_len]

        # terapkan mask
        scores.masked_fill_(attn_mask.bool(), -constants.EPS)

        # hitung output (softmax · value)
        return nn.functional.softmax(scores, dim=-1) @ v
        # shape: [batch, seq_len, value_dim]
