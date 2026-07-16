"""Thin multi-head attention wrapper built on
`torch.nn.functional.scaled_dot_product_attention` so callers can pass an
arbitrary per-batch boolean mask (needed for the decoder's block-restricted
cross-attention) -- `nn.MultiheadAttention`'s mask reshaping doesn't cleanly
support a mask that varies per batch element."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,Dh)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask=None) -> torch.Tensor:
        """attn_mask: bool, broadcastable to (B, H, Tq, Tk), True = attend."""
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)
        return self.out_proj(out)
