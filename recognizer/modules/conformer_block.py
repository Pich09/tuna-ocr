"""Macaron-style Conformer block: FF -> self-attn -> conv module -> FF."""
import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    """Depthwise Conv1d + GLU, 'same' padding -- no causal restriction needed
    since a chunk's frames are all visible to each other (this is the
    Conformer encoder, not the AR decoder)."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x).transpose(1, 2)          # (B, D, T)
        x = self.pointwise1(x)
        x = nn.functional.glu(x, dim=1)
        x = self.depthwise(x)
        x = self.bn(x)
        x = nn.functional.silu(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)                # (B, T, D)


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, conv_kernel: int, ff_expansion: int, dropout: float):
        super().__init__()
        self.ff1 = FeedForward(d_model, ff_expansion, dropout)
        self.ln_attn = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.conv_module = ConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForward(d_model, ff_expansion, dropout)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        attn_in = self.ln_attn(x)
        x = x + self.self_attn(attn_in, attn_in, attn_in)
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)
