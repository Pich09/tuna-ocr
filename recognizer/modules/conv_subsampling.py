"""Conv2d frontend that turns a fixed-width chunk image into a short frame
sequence, ready for the Conformer stack."""
import torch
import torch.nn as nn


class Conv2dSubsampling(nn.Module):
    """Input (N, 1, img_height, chunk_width) -> (N, T', d_model).
    Two stride-2 Conv2d(kernel=3, padding=1) + ReLU layers reduce both height
    and width by ~4x; height (a single text line has no independent second
    time axis) is collapsed into the feature dimension instead of kept as a
    second sequence axis."""

    def __init__(self, img_height: int, d_model: int, conv_channels: int = 128, in_channels: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        reduced_h = ((img_height + 1) // 2 + 1) // 2
        self.out_proj = nn.Linear(conv_channels * reduced_h, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)                       # (N, C, H', W')
        n, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2).reshape(n, w, c * h)  # (N, W', C*H')
        return self.out_proj(x)                # (N, W'=T', d_model)
