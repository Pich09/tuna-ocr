"""ConformerEncoder: runs the conv-subsampling + Conformer stack per chunk
(all chunks in a batch flattened into one dense pass, since chunks are
fixed-width), then stitches per-chunk outputs back into one per-line
sequence, trimming each non-first chunk's overlap-derived leading frames so
the stitched sequence has no duplicated content. Chunk index == block index
by construction -- this is also what the blockwise-AR decoder attends to.
"""
import math

import torch
import torch.nn as nn

from .conformer_block import ConformerBlock
from .conv_subsampling import Conv2dSubsampling
from .positional import SinusoidalPositionalEncoding


class ConformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.subsampling = Conv2dSubsampling(cfg.img_height, cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(cfg.d_model, cfg.encoder_attn_heads, cfg.encoder_conv_kernel,
                            cfg.encoder_ff_expansion, cfg.encoder_dropout)
            for _ in range(cfg.num_encoder_layers)
        ])
        # frames trimmed from the start of every non-first chunk, matching
        # the same ~4x reduction the conv frontend applies to chunk_overlap
        self.overlap_frames = max(0, math.ceil(cfg.chunk_overlap / 4))

    def forward(self, chunk_batch: torch.Tensor, chunks_per_line: torch.Tensor):
        """chunk_batch: (N, 1, img_height, chunk_width) -- all chunks from
        all lines in the batch, flattened in line order.
        chunks_per_line: (B,) how many of those N chunks belong to each line.
        Returns enc_out (B, T'_max, D), enc_lengths (B,), enc_block_ids (B, T'_max).
        """
        x = self.subsampling(chunk_batch)          # (N, F, D)
        x = x + self.pos_enc(x.shape[1]).unsqueeze(0)
        for block in self.blocks:
            x = block(x)

        device = x.device
        d_model = x.shape[-1]
        lines, block_ids_list = [], []
        offset = 0
        for n_chunks in chunks_per_line.tolist():
            line_chunks = x[offset:offset + n_chunks]     # (n_chunks, F, D)
            offset += n_chunks
            trimmed = [line_chunks[0]]
            for c in range(1, n_chunks):
                trimmed.append(line_chunks[c, self.overlap_frames:])
            line_seq = torch.cat(trimmed, dim=0)           # (T_line, D)
            block_ids = torch.cat([
                torch.full((t.shape[0],), c, dtype=torch.long, device=device)
                for c, t in enumerate(trimmed)
            ])
            lines.append(line_seq)
            block_ids_list.append(block_ids)

        enc_lengths = torch.tensor([seq.shape[0] for seq in lines], dtype=torch.long, device=device)
        t_max = int(enc_lengths.max())
        b = len(lines)
        enc_out = torch.zeros(b, t_max, d_model, device=device, dtype=x.dtype)
        enc_block_ids = torch.zeros(b, t_max, dtype=torch.long, device=device)
        for i, (seq, bids) in enumerate(zip(lines, block_ids_list)):
            enc_out[i, :seq.shape[0]] = seq
            enc_block_ids[i, :bids.shape[0]] = bids
        return enc_out, enc_lengths, enc_block_ids
