"""BlockwiseARDecoder: predicts a whole block of tokens per decoding step
(Blockwise Parallel Decoding, Stern et al. 2018, adapted for OCR) instead of
one token at a time. Autoregression happens BETWEEN blocks; within a block,
all `max_tokens_per_block` (K) token slots are produced from a single shared
hidden state via K parallel output heads, so no per-token loop is needed to
fill a block.

Concretely: the decoder runs an ordinary causal self-attention stack over
the (teacher-forced, shifted-right) flattened target sequence, with
cross-attention restricted so a position in block b only sees encoder
frames from blocks 0..b (never future blocks -- see ConformerEncoder's
1:1 chunk/block correspondence). The hidden state at the FIRST position of
block b (i.e. right after consuming the last real token of block b-1) is
computed purely from already-committed history, so applying the K-way
`block_head` to it yields all of block b's token logits in one shot,
without needing block b's own tokens as input. This is what makes it safe
to predict a whole block per step: a slot's prediction never depends on
another slot's true identity from the same block (no leakage), and at
inference no chicken-and-egg problem arises (block b's prefix only needs
block b-1's already-decided tokens).

Known v1 simplification: inference recomputes the whole prefix on every
block step (no KV-cache) -- fine for short OCR lines, noted as a future
optimization in recognizer/README.md.

Two-stage training (forward_sequential / decode_greedy_sequential): the
blockwise supervision above depends on ar_target's uniform-interval split
of each transcript into per-block token groups -- an approximation, since
none of these datasets provide real per-character alignment, and a group's
boundary frequently does not match where those characters actually appear
in that block's image region. Training blockwise from scratch teaches the
decoder to reproduce that approximate partitioning rather than real
image-to-text correspondence (observed in practice: AR val CER stuck well
above CTC's on the same encoder, despite near-zero AR train loss -- a
sign of fitting the wrong target, not slow learning).

forward_sequential instead runs plain, fully-sequential teacher forcing
over the TRUE (unsegmented) token sequence, with cross-attention over the
WHOLE line's encoder output (no per-block window) -- exactly like a
standard transformer decoder, so cross-attention is free to discover the
real alignment on its own (the same way CTC does). The recommended
training curriculum is to spend an initial fraction of steps in this mode
(see TrainConfig.sequential_ar_steps) so token_emb/pos_enc/layers/ln_final
-- all shared with blockwise mode -- learn genuine image-to-text
correspondence first, before switching to blockwise's parallel per-block
K-way head for the remainder of the run (for inference-time speed). Only
block_head vs token_head differ per mode; the shared trunk is what
benefits from this staging.
"""
import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .conformer_block import FeedForward
from .positional import SinusoidalPositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln_self = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln_cross = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, ff_expansion, dropout)

    def forward(self, x, enc_out, self_mask, cross_mask):
        h = self.ln_self(x)
        x = x + self.self_attn(h, h, h, attn_mask=self_mask)
        h = self.ln_cross(x)
        x = x + self.cross_attn(h, enc_out, enc_out, attn_mask=cross_mask)
        x = x + self.ff(x)
        return x


class BlockwiseARDecoder(nn.Module):
    def __init__(self, cfg, vocab_size: int, bos_id: int, pad_id: int):
        super().__init__()
        self.d_model = cfg.d_model
        self.K = cfg.max_tokens_per_block
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, cfg.d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(cfg.d_model, cfg.decoder_attn_heads, cfg.encoder_ff_expansion, cfg.encoder_dropout)
            for _ in range(cfg.num_decoder_layers)
        ])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.block_head = nn.Linear(cfg.d_model, self.K * vocab_size)
        # Separate per-position head for the sequential-AR training stage (see
        # module docstring's "Two-stage training") -- block_head predicts all K
        # slots of a block from ONE pre-block hidden state, which is a different
        # projection than an ordinary per-position next-token head, so the two
        # aren't shared, only token_emb/pos_enc/layers/ln_final are.
        self.token_head = nn.Linear(cfg.d_model, vocab_size)

    def _shift_right(self, flat_ids: torch.Tensor) -> torch.Tensor:
        b = flat_ids.shape[0]
        bos_col = torch.full((b, 1), self.bos_id, dtype=torch.long, device=flat_ids.device)
        return torch.cat([bos_col, flat_ids[:, :-1]], dim=1)

    def _masks(self, num_positions: int, enc_block_ids: torch.Tensor, enc_len_mask: torch.Tensor):
        device = enc_block_ids.device
        causal = torch.tril(torch.ones(num_positions, num_positions, dtype=torch.bool, device=device))
        self_mask = causal.unsqueeze(0).unsqueeze(0)  # (1,1,N,N), broadcasts over (B,H)

        block_of_pos = torch.arange(num_positions, device=device) // self.K  # (N,)
        allow = enc_block_ids.unsqueeze(1) <= block_of_pos.view(1, num_positions, 1)  # (B,N,T')
        allow = allow & enc_len_mask.unsqueeze(1)
        cross_mask = allow.unsqueeze(1)  # (B,1,N,T'), broadcasts over H
        return self_mask, cross_mask

    def _run_stack(self, flat_ids: torch.Tensor, enc_out: torch.Tensor, enc_lengths: torch.Tensor,
                   enc_block_ids: torch.Tensor) -> torch.Tensor:
        dec_in = self._shift_right(flat_ids)
        x = self.token_emb(dec_in) + self.pos_enc(dec_in.shape[1]).unsqueeze(0)
        t_enc = enc_out.shape[1]
        enc_len_mask = torch.arange(t_enc, device=enc_out.device).unsqueeze(0) < enc_lengths.unsqueeze(1)
        self_mask, cross_mask = self._masks(flat_ids.shape[1], enc_block_ids, enc_len_mask)
        for layer in self.layers:
            x = layer(x, enc_out, self_mask, cross_mask)
        return self.ln_final(x)

    def forward(self, enc_out: torch.Tensor, enc_lengths: torch.Tensor, enc_block_ids: torch.Tensor,
                ar_targets: torch.Tensor) -> torch.Tensor:
        """ar_targets: (B, num_blocks, K) teacher-forced targets.
        Returns logits: (B, num_blocks, K, vocab_size)."""
        b, num_blocks, k = ar_targets.shape
        assert k == self.K, f"ar_targets block width {k} != configured max_tokens_per_block {self.K}"
        flat = ar_targets.reshape(b, num_blocks * k)
        x = self._run_stack(flat, enc_out, enc_lengths, enc_block_ids)
        idx = torch.arange(0, num_blocks * k, k, device=x.device)
        h_prefix = x[:, idx, :]  # (B, num_blocks, D) -- hidden state right before each block starts
        return self.block_head(h_prefix).view(b, num_blocks, k, self.vocab_size)

    def forward_sequential(self, enc_out: torch.Tensor, enc_lengths: torch.Tensor,
                            seq_targets: torch.Tensor) -> torch.Tensor:
        """seq_targets: (B, L) teacher-forced flat token sequence (already
        includes a trailing <eos>, see data/dataset.py's ar_flat_target).
        Cross-attention sees the FULL line's encoder output -- no per-block
        window -- so alignment is discovered by attention, not assumed from a
        uniform split. Returns logits: (B, L, vocab_size)."""
        dec_in = self._shift_right(seq_targets)
        x = self.token_emb(dec_in) + self.pos_enc(dec_in.shape[1]).unsqueeze(0)
        l_dec = dec_in.shape[1]
        t_enc = enc_out.shape[1]
        enc_len_mask = torch.arange(t_enc, device=enc_out.device).unsqueeze(0) < enc_lengths.unsqueeze(1)
        causal = torch.tril(torch.ones(l_dec, l_dec, dtype=torch.bool, device=enc_out.device))
        self_mask = causal.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
        cross_mask = enc_len_mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,T') -- full visibility, padding excluded
        for layer in self.layers:
            x = layer(x, enc_out, self_mask, cross_mask)
        x = self.ln_final(x)
        return self.token_head(x)

    @torch.no_grad()
    def decode_greedy_sequential(self, enc_out: torch.Tensor, enc_lengths: torch.Tensor,
                                  max_len: int = 256) -> torch.Tensor:
        """Ordinary one-token-at-a-time greedy decoding (recomputes the whole
        prefix each step -- same known v1 no-KV-cache simplification as
        decode_greedy). Returns (B, <=max_len) token ids, NOT including the
        leading <bos>; caller truncates each sample at its first <eos>/<pad>."""
        b = enc_out.shape[0]
        device = enc_out.device
        t_enc = enc_out.shape[1]
        enc_len_mask = torch.arange(t_enc, device=device).unsqueeze(0) < enc_lengths.unsqueeze(1)
        cross_mask = enc_len_mask.unsqueeze(1).unsqueeze(1)
        generated = torch.full((b, 1), self.bos_id, dtype=torch.long, device=device)
        for _ in range(max_len):
            l_dec = generated.shape[1]
            x = self.token_emb(generated) + self.pos_enc(l_dec).unsqueeze(0)
            causal = torch.tril(torch.ones(l_dec, l_dec, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)
            for layer in self.layers:
                x = layer(x, enc_out, causal, cross_mask)
            x = self.ln_final(x)
            next_logits = self.token_head(x[:, -1, :])
            next_tok = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
        return generated[:, 1:]  # drop the leading <bos>

    @torch.no_grad()
    def decode_greedy(self, enc_out: torch.Tensor, enc_lengths: torch.Tensor, enc_block_ids: torch.Tensor,
                       max_blocks: int) -> torch.Tensor:
        """Greedy block-by-block decoding: predicts one block's K tokens per
        outer step (single parallel forward, no per-token loop), commits
        them, then advances. Returns (B, max_blocks*K) token ids -- caller
        truncates each sample at its first <eob>/<pad>/<eos>."""
        b = enc_out.shape[0]
        device = enc_out.device
        committed = torch.zeros((b, 0), dtype=torch.long, device=device)
        blocks_out = []
        for blk in range(max_blocks):
            dummy_current = torch.full((b, self.K), self.pad_id, dtype=torch.long, device=device)
            flat = torch.cat([committed, dummy_current], dim=1)
            x = self._run_stack(flat, enc_out, enc_lengths, enc_block_ids)
            h = x[:, blk * self.K, :]
            block_logits = self.block_head(h).view(b, self.K, self.vocab_size)
            block_tokens = block_logits.argmax(dim=-1)  # (B,K)
            committed = torch.cat([committed, block_tokens], dim=1)
            blocks_out.append(block_tokens)
        return torch.cat(blocks_out, dim=1)
