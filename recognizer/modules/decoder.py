"""BlockwiseARDecoder: predicts a whole block of tokens per decoding step
(Blockwise Parallel Decoding, Stern et al. 2018, adapted for OCR) instead of
one token at a time. Autoregression happens BETWEEN blocks; within a block,
all `max_tokens_per_block` (K) token slots are produced from a single shared
hidden state via K parallel output heads, so no per-token loop is needed to
fill a block.

Concretely: the decoder runs an ordinary causal self-attention stack over
the (teacher-forced, shifted-right) flattened target sequence, with
cross-attention over the WHOLE line's encoder output (all chunks, not
windowed to blocks 0..b -- see "Cross-attention window" below for why).
The hidden state at the FIRST position of block b (i.e. right after
consuming the last real token of block b-1) is computed purely from
already-committed history, so applying the K-way `block_head` to it
yields all of block b's token logits in one shot, without needing block
b's own tokens as input. This is what makes it safe to predict a whole
block per step: a slot's prediction never depends on another slot's true
identity from the same block (no leakage), and at inference no
chicken-and-egg problem arises (block b's prefix only needs block b-1's
already-decided tokens). Note this safety property is about SELF-attention
over decoder tokens (governed by the causal mask over already-committed
history); it has nothing to do with how much of the encoder's output
cross-attention is allowed to see, which is a separate, freely choosable
design knob -- see below.

Known v1 simplification: inference recomputes the whole prefix on every
block step (no KV-cache) -- fine for short OCR lines, noted as a future
optimization in recognizer/README.md.

Two-stage training (forward_sequential / decode_greedy_sequential): the
blockwise supervision above depends on ar_target's uniform-interval split
of each transcript into per-block token groups -- an approximation, since
none of these datasets provide real per-character alignment. Critically,
`data/dataset.py`'s `split_into_blocks` divides the transcript by raw
TOKEN COUNT (`n // num_blocks`), with no awareness of pixel width -- since
glyph width varies a lot (a run of narrow Latin characters vs a few wide
Khmer clusters), a block's assigned token slice frequently does NOT match
which characters are actually rendered in that block's image chunk, not
just near a boundary but potentially by a chunk or more. Training blockwise
from scratch teaches the decoder to reproduce that approximate partitioning
rather than real image-to-text correspondence (observed in practice: AR
val CER stuck well above CTC's on the same encoder, despite non-trivial AR
train loss -- a sign of fitting a target that's sometimes simply wrong,
not slow learning).

Cross-attention window (FIXED): block b's cross-attention used to be
restricted to encoder blocks 0..b (never later chunks), on the reasoning
that a block "shouldn't need to look ahead." Combined with the
token-count-uniform target mismatch above, this restriction actively made
things worse rather than just harder: when a block's assigned characters
are actually rendered in a LATER chunk than the string-split assigned them
to (a real, frequent occurrence, not an edge case), the old window made
correct prediction structurally impossible for that block -- the evidence
existed in the image but the mask hid it. Unlike the self-attention causal
mask (a genuine inference-time constraint -- future tokens haven't been
generated yet), there is no such constraint on the encoder side: the whole
image is encoded once, upfront, before any AR decoding happens, so nothing
is unsafe about letting a block see chunks after its own. Cross-attention
now sees the full line's encoder output in blockwise mode too, exactly
like `forward_sequential` already did -- removing one whole class of
otherwise-unrecoverable errors. This does NOT fix the token-count-split
mismatch itself (a block can still be trained against the wrong slice of
text), it only removes the *additional* failure of being unable to see
correct evidence even when it exists elsewhere in the image.

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


def truncate_at_eos(tokens: list, eos_id: int, pad_id: int) -> list:
    """Cuts a sequentially-decoded token list at its first <eos>/<pad>.
    decode_greedy_sequential always returns max_len positions (padded after
    each row's <eos>); scoring or displaying the raw row includes everything
    after the model said 'stop', which is how val_ar_cer values above 1.0
    (hypotheses longer than references) were produced."""
    out = []
    for t in tokens:
        if t == eos_id or t == pad_id:
            break
        out.append(t)
    return out


def truncate_blocks(tokens: list, k: int, eob_id: int, pad_id: int, num_blocks: int = None) -> list:
    """Assembles a blockwise-decoded token list into the real output: within
    each K-token block, keep tokens up to the first <eob>/<pad> (the rest of
    that block is padding, not content), then continue with the next block --
    <eob> ends a block, not the line, and uniform splitting legitimately
    produces empty middle blocks. `num_blocks` drops trailing blocks decoded
    past a line's real chunk count (batched decode runs every line to the
    batch max)."""
    out = []
    blocks = [tokens[i:i + k] for i in range(0, len(tokens), k)]
    if num_blocks is not None:
        blocks = blocks[:num_blocks]
    for blk in blocks:
        for t in blk:
            if t == eob_id or t == pad_id:
                break
            out.append(t)
    return out


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
    def __init__(self, cfg, vocab_size: int, bos_id: int, pad_id: int, eos_id: int = None, eob_id: int = None):
        super().__init__()
        self.d_model = cfg.d_model
        self.K = cfg.max_tokens_per_block
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.pad_id = pad_id
        # eos_id/eob_id carry no parameters, so checkpoints saved before they
        # existed still load. eos_id enables decode_greedy_sequential's early
        # stop; eob_id enables decode_greedy's post-<eob> commit masking. When
        # None, both decodes fall back to the old (correct but wasteful /
        # context-polluting) behavior.
        self.eos_id = eos_id
        self.eob_id = eob_id

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
        # enc_block_ids is unused here now -- kept in the signature since every
        # caller (forward, decode_greedy, _run_stack) already threads it through
        # from the encoder, and a future partial-window variant may want it
        # again. See the module docstring's "Cross-attention window (FIXED)"
        # section: cross-attention used to be restricted to encoder blocks 0..b
        # via enc_block_ids, which -- combined with split_into_blocks' token-
        # count-uniform (not pixel-width-aware) target split -- made correct
        # prediction structurally impossible whenever a block's assigned
        # characters were actually rendered in a later chunk. There is no
        # inference-time causality reason for that restriction (the whole
        # image is encoded upfront, unlike decoder tokens which are genuinely
        # generated in sequence), so cross-attention now sees the full line's
        # encoder output here too, exactly like forward_sequential already did.
        device = enc_block_ids.device
        causal = torch.tril(torch.ones(num_positions, num_positions, dtype=torch.bool, device=device))
        self_mask = causal.unsqueeze(0).unsqueeze(0)  # (1,1,N,N), broadcasts over (B,H)
        cross_mask = enc_len_mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,T') -- full visibility, padding excluded
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
                                  max_len: int = 256, repetition_penalty: float = 1.0,
                                  repetition_window: int = 8) -> torch.Tensor:
        """Ordinary one-token-at-a-time greedy decoding (recomputes the whole
        prefix each step -- same known v1 no-KV-cache simplification as
        decode_greedy). Returns (B, <=max_len) token ids, NOT including the
        leading <bos>; caller truncates each sample at its first <eos>/<pad>
        (see truncate_at_eos). Stops as soon as every row has emitted <eos>
        (when eos_id is configured): without this it always ran the full
        max_len=256 iterations -- measured as the dominant cost of the
        periodic eval -- and the post-<eos> positions are garbage anyway.
        Rows that finish early keep generating <pad> so the batch stays
        rectangular.

        `repetition_penalty` (CTRL-style: divides a positive logit, multiplies
        a negative one, so it always pushes the score down): 1.0 (default) is
        an exact no-op, preserving the original behavior for every existing
        caller -- notably recognizer/train.py's periodic eval, so past/live
        training runs' val_ar_cer curves and best.pt selection are unaffected
        by this. Only applied to token ids seen in the last
        `repetition_window` generated positions (not the whole history), so
        it targets the specific short-range loop/copy failure mode observed
        in eval (e.g. "GB+512 GB+512 GB+512") without discouraging legitimate
        longer-range reuse of common Khmer syllables/characters elsewhere in
        a line. Pass repetition_penalty > 1.0 (e.g. 1.3) to enable."""
        b = enc_out.shape[0]
        device = enc_out.device
        t_enc = enc_out.shape[1]
        enc_len_mask = torch.arange(t_enc, device=device).unsqueeze(0) < enc_lengths.unsqueeze(1)
        cross_mask = enc_len_mask.unsqueeze(1).unsqueeze(1)
        generated = torch.full((b, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_len):
            l_dec = generated.shape[1]
            x = self.token_emb(generated) + self.pos_enc(l_dec).unsqueeze(0)
            causal = torch.tril(torch.ones(l_dec, l_dec, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)
            for layer in self.layers:
                x = layer(x, enc_out, causal, cross_mask)
            x = self.ln_final(x)
            next_logits = self.token_head(x[:, -1, :])
            if repetition_penalty != 1.0:
                window_start = max(1, generated.shape[1] - repetition_window)
                recent = generated[:, window_start:]  # (B, w); never includes the leading <bos> at position 0
                recent_mask = torch.zeros_like(next_logits, dtype=torch.bool)
                recent_mask.scatter_(1, recent, True)
                positive = next_logits > 0
                next_logits = torch.where(recent_mask & positive, next_logits / repetition_penalty, next_logits)
                next_logits = torch.where(recent_mask & ~positive, next_logits * repetition_penalty, next_logits)
            next_tok = next_logits.argmax(dim=-1, keepdim=True)
            if self.eos_id is not None:
                next_tok = next_tok.masked_fill(finished.unsqueeze(1), self.pad_id)
                finished = finished | (next_tok.squeeze(1) == self.eos_id)
            generated = torch.cat([generated, next_tok], dim=1)
            if self.eos_id is not None and bool(finished.all()):
                break
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
            if self.eob_id is not None:
                # Slots after a block's first <eob>/<pad> were never supervised
                # (their teacher-forcing targets are <pad>, excluded by the CE's
                # ignore_index), so their argmax is noise. Committing that noise
                # hands later blocks a context that training never produced --
                # training always saw <pad> there. Mask to <pad> before
                # committing so inference context matches the training layout.
                # The <eob> itself IS supervised, so it stays.
                ended = (block_tokens == self.eob_id) | (block_tokens == self.pad_id)
                after_first_end = (ended.cumsum(dim=1) - ended.long()) > 0
                block_tokens = block_tokens.masked_fill(after_first_end, self.pad_id)
            committed = torch.cat([committed, block_tokens], dim=1)
            blocks_out.append(block_tokens)
        return torch.cat(blocks_out, dim=1)
