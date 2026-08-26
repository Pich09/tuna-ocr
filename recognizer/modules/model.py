"""Recognizer: ConformerEncoder + a CTC head (auxiliary monotonic-alignment
loss, over the full non-windowed encoder output) + BlockwiseARDecoder."""
import torch.nn as nn

from .decoder import BlockwiseARDecoder
from .encoder import ConformerEncoder


class Recognizer(nn.Module):
    def __init__(self, cfg, vocab_size: int, bos_id: int, pad_id: int, ctc_vocab_size: int = None,
                 eos_id: int = None, eob_id: int = None):
        super().__init__()
        self.encoder = ConformerEncoder(cfg)
        # `ctc_vocab_size` is the CTC head's OWN label-set size (blank
        # included). It is deliberately decoupled from the decoder's subword
        # vocab: CTC uses a small character label set so the encoder can
        # actually learn glyph->label mappings, while the decoder keeps the
        # shared subword tokenizer (see data/char_vocab.py). When omitted,
        # falls back to the legacy subword-CTC layout (vocab_size + 1 for
        # blank) so older checkpoints still load.
        self.ctc_vocab_size = ctc_vocab_size if ctc_vocab_size is not None else vocab_size + 1
        self.ctc_head = nn.Linear(cfg.d_model, self.ctc_vocab_size)
        # eos_id/eob_id are parameter-free (decode-time behavior only), so
        # checkpoints from before they existed still load -- see decoder.py.
        self.decoder = BlockwiseARDecoder(cfg, vocab_size, bos_id, pad_id, eos_id=eos_id, eob_id=eob_id)

    def forward(self, chunk_batch, chunks_per_line, ar_targets, mode: str = "blockwise",
               compute_ctc: bool = True, compute_ar: bool = True):
        """mode="blockwise" (default): ar_targets is (B, num_blocks, K), uses
        the parallel per-block decoder. mode="sequential": ar_targets is the
        flat (B, L) token sequence (data/dataset.py's ar_flat_target), uses
        plain one-token-at-a-time teacher forcing with unrestricted
        cross-attention -- see modules/decoder.py's "Two-stage training" note
        for why/when to use this mode.

        compute_ctc/compute_ar: set either False to skip that head's forward
        pass entirely (returns None for its logits) instead of just zero-
        weighting its loss -- for the CTC-only/AR-only ablation notebooks,
        which need the OTHER head to consume zero compute, not merely
        contribute zero gradient. See recognizer/train.py's compute_loss for
        how None logits are handled."""
        enc_out, enc_lengths, enc_block_ids = self.encoder(chunk_batch, chunks_per_line)
        ctc_logits = self.ctc_head(enc_out) if compute_ctc else None
        ar_logits = None
        if compute_ar:
            if mode == "sequential":
                ar_logits = self.decoder.forward_sequential(enc_out, enc_lengths, ar_targets)
            else:
                ar_logits = self.decoder(enc_out, enc_lengths, enc_block_ids, ar_targets)
        return ctc_logits, ar_logits, enc_lengths

    def encode(self, chunk_batch, chunks_per_line):
        """Encoder-only pass, for inference (decode_greedy needs enc_out/
        enc_lengths/enc_block_ids but not ar_targets)."""
        return self.encoder(chunk_batch, chunks_per_line)
