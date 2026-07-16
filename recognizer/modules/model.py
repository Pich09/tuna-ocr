"""Recognizer: ConformerEncoder + a CTC head (auxiliary monotonic-alignment
loss, over the full non-windowed encoder output) + BlockwiseARDecoder."""
import torch.nn as nn

from .decoder import BlockwiseARDecoder
from .encoder import ConformerEncoder


class Recognizer(nn.Module):
    def __init__(self, cfg, vocab_size: int, bos_id: int, pad_id: int):
        super().__init__()
        self.encoder = ConformerEncoder(cfg)
        # +1 for the CTC blank symbol -- distinct from the decoder's <eob>
        # end-of-block control token; never conflate the two.
        self.ctc_head = nn.Linear(cfg.d_model, vocab_size + 1)
        self.decoder = BlockwiseARDecoder(cfg, vocab_size, bos_id, pad_id)

    def forward(self, chunk_batch, chunks_per_line, ar_targets):
        enc_out, enc_lengths, enc_block_ids = self.encoder(chunk_batch, chunks_per_line)
        ctc_logits = self.ctc_head(enc_out)
        ar_logits = self.decoder(enc_out, enc_lengths, enc_block_ids, ar_targets)
        return ctc_logits, ar_logits, enc_lengths

    def encode(self, chunk_batch, chunks_per_line):
        """Encoder-only pass, for inference (decode_greedy needs enc_out/
        enc_lengths/enc_block_ids but not ar_targets)."""
        return self.encoder(chunk_batch, chunks_per_line)
