"""Character-level tokenizer shared by BOTH the CTC head and the AR decoder --
for the CTC-only / AR-only ablation notebooks (train_recognizer_ctc_only.ipynb
/ train_recognizer_ar_only.ipynb), which need "same tokenizer" comparability
between the two heads. Deliberately NOT used by the production two-head run
(train_recognizer_v2_scratch.ipynb): data/char_vocab.py's docstring documents
a measured dead end where the CTC head trained against the shared SUBWORD
vocab floors at loss ~2.1 and decodes every sample to the same string, while
character targets reach ~0.0001 and decode exactly -- so unifying onto
subwords was rejected. Unifying onto CHARACTERS instead works for both heads:
CTC already used character targets, and a character-level AR decoder is a
well-understood (if lower-compression) target space, just needing a larger
max_tokens_per_block than the subword decoder.

Implements two call interfaces on one object so it can be passed as BOTH
run_training(tokenizer=...) and run_training(..., char_vocab=...) without any
other train.py changes:
  - AR side (matches tokenizer.khmer_ocr_tokenizer.KhmerOcrTokenizer): pad_id/
    bos_id/eos_id/eob_id/vocab_size, encode_plain(text), decode(ids, strip_control=True)
  - CTC side (matches data.char_vocab.CharVocab): encode(text) [== encode_plain,
    the same ids feed both heads -- the whole point of this class], decode(ids),
    .size [CTC output classes including blank], .blank_id

Control tokens occupy ids 0-3 (PAD/BOS/EOS/EOB); real characters start at 4.
CTC's blank class is NOT one of these -- like modules/model.py's existing
"legacy subword-CTC" ctc_vocab_size=None fallback, it sits at the one extra
index past vocab_size, so .size == vocab_size + 1 and .blank_id == vocab_size.
"""
import json
from pathlib import Path

_CONTROL_NAMES = ["PAD", "BOS", "EOS", "EOB"]


class CharTokenizer:
    PAD, BOS, EOS, EOB = range(4)
    _NUM_CONTROL = 4

    def __init__(self, chars: list):
        self.chars = list(chars)
        self.pad_id, self.bos_id, self.eos_id, self.eob_id = self.PAD, self.BOS, self.EOS, self.EOB
        self.c2i = {c: i + self._NUM_CONTROL for i, c in enumerate(self.chars)}
        self.i2c = {i + self._NUM_CONTROL: c for i, c in enumerate(self.chars)}
        self.vocab_size = self._NUM_CONTROL + len(self.chars)
        self._control_ids = {self.pad_id, self.bos_id, self.eos_id, self.eob_id}

    # ---- AR-tokenizer interface ----
    def encode_plain(self, text: str) -> list:
        """Unknown characters are dropped (not mapped to <unk>) -- matches
        data/char_vocab.py's CharVocab.encode, which this replaces."""
        return [self.c2i[c] for c in text if c in self.c2i]

    def decode(self, ids, strip_control: bool = True) -> str:
        out = []
        for i in ids:
            if i in self.i2c:
                out.append(self.i2c[i])
            elif not strip_control and i in self._control_ids:
                out.append(f"<{_CONTROL_NAMES[i]}>")
        return "".join(out)

    # ---- CTC "char_vocab" interface -- same ids as encode_plain above, by
    # construction, so CTC and AR targets are literally the same tokenizer. ----
    encode = encode_plain

    @property
    def size(self) -> int:
        """Number of CTC output classes, including the blank."""
        return self.vocab_size + 1

    @property
    def blank_id(self) -> int:
        return self.vocab_size

    def to_json(self) -> str:
        return json.dumps({"chars": self.chars}, ensure_ascii=False)

    @classmethod
    def from_json(cls, blob: str) -> "CharTokenizer":
        return cls(json.loads(blob)["chars"])

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
