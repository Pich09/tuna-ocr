"""Character-level vocabulary for the CTC auxiliary head.

Why the CTC head does NOT share the subword tokenizer: CTC classifies every
encoder frame independently into a flat label set, so the label set has to be
something the encoder can actually learn to see. With the shared 8k-subword
vocab that means an ~8005-way decision per frame over whole words/word
pieces, each of which carries no compositional relationship to the glyphs on
screen -- a measured dead end: on 8 fixed samples the CTC loss floors at
~2.1 and every image decodes to the *same* string, while the identical
encoder with character targets reaches ~0.0001 and decodes all 8 exactly.
Characters recur across samples, so the encoder learns reusable glyph->label
mappings instead of trying to memorize one-shot word classes.

The AR decoder deliberately keeps the shared `Panhapich/khmer-sp-8k`
tokenizer -- it is the model's actual output and must stay consistent with
the rest of the project (see recognizer/README.md). This split is the
standard hybrid: CTC is an auxiliary signal that teaches the encoder to
read; it never has to agree with the decoder's label space.
"""
import json
from pathlib import Path

# id 0 is reserved as the CTC blank, so real characters start at 1.
BLANK_ID = 0


class CharVocab:
    def __init__(self, chars: list):
        self.chars = list(chars)
        self.c2i = {c: i + 1 for i, c in enumerate(self.chars)}
        self.i2c = {i + 1: c for i, c in enumerate(self.chars)}
        self.blank_id = BLANK_ID

    @property
    def size(self) -> int:
        """Number of CTC output classes, including the blank."""
        return len(self.chars) + 1

    def encode(self, text: str) -> list:
        """Unknown characters are dropped rather than mapped to a shared
        <unk> class: an <unk> label would teach the encoder to emit one
        symbol for visually unrelated glyphs, which is worse than simply not
        supervising those frames (CTC tolerates gaps -- other alignments
        still cover them)."""
        return [self.c2i[c] for c in text if c in self.c2i]

    def decode(self, ids: list) -> str:
        return "".join(self.i2c[i] for i in ids if i in self.i2c)

    def to_json(self) -> str:
        return json.dumps({"chars": self.chars}, ensure_ascii=False)

    @classmethod
    def from_json(cls, blob: str) -> "CharVocab":
        return cls(json.loads(blob)["chars"])

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharVocab":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def build_char_vocab(texts, min_count: int = 1) -> CharVocab:
    """Builds the vocab from transcripts. `min_count` drops one-off glyphs
    (usually OCR noise or a stray script) that would otherwise add classes
    the encoder can never learn from a single example."""
    counts = {}
    for t in texts:
        for c in t:
            counts[c] = counts.get(c, 0) + 1
    chars = sorted(c for c, n in counts.items() if n >= min_count)
    return CharVocab(chars)
