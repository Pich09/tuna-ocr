"""Wraps the project's shared tokenizer (Panhapich/khmer-sp-8k) for OCR use.

Confirmed against the upstream repo (huggingface.co/Panhapich/khmer-sp-8k):
khmer_segmentation.py's `KhmerTokenizer` class (segmentation pipeline +
SentencePiece) must be used instead of a bare `sentencepiece.SentencePieceProcessor`
-- the segmentation step (khmer-nltk word tokenize + gazetteer/loanword
masking) must never drift out of sync between training and inference, or
tokenization is silently wrong. The base tokenizer already reserves
<PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3, <MASK>=4 (looked up dynamically, not
hardcoded); this wrapper only appends new ids for what OCR needs on top:
<km>/<en>/<fr> language tags and <eob> (end-of-block, for the blockwise-AR
decoder) -- see recognizer/config.py's LANG_TOKENS/EXTRA_CONTROL_TOKENS.
"""
import importlib.util
import re
from pathlib import Path

from ..config import TOKENIZER_ASSETS_DIR, LANG_TOKENS, EXTRA_CONTROL_TOKENS

_TAGGED_SPAN_RE = re.compile(r"<(km|en|fr)>([^<]*)")
_KHMER_CHAR_RE = re.compile(r"[ក-៿]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


def _load_upstream_khmer_tokenizer(assets_dir: Path):
    """Dynamically imports khmer_segmentation.py from the downloaded assets
    dir (it's a vendored file, not an installed package) and constructs its
    KhmerTokenizer class."""
    module_path = assets_dir / "khmer_segmentation.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"{module_path} not found -- run "
            "`python -m recognizer.tokenizer.fetch_tokenizer` first."
        )
    spec = importlib.util.spec_from_file_location("khmer_segmentation", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    latin_exceptions_path = assets_dir / "latin_exceptions.json"
    return module.KhmerTokenizer(
        sp_model_path=str(assets_dir / "khmer_sp.model"),
        gazetteer_path=str(assets_dir / "gazetteer.json"),
        latin_exceptions_path=str(latin_exceptions_path) if latin_exceptions_path.exists() else None,
    )


def guess_dominant_lang(text: str) -> str:
    """Cheap heuristic for untagged real_data transcripts (all 5 configured
    sources are plain, untagged text): Khmer-Unicode-block char count vs
    ASCII-letter count. Khmer plurality -> 'km', else 'en' -- 'fr' is never
    auto-guessed here since no external source is French-dominant; it's
    only reachable via encode_tagged on data_gen-style inline-tagged text."""
    khmer_count = len(_KHMER_CHAR_RE.findall(text))
    ascii_count = len(_ASCII_LETTER_RE.findall(text))
    return "km" if khmer_count >= ascii_count else "en"


class KhmerOcrTokenizer:
    def __init__(self, assets_dir: Path = TOKENIZER_ASSETS_DIR):
        self._base = _load_upstream_khmer_tokenizer(Path(assets_dir))
        sp = self._base.sp
        self.base_vocab_size = sp.get_piece_size()
        self.pad_id = sp.piece_to_id("<PAD>")
        self.unk_id = sp.piece_to_id("<UNK>")
        self.bos_id = sp.piece_to_id("<BOS>")
        self.eos_id = sp.piece_to_id("<EOS>")
        self.mask_id = sp.piece_to_id("<MASK>")

        extra_tokens = LANG_TOKENS + EXTRA_CONTROL_TOKENS
        self.token2id = {t: self.base_vocab_size + i for i, t in enumerate(extra_tokens)}
        self.vocab_size = self.base_vocab_size + len(extra_tokens)

        self.km_id = self.token2id["<km>"]
        self.en_id = self.token2id["<en>"]
        self.fr_id = self.token2id["<fr>"]
        self.eob_id = self.token2id["<eob>"]
        self._lang_ids = {self.km_id, self.en_id, self.fr_id}
        self._control_ids = {self.pad_id, self.bos_id, self.eos_id, self.eob_id} | self._lang_ids

    def encode_tagged(self, tagged_text: str) -> list:
        """Parses data_gen-style inline '<km>...<en>...' text (matches
        data_gen/generate_lines.py's own serialization) into spans, emitting
        [lang_token_id] + base.encode(span_text) per span. Implemented for
        forward compatibility if data_gen output is ever added to training
        later -- not exercised by this pipeline's real_data-only sources."""
        ids = []
        for lang, span_text in _TAGGED_SPAN_RE.findall(tagged_text):
            span_text = span_text.strip()
            ids.append(self.token2id[f"<{lang}>"])
            if span_text:
                ids.extend(self._base.encode(span_text))
        return ids

    def encode_plain(self, text: str) -> list:
        """The encode path this pipeline actually uses: every real_data
        source is plain, untagged text, so the whole line is tagged with
        its guessed dominant language."""
        lang = guess_dominant_lang(text)
        return [self.token2id[f"<{lang}>"]] + self._base.encode(text)

    def decode(self, ids, strip_control: bool = True) -> str:
        """Strips control/lang tokens (unless strip_control=False) and
        decodes each contiguous run of base-vocab ids through the upstream
        tokenizer's own detokenization (which also strips the artificial
        word-boundary spaces segmentation introduced)."""
        pieces = []
        run = []
        for i in ids:
            if i in self._control_ids:
                if run:
                    pieces.append(self._base.decode(run))
                    run = []
                if not strip_control:
                    pieces.append(self._id_to_display(i))
            else:
                run.append(i)
        if run:
            pieces.append(self._base.decode(run))
        return "".join(pieces)

    def _id_to_display(self, i: int) -> str:
        for tok, tid in self.token2id.items():
            if tid == i:
                return tok
        if i == self.pad_id:
            return "<PAD>"
        if i == self.bos_id:
            return "<BOS>"
        if i == self.eos_id:
            return "<EOS>"
        return ""
