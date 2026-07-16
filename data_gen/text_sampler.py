"""Samples text lines/sentences from per-language corpora, with optional
code-mixing (splicing a short EN/FR span into a Khmer sentence) to mimic
real documents where acronyms, names, or citations appear inline.
"""
import random
from pathlib import Path

from .config import CORPORA_DIR, LANGUAGES, CODE_MIX_PROB

# Used only until real corpora are dropped into corpora/<lang>/*.txt.
_PLACEHOLDER = {
    "km": [
        "ព្រះរាជាណាចក្រកម្ពុជា",
        "ជំពូកទី១ បញ្ញត្តិទូទៅ",
        "អគ្គលេខាធិការដ្ឋាន នៃក្រុមប្រឹក្សារដ្ឋមន្ត្រី",
        "ការអភិវឌ្ឍសេដ្ឋកិច្ចនិងសង្គមកិច្ច",
        "សាកលវិទ្យាល័យភូមិន្ទភ្នំពេញ",
    ],
    "en": [
        "Ministry of Economy and Finance",
        "Table 1: Summary of results",
        "Phnom Penh, Cambodia",
        "See Appendix A for details",
        "Chapter 1: Introduction",
    ],
    "fr": [
        "Royaume du Cambodge",
        "Ministere des Affaires Etrangeres",
        "Annexe B: Methodologie",
        "Republique Francaise",
        "Introduction generale",
    ],
}


class TextSampler:
    def __init__(self, corpora_dir: Path = CORPORA_DIR, code_mix_prob: float = CODE_MIX_PROB):
        self.code_mix_prob = code_mix_prob
        self._sentences = {lang: self._load(corpora_dir / lang) for lang in LANGUAGES}
        for lang, sents in self._sentences.items():
            if not sents:
                self._sentences[lang] = _PLACEHOLDER[lang]

    @staticmethod
    def _load(lang_dir: Path):
        sentences = []
        if lang_dir.is_dir():
            for txt_file in sorted(lang_dir.glob("*.txt")):
                for line in txt_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        sentences.append(line)
        return sentences

    def sample_sentence(self, lang: str) -> str:
        return random.choice(self._sentences[lang])

    def sample_line(self, primary_lang: str = "km"):
        """Returns (text, spans) where spans is a list of (lang, substring)
        describing which language each part of the returned text is in.
        Used as the ground-truth for language-tagged decoder targets,
        e.g. '<km> ... <en> ... <km> ...'.
        """
        text = self.sample_sentence(primary_lang)
        spans = [(primary_lang, text)]

        if primary_lang == "km" and random.random() < self.code_mix_prob:
            other_lang = random.choice([l for l in ("en", "fr") if self._sentences[l]])
            splice = self.sample_sentence(other_lang)
            # Keep the splice short (looks like an acronym/name/citation, not a full sentence).
            splice = " ".join(splice.split()[:3])
            text = f"{text} {splice}"
            spans.append((other_lang, splice))

        return text, spans

    def sample_paragraph(self, primary_lang: str = "km", n_sentences: int = 4):
        return " ".join(self.sample_sentence(primary_lang) for _ in range(n_sentences))
