# Corpora

Plain-text corpora used to sample lines and paragraphs, one sentence/line per
row of a `.txt` file, grouped by language:

```
corpora/
  km/   *.txt   Khmer sentences (e.g. Khmer Wikipedia dump, news articles, book text)
  en/   *.txt   English sentences
  fr/   *.txt   French sentences
```

Any number of `.txt` files per folder is fine — `text_sampler.py` concatenates
and samples across all of them. Keep one sentence/line per row; blank lines
are skipped.

Until you add real corpora, `text_sampler.py` falls back to a small built-in
placeholder wordlist per language so the pipeline runs end-to-end for testing.
