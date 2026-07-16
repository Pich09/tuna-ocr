# tuna-ocr

Khmer OCR: synthetic + real training data generation, and a Conformer-encoder
/ blockwise-AR-decoder recognizer model, trained on real Hugging Face OCR
datasets.

## Packages

```
tuna-ocr/
  data_gen/     synthetic document generator (lines, pages, ID cards, letters,
                birth certificates) for the YOLO layout detector and OCR
                line recognizer -- see data_gen/README.md
  real_data/    pulls line images + transcripts from 5 external Hugging Face
                OCR datasets, slices them into overlapping chunks for
                Conformer-encoder input windowing -- see real_data/README.md
  recognizer/   the OCR recognizer model itself: Conformer encoder +
                blockwise-AR decoder, trained on real_data's output, using
                the shared Panhapich/khmer-sp-8k tokenizer -- see
                recognizer/README.md
  notebooks/    Colab/Kaggle/local training notebook + a data-pipeline
                exploration notebook
  example_images/  real reference photos (ID cards, birth certificate,
                official letters) the data_gen templates are modeled on
```

## Quick start

```bash
pip install -r data_gen/requirements.txt -r real_data/requirements.txt -r recognizer/requirements.txt

# pull real training data (5 configured sources, see real_data/config.py)
python -m real_data.generate_external_chunks --source all --num-samples 500

# fetch the shared tokenizer
python -m recognizer.tokenizer.fetch_tokenizer --out-dir recognizer/tokenizer/assets

# train
python -m recognizer.train \
    --real-data-dirs real_data/samples/deepcopy_khmer_text_recognition \
                     real_data/samples/chanrith_ocr_image_line \
                     real_data/samples/darayut_scene_text \
                     real_data/samples/soyvitou_handwritten \
                     real_data/samples/sokheng_synthetic_v1 \
    --tokenizer-dir recognizer/tokenizer/assets --run-name v1
```

Or use `notebooks/train_recognizer.ipynb`, which works unmodified on Colab,
Kaggle, and locally (auto batch sizing, periodic logging, checkpoint push to
the `Panhapich/tuna-ocr` Hugging Face repo) -- see its first cells for
per-platform one-time setup (HF token secret, GPU runtime, Kaggle internet
access).

Each package's own README has the full detail on its layout, conventions,
and known gaps.

## License

MIT -- see `LICENSE`.
