"""Essay-style Khmer PDF generator, tuned to visually match
example_images/image1.jpg (bordered A4 page, centered display-font title
with a small dated byline, then bold "N.label:" section headers each
followed by indented body paragraphs).

Adjustments made after comparing rendered output against image1.jpg:
  - Paths (fonts, output dir) are anchored to this file's directory instead
    of the cwd, so `python -m data_gen.generate_pdfs` and direct execution
    from any working directory land in the same place.
  - Added top whitespace before the title -- image1.jpg has a large gap
    between the border and the title, the previous version started the
    title almost immediately under the border.
  - Section headers now read "១.សេចក្តីផ្តើម:" (no space before the label,
    trailing colon) to match image1.jpg's heading style, instead of "១. ...".
  - Each section is now two indented sub-paragraphs (matching the two
    visibly indented text blocks per section in image1.jpg) instead of one
    single undifferentiated block.
  - make_paragraph() previously sampled *with* replacement, so the same
    sentence could appear twice in one paragraph (and the exact same
    paragraph could repeat verbatim across sections). Sentences are now
    drawn from a shuffled per-document pool without immediate reuse.
"""
from pathlib import Path
import random
from fpdf import FPDF

# ============================================================
# CONFIG
# ============================================================

NUM_DOCS = 10

SCRIPT_DIR = Path(__file__).resolve().parent
FONT_DIR = SCRIPT_DIR / "fonts" / "km"

BODY_FONT = FONT_DIR / "!KhmerOSSiemreap.ttf"
HEADER_FONT = FONT_DIR / "Moul-Regular.ttf"

OUTPUT_DIR = SCRIPT_DIR / "ocr_dataset"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# CONTENT
# ============================================================

TITLES = [
    "មូលបទសរសេរ",
    "សេចក្តីជូនដំណឹង",
    "របាយការណ៍",
    "ឯកសារសិក្សា",
    "សេចក្តីប្រកាស",
    "ព្រឹត្តិបត្រ",
]

SECTION_PAIRS = [
    ("សេចក្តីផ្តើម", "បញ្ហាគន្លឹះ"),
    ("គោលបំណង", "លទ្ធផល"),
    ("ការអនុវត្ត", "សេចក្តីសន្និដ្ឋាន"),
    ("ស្ថានភាពទូទៅ", "សំណើរ"),
]

SENTENCES = [
    "សេចក្តីជូនដំណឹងនេះមានគោលបំណងផ្តល់ព័ត៌មានជូនដល់សាធារណជនឱ្យបានជ្រាប។",
    "ប្រជាពលរដ្ឋអាចទាក់ទងអាជ្ញាធរពាក់ព័ន្ធដើម្បីទទួលបានព័ត៌មានបន្ថែម។",
    "អង្គភាពពាក់ព័ន្ធត្រូវអនុវត្តការងារឱ្យបានត្រឹមត្រូវតាមនីតិវិធីដែលបានកំណត់។",
    "ការរៀបចំឯកសារនេះត្រូវបានអនុវត្តឡើងដើម្បីសាកល្បងប្រព័ន្ធដំណើរការទិន្នន័យ។",
    "អ្នកទទួលខុសត្រូវត្រូវធ្វើការត្រួតពិនិត្យព័ត៌មានឱ្យបានច្បាស់លាស់មុនពេលប្រើប្រាស់។",
    "រាល់សកម្មភាពត្រូវអនុវត្តស្របតាមគោលការណ៍និងបទបញ្ជារបស់អង្គភាព។",
    "ការប្រមូលទិន្នន័យត្រូវអនុវត្តដោយគោរពតាមវិធានការសុវត្ថិភាពដែលបានកំណត់។",
    "សូមសហការគ្នាក្នុងការអនុវត្តការងារឱ្យបានមានប្រសិទ្ធភាពខ្ពស់។",
    "ការពិនិត្យនិងវាយតម្លៃនឹងត្រូវធ្វើឡើងជាបន្តបន្ទាប់។",
    "សាធារណជនអាចដាក់សំណើនិងផ្តល់មតិយោបល់តាមបណ្តាញព័ត៌មានផ្លូវការ។",
]

# ============================================================
# TEXT GENERATION
# ============================================================


class SentencePool:
    """Hands out sentences from a shuffled copy of SENTENCES without
    immediate repeats; reshuffles (excluding the just-used sentence) once
    exhausted, since a single document needs more sentences than the pool
    has unique entries.
    """

    def __init__(self, sentences):
        self._pool = []
        self._all = list(sentences)
        self._last = None

    def _refill(self):
        pool = list(self._all)
        random.shuffle(pool)
        if self._last is not None and pool[0] == self._last and len(pool) > 1:
            pool[0], pool[1] = pool[1], pool[0]
        self._pool = pool

    def next(self):
        if not self._pool:
            self._refill()
        sentence = self._pool.pop()
        self._last = sentence
        return sentence


def make_paragraph(pool: SentencePool, n_sentences=None):
    n = n_sentences or random.randint(2, 3)
    return " ".join(pool.next() for _ in range(n))


# ============================================================
# PDF GENERATION
# ============================================================

def create_pdf(pdf_path, txt_path):

    pdf = FPDF(
        orientation="P",
        unit="mm",
        format="A4"
    )

    pdf.add_page()

    pdf.set_auto_page_break(
        auto=True,
        margin=20
    )

    # Text margins are kept well inside the drawn border (10mm) so wrapped
    # lines never touch it -- with the default 10mm margin, text sat flush
    # against the border on every line after the first (which alone got the
    # +10mm first-line indent below).
    pdf.set_left_margin(22)
    pdf.set_right_margin(22)

    # Khmer shaping
    pdf.set_text_shaping(True)

    # Fonts
    pdf.add_font(
        "KhmerBody",
        fname=str(BODY_FONT)
    )

    pdf.add_font(
        "KhmerHeader",
        fname=str(HEADER_FONT)
    )

    # Border
    pdf.rect(
        10,
        10,
        190,
        277
    )

    title = random.choice(TITLES)

    sec1, sec2 = random.choice(
        SECTION_PAIRS
    )

    pool = SentencePool(SENTENCES)
    sec1_paras = [make_paragraph(pool), make_paragraph(pool)]
    sec2_paras = [make_paragraph(pool), make_paragraph(pool)]

    # Ground truth
    ground_truth = f"""
{title}

១.{sec1}:

{chr(10).join(sec1_paras)}

២.{sec2}:

{chr(10).join(sec2_paras)}
"""

    # =======================
    # TITLE
    # =======================

    # Tuned against pixel-row measurements of example_images/image1.jpg
    # (title text band starts at ~11% of page height).
    pdf.set_y(32)

    pdf.set_font(
        "KhmerHeader",
        size=16
    )

    pdf.cell(
        0,
        10,
        title,
        align="C",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.ln(5)

    pdf.set_font(
        "KhmerBody",
        size=10
    )

    pdf.cell(
        0,
        5,
        f"ភ្នំពេញ  *  ២០២៦",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.ln(6)

    # =======================
    # SECTION 1
    # =======================

    pdf.set_font(
        "KhmerHeader",
        size=13
    )

    pdf.cell(
        0,
        8,
        f"១.{sec1}:",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.ln(2)

    pdf.set_font(
        "KhmerBody",
        size=12
    )

    for para in sec1_paras:
        # first line indent
        pdf.set_x(
            pdf.l_margin + 10
        )
        pdf.multi_cell(
            0,
            8,
            para,
            align="J"
        )
        pdf.ln(3)

    pdf.ln(2)

    # =======================
    # SECTION 2
    # =======================

    pdf.set_font(
        "KhmerHeader",
        size=13
    )

    pdf.cell(
        0,
        8,
        f"២.{sec2}:",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.ln(2)

    pdf.set_font(
        "KhmerBody",
        size=12
    )

    for para in sec2_paras:
        pdf.set_x(
            pdf.l_margin + 10
        )
        pdf.multi_cell(
            0,
            8,
            para,
            align="J"
        )
        pdf.ln(3)

    pdf.output(
        str(pdf_path)
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            ground_truth.strip()
        )

# ============================================================
# DATASET
# ============================================================

if __name__ == "__main__":
    for i in range(NUM_DOCS):

        pdf_file = (
            OUTPUT_DIR /
            f"essay_{i:05d}.pdf"
        )

        txt_file = (
            OUTPUT_DIR /
            f"essay_{i:05d}.txt"
        )

        create_pdf(
            pdf_file,
            txt_file
        )

        if (i + 1) % 100 == 0:
            print(
                f"Generated {i+1}/{NUM_DOCS}"
            )

    print()
    print("Done.")
    print(
        f"Saved to: {OUTPUT_DIR.resolve()}"
    )
