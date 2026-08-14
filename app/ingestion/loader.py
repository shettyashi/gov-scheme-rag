"""
Loads raw government scheme PDFs and extracts per-page clean text + metadata.

Expected input layout:

    data/raw/<scheme_name>/<document_name>.pdf
    data/raw/<scheme_name>/<document_name>.meta.json   # optional sidecar

meta.json (optional, per PDF) should look like:
    {
        "source_url": "https://pmkisan.gov.in/Documents/Guidelines.pdf",
        "published_at": "2024-02-01"
    }

Without the sidecar, source_url falls back to the local file path — fine
for local dev, but fix before deploying, since "which official document
does this claim come from" is the entire point of the grounding guardrail.

Why per-page, not whole-document, extraction:
Chunking later needs to know which page a chunk came from, so citations
can say "Guidelines PDF, page 4" rather than just the document name.
This also lets you catch page-level extraction failures without losing
the rest of the document.

OCR fallback:
Real government PDFs are a mix of born-digital text and scanned images
(entire documents, or individual pages within an otherwise-text PDF —
e.g. a scanned annexure inserted into a typed guidelines document).
Rather than skip these, every page that looks low-confidence after
normal text extraction gets re-processed through Tesseract OCR at
300 DPI. OCR is meaningfully slower (~1-3s/page vs milliseconds) and
noisier than native text extraction — it WILL misread some characters,
especially numbers and tables — so every OCR'd page is flagged
`used_ocr=True` downstream so you can report what fraction of your
corpus is OCR-derived. That fraction is a real number worth including
in your eval writeup, since OCR pages are more likely to feed
hallucinated or garbled answers.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import pytesseract
from pdf2image import convert_from_path

OCR_DPI = 300


@dataclass
class RawPage:
    scheme_name: str
    source_url: str
    document_name: str
    page_number: int          # 1-indexed, matches what a human would cite
    raw_text: str
    is_low_confidence: bool   # True if even OCR extraction looked suspicious
    used_ocr: bool = False


def _load_sidecar(pdf_path: Path) -> dict:
    meta_path = pdf_path.with_suffix(".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _has_embedded_fonts(pdf_path: Path) -> bool:
    """
    Cheap upfront signal for whether a PDF is likely scanned. Doesn't
    gate OCR anymore (every low-confidence page gets OCR'd regardless),
    but it's still useful as a fast heads-up in the console before the
    slower page-by-page extraction runs.
    """
    try:
        result = subprocess.run(
            ["pdffonts", str(pdf_path)],
            capture_output=True, text=True, timeout=30,
        )
        return len(result.stdout.strip().splitlines()) > 2
    except Exception:
        return True


def _is_low_confidence_extraction(text: str) -> bool:
    """
    Heuristic flag, not a guarantee. A page that extracted almost no
    text, or mostly non-alphanumeric noise, is more likely garbled or
    scanned than genuinely empty.
    """
    stripped = text.strip()
    if len(stripped) < 20:
        return True
    alnum_ratio = sum(c.isalnum() for c in stripped) / len(stripped)
    return alnum_ratio < 0.3


def _ocr_page(pdf_path: Path, page_number: int) -> str:
    """
    Rasterize a single page and run Tesseract on it.

    lang="eng+hin": government circulars are routinely bilingual
    (Hindi header/subject line, English body, or vice versa). OCR'ing
    with English only silently mangles every Hindi character into
    Latin-alphabet noise instead of failing loudly — worse than doing
    nothing, since it looks like real extracted text. Add more language
    codes here (e.g. "+kan" for Kannada, "+tam" for Tamil) if your
    scheme set includes state-language documents; each additional
    language needs its own `apt-get install tesseract-ocr-<code>`
    traineddata installed first.
    """
    images = convert_from_path(
        str(pdf_path),
        dpi=OCR_DPI,
        first_page=page_number,
        last_page=page_number,
    )
    if not images:
        return ""
    return pytesseract.image_to_string(images[0], lang="eng+hin")


def load_pdf(pdf_path: Path, scheme_name: str) -> list[RawPage]:
    sidecar = _load_sidecar(pdf_path)
    source_url = sidecar.get("source_url", str(pdf_path))

    if not _has_embedded_fonts(pdf_path):
        print(f"  INFO: {pdf_path.name} has no embedded fonts — likely "
              f"fully scanned. Every page will go through OCR; this will "
              f"be slow for large documents.")

    pages: list[RawPage] = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            used_ocr = False

            if _is_low_confidence_extraction(text):
                print(f"  Native extraction weak on {pdf_path.name} "
                      f"page {i}/{total_pages} ({len(text.strip())} chars) "
                      f"— running OCR...")
                ocr_text = _ocr_page(pdf_path, i)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    used_ocr = True

            low_confidence = _is_low_confidence_extraction(text)
            if low_confidence:
                print(f"  WARNING: {pdf_path.name} page {i} still low-confidence "
                      f"after OCR ({len(text.strip())} chars) — flagged, "
                      f"likely a blank/image-only page. Excluded from chunking.")

            pages.append(
                RawPage(
                    scheme_name=scheme_name,
                    source_url=source_url,
                    document_name=pdf_path.stem,
                    page_number=i,
                    raw_text=text.strip(),
                    is_low_confidence=low_confidence,
                    used_ocr=used_ocr,
                )
            )

    return pages


def load_raw_pages(raw_dir: Path) -> list[RawPage]:
    """Walk data/raw/<scheme_name>/*.pdf and return per-page extracted text."""
    all_pages: list[RawPage] = []

    for scheme_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        scheme_name = scheme_dir.name
        for pdf_path in sorted(scheme_dir.glob("*.pdf")):
            print(f"Processing {pdf_path}")
            all_pages.extend(load_pdf(pdf_path, scheme_name))

    return all_pages


if __name__ == "__main__":
    import sys

    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    pages = load_raw_pages(raw_dir)
    print(f"\nLoaded {len(pages)} pages total from {raw_dir}")
    flagged = [p for p in pages if p.is_low_confidence]
    ocr_used = [p for p in pages if p.used_ocr]
    print(f"{len(ocr_used)} pages required OCR "
          f"({100*len(ocr_used)/max(len(pages),1):.1f}% of corpus)")
    if flagged:
        print(f"{len(flagged)} pages still flagged low_confidence "
              f"even after OCR — review these before trusting the pipeline")
