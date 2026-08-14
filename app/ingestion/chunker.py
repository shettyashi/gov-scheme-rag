"""
Chunks per-page PDF text into retrievable units, keeping page-number
provenance for citations.

Design note — same clause-splitting risk as before, now with an extra
wrinkle: PDFs also split content across PAGE boundaries, not just
paragraph boundaries (e.g. a table of eligibility criteria that
continues onto the next page). To handle this without losing page
citations, pages within a document are concatenated with a page-break
marker before chunking, and each chunk is tagged with the page it
STARTS on. A chunk that actually spans two pages will cite the first
page — good enough for "which document, roughly where" but not perfect;
know this before you trust page citations for prose that continuously
wraps in Adobe-reflow style.

Low-confidence pages (flagged by the loader as likely garbled/scanned)
are excluded from chunking entirely rather than silently embedded —
better to have a documented gap in coverage than confidently wrong
retrieval from garbage text.
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.loader import RawPage

CHUNK_SIZE = 800
CHUNK_OVERLAP = 160
PAGE_BREAK_MARKER = "\n\n<<PAGE_BREAK>>\n\n"


@dataclass
class Chunk:
    scheme_name: str
    source_url: str
    document_name: str
    page_number: int   # page the chunk STARTS on
    chunk_text: str
    chunk_index: int
    used_ocr: bool = False   # True if the starting page was OCR-derived


def _group_by_document(pages: list[RawPage]) -> dict[tuple[str, str], list[RawPage]]:
    groups: dict[tuple[str, str], list[RawPage]] = {}
    for page in pages:
        key = (page.scheme_name, page.document_name)
        groups.setdefault(key, []).append(page)
    return groups


def chunk_document(pages: list[RawPage]) -> list[Chunk]:
    """pages must all belong to the same (scheme_name, document_name), in order."""
    usable_pages = [p for p in pages if not p.is_low_confidence and p.raw_text]
    if not usable_pages:
        return []

    # Build full text with markers, and parallel offset maps for both
    # page number and OCR provenance.
    full_text = ""
    offset_to_page: list[tuple[int, int]] = []       # (char_offset_start, page_number)
    offset_to_ocr: list[tuple[int, bool]] = []        # (char_offset_start, used_ocr)
    for page in usable_pages:
        offset_to_page.append((len(full_text), page.page_number))
        offset_to_ocr.append((len(full_text), page.used_ocr))
        full_text += page.raw_text + PAGE_BREAK_MARKER

    def page_for_offset(offset: int) -> int:
        page_num = usable_pages[0].page_number
        for start_offset, pnum in offset_to_page:
            if start_offset <= offset:
                page_num = pnum
            else:
                break
        return page_num

    def ocr_for_offset(offset: int) -> bool:
        used_ocr = usable_pages[0].used_ocr
        for start_offset, flag in offset_to_ocr:
            if start_offset <= offset:
                used_ocr = flag
            else:
                break
        return used_ocr

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[PAGE_BREAK_MARKER, "\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(full_text)

    chunks: list[Chunk] = []
    search_cursor = 0
    for i, piece in enumerate(pieces):
        piece_clean = piece.replace(PAGE_BREAK_MARKER.strip(), "").strip()
        if not piece_clean:
            continue
        found_at = full_text.find(piece.strip()[:50], search_cursor)
        offset = found_at if found_at != -1 else search_cursor
        page_num = page_for_offset(offset)
        search_cursor = max(search_cursor, offset)

        chunks.append(
            Chunk(
                scheme_name=usable_pages[0].scheme_name,
                source_url=usable_pages[0].source_url,
                document_name=usable_pages[0].document_name,
                page_number=page_num,
                chunk_text=piece_clean,
                chunk_index=i,
                used_ocr=ocr_for_offset(offset),
            )
        )

    return chunks


def chunk_pages(pages: list[RawPage]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for _, doc_pages in _group_by_document(pages).items():
        doc_pages_sorted = sorted(doc_pages, key=lambda p: p.page_number)
        chunks.extend(chunk_document(doc_pages_sorted))
    return chunks
