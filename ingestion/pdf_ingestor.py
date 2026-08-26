"""
ingestion/pdf_ingestor.py

The production PDF ingestor. Orchestrates every other module in this
package into one `ingest(file_path) -> Document` call:

  1. Open the PDF (PyMuPDF preferred for rich per-line font metadata;
     pdfplumber-only degraded path when PyMuPDF is not installed).
  2. Extract raw per-page text, and per-line layout facts (font size,
     bold) when available.
  3. Detect and strip repeated running headers/footers across the whole
     document (`cleaner.TextCleaner`), then clean each page's text.
  4. Detect semantic sections / subsection hierarchy / bullet & numbered
     lists (`section_detector.SectionDetector`) -- chunking by MEANING,
     never by raw page boundaries.
  5. Extract tables as structured objects (`table_extractor.TableExtractor`).
  6. Extract images/figures/charts (`image_extractor.ImageExtractor`).
  7. Extract and normalize document metadata (`metadata_extractor.MetadataExtractor`).
  8. Assemble everything into one `Document`, compute aggregate
     statistics and a quality score, and return it.

This module deliberately does NOT do embedding/indexing -- that is a
retrieval-time concern for whatever merges this package's output with
the existing JSON-based `KnowledgeIndex` (see `utils.embed_texts` /
`build_faiss_index` for optional helpers a future merge step can use).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

from .base_ingestor import BaseIngestor, register_ingestor
from .cleaner import TextCleaner
from .document import Document, DocumentStatistics, PageInfo
from .image_extractor import ImageExtractor
from .metadata_extractor import MetadataExtractor
from .section import Section
from .section_detector import LineLayout, SectionDetector
from .table_extractor import TableExtractor
from .utils import detect_language

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _FITZ_AVAILABLE = False
    logger.warning(
        "PyMuPDF (fitz) is not installed. PDFIngestor will run in the degraded "
        "pdfplumber-only path: text/metadata extraction still works, but "
        "font-aware section detection and full image extraction are unavailable. "
        "Install PyMuPDF (`pip install pymupdf`) for the full pipeline."
    )

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _PDFPLUMBER_AVAILABLE = False


@register_ingestor
class PDFIngestor(BaseIngestor):
    """Production PDF ingestor. See module docstring for the full
    pipeline this orchestrates."""

    supported_extensions = (".pdf",)

    def __init__(
        self,
        image_output_dir: Optional[str] = None,
        # Defaults OFF: camelot is attempted on every page pdfplumber found
        # no table on, which for a full annual report (100-300+ pages,
        # mostly narrative) means hundreds of slow per-page camelot calls
        # (each one rasterizes the page) on a document that may only have
        # a handful of genuine financial-statement tables. On a batch of
        # several large annual reports this turns first-time ingestion
        # from seconds into many minutes-to-hours with no visible
        # progress in a chat UI. pdfplumber alone already catches the
        # large majority of real tables; enable this explicitly only if
        # you've confirmed specific documents have tables pdfplumber
        # misses and you're OK with the added ingestion time.
        use_camelot_fallback: bool = False,
        min_chunk_words: int = 20,
        # OFF by default. Vision captioning (see ingestion/image_captioner.py)
        # is the one genuinely slow step in this whole pipeline -- even with
        # the faster model + caching + logo/already-has-text skipping in that
        # module, a first-time run on a report with many uncaptioned charts
        # can still take minutes. That cost belongs in an explicit, offline
        # batch step (import_pdf_documents.py --caption-images), never inline
        # in a live chat request -- turning this on here would silently
        # reintroduce the "thinking UI never shows up" problem for the first
        # query after any new/changed PDF is dropped in.
        caption_images: bool = False,
    ) -> None:
        self.image_output_dir = image_output_dir or os.path.join(
            os.getcwd(), "documents", "extracted_images"
        )
        self.table_extractor = TableExtractor(use_camelot_fallback=use_camelot_fallback)
        self.image_extractor = ImageExtractor(output_dir=self.image_output_dir)
        self.metadata_extractor = MetadataExtractor()
        self.section_detector = SectionDetector()
        self.min_chunk_words = min_chunk_words
        self.caption_images = caption_images

    # -- public entry point -------------------------------------------------

    def ingest(self, file_path: str) -> Document:
        self.validate_file_exists(file_path)
        logger.info("Ingesting PDF: %s", file_path)

        document = Document(source_file=file_path, document_type="pdf")

        if _FITZ_AVAILABLE:
            pages_raw_text, line_layouts, page_dims = self._extract_with_fitz(file_path)
        else:
            pages_raw_text, page_dims = self._extract_with_pdfplumber(file_path)
            line_layouts = None
            document.extraction_warnings.append(
                "PyMuPDF not installed: using degraded pdfplumber-only text extraction "
                "(section detection falls back to plain-text heuristics)."
            )

        if not pages_raw_text:
            document.extraction_warnings.append("No pages could be extracted from this PDF.")
            document.quality_score = 0.0
            return document

        # -- clean text, detect/remove running headers & footers --------
        cleaner = TextCleaner()
        cleaner.detect_repeated_header_footer(pages_raw_text)
        cleaned_pages = [cleaner.clean_page(text) for text in pages_raw_text]

        document.pages = [
            PageInfo(
                page_number=i + 1,
                width=page_dims[i][0] if i < len(page_dims) else 0.0,
                height=page_dims[i][1] if i < len(page_dims) else 0.0,
                raw_text=pages_raw_text[i],
                cleaned_text=cleaned_pages[i],
                header=cleaner._detected_header,
                footer=cleaner._detected_footer,
            )
            for i in range(len(pages_raw_text))
        ]

        document.raw_text = "\n\n".join(pages_raw_text)
        document.cleaned_text = "\n\n".join(cleaned_pages)
        document.tokens = document.cleaned_text.split()
        document.language = detect_language(document.cleaned_text)

        # -- section detection --------------------------------------------
        if line_layouts is not None:
            cleaned_layouts = self._rebuild_layouts_with_cleaned_text(
                line_layouts, cleaner
            )
            document.sections = self.section_detector.detect_from_layout(cleaned_layouts)
        else:
            document.sections = self.section_detector.detect_from_plain_text(cleaned_pages)

        if not document.sections:
            document.extraction_warnings.append(
                "No semantic sections could be detected; retrieval for this "
                "document will need to fall back to page-level text."
            )

        # -- tables ----------------------------------------------------------
        try:
            document.tables = self.table_extractor.extract(file_path)
        except Exception as exc:
            logger.exception("Table extraction failed for %s", file_path)
            document.extraction_warnings.append(f"Table extraction failed: {exc}")

        # -- images ------------------------------------------------------------
        try:
            images, image_warnings = self.image_extractor.extract(file_path)
            document.images = images
            document.extraction_warnings.extend(image_warnings)

            if self.caption_images and images:
                from .image_captioner import caption_images as _caption_images
                page_text_by_number = {
                    p.page_number: p.cleaned_text for p in document.pages
                }
                n_calls = _caption_images(images, page_text_by_number)
                logger.info(
                    "Image captioning: %d vision call(s) made for %s "
                    "(%d image(s) skipped as logos/already-covered-by-text/cached).",
                    n_calls, file_path, len(images) - n_calls,
                )
        except Exception as exc:
            logger.exception("Image extraction failed for %s", file_path)
            document.extraction_warnings.append(f"Image extraction failed: {exc}")

        # -- metadata -----------------------------------------------------------
        text_sample = "\n".join(cleaned_pages[:2])
        document.metadata = self.metadata_extractor.extract(
            file_path=file_path,
            text_sample=text_sample,
            page_count=len(pages_raw_text),
            language=document.language,
        )
        document.checksum = document.metadata.checksum_sha256

        # -- statistics & quality score -------------------------------------
        document.statistics = self._compute_statistics(document)
        document.quality_score = document.compute_quality_score()

        logger.info(
            "Ingestion complete: %s (%d pages, %d sections, %d tables, %d images, "
            "quality=%.2f)",
            file_path,
            document.statistics.page_count,
            document.statistics.section_count,
            document.statistics.table_count,
            document.statistics.image_count,
            document.quality_score,
        )
        return document

    # -- PyMuPDF extraction path (preferred) -------------------------------

    def _extract_with_fitz(
        self, file_path: str
    ) -> Tuple[List[str], List[LineLayout], List[Tuple[float, float]]]:
        pages_raw_text: List[str] = []
        line_layouts: List[LineLayout] = []
        page_dims: List[Tuple[float, float]] = []

        doc = fitz.open(file_path)
        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                page_number = page_index + 1
                page_dims.append((page.rect.width, page.rect.height))

                page_text = page.get_text("text")
                pages_raw_text.append(page_text)

                try:
                    raw_dict = page.get_text("dict")
                except Exception as exc:
                    logger.warning(
                        "Failed to extract rich layout for page %d: %s", page_number, exc
                    )
                    continue

                for block in raw_dict.get("blocks", []):
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        if not spans:
                            continue
                        text = "".join(span.get("text", "") for span in spans).strip()
                        if not text:
                            continue
                        # A line's "effective" font size/boldness is taken
                        # from its largest span, since a heading line
                        # rendered with mixed spans (rare, but happens with
                        # inline styling) should still be judged by its
                        # most prominent visual weight.
                        dominant_span = max(spans, key=lambda s: s.get("size", 0))
                        is_bold = "bold" in dominant_span.get("font", "").lower()
                        line_layouts.append(
                            LineLayout(
                                text=text,
                                page_number=page_number,
                                font_size=dominant_span.get("size", 0.0),
                                is_bold=is_bold,
                            )
                        )
        finally:
            doc.close()

        return pages_raw_text, line_layouts, page_dims

    def _rebuild_layouts_with_cleaned_text(
        self, line_layouts: List[LineLayout], cleaner: TextCleaner
    ) -> List[LineLayout]:
        """Drop lines that match the detected running header/footer so
        they never masquerade as a heading, while keeping every other
        line's original font-size/bold metadata intact (only the raw
        page-level text goes through the rest of `TextCleaner`'s
        pipeline; per-line layout only needs header/footer stripped)."""
        header = cleaner._detected_header
        footer = cleaner._detected_footer
        if not header and not footer:
            return line_layouts
        kept = []
        for line in line_layouts:
            normalized = cleaner._normalize_for_comparison(line.text)
            if normalized in (header, footer):
                continue
            kept.append(line)
        return kept

    # -- pdfplumber-only degraded extraction path --------------------------

    def _extract_with_pdfplumber(
        self, file_path: str
    ) -> Tuple[List[str], List[Tuple[float, float]]]:
        if not _PDFPLUMBER_AVAILABLE:
            raise RuntimeError(
                "Neither PyMuPDF nor pdfplumber is installed; cannot extract "
                "any text from this PDF."
            )
        pages_raw_text: List[str] = []
        page_dims: List[Tuple[float, float]] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                pages_raw_text.append(page.extract_text() or "")
                page_dims.append((float(page.width), float(page.height)))
        return pages_raw_text, page_dims

    # -- statistics -----------------------------------------------------------

    def _compute_statistics(self, document: Document) -> DocumentStatistics:
        page_count = len(document.pages)
        word_count = len(document.tokens)
        character_count = len(document.cleaned_text)
        section_count = sum(1 for _ in document.iter_all_sections())
        table_count = len(document.tables)
        well_formed_table_count = sum(1 for t in document.tables if t.is_well_formed())
        image_count = len(document.images)
        average_words_per_page = round(word_count / page_count, 2) if page_count else 0.0

        return DocumentStatistics(
            page_count=page_count,
            word_count=word_count,
            character_count=character_count,
            section_count=section_count,
            table_count=table_count,
            well_formed_table_count=well_formed_table_count,
            image_count=image_count,
            average_words_per_page=average_words_per_page,
        )