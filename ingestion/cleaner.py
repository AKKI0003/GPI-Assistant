"""
ingestion/cleaner.py

Text cleaning for extracted PDF pages.

Handles, in order:
  1. Repeated header/footer detection and removal (computed ACROSS all
     pages of a document, since a single-page view can never tell a
     genuine one-off heading apart from a running header).
  2. Standalone page-number line removal.
  3. De-hyphenation of words broken across a line wrap (e.g.
     "manufac-\ntured" -> "manufactured").
  4. Rejoining of "soft" line wraps (a line that doesn't end in
     sentence-ending punctuation almost always continues the next line
     in the source PDF's reading order).
  5. OCR-artifact stripping (control characters, the Unicode replacement
     character, and repeated stray punctuation runs OCR engines often
     emit for graphical noise).
  6. Whitespace normalization.

Nothing here removes meaningful structural formatting: bullet glyphs,
numbered-list markers, and blank lines that separate paragraphs are all
deliberately preserved, since `section.detect_bullets` /
`detect_numbered_list` and `SectionDetector` both depend on them still
being present.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import List, Optional

logger = logging.getLogger(__name__)

_PAGE_NUMBER_LINE = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*of\s*\d{1,4})?\s*$", re.IGNORECASE)
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_REPLACEMENT_CHAR_RUN = re.compile(r"\ufffd+")
_STRAY_PUNCT_RUN = re.compile(r"([^\w\s])\1{3,}")  # e.g. "-----" or "....." noise
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_SENTENCE_END = re.compile(r"[.!?:;\"')\]]\s*$")
_BULLET_OR_NUMBERED_START = re.compile(r"^\s*([\u2022\u25CF\u25A0\-\*\u2013]|\d{1,3}[\.\)])\s+")


class TextCleaner:
    """Stateless-per-document text cleaner. Instantiate one per document
    (it caches the detected header/footer for that document only) rather
    than sharing an instance across documents."""

    def __init__(self, header_footer_threshold: float = 0.5) -> None:
        # A line must repeat on at least this fraction of pages to be
        # treated as a running header/footer rather than a genuine
        # heading that happens to appear more than once.
        self.header_footer_threshold = header_footer_threshold
        self._detected_header: Optional[str] = None
        self._detected_footer: Optional[str] = None

    # -- header/footer detection -----------------------------------------

    def detect_repeated_header_footer(self, pages_raw_text: List[str]) -> None:
        """Scan every page's first and last non-blank line; whatever line
        repeats (near-identically, ignoring embedded page numbers) across
        enough pages is recorded as the running header/footer for this
        document."""
        if len(pages_raw_text) < 3:
            return  # too few pages to distinguish "repeated" from "coincidence"

        first_lines: List[str] = []
        last_lines: List[str] = []
        for text in pages_raw_text:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                continue
            first_lines.append(self._normalize_for_comparison(lines[0]))
            last_lines.append(self._normalize_for_comparison(lines[-1]))

        self._detected_header = self._most_common_if_repeated(first_lines, len(pages_raw_text))
        self._detected_footer = self._most_common_if_repeated(last_lines, len(pages_raw_text))

        if self._detected_header:
            logger.debug("Detected repeated header: %r", self._detected_header)
        if self._detected_footer:
            logger.debug("Detected repeated footer: %r", self._detected_footer)

    def _most_common_if_repeated(self, normalized_lines: List[str], page_count: int) -> Optional[str]:
        if not normalized_lines:
            return None
        counts = Counter(normalized_lines)
        candidate, count = counts.most_common(1)[0]
        if not candidate:
            return None
        if count / page_count >= self.header_footer_threshold:
            return candidate
        return None

    @staticmethod
    def _normalize_for_comparison(line: str) -> str:
        """Strip digits before comparing lines, so a header/footer that
        includes a page number (e.g. "Annual Report 2024 - Page 3") still
        matches across pages."""
        return re.sub(r"\d+", "#", line).strip().lower()

    # -- per-page cleaning -------------------------------------------------

    def clean_page(self, raw_text: str) -> str:
        """Apply the full cleaning pipeline to a single page's raw text.
        `detect_repeated_header_footer` should be called once for the
        whole document BEFORE calling this per page."""
        text = self._strip_header_footer(raw_text)
        text = self._remove_page_number_lines(text)
        text = self._fix_hyphenation(text)
        text = self._rejoin_soft_wraps(text)
        text = self._remove_ocr_artifacts(text)
        text = self._normalize_whitespace(text)
        return text.strip()

    def _strip_header_footer(self, text: str) -> str:
        if not self._detected_header and not self._detected_footer:
            return text
        lines = text.splitlines()
        kept = []
        for line in lines:
            normalized = self._normalize_for_comparison(line.strip())
            if normalized and normalized in (self._detected_header, self._detected_footer):
                continue
            kept.append(line)
        return "\n".join(kept)

    @staticmethod
    def _remove_page_number_lines(text: str) -> str:
        lines = [line for line in text.splitlines() if not _PAGE_NUMBER_LINE.match(line)]
        return "\n".join(lines)

    @staticmethod
    def _fix_hyphenation(text: str) -> str:
        return _HYPHEN_LINEBREAK.sub(r"\1\2", text)

    @staticmethod
    def _rejoin_soft_wraps(text: str) -> str:
        """Join a line into the next one when it looks like a mid-sentence
        wrap rather than an intentional paragraph break: the line does
        not end in sentence punctuation, is not blank, and the next line
        is not itself a bullet/numbered-list start or blank (both of
        which signal a genuine new structural element, not a continued
        sentence)."""
        lines = text.split("\n")
        joined: List[str] = []
        buffer = ""
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if buffer:
                    joined.append(buffer)
                    buffer = ""
                joined.append("")  # preserve paragraph break
                continue
            if _BULLET_OR_NUMBERED_START.match(stripped):
                if buffer:
                    joined.append(buffer)
                    buffer = ""
                joined.append(stripped)
                continue
            if buffer and not _SENTENCE_END.search(buffer):
                buffer = f"{buffer} {stripped}"
            else:
                if buffer:
                    joined.append(buffer)
                buffer = stripped
        if buffer:
            joined.append(buffer)
        return "\n".join(joined)

    @staticmethod
    def _remove_ocr_artifacts(text: str) -> str:
        text = _CONTROL_CHARS.sub("", text)
        text = _REPLACEMENT_CHAR_RUN.sub("", text)
        text = _STRAY_PUNCT_RUN.sub(r"\1", text)
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = _MULTI_SPACE.sub(" ", text)
        text = _MULTI_BLANK_LINES.sub("\n\n", text)
        return text
