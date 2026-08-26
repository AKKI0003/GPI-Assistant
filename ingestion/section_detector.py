"""
ingestion/section_detector.py

Semantic section detection: chunk by MEANING, never by fixed page or
token boundaries.

The detector is driven by per-line "layout facts" (font size, whether
the line is bold, and the plain text) gathered per page by the caller
(`PDFIngestor`, using PyMuPDF's rich text-dict extraction when
available). From those facts it:

  1. Establishes the document's "body text" font size (the mode across
     all lines) -- anything meaningfully larger, or bold, is a heading
     candidate.
  2. Scores each heading candidate's hierarchy LEVEL from font size
     (bigger = shallower/higher-level) and from numbering depth (e.g.
     "3.2.1 ..." implies a deeper level than "3.2 ..." or "3 ...").
  3. Boosts confidence when the heading text matches the known
     financial/corporate section vocabulary (see `section.py`), without
     ever requiring a vocabulary match -- purely numeric/typographic
     headings ("14.2") are still detected.
  4. Builds a proper tree: a new heading at level N closes out (attaches
     accumulated body text to) the most recent open heading at level N,
     and becomes a child of the most recent still-open heading at level
     N-1.
  5. Falls back, when no rich per-line font data is available at all
     (the pdfplumber-only degraded path), to a plain regex/ALL-CAPS
     heuristic over page text -- still meaning-based, just lower
     confidence.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .section import KNOWN_SECTION_VOCABULARY, Section, detect_bullets, detect_numbered_list

logger = logging.getLogger(__name__)

_NUMBERING_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+){0,5})[\.\)]?\s+(\S.*)$")
_ALL_CAPS_HEADING = re.compile(r"^[A-Z][A-Z0-9 &\-/,]{3,80}$")
_TITLE_CASE_HEADING = re.compile(r"^([A-Z][a-zA-Z']*\s?){2,8}$")


@dataclass
class LineLayout:
    """One line of text plus the layout facts needed to judge whether it
    is a heading. `page_number` is 1-indexed to match the rest of the
    pipeline."""

    text: str
    page_number: int
    font_size: float = 0.0
    is_bold: bool = False


class SectionDetector:
    """Builds a tree of `Section` objects from a document's per-line
    layout data (preferred, font-aware path) or from plain page text
    (degraded fallback path when no font metadata is available)."""

    def __init__(
        self,
        heading_size_ratio: float = 1.15,
        max_heading_words: int = 18,
    ) -> None:
        # A line must be at least this many times larger than the body
        # text's modal font size to be considered a heading candidate on
        # size alone.
        self.heading_size_ratio = heading_size_ratio
        self.max_heading_words = max_heading_words

    # -- preferred, font-aware path ---------------------------------------

    def detect_from_layout(self, lines: List[LineLayout]) -> List[Section]:
        if not lines:
            return []

        body_font_size = self._estimate_body_font_size(lines)
        heading_threshold = body_font_size * self.heading_size_ratio

        roots: List[Section] = []
        # stack of (level, Section) for still-open ancestors, shallowest first
        open_stack: List[Tuple[int, Section]] = []
        current_leaf: Optional[Section] = None
        buffer_lines: List[str] = []
        buffer_start_page: Optional[int] = None

        def flush_buffer(end_page: Optional[int]) -> None:
            nonlocal buffer_lines, buffer_start_page
            if current_leaf is None or not buffer_lines:
                buffer_lines = []
                return
            text = "\n".join(buffer_lines)
            current_leaf.text = (current_leaf.text + "\n" + text).strip() if current_leaf.text else text
            current_leaf.bullet_points.extend(detect_bullets(text))
            current_leaf.numbered_points.extend(detect_numbered_list(text))
            if current_leaf.page_end is None or (end_page and end_page > current_leaf.page_end):
                current_leaf.page_end = end_page
            buffer_lines = []
            buffer_start_page = None

        for line in lines:
            is_heading, level, confidence = self._classify_line(
                line, body_font_size, heading_threshold
            )
            if is_heading:
                flush_buffer(line.page_number)
                new_section = Section(
                    title=line.text.strip(),
                    level=level,
                    page_start=line.page_number,
                    page_end=line.page_number,
                    confidence=confidence,
                )
                # Pop any open sections at this level or deeper -- a new
                # heading at level N always ends a previous heading at
                # level >= N.
                while open_stack and open_stack[-1][0] >= level:
                    open_stack.pop()
                if open_stack:
                    open_stack[-1][1].add_subsection(new_section)
                else:
                    new_section.level = 0
                    roots.append(new_section)
                open_stack.append((new_section.level, new_section))
                current_leaf = new_section
            else:
                if buffer_start_page is None:
                    buffer_start_page = line.page_number
                buffer_lines.append(line.text)

        flush_buffer(lines[-1].page_number if lines else None)
        return roots

    def _estimate_body_font_size(self, lines: List[LineLayout]) -> float:
        sizes = [round(line.font_size, 1) for line in lines if line.font_size > 0]
        if not sizes:
            return 10.0  # sane default when no font metadata exists at all
        return Counter(sizes).most_common(1)[0][0]

    def _classify_line(
        self,
        line: LineLayout,
        body_font_size: float,
        heading_threshold: float,
    ) -> Tuple[bool, int, float]:
        text = line.text.strip()
        if not text or len(text.split()) > self.max_heading_words:
            return False, 0, 0.0

        numbering_match = _NUMBERING_PATTERN.match(text)
        numbering_depth = numbering_match.group(1).count(".") if numbering_match else None

        looks_like_heading_typography = (
            line.font_size >= heading_threshold or line.is_bold
        )
        looks_like_heading_text = bool(numbering_match) or bool(_ALL_CAPS_HEADING.match(text))

        if not (looks_like_heading_typography or looks_like_heading_text):
            return False, 0, 0.0

        # Determine hierarchy level: numbering depth wins when present
        # (most reliable signal), otherwise derive a coarse level purely
        # from how much larger than body text the font is.
        if numbering_depth is not None:
            level = numbering_depth
        elif line.font_size >= heading_threshold * 1.6:
            level = 0
        elif line.font_size >= heading_threshold * 1.3:
            level = 1
        else:
            level = 2

        confidence = 0.6
        if numbering_match:
            confidence += 0.15
        if line.is_bold:
            confidence += 0.1
        clean_title = numbering_match.group(2) if numbering_match else text
        if self._matches_known_vocabulary(clean_title):
            confidence += 0.15
        confidence = min(confidence, 1.0)

        return True, level, confidence

    @staticmethod
    def _matches_known_vocabulary(title: str) -> bool:
        normalized = title.strip().lower()
        return any(normalized == known.lower() or normalized in known.lower()
                   or known.lower() in normalized for known in KNOWN_SECTION_VOCABULARY)

    # -- degraded fallback path (no font metadata available) --------------

    def detect_from_plain_text(self, pages_text: List[str]) -> List[Section]:
        """Fallback used when the underlying extraction backend could not
        supply per-line font metadata (e.g. pdfplumber-only mode without
        PyMuPDF installed). Headings are guessed from ALL-CAPS lines,
        numbered-heading patterns, and short Title Case lines standing
        alone on their own line -- strictly weaker than the font-aware
        path, so confidence scores are capped lower throughout."""
        lines: List[LineLayout] = []
        for page_number, page_text in enumerate(pages_text, start=1):
            for raw_line in page_text.splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    continue
                lines.append(LineLayout(text=stripped, page_number=page_number))
        return self._detect_from_plain_lines(lines)

    def _detect_from_plain_lines(self, lines: List[LineLayout]) -> List[Section]:
        roots: List[Section] = []
        open_stack: List[Tuple[int, Section]] = []
        current_leaf: Optional[Section] = None
        buffer_lines: List[str] = []

        def flush_buffer(end_page: Optional[int]) -> None:
            nonlocal buffer_lines
            if current_leaf is None or not buffer_lines:
                buffer_lines = []
                return
            text = "\n".join(buffer_lines)
            current_leaf.text = (current_leaf.text + "\n" + text).strip() if current_leaf.text else text
            current_leaf.bullet_points.extend(detect_bullets(text))
            current_leaf.numbered_points.extend(detect_numbered_list(text))
            if current_leaf.page_end is None or (end_page and end_page > current_leaf.page_end):
                current_leaf.page_end = end_page
            buffer_lines = []

        for line in lines:
            text = line.text
            too_long = len(text.split()) > self.max_heading_words
            numbering_match = None if too_long else _NUMBERING_PATTERN.match(text)
            is_all_caps = (not too_long) and bool(_ALL_CAPS_HEADING.match(text))
            is_heading = bool(numbering_match) or is_all_caps

            if is_heading:
                flush_buffer(line.page_number)
                if numbering_match:
                    level = numbering_match.group(1).count(".")
                    title = numbering_match.group(2).strip()
                else:
                    level = 0
                    title = text.strip()
                confidence = 0.45 if is_all_caps and not numbering_match else 0.55
                if self._matches_known_vocabulary(title):
                    confidence += 0.15
                new_section = Section(
                    title=title,
                    level=level,
                    page_start=line.page_number,
                    page_end=line.page_number,
                    confidence=min(confidence, 1.0),
                )
                while open_stack and open_stack[-1][0] >= level:
                    open_stack.pop()
                if open_stack:
                    open_stack[-1][1].add_subsection(new_section)
                else:
                    new_section.level = 0
                    roots.append(new_section)
                open_stack.append((new_section.level, new_section))
                current_leaf = new_section
            else:
                buffer_lines.append(text)

        flush_buffer(lines[-1].page_number if lines else None)

        if not roots:
            logger.warning(
                "SectionDetector: no headings detected via plain-text fallback; "
                "caller should treat this document as unsectioned (page-level only)."
            )
        return roots
