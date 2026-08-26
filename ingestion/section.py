"""
ingestion/section.py

Semantic section model. This is the single most important data
structure in the ingestion pipeline: retrieval quality for financial /
annual-report documents depends on chunking by MEANING (e.g. "Cash Flow"
as its own retrievable unit), not by raw page boundaries or fixed token
windows -- both of those routinely split a table or an argument in half
mid-thought.

A `Section` is a node in a tree: a top-level heading (e.g. "Financial
Statements") owns a list of child `Section`s (e.g. "Revenue", "Profit",
"Cash Flow", "Balance Sheet"), each of which may itself have further
children. Leaf sections carry the actual extracted text plus any
detected bullet/numbered list items.

`Chunk` is the flattened, retrieval-ready unit produced from a Section
(or a leaf subsection) -- this is what actually gets embedded / indexed,
carrying enough metadata (company, year, section, subsection, page,
document title, source, confidence) that a retriever never has to go
back to the original PDF to answer "where did this come from".
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from .utils import detect_years_in_text
from .financial_facts import extract_financial_facts

_CHUNK_ID_COUNTER = itertools.count(1)

# Recognized top-level heading vocabulary for financial / corporate
# documents. Used only as a *scoring boost* inside SectionDetector, never
# as a hard requirement -- a heading that doesn't match any of these is
# still detected purely from font-size/numbering heuristics.
KNOWN_SECTION_VOCABULARY: List[str] = [
    "Corporate Overview", "History", "Leadership", "Board of Directors",
    "Management Discussion", "Products", "Business Segments",
    "Financial Statements", "Financial Highlights", "Revenue", "Profit",
    "Profit and Loss", "Cash Flow", "Balance Sheet", "Notes to Accounts",
    "Auditor's Report", "Risk Factors", "Corporate Governance",
    "ESG", "Environmental Social Governance", "CSR",
    "Corporate Social Responsibility", "Shareholder Information",
    "Directors' Report", "Chairman's Statement", "Awards and Recognition",
    "Human Resources", "Sustainability", "Subsidiaries",
]

_BULLET_PATTERN = re.compile(r"^\s*[\u2022\u25CF\u25A0\-\*\u2013]\s+(.*)$")
_NUMBERED_PATTERN = re.compile(r"^\s*(\d{1,3})[\.\)]\s+(.*)$")


@dataclass
class Chunk:
    """A single retrieval-ready unit of text with full provenance."""

    chunk_id: str
    text: str
    document_title: str
    source: str
    section: str
    subsection: Optional[str] = None
    company: Optional[str] = None
    year: Optional[int] = None
    years_mentioned: List[int] = field(default_factory=list)
    page: Optional[int] = None
    confidence: float = 1.0
    is_table: bool = False
    is_image: bool = False
    financial_facts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "document_title": self.document_title,
            "source": self.source,
            "section": self.section,
            "subsection": self.subsection,
            "company": self.company,
            "year": self.year,
            "years_mentioned": self.years_mentioned,
            "page": self.page,
            "confidence": self.confidence,
            "is_table": self.is_table,
            "is_image": self.is_image,
            "financial_facts": [
                {"metric": f.metric, "year": f.year, "value": f.value,
                 "raw_label": f.raw_label, "raw_value": f.raw_value}
                for f in self.financial_facts
            ],
        }


@dataclass
class Section:
    """A node in the semantic section tree."""

    title: str
    level: int = 0                       # 0 = top-level chapter/heading
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    text: str = ""
    bullet_points: List[str] = field(default_factory=list)
    numbered_points: List[str] = field(default_factory=list)
    subsections: List["Section"] = field(default_factory=list)
    parent_title: Optional[str] = None
    confidence: float = 1.0

    # -- tree helpers ---------------------------------------------------

    def add_subsection(self, section: "Section") -> None:
        section.parent_title = self.title
        section.level = self.level + 1
        self.subsections.append(section)

    def iter_all(self) -> Iterator["Section"]:
        """Depth-first traversal of this section and every descendant."""
        yield self
        for child in self.subsections:
            yield from child.iter_all()

    def is_leaf(self) -> bool:
        return len(self.subsections) == 0

    def full_text(self) -> str:
        """This section's own text plus every descendant's text, in
        document order -- useful for a "give me everything under Risk
        Factors" style retrieval."""
        parts = [self.text] if self.text else []
        for child in self.subsections:
            child_text = child.full_text()
            if child_text:
                parts.append(child_text)
        return "\n\n".join(parts)

    def word_count(self) -> int:
        return len(self.full_text().split())

    # -- chunk production -------------------------------------------------

    def to_chunks(
        self,
        document_title: str,
        source: str,
        company: Optional[str] = None,
        year: Optional[int] = None,
        min_chunk_words: int = 20,
    ) -> List[Chunk]:
        """Flatten this section (and its leaf descendants) into
        retrieval-ready Chunks. Only leaf sections (or sections with text
        of their own, even if they also have children) produce a chunk --
        purely structural parent headings with no body text of their own
        do not, since an empty chunk carries no retrieval value."""
        chunks: List[Chunk] = []
        for node in self.iter_all():
            body = node.text.strip()
            if not body:
                continue
            # A leaf section whose own title (or the parent it just
            # closed out) matches a recognized heading is kept regardless
            # of word count. This matters most for roster-style pages --
            # "Board of Directors", "Management Team", "Leadership" --
            # which are mostly headshot photos with only a name and a
            # title as extractable text per person, routinely well under
            # any generic word-count floor despite being exactly the kind
            # of content someone will directly ask for by name. The
            # min_chunk_words floor still applies to everything else, to
            # keep out genuinely empty/noise fragments (stray page
            # numbers, running headers that slipped through).
            title_to_check = node.parent_title or node.title
            is_recognized_heading = any(
                known.lower() in title_to_check.lower() for known in KNOWN_SECTION_VOCABULARY
            )
            if not is_recognized_heading and len(body.split()) < min_chunk_words:
                continue
            parent = node.parent_title
            detected = detect_years_in_text(body)
            # Prefer years actually present in THIS chunk's own text over
            # the document-level year. A chunk that mentions exactly one
            # year is confidently tagged with it. A chunk that mentions
            # several (a 3-year comparative table, a CSR annexure listing
            # unrelated fiscal years) is NOT collapsed onto a single
            # `year` -- doing that is what caused 2024 figures to be
            # mislabeled as 2022 previously. `years_mentioned` always
            # carries the full list so callers can filter/match against
            # any of them. Only when the chunk mentions no year at all do
            # we fall back to the document's own cover-page year.
            if len(detected) == 1:
                chunk_year = detected[0]
            elif len(detected) > 1:
                chunk_year = None
            else:
                chunk_year = year
            chunks.append(
                Chunk(
                    chunk_id=f"chunk-{next(_CHUNK_ID_COUNTER):08d}",
                    text=body,
                    document_title=document_title,
                    source=source,
                    section=parent or node.title,
                    subsection=node.title if parent else None,
                    company=company,
                    year=chunk_year,
                    years_mentioned=detected,
                    page=node.page_start,
                    confidence=node.confidence,
                    # Many genuinely tabular notes (an EPS note, a small
                    # ratio-analysis table, ...) don't get picked up by
                    # pdfplumber's ruled-table detector at all -- they
                    # have no visible grid lines for it to find -- and
                    # end up chunked here, as plain narrative text,
                    # instead of through the dedicated table path in
                    # document.py (the only place this used to run).
                    # That meant a real, cleanly-labeled line item like
                    # page 104's "Profit for the year 29039.49 30156.52"
                    # never produced a financial fact at all, even
                    # though the exact same parser (already built to
                    # handle this plain "label + trailing numbers"
                    # shape) would have caught it immediately if it ever
                    # got the chance to look. Cheap and conservative --
                    # returns nothing for genuine prose -- so running it
                    # on every chunk, not just detected tables, is safe.
                    financial_facts=extract_financial_facts(body),
                )
            )
        return chunks

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "level": self.level,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
            "bullet_points": self.bullet_points,
            "numbered_points": self.numbered_points,
            "confidence": self.confidence,
            "subsections": [child.to_dict() for child in self.subsections],
        }


def detect_bullets(text: str) -> List[str]:
    """Extract bullet-list items from a block of text (one item per
    matching line). Returns the item text with the bullet glyph
    stripped."""
    items: List[str] = []
    for line in text.splitlines():
        match = _BULLET_PATTERN.match(line)
        if match:
            items.append(match.group(1).strip())
    return items


def detect_numbered_list(text: str) -> List[str]:
    """Extract numbered-list items (`1. ...`, `2) ...`) from a block of
    text, preserving the original number as a prefix so ordering is never
    lost even if items are later re-sorted."""
    items: List[str] = []
    for line in text.splitlines():
        match = _NUMBERED_PATTERN.match(line)
        if match:
            items.append(f"{match.group(1)}. {match.group(2).strip()}")
    return items