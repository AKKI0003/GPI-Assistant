"""
ingestion/document.py

The single, uniform output object every ingestor in this package (and
every future ingestor -- Word, PPTX, HTML, ...) must return. Nothing in
this pipeline ever hands back a raw dict: every ingestor produces one
`Document`, so downstream code (retrieval merging, indexing, a future
graph-generation feature) can rely on one stable shape regardless of
which file format it originated from.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from .image import Image
from .metadata import DocumentMetadata
from .section import Chunk, Section
from .financial_facts import extract_financial_facts
from .table import Table
from .utils import detect_years_in_text

logger = logging.getLogger(__name__)


# Bump this any time _detect_statement_heading, _detect_statement_from_row_labels,
# or any other logic that determines a chunk's `section`/heading/text
# changes. retrieval/pdf_retriever.py's on-disk cache is keyed by file
# checksum only -- an unchanged PDF is normally skipped and never
# re-ingested, which silently means a fix made HERE has zero effect on
# already-ingested chunks until someone remembers to run `import_pdf_
# documents.py --force`. That gap is exactly what let a labeling bug
# stay "fixed" in the code while the live cache kept serving chunks
# produced by the old, buggy version indefinitely. Storing this version
# in the cache lets pdf_retriever detect a logic change and force a
# full re-ingestion automatically, the same way a changed file checksum
# would.
# v3: fixed two extraction bugs that made PDF financial facts (esp. for
#     GPI's statements) unreliable -- (1) table_extractor no longer
#     splits a single label|value table down its own wide label/number
#     gutter (which had divorced every figure from its line item), and
#     (2) financial_facts' blocklist no longer matches "ratio" inside
#     "operations" (which had silently dropped every "Revenue from
#     operations" row) and now ignores sensitivity-note "Impact on ..."
#     rows. Bumped so the live cache re-ingests with the corrected logic.
# v4: two more financial_facts extraction fixes -- (1) "Total Income" no
#     longer matches balance-sheet "Total income tax assets/liabilities"
#     rows (a negative lookahead), which had surfaced spurious ~300-3,000
#     lakh "Total Income" facts next to the real 442,528.72; and (2) a
#     malformed-row guard drops Statement-of-Changes-in-Equity rows whose
#     equity columns collapsed to dashes so a real amount landed inside
#     the label (e.g. "Profit for the year - - 104320.37 -"), which had
#     recorded a P&L figure against the equity statement's WRONG opening-
#     balance year (this is how 2025's profit showed up as 2024's PAT).
#     Bumped so the live cache re-ingests without those false/mis-tagged
#     facts.
INGESTION_LOGIC_VERSION = 4

_STATEMENT_HEADING_PATTERNS = [
    # Both word orders are common, interchangeable conventions across
    # Indian annual reports depending on the company/auditor's preferred
    # format -- e.g. GPI's report says "CASH FLOW STATEMENT" while VST's
    # says "STATEMENT OF CASH FLOWS". Matching only one order is a real
    # gap, not a difference in what actually needs supporting: both are
    # equally standard, so both need to resolve to the same canonical
    # label rather than one company's convention silently working and
    # the other's silently not.
    (re.compile(r"cash\s*flow\s*statement|statement\s*of\s*cash\s*flows?", re.IGNORECASE),
     "Cash Flow Statement"),
    (re.compile(r"statement\s*of\s*profit\s*and\s*loss|profit\s*(&|and)\s*loss\s*account",
                re.IGNORECASE), "Statement of Profit and Loss"),
    (re.compile(r"balance\s*sheet", re.IGNORECASE), "Balance Sheet"),
    (re.compile(r"statement\s*of\s*changes\s*in\s*equity", re.IGNORECASE),
     "Statement of Changes in Equity"),
]

_UNIT_OF_MEASURE_PATTERN = re.compile(
    r"(?:(?:\u20b9|rs\.?|inr|usd)\s*)?"
    r"(?:in\s+|amounts?\s+in\s+)?"
    r"(lakh|lakhs|crore|crores|million|millions|billion|billions|thousand|thousands)"
    r"(?:\s+of\s+rupees)?",
    re.IGNORECASE,
)


_HEADING_MODIFIER_WORDS = {
    "standalone", "consolidated", "unconsolidated", "audited", "unaudited",
    "annexure", "provisional", "reformatted", "restated",
}


def _looks_like_real_heading(top_slice: str, match: "re.Match") -> bool:
    """
    True if this regex match looks like an actual, standalone statement
    heading -- not just the SAME phrase appearing mid-note as a sub-label.

    Real statement page titles ("STATEMENT OF PROFIT AND LOSS", "BALANCE
    SHEET") sit alone on their own line, at the top of that statement's
    own page, with nothing else glued onto the same line -- except for a
    handful of completely standard modifier words that commonly prefix
    the real title itself, e.g. "STANDALONE STATEMENT OF PROFIT AND
    LOSS" / "CONSOLIDATED BALANCE SHEET". Those are still genuine
    headings, just with an extra word in front; they must not be
    rejected the same way an unrelated sentence would be.

    But the identical phrase also appears constantly INSIDE completely
    unrelated notes as a sub-classification label -- e.g. an Income Tax
    note listing which figures were "recognised in the Statement of
    profit and loss" versus "recognised in Other Comprehensive Income"
    is a totally standard tax note that could be sitting on a Balance
    Sheet note's page, not the P&L page itself. The tell-tale difference:
    that kind of reference is always followed immediately by a colon
    introducing what comes next ("Statement of profit and loss:",
    "Statement of profit and loss: Current tax ..."), whereas a genuine
    page heading is never followed by a colon -- the next thing on the
    page is the unit-of-measure line, a date range, or the table itself,
    never a colon-and-continuation on the same line.

    So a match only counts as a real heading if:
      1. It starts its own line (only whitespace, OR one of the known
         modifier words above, OR another recognized statement heading
         [the side-by-side case, e.g. "BALANCE SHEET   STATEMENT OF
         PROFIT AND LOSS" as two column headings sharing one line]
         appears before it on that line) -- rules out a heading phrase
         appearing mid-sentence, e.g. "...amounts recognised in the
         Statement of Profit and Loss during the year were...".
      2. It is NOT immediately followed by a colon (skipping only
         whitespace) -- rules out the tax-note-style sub-label case
         above, which is the single most common false positive.
    """
    line_start = top_slice.rfind("\n", 0, match.start()) + 1
    before_on_line = top_slice[line_start:match.start()].strip()
    if before_on_line:
        words = before_on_line.lower().replace(",", " ").split()
        is_known_modifier = bool(words) and all(w in _HEADING_MODIFIER_WORDS for w in words)
        is_another_heading = (
            len(before_on_line) <= 40
            and any(p.search(before_on_line) for p, _ in _STATEMENT_HEADING_PATTERNS)
        )
        if not (is_known_modifier or is_another_heading):
            return False

    after = top_slice[match.end():].lstrip(" \t")
    if after.startswith(":"):
        return False  # e.g. "Statement of profit and loss:" -- a sub-label, not a title

    return True


def _detect_statement_heading(page_text, table_bbox=None, page_width=None):
    """Return the canonical statement name ONLY when the page text gives
    an unambiguous signal that this specific table IS that statement --
    not merely a page that mentions it somewhere.

    Guards, added after this label was seen actively hurting retrieval
    instead of helping it:

    1. Only the first ~600 characters of the page are searched. A real
       statement heading sits at the top of its own page; further down,
       the SAME page frequently cross-references a DIFFERENT statement
       in a footnote ("refer to Note 34 for the Cash Flow Statement"),
       which previously caused a P&L page's table to be mislabeled
       "Cash Flow Statement" purely because that phrase appeared
       somewhere on the page.

    2. Some reports print two statements SIDE BY SIDE on one wide page
       (e.g. VST's annual report puts "BALANCE SHEET" and "STATEMENT OF
       PROFIT AND LOSS" as two column headings on the same landscape
       page, rather than GPI's convention of one statement per page).
       Plain page text extraction reads left-to-right, so both heading
       phrases appear in the top slice in the SAME left-to-right order
       as their columns. When `table_bbox` and `page_width` are given,
       whichever heading appears earlier in the text is assumed to
       belong to the left-hand table, and whichever appears later
       belongs to the right-hand table -- matched against which half of
       the page this specific table's own bbox sits in. This is a
       heuristic, not a proof, but it's a safe one: getting it wrong
       only costs the boost this label provides (same fallback as
       before), it can't actively corrupt data.

    3. If more than one distinct statement name matches AND we can't
       disambiguate by position (no bbox given, or the guess above still
       can't tell them apart), the page is treated as genuinely
       ambiguous -- return None rather than guess, so the caller falls
       back to the older, safer generic "Financial Statements" label.

    4. A match only counts at all if `_looks_like_real_heading` confirms
       it's a standalone page title, not the same phrase surfacing as a
       sub-label INSIDE some other, unrelated note (see that function's
       docstring for the tax-note example that motivated this guard --
       it was mislabeling Balance-Sheet notes as the Profit & Loss
       statement itself, purely because a tax note on that page happened
       to mention "Statement of profit and loss:" as a column label).
    """
    if not page_text:
        return None
    top_slice = page_text[:600]
    matches = []  # (canonical_name, character_offset_of_first_match)
    for pattern, canonical_name in _STATEMENT_HEADING_PATTERNS:
        m = pattern.search(top_slice)
        if m and _looks_like_real_heading(top_slice, m):
            matches.append((canonical_name, m.start()))

    distinct_names = {name for name, _ in matches}
    if len(distinct_names) == 1:
        return matches[0][0]
    if len(distinct_names) < 2:
        return None

    if table_bbox is not None and page_width:
        matches.sort(key=lambda pair: pair[1])  # left-to-right text order
        table_x0 = table_bbox[0]
        is_left_half = table_x0 < (page_width / 2)
        # First-appearing heading -> left column; last-appearing -> right
        # column. With exactly two distinct headings (the common case)
        # this cleanly resolves; with more than two, still take the
        # nearer end rather than giving up.
        return matches[0][0] if is_left_half else matches[-1][0]

    return None


# Same boilerplate keywords used in retrieval/pdf_retriever.py's
# always-on certification penalty -- kept in sync manually since the two
# modules serve different purposes here (this one TRIMS the boilerplate
# out of legitimate data chunks; that one penalizes chunks that are
# genuinely dominated by it).
_CERTIFICATION_MARKERS = (
    "as per our report of even date", "for and on behalf of the board",
    "chartered accountants", "firm registration number",
)


def _trim_trailing_certification_block(text: str) -> str:
    """Cut off a trailing director/auditor certification block from a
    table's text, WITHOUT discarding the real data that precedes it.

    Camelot's "stream" flavor in particular tends to capture an entire
    page as one table region, so a genuine Cash Flow Statement/P&L/
    Balance Sheet table frequently comes back with the page's trailing
    signature block ("As per our report of even date... Chief Financial
    Officer... (DIN 00048606)...") glued onto the end of the SAME chunk.
    A blunt "this chunk mentions certification keywords, penalize it"
    rule would wrongly demote that chunk even though the vast majority
    of it is exactly the data being asked for -- so this trims the
    boilerplate tail off instead, which both prevents the mislabeling
    and (as a side benefit) stops every statement table from wasting
    context-window budget on a name-and-signature block no question
    ever needs answered.

    Only trims when the earliest certification marker appears in the
    LATTER half of the text -- a chunk that's mostly boilerplate from
    the start (the standalone certification-only table camelot
    sometimes also extracts as its own separate region) is left alone
    here; that case is handled by the always-on penalty in
    pdf_retriever.py instead, since there's no real data to preserve.
    """
    lower = text.lower()
    earliest = min(
        (lower.find(m) for m in _CERTIFICATION_MARKERS if m in lower),
        default=-1,
    )
    if earliest > len(text) * 0.5:
        return text[:earliest].rstrip()
    return text


def _detect_unit_of_measure(page_text):
    if not page_text:
        return None
    for line in page_text.splitlines():
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) > 120:
            continue
        has_currency_cue = re.search(r"\u20b9|rs\.?\s|inr|rupees|usd", line_stripped, re.IGNORECASE)
        looks_like_unit_note = (
            "unless" in line_stripped.lower()
            or line_stripped.startswith("(")
            or bool(has_currency_cue)
        )
        if looks_like_unit_note and _UNIT_OF_MEASURE_PATTERN.search(line_stripped):
            return line_stripped
    return None


def _normalize_for_loose_match(text: str) -> str:
    """Collapse whitespace-between-every-letter artifacts ("P r o f i t")
    that some PDF fonts/encodings produce, and strip all whitespace, so a
    substring check still finds a phrase regardless of how pdfplumber
    happened to space out the glyphs on that particular page. Used only
    for the row-label content classifier below, where false positives
    are cheap to avoid (each phrase is distinctive) and false negatives
    from spacing artifacts were the actual observed failure mode."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


# Row-label phrases that only ever appear as actual line items WITHIN
# one specific statement -- as opposed to _STATEMENT_HEADING_PATTERNS,
# which look at the PAGE's heading/title text. This is a second,
# independent signal that classifies a table by what's actually IN it,
# which is immune to two real failure modes seen with heading-only
# detection: (1) a notes/schedule page's own heading not matching any
# statement pattern at all while still needing a page-level label carried
# over from context, and (2) PDF text extraction inserting stray spaces
# between individual letters for certain embedded fonts, which silently
# breaks a whole-word regex match on the heading but does not break a
# fully-normalized (whitespace-stripped) substring check like this one.
# Ordered most-specific-first; a table only needs 2 distinct hits from
# ONE statement's list (and zero from a competing one) to be classified.
_STATEMENT_ROW_LABEL_SETS = {
    "Cash Flow Statement": [
        "cashflowfromoperatingactivities", "cashflowfrominvestingactivities",
        "cashflowfromfinancingactivities", "netincreaseindecreaseincash",
        "cashandcashequivalentsattheendoftheyear",
        "cashandcashequivalentsatthebeginningoftheyear",
    ],
    "Statement of Profit and Loss": [
        "revenuefromoperations", "totalincome", "totalexpenses",
        "profitbeforetax", "profitfortheyear", "earningspershare",
        "taxexpense", "otherincome",
    ],
    "Balance Sheet": [
        "totalassets", "totalequityandliabilities", "noncurrentassets",
        "currentassets", "noncurrentliabilities", "currentliabilities",
        "propertyplantandequipment",
    ],
    "Statement of Changes in Equity": [
        "balanceatthebeginningofthereportingperiod",
        "balanceattheendofthereportingperiod", "othercomprehensiveincome",
        "totalcomprehensiveincomefortheyear",
    ],
}


def _detect_statement_from_row_labels(table_text: str) -> Optional[str]:
    """Classify a table by scanning ITS OWN text for row-label phrases
    that are distinctive to one specific statement. Returns the
    canonical label only when exactly one statement has 2+ distinct
    matches and every other statement has none -- a single incidental
    match (e.g. "other income" appearing as a line inside a note that
    isn't the P&L itself) isn't enough on its own, since some phrases are
    reused loosely across notes."""
    normalized = _normalize_for_loose_match(table_text)
    hits: Dict[str, int] = {}
    for label, phrases in _STATEMENT_ROW_LABEL_SETS.items():
        count = sum(1 for phrase in phrases if phrase in normalized)
        if count:
            hits[label] = count

    qualifying = [label for label, count in hits.items() if count >= 2]
    if len(qualifying) == 1:
        return qualifying[0]
    return None


@dataclass
class PageInfo:
    """Per-page bookkeeping, kept separately from Section so page
    numbers survive even for content a SectionDetector could not
    confidently assign to a heading."""

    page_number: int
    width: float = 0.0
    height: float = 0.0
    raw_text: str = ""
    cleaned_text: str = ""
    header: Optional[str] = None
    footer: Optional[str] = None

    @property
    def word_count(self) -> int:
        return len(self.cleaned_text.split()) if self.cleaned_text else len(self.raw_text.split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "header": self.header,
            "footer": self.footer,
            "word_count": self.word_count,
        }


@dataclass
class DocumentStatistics:
    """Aggregate, computed-once statistics about an ingested document.
    Kept as real fields (not re-derived on every access) so callers can
    cheaply log/compare ingestion runs across hundreds of files."""

    page_count: int = 0
    word_count: int = 0
    character_count: int = 0
    section_count: int = 0
    table_count: int = 0
    image_count: int = 0
    well_formed_table_count: int = 0
    average_words_per_page: float = 0.0
    extraction_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_count": self.page_count,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "section_count": self.section_count,
            "table_count": self.table_count,
            "image_count": self.image_count,
            "well_formed_table_count": self.well_formed_table_count,
            "average_words_per_page": self.average_words_per_page,
            "extraction_duration_seconds": self.extraction_duration_seconds,
        }


@dataclass
class Document:
    """The uniform ingestion output object.

    Every ingestor (`PDFIngestor` today; a future `DocxIngestor`,
    `PptxIngestor`, `HtmlIngestor`, etc.) must construct and return one
    of these. Nothing downstream should ever need to know which concrete
    ingestor produced it.
    """

    source_file: str
    document_type: str = "pdf"
    checksum: str = ""
    language: str = "unknown"

    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    pages: List[PageInfo] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    images: List[Image] = field(default_factory=list)

    raw_text: str = ""
    cleaned_text: str = ""
    tokens: List[str] = field(default_factory=list)

    statistics: DocumentStatistics = field(default_factory=DocumentStatistics)
    extraction_warnings: List[str] = field(default_factory=list)
    quality_score: float = 0.0

    ingested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # -- convenience accessors -------------------------------------------

    def iter_all_sections(self) -> Iterator[Section]:
        """Depth-first walk across every top-level section and all of
        its descendants."""
        for section in self.sections:
            yield from section.iter_all()

    def find_section(self, title: str, case_sensitive: bool = False) -> Optional[Section]:
        """Find the first section (at any depth) whose title matches."""
        target = title if case_sensitive else title.lower()
        for section in self.iter_all_sections():
            candidate = section.title if case_sensitive else section.title.lower()
            if candidate == target:
                return section
        return None

    def tables_on_page(self, page_number: int) -> List[Table]:
        return [table for table in self.tables if table.page_number == page_number]

    def images_on_page(self, page_number: int) -> List[Image]:
        return [image for image in self.images if image.page_number == page_number]

    def iter_chunks(self, min_chunk_words: int = 20) -> Iterator[Chunk]:
        """Yield every retrieval-ready Chunk across all top-level
        sections, PLUS one chunk per extracted table. This -- not raw
        pages, not fixed token windows -- is the unit that should be
        handed to an embedding/index step.

        Tables get their own chunk path (rather than being folded into
        section text) because `TableExtractor` already did the hard work
        of recovering real rows/columns from the PDF -- the one thing
        that actually contains the numbers someone asks about in a
        financial document. Without this, a P&L / balance sheet / cash
        flow table is extracted successfully and then silently never
        becomes searchable, since PyMuPDF's plain per-page text already
        went into `sections` as jumbled, column-less running text.
        """
        title = self.metadata.title or self.source_file
        company = self.metadata.company
        year = self.metadata.year
        for section in self.sections:
            for chunk in section.to_chunks(
                document_title=title,
                source=self.source_file,
                company=company,
                year=year,
                min_chunk_words=min_chunk_words,
            ):
                yield chunk

        page_text_by_number: Dict[int, str] = {
            p.page_number: (p.raw_text or "") for p in self.pages
        }
        page_width_by_number: Dict[int, float] = {
            p.page_number: p.width for p in self.pages
        }

        for table in self.tables:
            text = table.to_markdown()
            if not text.strip():
                continue
            text = _trim_trailing_certification_block(text)

            # `table.caption` is never populated by either the pdfplumber
            # or camelot extraction path (neither backend looks at the
            # text surrounding a table's own bounding box), so it is
            # always None in practice and the `if table.caption:` branch
            # below never fires. Two things routinely sit on the page
            # just above/around a financial-statement table that never
            # make it into `text` (which only ever holds the extracted
            # grid): (1) the actual statement heading ("Statement of
            # Profit and Loss", "Balance Sheet", "Cash Flow Statement"),
            # and (2) the unit-of-measure line ("(₹ in Lakhs)", "Rs. in
            # Crores", "(All amounts in INR Lakhs unless stated
            # otherwise)"). Without (2) in particular, nothing in the
            # context ever tells the model which unit the numbers are
            # actually in, which is exactly what let the LLM's own
            # (previously crore-biased) default guess decide the unit
            # instead of the source document. Recover both from the
            # table's own page text and fold them into this chunk's text
            # explicitly, so the unit and statement name travel with the
            # numbers wherever this chunk is retrieved.
            page_text = page_text_by_number.get(table.page_number, "")
            # Content-based classification (this table's OWN row labels)
            # is checked FIRST and takes priority over the page-heading
            # guess below: it's immune to page headers/footers, footnote
            # cross-references to a different statement, and PDF font
            # encodings that insert stray spaces between individual
            # letters (which silently breaks a whole-word heading regex
            # but not a whitespace-stripped substring check). Only fall
            # back to the heading-based guess when the table's own
            # content doesn't give a confident answer -- e.g. a
            # standalone data table with a heading above it but few of
            # its own distinctive row-label phrases.
            detected_heading = (
                _detect_statement_from_row_labels(text)
                or _detect_statement_heading(
                    page_text,
                    table_bbox=table.bounding_box,
                    page_width=page_width_by_number.get(table.page_number),
                )
            )
            detected_unit = _detect_unit_of_measure(page_text)
            prefix_lines = []
            if detected_heading:
                prefix_lines.append(f"### {detected_heading}")
            if detected_unit:
                prefix_lines.append(f"(Unit of measure: {detected_unit})")
            if prefix_lines:
                text = "\n".join(prefix_lines) + "\n\n" + text

            # Every table WITHOUT a detected caption was previously
            # defaulted straight to "Financial Statements" -- which
            # mislabeled every uncaptioned table in the document (CSR
            # spend annexures, outstanding-payments schedules, roster
            # tables, anything else pdfplumber pulled out without a
            # heading immediately above it) as if it were a genuine P&L/
            # balance-sheet table. That false label then earned those
            # CSR/administrative tables the financial-topic scoring
            # boost in retrieval/pdf_retriever.py, crowding out the
            # REAL financial-statement table for the same query. Only
            # tag a table "Financial Statements" when its own caption OR
            # content actually looks financial; otherwise fall back to a
            # neutral, non-boosted label.
            table_text_lower = text.lower()
            # CSR-spend annexures ("average net profit", "prescribed CSR
            # expenditure", "amount spent"...) legitimately contain the
            # word "profit" and rupee figures, so a plain keyword-hit
            # check alone still mislabels them as financial statements.
            # Checked first so it always wins over the generic financial
            # check below.
            is_csr_annexure = any(
                kw in table_text_lower
                for kw in (
                    "average net profit", "prescribed csr expenditure",
                    "unspent csr account", "csr committee",
                    "csr expenditure", "due date of payment",
                )
            )
            # Every financial statement in these reports ends with the
            # SAME director/auditor certification block ("As per our
            # report of even date", signatures, DIN numbers, "Chief
            # Financial Officer", "Chartered Accountants"...) on the
            # same physical page as the real table. Because
            # `detected_heading` is computed per-PAGE, this boilerplate
            # was inheriting the exact same canonical label ("Cash Flow
            # Statement", "Statement of Profit and Loss"...) as the real
            # data table sitting right above it -- making the two
            # completely indistinguishable to the `requested_statement`
            # +0.35 exact-section-match boost in retrieval/
            # pdf_retriever.py. Once both chunks share one label, that
            # boost can no longer tell them apart, and this content-free
            # signature block was winning often enough to be the actual
            # answer for "give me the cash flow statement" / "complete
            # P&L statement" -- explaining both the wrong-table answers
            # and the follow-up surfacing director/CFO names instead of
            # figures. Checked before `detected_heading` so it always
            # overrides the inherited page-level label.
            is_certification_block = sum(
                1 for kw in (
                    "as per our report of even date", "for and on behalf of the board",
                    "chartered accountants", "firm registration number",
                    "chief financial officer", "membership no",
                    "din ", "whole-time director",
                )
                if kw in table_text_lower
            ) >= 2
            if table.caption:
                section_label = table.caption
            elif is_csr_annexure:
                section_label = "CSR Disclosure"
            elif is_certification_block:
                section_label = "Statement Certification"
            elif detected_heading:
                # Use the SPECIFIC statement name recovered above rather
                # than a generic "Financial Statements" bucket. Lumping
                # the P&L, balance sheet, and cash flow statement under
                # one identical label made them indistinguishable to
                # retrieval/pdf_retriever.py's section-based scoring: a
                # query asking specifically for the cash flow statement
                # had no section-level signal to prefer the cash-flow
                # table chunk over the P&L/balance-sheet chunks sharing
                # the same label, so ranking fell back to raw semantic
                # similarity alone -- exactly where the cash flow table
                # (often extracted with lower confidence; see the
                # camelot-fallback notes above) could lose out and never
                # surface for a cash-flow-specific question.
                section_label = detected_heading
            elif any(
                kw in table_text_lower
                for kw in (
                    "revenue", "profit", "balance sheet", "cash flow",
                    "turnover", "ebitda", "income statement", "net worth",
                    "earnings per share", "dividend",
                )
            ):
                section_label = "Financial Statements"
            else:
                section_label = "Table"

            # Per-table year detection (see detect_years_in_text): a
            # table's own header/caption/rows are a far more reliable
            # signal of which fiscal year(s) it covers than the
            # document's single cover-page year, especially for the
            # 3-year comparative tables that make up most of an annual
            # report's actual financial statements.
            detected_years = detect_years_in_text(
                (table.caption or "") + " " + " ".join(table.header) + " " + text
            )
            if len(detected_years) == 1:
                table_year = detected_years[0]
            elif len(detected_years) > 1:
                table_year = None
            else:
                table_year = year

            yield Chunk(
                chunk_id=f"chunk-table-{table.table_id}",
                text=text,
                document_title=title,
                source=self.source_file,
                section=section_label,
                subsection=None,
                company=company,
                year=table_year,
                years_mentioned=detected_years,
                page=table.page_number,
                confidence=table.confidence,
                is_table=True,
                financial_facts=extract_financial_facts(text),
            )

        # Captioned images (charts, infographics, logos, photos) --
        # WITHOUT this, a vision-model caption is computed and stored on
        # the Image object but never becomes a retrievable unit at all,
        # so "describe the ITC logo" or "what does the revenue chart on
        # page 6 show" can never be answered no matter how good the
        # caption is. Only images that actually got a caption (either
        # from an offline `--caption-images` run, or none if that step
        # was skipped) produce a chunk here -- this adds zero ingestion
        # cost on its own, it just stops throwing away work already done.
        for image in self.images:
            if not image.caption:
                continue
            kind = "Logo" if image.is_likely_logo() else "Image"
            detected_years = detect_years_in_text(image.caption)
            yield Chunk(
                chunk_id=f"chunk-image-{image.image_id}",
                text=image.caption,
                document_title=title,
                source=self.source_file,
                section=kind,
                subsection=None,
                company=company,
                year=detected_years[0] if len(detected_years) == 1 else year,
                years_mentioned=detected_years,
                page=image.page_number,
                confidence=0.85,
                is_table=False,
                is_image=True,
            )

    def all_chunks(self, min_chunk_words: int = 20) -> List[Chunk]:
        return list(self.iter_chunks(min_chunk_words=min_chunk_words))

    def compute_quality_score(self) -> float:
        """A cheap, deterministic 0-1 quality signal used to flag
        documents worth a human's attention (e.g. a scanned/OCR-only PDF
        that yielded almost no extractable text, or a table-heavy report
        where extraction mostly failed). Not a substitute for manual QA
        on a genuinely important document, but very effective for
        triaging which of a few hundred ingested PDFs need a second
        look.
        """
        score = 1.0

        if self.statistics.page_count == 0:
            return 0.0

        # Penalize very low text yield per page (a common signature of a
        # scanned, image-only PDF that this pipeline cannot OCR).
        if self.statistics.average_words_per_page < 20:
            score -= 0.4
        elif self.statistics.average_words_per_page < 60:
            score -= 0.15

        # Penalize a low ratio of well-formed tables when tables exist at
        # all -- a document that is mostly malformed tables signals a
        # layout pdfplumber/camelot struggled with.
        if self.statistics.table_count > 0:
            well_formed_ratio = (
                self.statistics.well_formed_table_count / self.statistics.table_count
            )
            score -= (1 - well_formed_ratio) * 0.3

        # Penalize a document where no semantic sections were detected at
        # all -- everything downstream degrades to page-level chunking.
        if self.statistics.section_count == 0:
            score -= 0.3

        # Penalize accumulated extraction warnings, capped so a handful
        # of minor warnings don't dominate the score.
        warning_penalty = min(len(self.extraction_warnings) * 0.03, 0.2)
        score -= warning_penalty

        return round(max(0.0, min(1.0, score)), 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "document_type": self.document_type,
            "checksum": self.checksum,
            "language": self.language,
            "metadata": self.metadata.to_dict(),
            "pages": [page.to_dict() for page in self.pages],
            "sections": [section.to_dict() for section in self.sections],
            "tables": [table.to_dict() for table in self.tables],
            "images": [image.to_dict() for image in self.images],
            "statistics": self.statistics.to_dict(),
            "extraction_warnings": self.extraction_warnings,
            "quality_score": self.quality_score,
            "ingested_at": self.ingested_at,
        }