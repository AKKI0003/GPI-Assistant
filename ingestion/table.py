"""
ingestion/table.py

Structured table model. Financial documents are mostly tables (balance
sheets, P&L statements, cash-flow statements), so tables are NEVER
flattened into prose here -- rows, columns, header, page number,
bounding box, and an extraction-confidence score are all preserved as
first-class structured data. This is required for two reasons:

1. A flattened "Revenue was 100, 120, 140 in 2022, 2023, 2024" sentence
   is lossy and ambiguous the moment a table has more than one or two
   columns (which years map to which values? which row is which line
   item?).
2. Any future feature that renders a chart or re-derives a computed
   metric (e.g. YoY growth) needs real rows/columns to compute from, not
   a paraphrased sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

BoundingBox = Tuple[float, float, float, float]  # (x0, top, x1, bottom)


@dataclass
class Table:
    """A single extracted table."""

    table_id: str
    page_number: int
    rows: List[List[str]] = field(default_factory=list)
    header: List[str] = field(default_factory=list)
    bounding_box: Optional[BoundingBox] = None
    confidence: float = 0.0
    extraction_method: str = "unknown"   # "pdfplumber" | "camelot" | "fallback" | "text_fallback"
    caption: Optional[str] = None
    # Set ONLY when structured row/column extraction lost the table's
    # label column entirely (see table_extractor.py's
    # `_label_column_missing` check) and even a text-alignment retry
    # couldn't recover it. In that case plain, reading-order text
    # extraction of the same region is used instead -- e.g. a genuine
    # Statement of Profit and Loss otherwise turns into unlabeled rows
    # like "|  |  | 32698.28 |  |" with no way to tell that number is
    # Net Profit, whereas the plain-text version reads
    # "VII Profit for the year (V - VI) 32698.28 32023.11" -- not a
    # clean grid, but every figure stays attached to its label, which is
    # what actually matters for retrieval/answering.
    raw_text_override: Optional[str] = None

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        if self.header:
            return len(self.header)
        return max((len(row) for row in self.rows), default=0)

    def is_well_formed(self) -> bool:
        """A table is 'well formed' if every row has the same column
        count as the header (or, absent a header, as the first row).
        Used to sanity-check extraction quality and feed into the
        document-level quality score."""
        if not self.rows:
            return False
        expected = len(self.header) if self.header else len(self.rows[0])
        if expected == 0:
            return False
        return all(len(row) == expected for row in self.rows)

    def to_records(self) -> List[dict]:
        """Return the table as a list of {header: value} dicts, one per
        row -- convenient for callers (e.g. a future chart generator)
        that want named-column access rather than raw index access."""
        if not self.header:
            return [{"col_%d" % i: cell for i, cell in enumerate(row)} for row in self.rows]
        records = []
        for row in self.rows:
            record = {}
            for i, cell in enumerate(row):
                key = self.header[i] if i < len(self.header) else f"col_{i}"
                record[key] = cell
            records.append(record)
        return records

    def to_markdown(self) -> str:
        """Render this table as a markdown table, preserving row/column
        structure -- this (not a flattened sentence) is what should be
        embedded/indexed, so a query about a specific line item and year
        can actually be answered from a real cell value rather than the
        model guessing from prose. Includes the caption as a heading
        line when available, since that's often the only place a table's
        subject ("Operating Performance", "Balance Sheet") is stated.

        If this table's label column was lost during structured
        extraction (see `raw_text_override`), plain reading-order text
        is returned instead -- an unlabeled grid of numbers is worse
        than useless for answering a question about a specific line
        item, since there's no way to tell which figure is which."""
        if self.raw_text_override:
            lines: List[str] = []
            if self.caption:
                lines.append(f"### {self.caption}")
            lines.append(self.raw_text_override)
            return "\n".join(lines)

        lines: List[str] = []
        if self.caption:
            lines.append(f"### {self.caption}")

        # pdfplumber's ruling-line detection routinely invents extra,
        # entirely-empty columns on unruled/lightly-ruled financial
        # tables (a faint gridline or column guide gets treated as a
        # real column boundary). Left in, these show up as a wall of
        # "|  |  |" gaps between the real label/number cells -- which is
        # exactly what made a balance sheet render with "most columns
        # empty" even though the real data was fully extracted. Drop any
        # column index that is blank across the header AND every row
        # before rendering, then re-render every row/header with that
        # column removed so everything still lines up.
        all_rows_for_scan = ([self.header] if self.header else []) + self.rows
        num_cols = max((len(r) for r in all_rows_for_scan), default=0)
        if num_cols and all_rows_for_scan:
            keep_cols = [
                i for i in range(num_cols)
                if any(
                    (row[i] if i < len(row) else "").strip()
                    for row in all_rows_for_scan
                )
            ]
        else:
            keep_cols = list(range(num_cols))

        def _project(row: List[str]) -> List[str]:
            return [str(row[i]) if i < len(row) and row[i] is not None else "" for i in keep_cols]

        header = _project(self.header) if self.header else []
        if header:
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in self.rows:
            cells = _project(row)
            if not any(cell.strip() for cell in cells):
                # A row with every cell blank carries no information --
                # pdfplumber/camelot both routinely emit these as visual
                # spacer rows that existed in the PDF's layout but were
                # never real data. Rendering them verbatim is what
                # produced ugly runs of "|  |  |  |" in answers; skipping
                # them loses nothing since there's no cell value here for
                # any question to actually reference.
                continue
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "page_number": self.page_number,
            "header": self.header,
            "rows": self.rows,
            "bounding_box": self.bounding_box,
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
            "caption": self.caption,
            "raw_text_override": self.raw_text_override,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
        }