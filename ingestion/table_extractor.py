"""
ingestion/table_extractor.py

Table extraction, kept structured end-to-end (rows/columns/header/page/
bounding-box/confidence) -- never flattened into prose. See table.py's
module docstring for why this matters for financial documents
specifically.

Primary backend: pdfplumber, which gives reliable row/column structure
plus bounding boxes for most native (non-scanned) financial-report
tables. Secondary/optional backend: camelot, which is attempted only for
pages where pdfplumber found nothing at all, since camelot is
noticeably slower per page (it rasterizes / runs its own line-detection
pass) and there is no benefit re-running it on pages that already
extracted cleanly.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import logging
import os
import re
from typing import List, Optional

from .table import Table

logger = logging.getLogger(__name__)

_TABLE_ID_COUNTER = itertools.count(1)

# Below this many real (non-empty) data cells, a page's pdfplumber-
# detected tables are treated as not worth keeping if the page itself
# looks like a genuine financial statement -- see `extract()`.
_MIN_USABLE_CELLS = 6

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not installed; table extraction will be unavailable.")

try:
    import camelot
    _CAMELOT_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _CAMELOT_AVAILABLE = False
    logger.info("camelot not installed; falling back to pdfplumber-only table extraction.")

# How long to wait for ONE camelot.read_pdf() batch (covering all pages
# routed to the fallback for a single source file) before giving up on
# it -- see _camelot_fallback's docstring/comments for why this exists
# at all. Override with CAMELOT_FALLBACK_TIMEOUT_SECONDS if a
# particularly large/complex document set genuinely needs longer.
CAMELOT_FALLBACK_TIMEOUT_SECONDS = int(os.getenv("CAMELOT_FALLBACK_TIMEOUT_SECONDS", "120"))


class TableExtractor:
    """Extracts every table in a PDF as a list of structured `Table`
    objects, one per detected table (a single page may contain zero,
    one, or several)."""

    def __init__(self, use_camelot_fallback: bool = True) -> None:
        self.use_camelot_fallback = use_camelot_fallback and _CAMELOT_AVAILABLE

    def extract(self, file_path: str) -> List[Table]:
        if not _PDFPLUMBER_AVAILABLE:
            logger.error("Cannot extract tables: pdfplumber is not installed.")
            return []

        tables: List[Table] = []
        pages_needing_fallback: List[int] = []

        with pdfplumber.open(file_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                page_tables = self._extract_page_tables(page, page_index)
                # A page can come back with tables that are technically
                # non-empty (find_tables() detected *something*) but
                # functionally useless -- e.g. one detected "table" that
                # is only the header row, plus another that's a narrow,
                # spurious vertical sliver of entirely blank cells caught
                # from an unrelated ruling line. This happened on a real annual report's
                # own Cash Flow Statement page: pdfplumber's line-based
                # detection fragmented the real (unruled) table into
                # near-empty garbage instead of ever finding its true
                # extent, so the earlier "only fall back when a page
                # produced ZERO tables" check let this page slip through
                # with its real data never recovered. Checking the
                # page's own text for financial-statement signals AND
                # how little real content the detected tables actually
                # hold catches this "technically non-empty but useless"
                # case too.
                usable = self._usable_content_score(page_tables)
                looks_financial = self._looks_like_financial_statement(page)
                if not page_tables:
                    pages_needing_fallback.append(page_index)
                elif usable < _MIN_USABLE_CELLS and looks_financial:
                    # Try text strategy first for unruled financial tables
                    # before falling back to slower camelot
                    try:
                        text_strategy_tables = self._extract_page_tables_text_strategy(page, page_index)
                        if self._usable_content_score(text_strategy_tables) > usable:
                            page_tables = text_strategy_tables
                            usable = self._usable_content_score(page_tables)
                    except Exception:
                        pass
                    if usable < _MIN_USABLE_CELLS:
                        pages_needing_fallback.append(page_index)
                        # Don't keep the near-empty/garbage tables alongside
                        # whatever the fallback below finds -- they'd just
                        # add noise (e.g. a lone header row with no figures)
                        # without adding any real information.
                else:
                    tables.extend(page_tables)

        if self.use_camelot_fallback and pages_needing_fallback:
            tables.extend(self._camelot_fallback(file_path, pages_needing_fallback))

        return tables

    _FINANCIAL_STATEMENT_HEADINGS = (
        "cash flow statement", "statement of profit and loss",
        "balance sheet", "statement of changes in equity",
        "profit and loss account",
    )

    @classmethod
    def _looks_like_financial_statement(cls, page) -> bool:
        """Cheap, page-level check: does this page's own text look like
        a real financial-statement page (by heading and by having a lot
        of currency-shaped numbers), regardless of whether pdfplumber's
        table detection actually found it. Used only to decide whether a
        page with junk/near-empty detected tables deserves the extra
        camelot-fallback attempt below -- an ordinary prose page with a
        junky one-off table detection isn't worth the extra processing
        time."""
        try:
            text = (page.extract_text() or "").lower()
        except Exception:
            return False
        if any(h in text for h in cls._FINANCIAL_STATEMENT_HEADINGS):
            return True
        # Fallback signal: a page dense with multi-digit, comma/decimal
        # formatted figures (typical of a financial table) even without
        # a recognized heading on this particular page (continuation
        # pages of a statement often don't repeat the heading).
        number_hits = len(re.findall(r"\d{2,3}(?:,\d{3})*\.\d{2}", text))
        return number_hits >= 8

    @staticmethod
    def _usable_content_score(tables: List[Table]) -> int:
        """Counts real, non-empty data cells across a page's detected
        tables (using the recovered raw_text_override length as a proxy
        when that fallback already kicked in). A page whose only
        "tables" are a bare header row and an empty sliver scores
        ~0-2, versus dozens for a genuinely-captured statement.

        A table can also score deceptively HIGH here while still being
        useless: pdfplumber's "lines" strategy can return a bounding
        box too narrow to include the label column at all (common on
        unruled financial-statement tables where the real label column
        sits well to the left of the box pdfplumber inferred). Both
        per-table fallbacks above operate WITHIN that same wrong box,
        so they can recover a garbled structure but never the missing
        labels -- the result is a table full of real-looking digits
        with zero row labels (e.g. "30.00 | 233868.00" with no
        "Adjustments for depreciation" etc. next to it), which is not
        meaningfully answerable even though it's technically non-empty.
        Such a table is scored as if it were nearly empty, so the page
        still qualifies for the camelot fallback below instead of
        silently keeping an unusable, unlabeled number dump.
        """
        total = 0
        for t in tables:
            if t.raw_text_override:
                # Already recovered via the per-table fallback -- treat
                # as sufficiently usable regardless of row/col cell count.
                total += 50
                continue
            rows_with_header = ([t.header] if t.header else []) + t.rows
            if TableExtractor._label_column_missing(rows_with_header):
                total += 1
                continue
            for row in t.rows:
                total += sum(1 for c in row if c and c.strip())
        return total

    # -- pdfplumber path --------------------------------------------------

    def _extract_page_tables(self, page, page_index: int) -> List[Table]:
        results: List[Table] = []
        try:
            found = page.find_tables()
        except Exception as exc:  # pdfplumber can raise on malformed pages
            logger.warning("pdfplumber failed to find tables on page %d: %s", page_index, exc)
            return results

        # If no tables found but page looks like a financial statement,
        # try the "text" strategy (whitespace-based) which works for unruled tables
        if not found and self._looks_like_financial_statement(page):
            try:
                found = page.find_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})
            except Exception as exc:
                logger.warning("pdfplumber text strategy failed on page %d: %s", page_index, exc)
                found = []

        for raw_table in found:
            try:
                extracted_rows = raw_table.extract()
            except Exception as exc:
                logger.warning("pdfplumber failed to extract a table on page %d: %s", page_index, exc)
                continue
            if not extracted_rows:
                continue

            cleaned_rows = [
                [cell.strip() if isinstance(cell, str) else (cell or "") for cell in row]
                for row in extracted_rows
            ]

            # ── Recover dropped row-label columns ──────────────────────
            # pdfplumber's DEFAULT table detection ("lines" strategy)
            # infers column boundaries from visible ruling lines. Many
            # financial-statement tables (Statement of Profit and Loss,
            # Balance Sheet, Cash Flow) have NO vertical rule between the
            # left-hand "Particulars" label column and the right-hand
            # number columns -- labels are just left-aligned text next to
            # right-aligned figures, with no drawn line separating them.
            # When that happens, the "lines" strategy can miss the label
            # column's boundary entirely and the whole column comes back
            # BLANK on every row -- e.g. a real P&L table turning into
            # rows like "|  |  | 32698.28 |  |" with the "Profit for the
            # Year" label silently gone, leaving an unlabeled number an
            # LLM has no way to identify. This previously caused a
            # correctly-extracted Net Profit figure to be functionally
            # invisible, which led the model to either say the figure
            # wasn't in the context, or invent a plausible-sounding wrong
            # number rather than admit it couldn't find it.
            #
            # Fix: if the FIRST column is empty on most rows (a strong
            # signal the label column was lost), retry this exact table
            # region using the "text" strategy instead, which infers
            # columns from text alignment/whitespace gaps rather than
            # requiring a drawn line -- this is the standard pdfplumber
            # remedy for unruled financial tables. Keep whichever
            # extraction actually has labels.
            raw_text_fallback = None
            needs_fallback = self._label_column_missing(cleaned_rows) or self._looks_over_fragmented(cleaned_rows)
            if needs_fallback:
                widened_bbox = self._widen_bbox(page, raw_table.bbox, found)
                raw_text_fallback = self._extract_bbox_as_text(page, widened_bbox)

            header, body_rows = self._split_header(cleaned_rows)
            confidence = self._estimate_confidence(header, body_rows)
            if raw_text_fallback:
                # This table is no longer a clean grid -- its
                # reliability comes from preserved label/number
                # adjacency in reading order, not from row/column
                # consistency, so it's given a fixed, deliberately
                # moderate confidence rather than the (misleadingly
                # low, since rows/cols look "inconsistent") grid-based
                # estimate above.
                confidence = 0.75

            table = Table(
                table_id=f"table-{next(_TABLE_ID_COUNTER):06d}",
                page_number=page_index,
                rows=body_rows,
                header=header,
                bounding_box=tuple(raw_table.bbox) if raw_table.bbox else None,
                confidence=confidence,
                extraction_method="text_fallback" if raw_text_fallback else "pdfplumber",
                raw_text_override=raw_text_fallback,
            )
            results.append(table)
        return results

    def _extract_page_tables_text_strategy(self, page, page_index: int) -> List[Table]:
        """Extract tables using pdfplumber's text strategy (whitespace-based)
        which works for unruled financial statement tables."""
        results: List[Table] = []
        try:
            found = page.find_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})
        except Exception as exc:
            logger.warning("pdfplumber text strategy failed on page %d: %s", page_index, exc)
            return results

        for raw_table in found:
            try:
                extracted_rows = raw_table.extract()
            except Exception as exc:
                logger.warning("pdfplumber failed to extract a table on page %d: %s", page_index, exc)
                continue
            if not extracted_rows:
                continue

            cleaned_rows = [
                [cell.strip() if isinstance(cell, str) else (cell or "") for cell in row]
                for row in extracted_rows
            ]

            raw_text_fallback = None
            needs_fallback = self._label_column_missing(cleaned_rows) or self._looks_over_fragmented(cleaned_rows)
            if needs_fallback:
                widened_bbox = self._widen_bbox(page, raw_table.bbox, found)
                raw_text_fallback = self._extract_bbox_as_text(page, widened_bbox)

            header, body_rows = self._split_header(cleaned_rows)
            confidence = self._estimate_confidence(header, body_rows)
            if raw_text_fallback:
                confidence = 0.75

            table = Table(
                table_id=f"table-{next(_TABLE_ID_COUNTER):06d}",
                page_number=page_index,
                rows=body_rows,
                header=header,
                bounding_box=tuple(raw_table.bbox) if raw_table.bbox else None,
                confidence=confidence,
                extraction_method="text_fallback",
                raw_text_override=raw_text_fallback,
            )
            results.append(table)
        return results

    @staticmethod
    def _clamp_bbox(page, bbox):
        """pdfplumber's `find_tables()` occasionally returns a bounding
        box that overflows the page edge by a few points (a rendering
        artifact from background shading/bleed near the border on some
        wide landscape financial-statement pages). `page.within_bbox()`
        raises ValueError if the box isn't FULLY inside the page, which
        was silently defeating both the text-strategy retry and the
        plain-text fallback exactly when they were needed most (e.g.
        some real-world PDFs have this off-by-a-few-points overflow).
        Clamp the box to the page's own bounds first so a small overflow
        can never block recovery."""
        x0, top, x1, bottom = bbox
        x0 = max(0.0, min(x0, page.width))
        x1 = max(0.0, min(x1, page.width))
        top = max(0.0, min(top, page.height))
        bottom = max(0.0, min(bottom, page.height))
        if x1 <= x0 or bottom <= top:
            return None
        return (x0, top, x1, bottom)

    @staticmethod
    def _widen_bbox(page, bbox, sibling_tables) -> tuple:
        """Widen `bbox` in BOTH directions to recover content the
        original table-region detection cut off, WITHOUT ever crossing
        into a sibling table's own territory.

        This replaces an earlier version that had a real bug: it widened
        the left edge out to the *leftmost sibling's own left edge*
        (`t.bbox[0]`) rather than the *nearest sibling's right edge*
        (`t.bbox[2]`). On a page with only two side-by-side tables that
        happened to still "work" by accident (there's only one other
        table to be wrong about) -- but on a page with THREE or more
        table regions (confirmed on a real annual report's Balance Sheet + Statement
        of Profit and Loss page, which pdfplumber split into three
        separate regions), widening the rightmost table leftward used
        the FARTHEST sibling's left edge as the new boundary, swallowing
        BOTH other tables' columns into one extraction -- producing a
        single "table" whose rows jumble Balance Sheet line items and
        P&L line items together on the same line, with erratic spacing
        between them. That is the exact "random column divider / random
        spaces" symptom reported after re-ingesting.

        Correct rule: widen left only as far as the RIGHT edge of the
        nearest table sitting to the left (or the page's own left edge
        if nothing is there), and widen right only as far as the LEFT
        edge of the nearest table sitting to the right (or the page's
        own right edge). Both directions are computed independently and
        take the CLOSEST blocking sibling, not the farthest.
        """
        if not bbox:
            return bbox
        x0, top, x1, bottom = bbox
        this_height = bottom - top

        def _is_real_sibling(t) -> bool:
            # A genuine neighboring statement/table has substantial
            # vertical extent, comparable to the table being widened.
            # pdfplumber sometimes also detects tiny, spurious fragments
            # -- e.g. a single isolated "Note" header cell only ~10
            # points tall -- as their own separate "table". Treating
            # one of those as a real blocking sibling was stopping a
            # genuine table's widening just a few points short of its
            # own label column (confirmed on a real annual report's Balance Sheet:
            # a 1-line junk fragment sitting exactly where the real
            # label column starts blocked the real table from ever
            # reaching it, leaving every row numbers-only with no
            # labels at all). Require at least 30% of the current
            # table's own height, or an absolute 40pt floor for cases
            # where the current table itself is short.
            if not t.bbox:
                return False
            h = t.bbox[3] - t.bbox[1]
            return h >= max(this_height * 0.3, 40)

        real_siblings = [t for t in sibling_tables if t.bbox and t.bbox != bbox and _is_real_sibling(t)]

        left_blockers = [t.bbox[2] for t in real_siblings if t.bbox[2] <= x0]
        new_x0 = max(left_blockers) if left_blockers else 0.0

        right_blockers = [t.bbox[0] for t in real_siblings if t.bbox[0] >= x1]
        new_x1 = min(right_blockers) if right_blockers else page.width

        return (new_x0, top, new_x1, bottom)

    @staticmethod
    def _extract_bbox_as_text(page, bbox) -> Optional[str]:
        """Plain, reading-order text extraction of a table's exact
        bounding-box region -- the reliable fallback when structured
        row/column extraction can't keep a label glued to its number.
        `x_tolerance=2` keeps adjacent words on financial-statement
        lines (label text immediately followed by a rupee figure) from
        being incorrectly merged into one run.

        Many annual-report pages are wide, LANDSCAPE two-column layouts:
        a numeric statement (e.g. the Cash Flow Statement) on one side,
        an unrelated narrative note ("Company Information", "Basis of
        Measurement", ...) on the other. A single flat top-to-bottom
        pass across the WHOLE bbox width sorts words purely by vertical
        position, so a statement line and a narrative line that happen
        to sit at the same height get READ AS IF THEY WERE ONE LINE --
        e.g. "Profit Before Tax 36960.90 39469.13" ending up glued
        next to "public limited Company domiciled in India with". The
        numbers are technically still present afterwards, but a reader
        (human or model) can no longer tell where the real statement
        rows end and the unrelated narrative begins, and can end up
        citing the wrong row entirely.

        Fix: look for a genuine COLUMN GUTTER -- the largest gap between
        adjacent words' left edges, if it's wide and sits roughly in the
        middle of the bbox (not just one page-normal word-gap, and not
        off at one edge, which would just be a table's own last column).
        If found, extract the left half and right half as two SEPARATE,
        independently-ordered text passes (left column fully top-to-
        bottom, then right column fully top-to-bottom) instead of one
        interleaved pass. A normal single-column table has no such gutter
        and falls through to the original flat extraction unchanged.
        """
        try:
            safe_bbox = TableExtractor._clamp_bbox(page, bbox) if bbox else None
            if bbox and not safe_bbox:
                return None
            region = page.within_bbox(safe_bbox) if safe_bbox else page

            gutter = TableExtractor._find_column_gutter(region)
            if gutter is not None and not TableExtractor._gutter_is_intra_table_label_value(region, gutter):
                x0, top, x1, bottom = region.bbox
                left = region.within_bbox((x0, top, gutter, bottom))
                right = region.within_bbox((gutter, top, x1, bottom))
                left_text = (left.extract_text(layout=False, x_tolerance=2) or "").strip()
                right_text = (right.extract_text(layout=False, x_tolerance=2) or "").strip()
                parts = [t for t in (left_text, right_text) if t]
                if len(parts) == 2:
                    # Keep both columns, but never let one column's lines
                    # sort in among the other's -- each column is fully
                    # self-contained, in its own reading order, with a
                    # clear separator so a table-row-finder downstream
                    # doesn't accidentally treat the boundary as a
                    # continuation of the same row/sentence.
                    return parts[0] + "\n\n[unrelated adjacent column omitted]\n\n" + parts[1]
                # If only one side actually had content, that side alone
                # is the real table -- the "gutter" was probably just
                # this table's own wide blank margin, not a second column.
                if len(parts) == 1:
                    return parts[0]
            # No gutter, OR the "gutter" is really the wide blank gap
            # between a single table's label column and its own value
            # columns (see _gutter_is_intra_table_label_value). In the
            # latter case splitting would divorce every label from its
            # numbers; a single flat pass instead sorts words by
            # (top, x) so each label and the figures on its own row land
            # on the SAME output line -- exactly what a downstream
            # row/fact parser needs.
            text = region.extract_text(layout=False, x_tolerance=2)
            return text.strip() if text and text.strip() else None
        except Exception as exc:
            logger.warning("Plain-text bbox fallback failed for a table: %s", exc)
            return None

    @staticmethod
    def _find_column_gutter(region) -> Optional[float]:
        """Return the x-coordinate of a genuine two-column gutter within
        *region*, or None if this looks like an ordinary single-column
        area. A genuine gutter is the SINGLE largest gap between two
        adjacent words' left edges, wide enough to be a real page-layout
        column break (not just normal word/column spacing), and located
        roughly in the middle two-thirds of the region's width (a gap
        hugging either edge is much more likely to be a table's own last
        blank column than a second, unrelated column of content)."""
        try:
            words = region.extract_words()
        except Exception:
            return None
        if len(words) < 20:
            return None
        x0, _top, x1, _bottom = region.bbox
        width = x1 - x0
        if width < 400:
            # Two side-by-side content columns on a normal-width page
            # would each be uncomfortably narrow to read -- this pattern
            # is specific to wide/landscape pages.
            return None

        xs = sorted(set(round(w["x0"], 1) for w in words))
        if len(xs) < 4:
            return None

        gaps = [(xs[i + 1] - xs[i], xs[i], xs[i + 1]) for i in range(len(xs) - 1)]
        gap_size, left_edge, right_edge = max(gaps)

        midpoint = left_edge + gap_size / 2
        in_middle_band = (x0 + width * 0.25) <= midpoint <= (x0 + width * 0.75)
        if gap_size >= 60 and in_middle_band:
            return midpoint
        return None

    @staticmethod
    def _gutter_is_intra_table_label_value(region, gutter: float) -> bool:
        """Distinguish a REAL two-column page (a numeric statement beside
        an UNRELATED prose narrative -- the case the gutter split exists
        to separate) from a SINGLE financial table whose own label column
        and value columns merely have a wide blank gap between them.

        This matters because the second case is extremely common in
        annual-report P&L / note tables: short text labels on the left
        ("Revenue from operations", "Total revenue from operations
        (A+B)"), right-aligned figures far to the right, and a wide blank
        band in between that _find_column_gutter happily flags as a
        "column gutter". Splitting there is catastrophic -- it emits all
        the labels as one block and all the numbers as a separate block,
        so EVERY figure is divorced from the line item it belongs to
        (confirmed on GPI's statements, where it left the deterministic
        fact extractor with nothing to bind "Profit before tax" to
        "108656.57"). Such a table must be read as ONE flat pass, not
        split.

        The tell is unambiguous on real pages:
          * an unrelated narrative to the right is PROSE -- mostly words,
            very few numbers;
          * a table's own value block to the right is NUMBERS, and each
            of those number rows sits at the SAME vertical position as a
            text label on the left (they are the same table row).

        So this returns True ("do not split -- one table") only when all
        three hold: the right half is predominantly numeric, the left
        half is predominantly non-numeric labels, AND a meaningful
        fraction of rows pair a left-side label with a right-side number.
        A genuine statement|narrative page fails at least one (its
        narrative side isn't numeric, or its lines don't line up
        row-for-row with the other side), so it still splits as before.
        """
        try:
            words = region.extract_words()
        except Exception:
            return False
        if not words:
            return False

        def _is_num(tok: str) -> bool:
            return bool(re.match(r"^[-(]*\d[\d,.)/]*%?$", tok or ""))

        left_words = [w for w in words if w["x1"] <= gutter + 2]
        right_words = [w for w in words if w["x0"] >= gutter - 2]
        if not left_words or not right_words:
            return False

        left_num_ratio = sum(1 for w in left_words if _is_num(w["text"])) / len(left_words)
        right_num_ratio = sum(1 for w in right_words if _is_num(w["text"])) / len(right_words)
        if not (right_num_ratio >= 0.5 and left_num_ratio <= 0.25):
            return False

        # Row alignment: bucket words into ~3pt-tall rows, then count the
        # rows that have BOTH a left-side label word and a right-side
        # numeric word -- i.e. a real "label ... value value" table row.
        rows: dict = {}
        for w in words:
            rows.setdefault(round(w["top"] / 3.0), []).append(w)
        aligned = 0
        for ws in rows.values():
            has_left_label = any(w["x1"] <= gutter + 2 and not _is_num(w["text"]) for w in ws)
            has_right_num = any(w["x0"] >= gutter - 2 and _is_num(w["text"]) for w in ws)
            if has_left_label and has_right_num:
                aligned += 1
        return (aligned / max(len(rows), 1)) >= 0.30

    @staticmethod
    def _label_column_missing(rows: List[List[str]]) -> bool:
        """True if the first column doesn't contain real text labels --
        either because it's empty on most data rows, or because it's
        populated but with numbers/dashes/whitespace-only fragments
        rather than alphabetic label text (a subtler failure: pdfplumber
        assigned SOME content to column 0, but it's stray digits that
        got misaligned into the label column's position rather than the
        real "Adjustments for depreciation"-style text -- e.g. a row
        like ['30.00', '233868.00'] has a non-empty first cell that is
        still not a label). Either case means this table has numbers
        with nothing identifying what they represent."""
        if len(rows) < 3:
            return False
        has_other_content = any(
            any(c.strip() for c in r[1:]) for r in rows if len(r) > 1
        )
        if not has_other_content:
            return False
        empty_first = sum(1 for r in rows if r and not r[0].strip())
        if (empty_first / len(rows)) >= 0.6:
            return True
        # First column has content, but check whether it's real label
        # text (contains letters) or just numeric/symbol noise.
        non_empty_first = [r[0].strip() for r in rows if r and r[0].strip()]
        if not non_empty_first:
            return False
        alphabetic_labels = sum(1 for v in non_empty_first if re.search(r"[A-Za-z]{3,}", v))
        return (alphabetic_labels / len(non_empty_first)) < 0.4

    @staticmethod
    def _looks_over_fragmented(rows: List[List[str]]) -> bool:
        """True if the label column exists but has been sliced into too
        many spurious fragment columns instead of one real label column
        (e.g. a single label like "Profit for the year" split across
        several adjacent short cells). Signal: many short (<=3 char)
        alphabetic-only cells sitting among the row's non-numeric cells."""
        if len(rows) < 3:
            return False
        widths = [len(r) for r in rows if r]
        if not widths:
            return False
        max_width = max(widths)
        if max_width < 5:
            return False
        short_fragment_count = 0
        total_nonnumeric = 0
        for r in rows:
            for c in r:
                c = c.strip()
                if not c or re.fullmatch(r"[\d.,()\-]+", c):
                    continue
                total_nonnumeric += 1
                if len(c) <= 3 and re.fullmatch(r"[A-Za-z]+", c):
                    short_fragment_count += 1
        if total_nonnumeric < 4:
            return False
        return (short_fragment_count / total_nonnumeric) >= 0.3

    @staticmethod
    def _split_header(rows: List[List[str]]) -> tuple:
        if not rows:
            return [], []
        first_row = rows[0]
        # A first row is treated as a header when every cell is
        # non-empty and short -- a reasonable proxy for "these are
        # column labels, not a data row" without needing NLP.
        looks_like_header = all(cell and len(cell) <= 40 for cell in first_row)
        if looks_like_header:
            return first_row, rows[1:]
        return [], rows

    @staticmethod
    def _estimate_confidence(header: List[str], rows: List[List[str]]) -> float:
        if not rows:
            return 0.0
        expected_cols = len(header) if header else len(rows[0])
        if expected_cols == 0:
            return 0.0
        consistent = sum(1 for row in rows if len(row) == expected_cols)
        consistency_ratio = consistent / len(rows)
        non_empty_ratio = sum(
            1 for row in rows for cell in row if cell.strip()
        ) / max(sum(len(row) for row in rows), 1)
        return round(0.5 * consistency_ratio + 0.5 * non_empty_ratio, 3)

    # -- camelot fallback --------------------------------------------------

    def _camelot_fallback(self, file_path: str, page_numbers: List[int]) -> List[Table]:
        results: List[Table] = []
        if not page_numbers:
            return results
        pages_arg = ",".join(str(n) for n in page_numbers)

        # camelot.read_pdf() has NO timeout of its own -- on a page with
        # a genuinely huge/dense table, or one that trips a known slow
        # path in its stream-alignment algorithm, this call can run for
        # a very long time (occasionally close to indefinitely) with
        # ZERO console output while it does: no exception, no progress
        # log, nothing -- which is indistinguishable from the whole
        # ingestion process having frozen. `signal.alarm` can't be used
        # here since it doesn't exist on Windows; running the call on a
        # background thread and bounding how long we WAIT for it works
        # on every platform. Note this can't forcibly kill the camelot
        # call itself (Python threads aren't killable) -- if it times
        # out, that one background thread keeps running until camelot
        # itself eventually finishes or errors, using some CPU in the
        # background, but ingestion moves on immediately rather than the
        # entire multi-file run being blocked on it. These pages simply
        # keep whatever pdfplumber's own extraction already produced
        # (however imperfect) instead of waiting indefinitely.
        def _run_camelot() -> list:
            # "stream" (whitespace/alignment-based, like pdfplumber's
            # text strategy) is tried FIRST here, not "lattice"
            # (ruling-line-based). The pages routed into this fallback
            # are specifically the ones where line-based detection
            # already failed or produced garbage (see extract()) --
            # trying lattice first on them again would just repeat the
            # same failure mode camelot shares with pdfplumber's default
            # for unruled financial-statement tables. "lattice" is kept
            # as the secondary attempt for the genuinely-zero-tables
            # case, where a page might have ruling lines pdfplumber's
            # OWN detector simply missed.
            camelot_tables = camelot.read_pdf(file_path, pages=pages_arg, flavor="stream")
            if len(camelot_tables) == 0:
                camelot_tables = camelot.read_pdf(file_path, pages=pages_arg, flavor="lattice")
            return camelot_tables

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_run_camelot)
        try:
            camelot_tables = future.result(timeout=CAMELOT_FALLBACK_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "camelot fallback for pages %s exceeded %ds -- giving up on "
                "these pages and keeping whatever pdfplumber's own extraction "
                "already found for them (this is what a PDF hanging ingestion "
                "for many minutes with no log output was: camelot silently "
                "stuck on one of these pages, not the app being frozen).",
                pages_arg, CAMELOT_FALLBACK_TIMEOUT_SECONDS,
            )
            # shutdown(wait=False) is essential here -- the default
            # wait=True (which a `with ThreadPoolExecutor(...):` block
            # would trigger on exit) blocks until the background thread
            # actually finishes, which defeats the entire point of the
            # timeout above: the caller would still end up waiting the
            # FULL, possibly-unbounded camelot runtime, just with extra
            # steps. wait=False lets this function return immediately;
            # the orphaned thread cleans itself up once camelot itself
            # eventually finishes or errors.
            executor.shutdown(wait=False)
            return results
        except Exception as exc:
            logger.warning("camelot fallback failed for pages %s: %s", pages_arg, exc)
            executor.shutdown(wait=False)
            return results
        else:
            executor.shutdown(wait=False)

        for camelot_table in camelot_tables:
            try:
                df = camelot_table.df
                rows = df.values.tolist()
                header = rows[0] if rows else []
                body_rows = rows[1:] if rows else []
                page_number = int(camelot_table.page)
                results.append(
                    Table(
                        table_id=f"table-{next(_TABLE_ID_COUNTER):06d}",
                        page_number=page_number,
                        rows=body_rows,
                        header=header,
                        bounding_box=None,
                        confidence=float(getattr(camelot_table, "accuracy", 50.0)) / 100.0,
                        extraction_method="camelot",
                    )
                )
            except Exception as exc:
                logger.warning("Failed to normalize a camelot table: %s", exc)
                continue
        return results
