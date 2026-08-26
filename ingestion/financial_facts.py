"""
ingestion/financial_facts.py

Deterministic extraction of named financial line items (PAT, PBT, Tax
Expense, Revenue, EBITDA, ...) from a table chunk's own text, with each
value tagged to the ACTUAL YEAR OF ITS OWN COLUMN -- not the chunk's
single `year` field, which is None or wrong the moment a table has more
than one year's column (the normal case: almost every real financial
statement shows the current year AND at least one comparative year side
by side).

Why this exists
----------------
The LLM was repeatedly giving the correct PAT for one year in a
multi-year question, then either recomputing a DIFFERENT year from PBT
(wrong), or saying that year "isn't mentioned" (also wrong, when it
plainly was in the same table). Both symptoms trace to the same root
cause: nothing in the pipeline ever recorded "this exact number is
2024's PAT" as a fact -- only "this chunk's overall year is X" (a single
value) or "these years are mentioned somewhere in this chunk" (a list,
with no per-value mapping back to which column each number came from).
The LLM was left to reconstruct that column-to-year mapping itself, at
read time, from raw table text -- exactly the kind of positional
inference a smaller model gets wrong under load.

This module does that mapping once, deterministically, at ingestion
time, so answering "PAT for 2024" becomes a dictionary lookup instead of
something the LLM has to re-derive from scratch on every question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Canonical metric name -> regex patterns that match a row's OWN label
# (never a value, never a surrounding sentence). Ordered so a more
# specific pattern (e.g. "profit before tax") is tried before a more
# general one that could accidentally also match it.
_METRIC_PATTERNS = {
    "PAT": [
        r"profit\s+for\s+the\s+year",
        r"profit\s+for\s+the\s+period",
        r"profit\s+after\s+tax",
        r"net\s+profit(?!\s*(?:margin|ratio))",
    ],
    "PBT": [
        r"profit\s+before\s+tax",
        r"profit\s*/\s*\(loss\)\s+before\s+tax",
    ],
    "Tax Expense": [
        r"total\s+tax\s+expense",
        r"^\s*tax\s+expense",
        r"income\s+tax\s+expense(?!\s+recognised)",
    ],
    "Revenue": [
        r"revenue\s+from\s+operations",
        r"total\s+revenue",
    ],
    "Total Income": [
        # Negative lookahead so this does NOT also match balance-sheet
        # rows like "Total income tax assets" / "Total income tax
        # liabilities", which are NOT the P&L "Total income (I+II)" line
        # and were otherwise surfacing as spurious ~300-3,000 lakh
        # "Total Income" facts (e.g. 326.41, 3157.80) sitting right next
        # to the real 442,528.72.
        r"total\s+income(?!\s+tax)",
    ],
    "EBITDA": [
        r"\bebitda\b",
    ],
    "EPS (Basic)": [
        r"basic\s+earnings\s+per\s+share",
    ],
    "EPS (Diluted)": [
        r"diluted\s+earnings\s+per\s+share",
    ],
}

# A row is disqualified from matching ANY metric if its label matches
# one of the blocklists below -- these are the exact mid-word / cross-
# reference / notes / ratio phrasings that previously caused the wrong
# row (or a garbled fragment of the right one) to be read as if it were
# the real line item.
#
# History: a row literally labeled "PAT Margin (%)" or "Profit After Tax
# as % of Revenue" matched the PAT/PBT pattern just fine (the metric
# regex only checks the label CONTAINS "profit after tax"; nothing stops
# it from also containing "margin"), and its PERCENTAGE value got
# presented as the real absolute PAT figure. Blocklisting
# margin/ratio/percentage labels catches this for every metric at once.
#
# Split into two kinds of match on purpose:
#
# _BLOCKLIST_WORDS are matched only as WHOLE WORDS (\bword\b), never as a
# bare substring. This is not a style choice -- it fixes a real,
# silently-catastrophic bug: "ratio" is a substring of "ope-ratio-ns",
# so a plain `"ratio" in label_lower` test discarded EVERY
# "Revenue from operations" row, for every company, which is the entire
# reason revenue never got extracted from a primary P&L. Whole-word
# matching still blocks the rows these terms are actually meant for
# ("PAT Margin", "Current Ratio", "... as a percentage") without eating
# "operations" (or any other word that merely contains those letters).
_BLOCKLIST_WORDS = ("margin", "ratio", "percentage")

# _BLOCKLIST_SUBSTR are genuine phrases/fragments that are safe to match
# anywhere in a label.
_BLOCKLIST_SUBSTR = (
    "recognised in", "recognized in", "reconcil", "refer note", "as per",
    "attributable to", "before exceptional", "before share of",
    "% of", "as a % ",
    # Sensitivity-analysis / market-risk notes phrase a row as the
    # HYPOTHETICAL effect of a rate move -- e.g. "Impact on profit
    # before tax as at the end of the reporting year (110.49)" -- which
    # is NOT the actual line item. Matching the PBT pattern on that row
    # produced a spurious "PBT = 110.49" for GPI. These phrasings never
    # appear on a genuine P&L line-item label, so blocking them removes
    # the false facts without touching any real one.
    "impact on", "impact of", "sensitivity", "effect of",
)


def _label_is_blocked(label_lower: str) -> bool:
    if any(b in label_lower for b in _BLOCKLIST_SUBSTR):
        return True
    return any(
        re.search(rf"\b{re.escape(w)}\b", label_lower) for w in _BLOCKLIST_WORDS
    )

_YEAR_RE = re.compile(r"(20\d{2})")
_NUMBER_RE = re.compile(r"-?\(?\d[\d,]*\.?\d*\)?")
_NUMBER_TOKEN_RE = re.compile(r"^-?\(?\d[\d,]*\.?\d*\)?%?$")


@dataclass
class FinancialFact:
    metric: str          # canonical name, e.g. "PAT"
    year: int            # the actual fiscal year THIS value belongs to
    value: float
    raw_label: str        # the row label as it actually appeared
    raw_value: str         # the cell text as it actually appeared


def _parse_number(cell: str) -> Optional[float]:
    cell = cell.strip().replace(",", "").rstrip("%")
    if not cell:
        return None
    negative = cell.startswith("(") and cell.endswith(")")
    cell = cell.strip("()")
    m = re.fullmatch(r"-?\d+\.?\d*", cell)
    if not m:
        return None
    try:
        val = float(cell)
    except ValueError:
        return None
    return -val if negative else val


def _parse_row(line: str) -> List[str]:
    """
    Split one table row into [label, value1, value2, ...], handling
    BOTH of the two shapes these PDFs actually produce:

    - Markdown pipe tables ("| Profit for the year | 30156.52 | 32698.28 |"),
      the tidy case.
    - Plain whitespace-separated text ("VII Profit for the year (V - VI)
      30156.52 32698.28"), which is what `ingestion/table_extractor.py`'s
      plain-text fallback actually outputs for the majority of real
      statement pages -- an earlier version of this module only handled
      the pipe-table shape and would have silently never matched a
      single real row in practice.

    For the plain-text shape: read tokens from the RIGHT, keeping every
    token that looks like a plain number (or a parenthesized negative,
    or a percentage) as a value; the first token that doesn't match
    stops the scan, and everything up to and including that token is
    the label. This correctly leaves a label like "VII Profit for the
    year (V - VI)" intact (its own parentheses contain letters, not a
    lone number, so "(V" and "VI)" don't match the number pattern) while
    still peeling off trailing figures like "30156.52" or "(297.86)".
    """
    line = line.strip()
    if not line:
        return []
    if line.startswith("|"):
        s = line[1:]
        if s.endswith("|"):
            s = s[:-1]
        return [c.strip() for c in s.split("|")]

    tokens = line.split()
    values: List[str] = []
    i = len(tokens) - 1
    while i >= 0 and _NUMBER_TOKEN_RE.match(tokens[i]) and _parse_number(tokens[i]) is not None:
        values.insert(0, tokens[i])
        i -= 1
    label = " ".join(tokens[: i + 1])
    return [label] + values


def _split_table_lines(text: str) -> List[str]:
    return [l for l in text.split("\n") if l.strip()]


def _is_separator_row(cells: List[str]) -> bool:
    joined = "".join(cells)
    return bool(joined) and all(ch in "-: " for ch in joined)


_MONTH_NAMES = (
    r"Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|"
    r"Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?"
)
# Indian financial statements often use DD.MM.YYYY or DD/MM/YYYY format
# for comparative year headers (e.g., "31.3.2024 31.3.2023" or "31/03/2024 31/03/2023")
_DATE_HEADER_DDMMYYYY = rf"\d{{1,2}}[./]\d{{1,2}}[./]20\d{{2}}"
# Also support "Month DD, YYYY" format (e.g., "March 31, 2023")
_DATE_HEADER_MONTH_DD_YYYY = rf"({_MONTH_NAMES})\s+\d{{1,2}},?\s+20\d{{2}}"
_STRONG_DATE_HEADER_RE = re.compile(
    r"(as\s+at|as\s+on|for\s+the\s+year\s+ended|year\s+ended|"
    rf"\d{{1,2}}(st|nd|rd|th)?\s+({_MONTH_NAMES}),?\s+20\d{{2}}"
    rf"|{_DATE_HEADER_DDMMYYYY}"
    rf"|{_DATE_HEADER_MONTH_DD_YYYY})",
    re.IGNORECASE,
)


def _resolve_ordered_years(lines: List[str]) -> List[int]:
    """
    Find the column order of fiscal years for this table by scanning
    the first few RAW lines (not parsed cells) for 4-digit years, in
    reading order -- e.g. "31st March, 2024 31st March, 2023" yields
    [2024, 2023]. Deliberately regex-scans raw text rather than parsed
    row cells: the plain-text fallback extractor (see
    `ingestion/table_extractor.py`) has no real column boundaries at
    all for a table it couldn't reconstruct a grid for, just a left-to-
    right stream of tokens per line -- but the YEARS still appear in
    the same order the corresponding data columns do, so that reading
    order is itself a reliable enough signal, even without knowing
    exactly where one cell ends and the next begins.

    Only a line that ALSO looks like a genuine date header ("as at",
    "as on", "year ended", or a formatted date like "31st March, 2025")
    is accepted -- a bare line with 2+ year-shaped numbers isn't enough
    on its own. This matters because real financial-note text routinely
    contains other, completely unrelated year-like numbers before the
    real date header ever appears -- e.g. "Government Securities
    (Maturity in year 2026 to 2028)" in an investments note, which a
    plain "does this line contain 2+ years" check would have wrongly
    accepted as the table's column years, mislabeling every real figure
    in that note with those investment-maturity years instead of the
    statement's actual fiscal years. Requiring one of these specific,
    standard header phrasings (which is how every real column-date row
    in these reports is actually formatted) avoids that without losing
    real matches.

    For tables with only a single year column (e.g., Statement of
    Changes in Equity, or single-year P&L), we also accept a single
    year IF the table contains known financial metric labels in its
    data rows -- this avoids false positives from investment notes
    while still capturing legitimate single-year tables.

    Returns [] if no such line is found in the first few lines, in
    which case this table produces no facts at all -- conservative by
    design, matching the rest of this module.
    """
    # First pass: look for multi-year header (preferred)
    for line in lines[:8]:
        if not _STRONG_DATE_HEADER_RE.search(line):
            continue
        years = [int(y) for y in _YEAR_RE.findall(line)]
        if len(years) >= 2:
            return years

    # Second pass: if no multi-year header, check for single-year header
    # but ONLY if the table appears to contain financial metrics
    has_financial_metric = False
    for line in lines:
        line_lower = line.lower()
        for metric, patterns in _METRIC_PATTERNS.items():
            if any(re.search(p, line_lower) for p in patterns):
                has_financial_metric = True
                break
        if has_financial_metric:
            break

    if has_financial_metric:
        for line in lines[:8]:
            if not _STRONG_DATE_HEADER_RE.search(line):
                continue
            years = [int(y) for y in _YEAR_RE.findall(line)]
            if len(years) == 1:
                return years

    return []


def extract_financial_facts(chunk_text: str) -> List[FinancialFact]:
    """
    Scan one table chunk's text for known financial line items, and
    return one FinancialFact per (line item, year) pair actually found
    -- each with its OWN year, resolved from the table's own column
    order, not a single value guessed for the whole table.

    Deliberately conservative: a row only produces facts when the
    table's column-year order AND that row's own metric label are both
    unambiguous, and its trailing numeric values line up with that
    order. A table this can't confidently parse simply produces no
    facts (falls back to normal retrieval/LLM reading) rather than
    guessing.
    """
    facts: List[FinancialFact] = []
    if not chunk_text:
        return facts

    lines = _split_table_lines(chunk_text)
    if len(lines) < 2:
        return facts

    years = _resolve_ordered_years(lines)
    if not years:
        return facts

    for line in lines:
        cells = _parse_row(line)
        if not cells or _is_separator_row(cells):
            continue
        label = cells[0].strip()
        label_lower = label.lower()
        if not label or len(label) < 4:
            continue
        if _label_is_blocked(label_lower):
            continue
        # A clean statement line-item label is roman-numeral / word text
        # only -- "VII Profit for the year (V-VI)", "III Total income
        # (I+II)" -- and NEVER contains an actual figure. When a row from
        # a many-columned statement (in practice almost always the
        # Statement of Changes in Equity) extracts malformed -- its
        # equity-component columns collapse to dashes so a real amount
        # lands INSIDE cells[0], e.g. "Profit for the year - - 104320.37
        # -" -- both that row's label AND its column/year alignment are
        # untrustworthy. Mining it produced a "Profit for the year" fact
        # tagged with the equity statement's opening-balance year (the
        # WRONG year, and not even a P&L figure): that is exactly how
        # 2025's profit showed up as 2024's PAT. A decimal amount or a
        # 4+ digit run inside the label is a reliable marker of this
        # misparse -- a genuine note-reference number is at most a couple
        # of digits and never decimal -- so skip the row entirely rather
        # than record a mis-tagged fact.
        if re.search(r"\d\.\d|\d{4,}", label):
            continue

        matched_metric = None
        for metric, patterns in _METRIC_PATTERNS.items():
            if any(re.search(p, label_lower) for p in patterns):
                matched_metric = metric
                break
        if matched_metric is None:
            continue

        values = cells[1:]
        if not values:
            continue
        # A row can have extra leading numeric-looking tokens before
        # the real figures (a "Note" reference number, e.g. "V Profit
        # before tax 26 39469.13 42888.84" where "26" is a footnote
        # reference, not a value) -- keep only the LAST len(years)
        # tokens, since the actual data figures are always the
        # trailing ones and this keeps them correctly aligned to the
        # year order even when a stray reference number was swept up
        # as if it were an extra column.
        if len(values) > len(years):
            values = values[-len(years):]
        elif len(values) < len(years):
            continue  # can't confidently align a partial row

        for year, raw_value in zip(years, values):
            # Second line of defense beyond the label blocklist above:
            # every metric this module tracks (PAT, PBT, Revenue,
            # EBITDA, EPS, ...) is an absolute rupee/per-share figure,
            # never a percentage. A literal "%" on the cell itself is a
            # strong signal this is actually a margin/ratio row whose
            # LABEL happened not to contain any of the blocklisted
            # words (an unusual phrasing, a typo, a merged/split cell)
            # -- rather than silently stripping the "%" and returning
            # the bare number as if it were the real absolute figure
            # (which is exactly how "PAT: 15.20%" got presented as
            # GPI's actual profit after tax), skip it entirely.
            if raw_value.strip().endswith("%"):
                continue
            value = _parse_number(raw_value)
            if value is None:
                continue
            facts.append(FinancialFact(
                metric=matched_metric, year=year, value=value,
                raw_label=label, raw_value=raw_value,
            ))

    return facts