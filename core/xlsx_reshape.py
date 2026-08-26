"""
core/xlsx_reshape.py

Some real-world spreadsheets aren't "one row per record, one header
row" -- they're cross-tab / pivoted MIS-style reports: a handful of
identifier columns on the left (e.g. a product/SKU name), then a WIDE
block of data columns grouped under merged multi-row headers (e.g. one
merged header cell reading "state 1a" spanning three actual data
columns underneath it labeled "Plan" / "MTD" / "%Ach").

`pd.read_excel(..., header=0)` -- what `core/data_analyst.py` normally
uses -- has no way to represent that: it only understands ONE header
row, so every column that sat under a merged group header (which Excel
only stores on that group's own top-left cell, leaving the rest blank)
comes back as "Unnamed: N", and the real second header row (Plan/MTD/
%Ach) gets read as if it were a data row. Filtering/grouping/aggregating
on a sheet loaded that way can't work at all -- there's no "State"
column, no "Metric" column, nothing to group by except accidents of
position.

This module detects that shape (purely from the GRID'S OWN structure --
blank-vs-filled cells, numeric density, distinct-value counts -- never
from any hardcoded column/sheet name) and reshapes it into a plain tidy
table: one row per (identifier..., group, metric, value). That tidy
table is a completely ordinary flat DataFrame, so every existing
filter/group-by/aggregate/outlier code path in data_analyst.py runs on
it completely unchanged -- this module's only job is turning a shape
that engine can't query into one it already can.

Sheets that are already flat (the common case) are left byte-for-byte
alone -- detection is deliberately conservative, and the existing
simple-table loading path is never touched.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

# How many of a sheet's leading rows to even consider as possible
# header rows. A real multi-level MIS header is almost always 2-4 rows
# (title / group names / sub-metric names); capping this keeps the
# detector from ever mistaking a large chunk of a genuinely flat,
# numeric-heavy table for "more header rows".
_MAX_HEADER_ROWS = 5

_NUMERIC_RE = re.compile(r"^-?\(?\d[\d,]*\.?\d*%?\)?$")


def _is_blank(value) -> bool:
    if value is None:
        return True
    try:
        if pd is not None and pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def _looks_numeric(value) -> bool:
    if _is_blank(value):
        return False
    if isinstance(value, (int, float)):
        return True
    s = str(value).strip()
    if s in ("-", "—", "–"):
        return False
    return bool(_NUMERIC_RE.match(s))


def _row_numeric_density(row: List) -> float:
    non_blank = [v for v in row if not _is_blank(v)]
    if not non_blank:
        return 0.0
    numeric = [v for v in non_blank if _looks_numeric(v)]
    return len(numeric) / len(non_blank)


def sheet_needs_reshaping(flat_df: "pd.DataFrame") -> bool:
    """
    True if a normally-loaded (header=0) DataFrame shows the tell-tale
    signs of a multi-row merged header that pandas couldn't represent:
    a large fraction of "Unnamed: N" columns (pandas's own marker for a
    blank header cell), or the first data row still being full of
    header-shaped text rather than real values.

    Deliberately conservative -- a normal, already-flat sheet (the vast
    majority of real spreadsheets) will not trip this, and this
    function is the ONLY thing that decides whether the reshape path
    below ever runs at all.
    """
    if flat_df.empty or len(flat_df.columns) < 2:
        return False
    unnamed = sum(1 for c in flat_df.columns if re.match(r"^Unnamed: \d+$", str(c)))
    if unnamed / len(flat_df.columns) >= 0.3:
        return True
    # A second signal, independent of column naming: the first couple
    # of "data" rows are themselves mostly non-numeric, short, header-
    # shaped text (a second header row pandas mistook for data).
    first_rows = flat_df.head(2)
    for _, row in first_rows.iterrows():
        density = _row_numeric_density(list(row))
        if density < 0.2 and unnamed / len(flat_df.columns) >= 0.15:
            return True
    return False


def _detect_header_row_count(raw_rows: List[List]) -> int:
    """
    Walk down from the top of the raw grid and count how many leading
    rows are "header-like" (low numeric density) before hitting a row
    that looks like real data (numeric-heavy). Capped at
    `_MAX_HEADER_ROWS` as a safety bound.
    """
    count = 0
    for row in raw_rows[:_MAX_HEADER_ROWS]:
        if _row_numeric_density(row) >= 0.4:
            break
        count += 1
    return max(count, 1)


def _ffill_row(row: List) -> List:
    """Forward-fill a single header row left-to-right -- reconstructs
    a merged cell's label into the blank cells Excel leaves after it."""
    filled = []
    last = None
    for v in row:
        if not _is_blank(v):
            last = v
            filled.append(v)
        else:
            filled.append(last)
    return filled


def _is_degenerate_row(row: List) -> bool:
    """A header row that has only one distinct non-blank value across
    the WHOLE row doesn't distinguish any columns from each other (a
    report title spanning the full width, e.g. "Biscuits - G...") --
    it's real, but useless for building a composite column name, so
    it's dropped from the join (still consumed as a header row)."""
    distinct = {str(v).strip() for v in row if not _is_blank(v)}
    return len(distinct) <= 1


def _detect_id_column_count(raw_rows: List[List], header_row_count: int) -> int:
    """
    Identifier columns (kept as-is, never melted) are the leading
    columns that never belong to any merged GROUP header -- i.e. every
    header row above the last one is blank for that column in the
    ORIGINAL (pre-ffill) grid. The last header row (the immediate
    sub-metric labels) is allowed to be blank or filled for id columns
    either way, since some reports label the id column itself there
    (e.g. "BRANDS") and some don't.
    """
    if header_row_count <= 1:
        return 1
    group_rows = raw_rows[: header_row_count - 1]
    if not group_rows:
        return 1
    max_cols = max(len(r) for r in group_rows)
    id_count = 0
    for col_idx in range(max_cols):
        any_filled = any(
            col_idx < len(r) and not _is_blank(r[col_idx])
            for r in group_rows
        )
        if any_filled:
            break
        id_count += 1
    return max(id_count, 1)


_METRIC_HINT_WORDS = (
    "plan", "mtd", "ytd", "ach", "act", "actual", "target", "tgt",
    "budget", "var", "variance", "growth", "rev", "revised",
    "ly", "last year", "prior", "forecast", "fcst",
)

_ROLLUP_HINT_WORDS = (
    "total", "grand total", "sub total", "subtotal", "all ", "overall",
    "combined", "consolidated", "summary", "aggregate", "sum",
)
_ROLLUP_BRANCH_RE = re.compile(r"^(branch|zone|region|cluster|division|group)\s*\d+$", re.IGNORECASE)

_TRAILING_NUMBER_RE = re.compile(r"^(.*?)[\s]*\d+(?:\.\d+)?$")


def _hierarchy_family(value: str) -> str:
    """
    The "family" a single identifier value belongs to: its text with
    any trailing number (and separating space/punctuation) stripped --
    "SEGMENT 3" -> "SEGMENT", "SKU 1.24" -> "SKU", "FRANCHISE 10" ->
    "FRANCHISE". A value with no trailing number (most rollup labels:
    "TOTAL", "ALL EUROPE") keeps its full text as its own family, which
    is exactly what correctly separates them from the numbered ones.
    """
    v = value.strip()
    m = _TRAILING_NUMBER_RE.match(v)
    if m and m.group(1).strip():
        return m.group(1).strip().upper()
    return v.upper()


def detect_hierarchy_levels(values: "pd.Series") -> Optional[Tuple["pd.Series", "pd.Series"]]:
    """
    Generic, DATA-DRIVEN detection of hierarchy levels mixed into one
    identifier column -- e.g. a "BRANDS" column that holds SEGMENT-,
    FRANCHISE-, and SKU-level rows all at once with nothing structurally
    marking which is which (a flattened outline/tree report, extremely
    common in MIS exports). Returns (level_numbers, level_names) as two
    Series aligned to `values`, or None if this column doesn't actually
    show that pattern (e.g. it's already a flat, single-level list of
    names).

    Deliberately contains NO hardcoded business vocabulary -- no
    "segment"/"franchise"/"SKU"/"product" keyword list to maintain per
    report. Those words only show up in this module's docstrings/
    examples because they happen to be THIS specific report's own
    words; the actual algorithm discovers whatever level names a given
    column's OWN data happens to use, so the exact same code handles a
    completely different report (e.g. "REGION"/"STORE"/"SKU", or
    "DIVISION"/"DEPARTMENT"/"EMPLOYEE") with no changes:

      1. Group the column's distinct values by "family" (see
         _hierarchy_family) -- text with any trailing number stripped.
      2. A small, genuinely universal set of rollup words (total / all /
         subtotal / overall / sum -- ordinary business-reporting
         vocabulary, not specific to any one industry or report) marks a
         family as the topmost rollup level (0) regardless of its size.
         This reuses the exact same _ROLLUP_HINT_WORDS already used for
         Is_Rollup detection elsewhere in this module, so the two stay
         consistent with each other by construction.
      3. Every other family is ranked by how many distinct values it
         has, ascending: fewer distinct values means a coarser, higher-
         level grouping. 5 segments vs. 10 franchises vs. ~100 SKUs is a
         completely general shape (a coarse category always has fewer
         members than a fine one), not something specific to this
         report -- that ordering becomes the level number (1, 2, 3, ...).
      4. Each family's own text IS its level name -- read directly off
         the data, never chosen by a developer ahead of time.

    Requires at least 2 distinct families and a reasonable amount of
    data to be worth doing at all; returns None otherwise (a genuinely
    flat column is left completely alone).
    """
    non_null = values.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if len(non_null) < 8:
        return None

    distinct_values = non_null.unique()
    family_members: dict = {}
    for v in distinct_values:
        fam = _hierarchy_family(v)
        family_members.setdefault(fam, []).append(v)

    if len(family_members) < 2:
        return None  # only one family present -- not a mixed hierarchy

    def is_rollup_family(fam: str) -> bool:
        low = fam.lower()
        return any(w in low for w in _ROLLUP_HINT_WORDS) or bool(_ROLLUP_BRANCH_RE.match(fam))

    rollup_families = {f for f in family_members if is_rollup_family(f)}
    other_families = sorted(
        (f for f in family_members if f not in rollup_families),
        key=lambda f: len(family_members[f]),
    )
    # If EVERYTHING would end up in one bucket (no real multi-level
    # split beyond the rollup family, if any), this isn't a meaningful
    # hierarchy either.
    if len(other_families) < 2 and not rollup_families:
        return None

    family_level = {f: 0 for f in rollup_families}
    family_name = {f: f.title() for f in rollup_families}
    for rank, fam in enumerate(other_families, start=1):
        family_level[fam] = rank
        family_name[fam] = fam.title()

    value_to_family = {v: _hierarchy_family(v) for v in distinct_values}
    str_values = values.astype(str).str.strip()
    levels = str_values.map(lambda v: family_level.get(value_to_family.get(v)))
    names = str_values.map(lambda v: family_name.get(value_to_family.get(v)))
    return levels, names


def _looks_like_metric_level(values: List[str]) -> bool:
    """
    True if a header level's distinct values look like report METRICS
    (Plan / MTD / %Ach / Target / ...) rather than real entity names
    (states, products, customers). This is what lets the reshaper name
    this level "Metric" instead of a meaningless "Group2" -- critical
    because without that name, nothing downstream (the LLM building a
    query spec, or a person reading the result) has any way to know
    this column must be filtered to ONE value before aggregating,
    rather than summed/averaged across fundamentally different kinds of
    number (e.g. a percentage achievement figure and a raw volume
    figure) as if they were the same measurement.

    Deliberately keyword-based rather than positional (never assumes
    "the second header level is always the metric level") so a report
    with the group/metric levels in a different order still gets
    detected correctly. Also true for a small, low-cardinality set of
    short labels in general (most real MIS metric sets are 2-6 values),
    since that alone is already a strong signal even without a keyword
    hit -- a real entity level (states, customers, products) almost
    always has many more distinct values than a metric level does.
    """
    distinct = {str(v).strip().lower() for v in values if v}
    if not distinct:
        return False
    hits = sum(1 for v in distinct if any(w in v for w in _METRIC_HINT_WORDS))
    if hits >= 1:
        return True
    return 1 <= len(distinct) <= 6 and all(len(v) <= 20 for v in distinct)


def _looks_like_rollup(value: str) -> bool:
    """
    True if a GROUP-level value (e.g. a "state") looks like a rollup/
    subtotal/grand-total row rather than one real individual entity --
    "ALL EUROPE", "TOTAL", "BRANCH 3" (a subtotal of the states listed
    just before it in the same report) are all real values that will
    sit in the exact same column as genuine individual states, with no
    structural difference the reshaper alone can detect -- only the
    LABEL gives it away. This is inherently a heuristic (a real place
    could coincidentally be named "Total River" or similar), so it's
    surfaced as an extra "Is_Rollup" column for the query layer/prompt
    to use, NEVER used to silently drop rows here -- a false positive
    only means a row gets flagged, not deleted, and the person asking
    can always override it explicitly if they actually want the
    rollups compared too.
    """
    v = value.strip().lower()
    if any(word in v for word in _ROLLUP_HINT_WORDS):
        return True
    if _ROLLUP_BRANCH_RE.match(value.strip()):
        return True
    return False


_ENTITY_NAME_WORDS = (
    "state", "branch", "zone", "region", "district", "town", "city",
    "customer", "client", "product", "brand", "sku", "segment",
    "division", "territory", "outlet", "distributor", "channel", "area",
)


def _outer_level_name_from_sheet(sheet_name: Optional[str]) -> Optional[str]:
    """
    Best-effort: if the sheet's own name says what its outer group level
    represents (e.g. "State wise Volume MIS MTD" -> "State"), use that
    instead of a meaningless "Group". This is exactly the kind of
    context a person glances at the sheet tab to get instantly, but an
    LLM building a query spec from a bare column list never sees at all
    unless it's put IN the column name. Falls back to None (caller keeps
    the generic "Group") when nothing recognizable is found -- never
    guesses at a name with no basis in the sheet itself.
    """
    if not sheet_name:
        return None
    lower = sheet_name.lower()
    for word in _ENTITY_NAME_WORDS:
        if word in lower:
            return word.capitalize()
    return None


def reshape_pivoted_sheet(raw_df: "pd.DataFrame", sheet_name: Optional[str] = None) -> Optional["pd.DataFrame"]:
    """
    Takes a sheet read with `header=None` (so every row -- including
    the real header rows -- is still plain grid data) and returns a
    tidy long-format DataFrame: identifier column(s) unchanged, plus
    "Group" (the outer merged header, e.g. a state/branch/segment name)
    and "Metric" (the inner sub-header, e.g. Plan/MTD/%Ach) and "Value".

    If the sheet has 3+ header levels, additional "Group2", "Group3",
    ... columns are added for each extra level, so a report with an
    even deeper nesting than the 2-level example above still reshapes
    into something fully queryable, without this needing to hardcode
    how many levels to expect.

    Returns None if the sheet doesn't have enough structure to safely
    reshape (falls back to the normal flat load untouched).
    """
    if pd is None:
        return None
    raw_rows = raw_df.values.tolist()
    if len(raw_rows) < 3:
        return None

    header_row_count = _detect_header_row_count(raw_rows)
    if header_row_count < 2:
        return None  # looks like an ordinary single-header table -- not our job

    header_rows = raw_rows[:header_row_count]
    data_rows = raw_rows[header_row_count:]
    if not data_rows:
        return None

    ffilled_header_rows = [_ffill_row(r) for r in header_rows]
    usable_header_rows = [r for r in ffilled_header_rows if not _is_degenerate_row(r)]
    if not usable_header_rows:
        return None

    id_col_count = _detect_id_column_count(header_rows, header_row_count)
    n_cols = max(len(r) for r in raw_rows)

    # Build a composite label per data column from every usable header
    # level, e.g. ["state 1a", "DEC PLAN"] -> "state 1a | DEC PLAN".
    composite_labels: List[Tuple[str, ...]] = []
    for col_idx in range(n_cols):
        parts = []
        for hrow in usable_header_rows:
            v = hrow[col_idx] if col_idx < len(hrow) else None
            if not _is_blank(v):
                parts.append(str(v).strip())
        composite_labels.append(tuple(dict.fromkeys(parts)))  # de-dup consecutive repeats, keep order

    id_labels = []
    for col_idx in range(id_col_count):
        last_row = header_rows[-1]
        v = last_row[col_idx] if col_idx < len(last_row) else None
        id_labels.append(str(v).strip() if not _is_blank(v) else f"Column {col_idx + 1}")

    df = pd.DataFrame(data_rows, columns=[f"__col{i}" for i in range(n_cols)])
    # Drop fully-empty rows/columns that often trail a real data block.
    df = df.dropna(axis=0, how="all")
    if df.empty:
        return None

    id_cols = [f"__col{i}" for i in range(id_col_count)]
    value_cols = [f"__col{i}" for i in range(id_col_count, n_cols) if composite_labels[i]]
    if not value_cols:
        return None

    tidy = df.melt(id_vars=id_cols, value_vars=value_cols, var_name="__key", value_name="Value")
    key_to_label = {f"__col{i}": composite_labels[i] for i in range(id_col_count, n_cols)}
    max_depth = max((len(v) for v in key_to_label.values()), default=1)

    # Name each header level semantically instead of "Group"/"Group2"/...
    # -- this is what actually lets a query spec (built by an LLM that
    # only ever sees column NAMES and a few sample values, never the
    # original spreadsheet's visual layout) know it must filter to ONE
    # metric before aggregating, rather than blending e.g. a "%Ach"
    # percentage and a "MTD" volume figure into one meaningless max()/
    # sum(). Checked from the LAST level backward: a metric level is
    # almost always the innermost/last one (Plan/MTD/%Ach sitting under
    # a state), and this only actually renames a level when its values
    # clearly look like report metrics (see _looks_like_metric_level) --
    # a level that doesn't match keeps a generic "GroupN" name rather
    # than guessing.
    level_values: List[List[str]] = [[] for _ in range(max_depth)]
    for labels in key_to_label.values():
        for depth_idx in range(max_depth):
            if depth_idx < len(labels):
                level_values[depth_idx].append(labels[depth_idx])

    group_col_names: List[str] = []
    metric_level_idx: Optional[int] = None
    for depth_idx in range(max_depth):
        if metric_level_idx is None and _looks_like_metric_level(level_values[depth_idx]):
            group_col_names.append("Metric")
            metric_level_idx = depth_idx
        elif depth_idx == 0:
            group_col_names.append(_outer_level_name_from_sheet(sheet_name) or "Group")
        else:
            group_col_names.append(f"Group{depth_idx + 1}")

    for depth_idx, col_name in enumerate(group_col_names):
        tidy[col_name] = tidy["__key"].map(lambda k: (key_to_label.get(k) or ("",) * (depth_idx + 1))[depth_idx] if depth_idx < len(key_to_label.get(k, ())) else "")
    tidy = tidy.drop(columns=["__key"])

    # Flag rows whose OUTERMOST group value looks like a rollup/subtotal
    # (see _looks_like_rollup) -- the outermost level is used because
    # that's structurally where a report's own subtotal columns (a
    # "BRANCH 3" column summing the states listed before it) live in
    # every real example seen. Never drops anything; purely an extra
    # signal for the query layer.
    outer_group_col = group_col_names[0]
    tidy["Is_Rollup"] = tidy[outer_group_col].apply(
        lambda v: _looks_like_rollup(str(v)) if v else False
    )

    rename = {f"__col{i}": id_labels[i] for i in range(id_col_count)}
    tidy = tidy.rename(columns=rename)

    tidy["Value"] = pd.to_numeric(tidy["Value"], errors="coerce")

    # Apply generic hierarchy-level detection to each identifier column
    # (e.g. "BRANDS" mixing SEGMENT/FRANCHISE/SKU rows together -- see
    # detect_hierarchy_levels' docstring). This is what lets a query
    # like "grouped by segment" or "at the SKU/product level" actually
    # filter to the right rows: without it, there was no way to tell
    # "SEGMENT 3" and "SKU 1.24" apart except by hardcoding those exact
    # words into the app, which breaks the moment a different report
    # uses different vocabulary for the same kind of mixed hierarchy.
    extra_cols: List[str] = []
    for id_col in id_labels:
        detected = detect_hierarchy_levels(tidy[id_col])
        if detected is None:
            continue
        levels, names = detected
        level_col, name_col = f"{id_col}_Level", f"{id_col}_Level_Name"
        tidy[level_col] = levels
        tidy[name_col] = names
        extra_cols.extend([level_col, name_col])
        # A row whose identifier-column value fell into a detected
        # ROLLUP family (level 0 -- "TOTAL", "ALL EUROPE", etc.) is a
        # subtotal by this column too, not just by the outer group
        # column Is_Rollup already checked. OR it in rather than
        # overwrite, so a row flagged as a rollup by EITHER signal stays
        # flagged.
        tidy["Is_Rollup"] = tidy["Is_Rollup"] | (levels == 0)

    ordered_cols = id_labels + group_col_names + extra_cols + ["Is_Rollup", "Value"]
    tidy = tidy[ordered_cols]
    tidy = tidy.dropna(subset=["Value"])
    return tidy if not tidy.empty else None