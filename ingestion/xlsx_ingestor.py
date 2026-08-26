"""
ingestion/xlsx_ingestor.py

Ingestion for .xlsx / .xls / .csv spreadsheet data files (e.g. primary
sales registers, distributor/retailer lists, monthly MIS dumps) into the
same IngestedChunk shape ingest/ingest.py already produces for PDFs and
PPTX, so import_documents.py and retrieval/knowledge_index.py need no
changes to pick these up.

Why this exists
---------------
Before this module, .xlsx/.csv was not a supported ingestion format at all --
ingest_file()/ingest_directory() only recognized .pdf/.pptx/.ppt/image
extensions. Dropping a spreadsheet into documents/raw/ silently produced
zero chunks, so every question about it later came back as "not
explicitly mentioned in the context" -- not because the assistant was
being overly cautious, but because nothing from that file ever entered
the knowledge base in the first place.

Two different retrieval-shaped outputs, from the SAME sheet
-------------------------------------------------------------
A pure "chunk the raw rows and embed them" approach is exactly wrong for
a question like "analyze primary sales": with an 87,000+ row sheet, any
single retrieval call only ever sees a handful of chunks (RETRIEVAL_TOP_K
/ CONTEXT_MAX_CHARS both cap this hard, and correctly so -- the LLM's own
context window is finite). Answering an aggregate question ("total sales
by sales center", "how many distinct sales centers") from a random
sample of a few hundred rows out of 87,000 produces a plausible-looking
but WRONG number, and a truncated list of sales centers (exactly what was
reported) is the visible symptom of that sampling.

So this ingestor produces two kinds of chunk:

  1. Row-batch chunks: the raw data, `ROWS_PER_CHUNK` rows at a time with
     the header repeated in every chunk, so a question about a SPECIFIC
     row/entry ("what did sales center X sell on date Y") can still be
     answered by ordinary retrieval.

  2. Aggregate chunks: computed ONCE, over the ENTIRE sheet, not a
     sample -- total/count per distinct value of every categorical
     column crossed with every numeric column, plus an overall-totals
     and column-inventory chunk. These carry the real answer to "how
     many sales centers", "total primary sales", "top N by volume", etc.
     regardless of which few chunks a later retrieval call happens to
     pick, and regardless of context-budget truncation.

Large-file handling
--------------------
openpyxl's `read_only=True` mode streams rows instead of loading the
whole worksheet's cell-object graph into memory, so an 87k-row sheet is
handled the same way a 300-page PDF already is elsewhere in this
package: memory use stays flat and there's no hard row-count limit.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import openpyxl
    _OPENPYXL_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _OPENPYXL_AVAILABLE = False

# How many data rows go into one retrievable "row-batch" chunk. Kept
# small enough that a chunk stays well under ingest.ingest.MAX_CHUNK_CHARS
# for a typical ~8-12 column sales register, while still keeping the
# TOTAL chunk count for an 87k-row sheet manageable (~870 chunks at the
# default of 100 rows/chunk, versus 87,000 one-row chunks).
ROWS_PER_CHUNK = int(os.getenv("XLSX_ROWS_PER_CHUNK", "100"))

# A column is treated as "categorical" (worth aggregating BY) if it has
# a reasonably small number of distinct values relative to the row
# count -- a sales-center/region/product/month column, not a free-text
# remarks field or a per-row unique ID.
_MAX_CATEGORY_CARDINALITY = 500
_MIN_CATEGORY_ROWS = 2  # a column with every value unique isn't a category


def _col_letter_free_normalize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _looks_numeric(values: List[Any]) -> bool:
    numeric_count = 0
    total = 0
    for v in values:
        if v is None or v == "":
            continue
        total += 1
        if isinstance(v, (int, float)):
            numeric_count += 1
        else:
            try:
                float(str(v).replace(",", ""))
                numeric_count += 1
            except ValueError:
                pass
    return total > 0 and (numeric_count / total) >= 0.9


def _to_number(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


def _read_sheet_rows(ws) -> Tuple[List[str], List[List[Any]]]:
    """Stream every row of a worksheet (read_only mode). Returns
    (header, data_rows). The first non-empty row is treated as the
    header, matching how these registers are laid out in practice.

    IMPORTANT: openpyxl's read_only mode trusts the worksheet's stored
    <dimension> tag to decide how many rows/columns to iterate, instead
    of scanning the actual sheet data. That tag is only reliably correct
    for files openpyxl itself wrote. Files produced or re-saved by many
    other tools (SAP/Tally/Zoho exports, or Excel after rows were added
    beyond the sheet's originally-saved range) can leave a stale,
    understated dimension -- which silently truncates iteration to that
    stale row count with no error or warning, well short of the sheet's
    real data. `reset_dimensions()` forces openpyxl to actually scan and
    recompute the true used range before iterating, which is the fix for
    exactly the "only ~1,200 of 87,000+ rows ingested" symptom.
    """
    try:
        ws.reset_dimensions()
    except AttributeError:
        pass  # older openpyxl without this method; iter_rows still works, just trusts the tag

    header: List[str] = []
    data_rows: List[List[Any]] = []
    for row in ws.iter_rows(values_only=True):
        if not header:
            if all(c is None for c in row):
                continue  # skip leading blank rows
            header = [_col_letter_free_normalize(c) or f"col_{i+1}" for i, c in enumerate(row)]
            continue
        if all(c is None for c in row):
            continue  # skip fully blank rows anywhere in the sheet
        data_rows.append(list(row))
    return header, data_rows


def _classify_columns(
    header: List[str], data_rows: List[List[Any]]
) -> Tuple[List[int], List[int]]:
    """Returns (categorical_column_indices, numeric_column_indices)."""
    n_cols = len(header)
    categorical: List[int] = []
    numeric: List[int] = []
    n_rows = len(data_rows)

    for col_idx in range(n_cols):
        col_values = [row[col_idx] if col_idx < len(row) else None for row in data_rows]
        non_null = [v for v in col_values if v not in (None, "")]
        if not non_null:
            continue
        if _looks_numeric(non_null):
            numeric.append(col_idx)
            continue
        distinct = {str(v) for v in non_null}
        if _MIN_CATEGORY_ROWS <= len(distinct) <= min(_MAX_CATEGORY_CARDINALITY, max(1, n_rows // 2)):
            categorical.append(col_idx)

    return categorical, numeric


def _format_row_batch_markdown(header: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        cells = [str(row[i]) if i < len(row) and row[i] is not None else "" for i in range(len(header))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _build_aggregate_chunks(
    sheet_name: str,
    source_file: str,
    header: List[str],
    data_rows: List[List[Any]],
    categorical_idx: List[int],
    numeric_idx: List[int],
) -> List["IngestedChunk"]:
    from ingest.ingest import IngestedChunk  # local import: avoids a cycle at module load

    chunks: List[IngestedChunk] = []
    n_rows = len(data_rows)

    # 1. Whole-sheet inventory chunk -- total row count, every column
    #    name, and (for numeric columns) a grand total/average/min/max
    #    computed over ALL rows, not a sample. This alone answers "how
    #    many rows/entries are there" and "what's the overall total"
    #    correctly regardless of retrieval sampling.
    overview_lines = [
        f"Sheet '{sheet_name}' in {os.path.basename(source_file)}: {n_rows} data row(s), "
        f"{len(header)} column(s): {', '.join(header)}.",
    ]
    for idx in numeric_idx:
        col_name = header[idx]
        values = [_to_number(row[idx]) for row in data_rows if idx < len(row)]
        values = [v for v in values if v is not None]
        if not values:
            continue
        overview_lines.append(
            f"- {col_name}: total = {sum(values):,.2f}, average = {sum(values)/len(values):,.2f}, "
            f"min = {min(values):,.2f}, max = {max(values):,.2f}, non-empty rows = {len(values)}."
        )
    chunks.append(IngestedChunk(
        text=_clean_local("\n".join(overview_lines)),
        source_file=source_file,
        page_or_slide=0,
        section=f"{sheet_name} — Overview & Totals",
        category="company",
        record={"name": f"{sheet_name} overview", "sheet": sheet_name},
        chunk_type="text",
        confidence=1.0,
    ))

    # 2. Per-categorical-column x per-numeric-column aggregate chunks --
    #    the actual "analyze primary sales by X" answer, computed once
    #    over every row.
    for cat_idx in categorical_idx:
        cat_name = header[cat_idx]
        for num_idx in numeric_idx:
            num_name = header[num_idx]
            totals: Dict[str, float] = defaultdict(float)
            counts: Dict[str, int] = defaultdict(int)
            for row in data_rows:
                if cat_idx >= len(row):
                    continue
                key = _col_letter_free_normalize(row[cat_idx]) or "(blank)"
                val = _to_number(row[num_idx]) if num_idx < len(row) else None
                if val is not None:
                    totals[key] += val
                    counts[key] += 1
            if not totals:
                continue
            ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
            lines = [
                f"{num_name} grouped by {cat_name} — sheet '{sheet_name}' in "
                f"{os.path.basename(source_file)} (computed over all {n_rows} rows, "
                f"{len(ranked)} distinct {cat_name} value(s)):"
            ]
            for key, total in ranked:
                lines.append(f"- {key}: {total:,.2f} (across {counts[key]} row(s))")
            chunks.append(IngestedChunk(
                text=_clean_local("\n".join(lines)),
                source_file=source_file,
                page_or_slide=0,
                section=f"{sheet_name} — {num_name} by {cat_name}",
                category="company",
                record={"name": f"{num_name} by {cat_name}", "sheet": sheet_name},
                chunk_type="text",
                confidence=1.0,
            ))

    return chunks


def _clean_local(text: str) -> str:
    return text.strip()


def _read_csv_rows(path: str) -> Tuple[List[str], List[List[Any]]]:
    """Read a CSV file and return (header, data_rows) using the same
    format as _read_sheet_rows for XLSX. Uses Python's built-in csv module
    with automatic delimiter detection."""
    import csv

    data_rows: List[List[Any]] = []
    header: List[str] = []

    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            # Sniff delimiter
            sample = f.read(8192)
            f.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter if sample else ","
            reader = csv.reader(f, delimiter=delimiter)

            for row in reader:
                if not header:
                    if all(c is None or c == "" for c in row):
                        continue
                    header = [_col_letter_free_normalize(c) or f"col_{i+1}" for i, c in enumerate(row)]
                    continue
                if all(c is None or c == "" for c in row):
                    continue
                data_rows.append(list(row))
    except UnicodeDecodeError:
        # Fallback for non-UTF8 CSVs (common in some regions)
        with open(path, "r", encoding="cp1252", newline="") as f:
            sample = f.read(8192)
            f.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter if sample else ","
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if not header:
                    if all(c is None or c == "" for c in row):
                        continue
                    header = [_col_letter_free_normalize(c) or f"col_{i+1}" for i, c in enumerate(row)]
                    continue
                if all(c is None or c == "" for c in row):
                    continue
                data_rows.append(list(row))

    return header, data_rows


def ingest_csv(path: str) -> List["IngestedChunk"]:
    """Ingest a .csv file. Returns a list of IngestedChunk with aggregate
    chunks first, then row-batch chunks."""
    from ingest.ingest import IngestedChunk  # local import: avoids a cycle at module load

    all_chunks: List["IngestedChunk"] = []

    header, data_rows = _read_csv_rows(path)
    if not header or not data_rows:
        logger.info("CSV %s has no data rows; skipping.", path)
        return []

    logger.info(
        "Ingesting CSV %s: %d rows, %d columns.",
        os.path.basename(path), len(data_rows), len(header),
    )

    sheet_name = os.path.splitext(os.path.basename(path))[0]
    categorical_idx, numeric_idx = _classify_columns(header, data_rows)

    all_chunks.extend(_build_aggregate_chunks(
        sheet_name, path, header, data_rows, categorical_idx, numeric_idx,
    ))

    for start in range(0, len(data_rows), ROWS_PER_CHUNK):
        batch = data_rows[start:start + ROWS_PER_CHUNK]
        end = start + len(batch)
        text = _format_row_batch_markdown(header, batch)
        from ingest.ingest import IngestedChunk  # local import
        all_chunks.append(IngestedChunk(
            text=_clean_local(
                f"CSV '{sheet_name}' rows {start + 1}-{end} "
                f"(of {len(data_rows)} total):\n\n{text}"
            ),
            source_file=path,
            page_or_slide=0,
            section=f"{sheet_name} — rows {start + 1}-{end}",
            category="company",
            record={"name": f"{sheet_name} rows {start+1}-{end}", "sheet": sheet_name},
            chunk_type="text",
            confidence=1.0,
        ))

    logger.info(
        "Finished %s: %d chunk(s) total (%d aggregate + row-batches).",
        path, len(all_chunks),
        sum(1 for c in all_chunks if "rows " not in c.section or "— rows" not in c.section),
    )
    return all_chunks


def ingest_xlsx(path: str) -> List["IngestedChunk"]:
    """Ingest one .xlsx/.xls workbook. Returns a list of IngestedChunk:
    aggregate chunks first (so they're never the ones dropped if a
    downstream cap on chunks-per-source ever applies), then row-batch
    chunks covering every row in every sheet."""
    from ingest.ingest import IngestedChunk  # local import: avoids a cycle at module load

    if not _OPENPYXL_AVAILABLE:
        logger.error("Cannot ingest %s: openpyxl is not installed.", path)
        return []

    all_chunks: List[IngestedChunk] = []
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception:
        logger.exception("Failed to open workbook: %s", path)
        return []

    try:
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            header, data_rows = _read_sheet_rows(ws)
            if not header or not data_rows:
                logger.info("Sheet '%s' in %s has no data rows; skipping.", sheet_name, path)
                continue

            logger.info(
                "Ingesting sheet '%s' from %s: %d rows, %d columns.",
                sheet_name, os.path.basename(path), len(data_rows), len(header),
            )

            categorical_idx, numeric_idx = _classify_columns(header, data_rows)

            all_chunks.extend(_build_aggregate_chunks(
                sheet_name, path, header, data_rows, categorical_idx, numeric_idx,
            ))

            for start in range(0, len(data_rows), ROWS_PER_CHUNK):
                batch = data_rows[start:start + ROWS_PER_CHUNK]
                end = start + len(batch)
                text = _format_row_batch_markdown(header, batch)
                all_chunks.append(IngestedChunk(
                    text=_clean_local(
                        f"Sheet '{sheet_name}' rows {start + 1}-{end} "
                        f"(of {len(data_rows)} total):\n\n{text}"
                    ),
                    source_file=path,
                    page_or_slide=0,
                    section=f"{sheet_name} — rows {start + 1}-{end}",
                    category="company",
                    record={"name": f"{sheet_name} rows {start+1}-{end}", "sheet": sheet_name},
                    chunk_type="text",
                    confidence=1.0,
                ))
    finally:
        wb.close()

    logger.info(
        "Finished %s: %d chunk(s) total (%d aggregate + row-batches).",
        path, len(all_chunks),
        sum(1 for c in all_chunks if "rows " not in c.section or "— rows" not in c.section),
    )
    return all_chunks
