"""
ingestion/complex_xlsx_ingestor.py

Ingestion for COMPLEX .xlsx/.xls spreadsheet files with:
- Multi-row headers (2-3+ rows)
- Merged cells spanning columns
- Hierarchical row structure (SKU - FRANCHISE - SEGMENT - SUB SEGMENT)
- Repeating column triplets (Plan / MTD / %ACH per geography)
- Multiple time-period sections (DEC, MAY, etc.)
- Special values ("-", "#DIV/0!", blanks)
- Blank rows as section boundaries

Outputs IngestedChunk objects compatible with the existing knowledge index.
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import openpyxl
    from openpyxl.utils import get_column_letter
    _OPENPYXL_AVAILABLE = True
except ImportError:
    _OPENPYXL_AVAILABLE = False


@dataclass
class MergedCellRange:
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    value: Any


@dataclass
class ColumnDefinition:
    name: str
    level: int
    geography: str
    metric_type: str
    segment: Optional[str] = None
    original_indices: List[int] = None
    is_hierarchy_label: bool = False


@dataclass
class RowData:
    hierarchy: Dict[str, str]
    is_total_row: bool
    values: Dict[str, float]
    raw_values: Dict[str, Any]
    row_type: str


SECTION_KEYWORDS = {
    "FRANCHISE": "FRANCHISE",
    "SEGMENT": "SEGMENT",
    "SUB SEGMENT": "SUB SEGMENT",
    "BRAND": "BRAND",
    "TOTAL": "TOTAL",
    "ALL EUROPE": "TOTAL",
}

METRIC_PATTERNS = {
    "PLAN": re.compile(r"^(DEC|MAY|JUN|JUL|AUG|SEP|OCT|NOV|JAN|FEB|MAR|APR)\s*PLAN$", re.IGNORECASE),
    "PLAN_REV": re.compile(r"^(DEC|MAY|JUN|JUL|AUG|SEP|OCT|NOV|JAN|FEB|MAR|APR)\s*PLAN\s*REV$", re.IGNORECASE),
    "MTD": re.compile(r"^(DEC|MAY|JUN|JUL|AUG|SEP|OCT|NOV|JAN|FEB|MAR|APR)\s*MTD$", re.IGNORECASE),
    "PCT_ACH": re.compile(r"^(DEC|MAY|JUN|JUL|AUG|SEP|OCT|NOV|JAN|FEB|MAR|APR)\s*%?ACH$", re.IGNORECASE),
}

GEOGRAPHY_MARKERS = {"ALL EUROPE", "BRANCH", "STATE"}


def _normalize_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return str(v).strip()
    return str(v).strip()


def _to_number(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if s in ("-", "#DIV/0!", "N/A", "NA", "n/a", "#N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _looks_numeric(v: Any) -> bool:
    return _to_number(v) is not None


def _is_section_header(val: str) -> Optional[str]:
    if not val:
        return None
    val_upper = val.upper().strip()
    for kw, typ in SECTION_KEYWORDS.items():
        if val_upper.startswith(kw):
            return typ
    return None


def _detect_metric_type(header: str) -> Tuple[Optional[str], Optional[str]]:
    if not header:
        return None, None
    h = header.strip().upper()
    for metric, pattern in METRIC_PATTERNS.items():
        match = pattern.match(h)
        if match:
            period = match.group(1).upper()
            return period, metric
    if h in ("PLAN", "MTD", "%ACH", "ACH", "PCT", "PERCENT"):
        return None, h
    return None, None


def _is_geography_header(header: str) -> bool:
    if not header:
        return False
    h = header.strip().upper()
    return any(marker in h for marker in GEOGRAPHY_MARKERS) or h in ("TOTAL", "GRAND TOTAL")


class ComplexXLSXParser:
    def __init__(self, ws, source_file: str):
        self.ws = ws
        self.source_file = source_file
        self.merged_ranges = []
        self._load_merged_cells()

    def _load_merged_cells(self):
        if hasattr(self.ws, 'merged_cells'):
            for merged_range in self.ws.merged_cells.ranges:
                min_row, min_col, max_row, max_col = (
                    merged_range.min_row, merged_range.min_col,
                    merged_range.max_row, merged_range.max_col
                )
                cell = self.ws.cell(row=min_row, column=min_col)
                self.merged_ranges.append(MergedCellRange(
                    min_row=min_row, max_row=max_row,
                    min_col=min_col, max_col=max_col,
                    value=cell.value
                ))

    def _get_cell_value(self, row: int, col: int) -> Any:
        for mr in self.merged_ranges:
            if mr.min_row <= row <= mr.max_row and mr.min_col <= col <= mr.max_col:
                return mr.value
        cell = self.ws.cell(row=row, column=col)
        return cell.value

    def _read_all_rows(self) -> List[List[Any]]:
        try:
            self.ws.reset_dimensions()
        except AttributeError:
            pass

        rows = []
        for row in self.ws.iter_rows(values_only=False):
            row_data = []
            for cell in row:
                row_data.append(cell.value)
            if any(v is not None and v != "" for v in row_data):
                rows.append(row_data)
        return rows

    def _detect_header_rows(self) -> List[int]:
        """Detect which rows are header rows (typically 2-3 rows: geography + metric headers)."""
        max_check = 6
        max_row = self.ws.max_row if self.ws.max_row is not None else 100
        max_check = min(6, max_row)
        max_col = self.ws.max_column if self.ws.max_column is not None else 26

        candidate_rows = []
        for i in range(max_check):
            row_idx = i + 1
            has_geo = False
            text_col_count = 0
            
            for col_idx in range(1, max_col + 1):
                val = self._get_cell_value(row_idx, col_idx)
                if val is not None:
                    s = str(val).strip()
                    if s and not _looks_numeric(s):
                        text_col_count += 1
                    if _is_geography_header(s):
                        has_geo = True
            
            # Title row check
            if row_idx == 1:
                non_empty_count = sum(1 for col_idx in range(1, max_col + 1) 
                    if self._get_cell_value(1, col_idx) is not None and str(self._get_cell_value(1, col_idx)).strip())
                if non_empty_count <= 1:
                    continue
            
            # A header row has text in multiple columns, or has geography
            if text_col_count >= 3 or has_geo:
                candidate_rows.append(i)
        
        # If no candidates found with >=3 text cols, check for geography row
        if not candidate_rows:
            for i in range(max_check):
                row_idx = i + 1
                has_geo = False
                for col_idx in range(1, max_col + 1):
                    val = self._get_cell_value(row_idx, col_idx)
                    if val is not None:
                        s = str(val).strip()
                        if _is_geography_header(s):
                            has_geo = True
                            break
                if has_geo:
                    candidate_rows.append(i)
        
        # If still no candidates, check for rows with >=3 text columns (metric headers)
        if not candidate_rows:
            for i in range(max_check):
                row_idx = i + 1
                text_col_count = 0
                for col_idx in range(1, max_col + 1):
                    val = self._get_cell_value(row_idx, col_idx)
                    if val is not None:
                        s = str(val).strip()
                        if s and not _looks_numeric(s):
                            text_col_count += 1
                if text_col_count >= 3:
                    candidate_rows.append(i)
        
        # Take the last 2-3 candidates as headers
        if len(candidate_rows) >= 2:
            header_rows = candidate_rows[-3:]
        elif candidate_rows:
            header_rows = candidate_rows
        else:
            # Fallback: assume rows 1, 2 are headers (0-indexed)
            header_rows = [1, 2]
        
        return header_rows

    def _build_column_definitions(
        self, raw_rows: List[List[Any]], header_rows: List[int]
    ) -> List[ColumnDefinition]:
        if not header_rows:
            return []

        num_cols = max(len(row) for row in raw_rows[:max(header_rows)+1]) if header_rows else 0
        columns = []

        merged_map = {}
        for mr in self.merged_ranges:
            if mr.min_row - 1 < len(header_rows):
                for r in range(mr.min_row - 1, mr.max_row):
                    for c in range(mr.min_col - 1, mr.max_col):
                        merged_map[(r, c)] = mr

        for col_idx in range(num_cols):
            header_vals = []
            for hr_idx in header_rows:
                val = self._get_cell_value(hr_idx + 1, col_idx + 1)
                header_vals.append(_normalize_value(val))

            col_def = self._analyze_column(header_vals, col_idx, header_rows, merged_map)
            columns.append(col_def)

        columns = self._group_triplets(columns)
        return columns

    def _analyze_column(
        self, header_vals: List[str], col_idx: int,
        header_rows: List[int], merged_map: Dict
    ) -> ColumnDefinition:
        primary = header_vals[-1] if header_vals else ""
        secondary = header_vals[-2] if len(header_vals) > 1 else ""
        tertiary = header_vals[-3] if len(header_vals) > 2 else ""

        period, metric = None, None
        for val in reversed(header_vals):
            p, m = _detect_metric_type(val)
            if p or m:
                period, metric = p, m
                break

        geo = None
        for val in header_vals:
            if _is_geography_header(val):
                geo = val
                break

        is_hierarchy = False
        hierarchy_type = None
        for val in header_vals:
            ht = _is_section_header(val)
            if ht:
                is_hierarchy = True
                hierarchy_type = ht
                break

        name_parts = []
        if geo:
            name_parts.append(geo)
        if period:
            name_parts.append(period)
        if metric:
            name_parts.append(metric)
        if primary and not (geo or period or metric):
            name_parts.append(primary)

        col_name = " > ".join(name_parts) if name_parts else f"col_{col_idx+1}"

        level = 0
        if hierarchy_type:
            level = 1
        elif geo:
            level = 0
        elif metric:
            level = 1

        return ColumnDefinition(
            name=col_name,
            level=level,
            geography=geo or "",
            metric_type=metric or "",
            segment=hierarchy_type or "",
            original_indices=[col_idx],
            is_hierarchy_label=is_hierarchy,
        )

    def _group_triplets(self, columns: List[ColumnDefinition]) -> List[ColumnDefinition]:
        return columns

    def _parse_data_rows(
        self, raw_rows: List[List[Any]], header_rows: List[int],
        columns: List[ColumnDefinition]
    ) -> List[RowData]:
        data_start = max(header_rows) + 1 if header_rows else 0
        data_rows = []

        current_hierarchy = {"FRANCHISE": "", "SEGMENT": "", "SUB SEGMENT": "", "BRAND": ""}

        for row_idx in range(data_start, len(raw_rows)):
            raw_row = raw_rows[row_idx]
            if not raw_row or all(v is None or v == "" for v in raw_row):
                continue

            first_val = _normalize_value(raw_row[0]) if raw_row else ""
            section_type = _is_section_header(first_val)

            is_total = False
            row_type = "SKU"

            if section_type:
                current_hierarchy[section_type] = first_val
                is_total = True
                row_type = section_type
            elif "TOTAL" in first_val.upper() or "ALL EUROPE" in first_val.upper():
                is_total = True
                row_type = "TOTAL"
            elif "SKU" in first_val.upper() or first_val.upper().startswith("SKU"):
                row_type = "SKU"

            values = {}
            raw_values = {}
            for col in columns:
                for orig_idx in col.original_indices:
                    if orig_idx < len(raw_row):
                        raw_val = raw_row[orig_idx]
                        num_val = _to_number(raw_val)
                        key = col.name
                        raw_values[key] = raw_val
                        if num_val is not None:
                            values[key] = num_val
                        break

            data_rows.append(RowData(
                hierarchy=current_hierarchy.copy(),
                is_total_row=is_total,
                values=values,
                raw_values=raw_values,
                row_type=row_type,
            ))

        return data_rows

    def parse(self) -> Tuple[List[ColumnDefinition], List[RowData], Dict[str, Any]]:
        raw_rows = self._read_all_rows()
        if not raw_rows:
            return [], [], {}

        header_rows = self._detect_header_rows()
        columns = self._build_column_definitions(raw_rows, header_rows)
        data_rows = self._parse_data_rows(raw_rows, header_rows, columns)

        metadata = {
            "sheet_name": self.ws.title,
            "source_file": os.path.basename(self.source_file),
            "total_rows": len(data_rows),
            "num_columns": len(columns),
            "header_rows": len(header_rows),
        }

        return columns, data_rows, metadata


def _build_complex_chunks(
    sheet_name: str,
    source_file: str,
    columns: List[ColumnDefinition],
    rows: List[RowData],
    metadata: Dict[str, Any],
) -> List["IngestedChunk"]:
    from ingest.ingest import IngestedChunk

    chunks = []
    n_rows = len(rows)

    if n_rows == 0:
        return chunks

    overview_lines = [
        f"Complex Sheet '{sheet_name}' in {os.path.basename(source_file)}: {n_rows} data row(s), "
        f"{len(columns)} column(s) parsed with hierarchical structure.",
        f"Header rows: {metadata.get('header_rows', 0)}",
        f"Hierarchy levels found: {set(r.row_type for r in rows)}",
        f"Geographies: {set(c.geography for c in columns if c.geography)}",
        f"Metrics: {set(c.metric_type for c in columns if c.metric_type)}",
    ]
    chunks.append(IngestedChunk(
        text="\n".join(overview_lines),
        source_file=source_file,
        page_or_slide=0,
        section=f"{sheet_name} — Complex Overview",
        category="company",
        record={"name": f"{sheet_name} complex overview", "sheet": sheet_name},
        chunk_type="text",
        confidence=1.0,
    ))

    geo_groups = defaultdict(list)
    for col in columns:
        if col.geography:
            geo_groups[col.geography].append(col)

    for geo, geo_cols in geo_groups.items():
        metric_groups = defaultdict(list)
        for col in geo_cols:
            metric_groups[col.metric_type].append(col)

        for metric, metric_cols in metric_groups.items():
            hierarchy_totals = defaultdict(lambda: defaultdict(float))
            hierarchy_counts = defaultdict(lambda: defaultdict(int))

            for row in rows:
                if not row.values:
                    continue
                hier_key = " > ".join(f"{k}:{v}" for k, v in row.hierarchy.items() if v)
                if not hier_key:
                    hier_key = row.row_type

                for col in metric_cols:
                    for orig_idx in col.original_indices:
                        if num_val := row.values.get(col.name):
                            hierarchy_totals[hier_key][col.name] += num_val
                            hierarchy_counts[hier_key][col.name] += 1

            if not hierarchy_totals:
                continue

            lines = [
                f"{metric} for {geo} — sheet '{sheet_name}' in "
                f"{os.path.basename(source_file)} (computed over {n_rows} rows):"
            ]
            for hier_key in sorted(hierarchy_totals.keys()):
                lines.append(f"- {hier_key}:")
                for col_name, total in sorted(hierarchy_totals[hier_key].items(), key=lambda x: -x[1]):
                    cnt = hierarchy_counts[hier_key][col_name]
                    lines.append(f"  {col_name}: {total:,.2f} (across {cnt} rows)")

            chunks.append(IngestedChunk(
                text="\n".join(lines),
                source_file=source_file,
                page_or_slide=0,
                section=f"{sheet_name} — {metric} by {geo}",
                category="company",
                record={"name": f"{metric} by {geo}", "sheet": sheet_name},
                chunk_type="text",
                confidence=1.0,
            ))

    ROWS_PER_CHUNK = 50
    for start in range(0, len(rows), ROWS_PER_CHUNK):
        batch = rows[start:start + ROWS_PER_CHUNK]

        col_names = [c.name for c in columns if not c.is_hierarchy_label]
        if not col_names:
            col_names = [c.name for c in columns]

        header_line = ["Hierarchy", "Type"] + col_names
        md_lines = ["| " + " | ".join(header_line) + " |",
                    "|" + "|".join(["---"] * len(header_line)) + "|"]

        for row in batch:
            hier_str = " > ".join(f"{k}:{v}" for k, v in row.hierarchy.items() if v) or "—"
            cells = [hier_str, row.row_type]
            for col_name in col_names:
                val = row.values.get(col_name)
                if val is not None:
                    cells.append(f"{val:,.2f}")
                else:
                    raw = row.raw_values.get(col_name)
                    cells.append("" if raw is None else str(raw))
            md_lines.append("| " + " | ".join(cells) + " |")

        batch_text = "\n".join(md_lines)
        chunks.append(IngestedChunk(
            text=f"Sheet '{sheet_name}' rows {start+1}-{start+len(batch)} (of {n_rows}):\n\n{batch_text}",
            source_file=source_file,
            page_or_slide=0,
            section=f"{sheet_name} — rows {start+1}-{start+len(batch)}",
            category="company",
            record={"name": f"{sheet_name} rows {start+1}-{start+len(batch)}", "sheet": sheet_name},
            chunk_type="text",
            confidence=0.9,
        ))

    return chunks


def ingest_complex_xlsx(path: str) -> List["IngestedChunk"]:
    if not _OPENPYXL_AVAILABLE:
        logger.error("Cannot ingest %s: openpyxl is not installed.", path)
        return []

    try:
        wb = openpyxl.load_workbook(path, read_only=False, data_only=True)
    except Exception:
        logger.exception("Failed to open workbook: %s", path)
        return []

    all_chunks = []
    try:
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]

            parser = ComplexXLSXParser(ws, path)
            columns, rows, metadata = parser.parse()

            if not columns or not rows:
                logger.info("Sheet '%s' in %s produced no parseable data; skipping.", sheet_name, path)
                continue

            logger.info(
                "Parsed complex sheet '%s' from %s: %d rows, %d columns, %d hierarchy types.",
                sheet_name, os.path.basename(path), len(rows), len(columns),
                len(set(r.row_type for r in rows))
            )

            sheet_chunks = _build_complex_chunks(sheet_name, path, columns, rows, metadata)
            all_chunks.extend(sheet_chunks)

    finally:
        wb.close()

    logger.info(
        "Finished complex XLSX %s: %d chunk(s) total.",
        path, len(all_chunks)
    )
    return all_chunks