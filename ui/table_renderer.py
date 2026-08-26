"""
ui/table_renderer.py

Rich table renderer for Direct RAG and LLM results.
"""

from typing import Any, Dict, List

from rich import box
from rich.table import Table

_PRIORITY_COLS = ["id", "name", "head", "owner", "status", "description"]


def build_table(title: str, records: List[Dict[str, Any]]) -> Table:
    """
    Build a Rich Table from a list of record dicts.
    Column order: priority columns first, then any remaining keys.
    """
    all_keys: List[str] = []
    seen = set()
    for rec in records:
        for k in rec:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    ordered = [k for k in _PRIORITY_COLS if k in seen]
    ordered += [k for k in all_keys if k not in ordered]

    table = Table(
        title=title,
        box=box.ROUNDED,
        show_lines=True,
        border_style="cyan",
        header_style="bold cyan",
    )

    for col in ordered:
        table.add_column(col.replace("_", " ").upper(), overflow="fold")

    for rec in records:
        row = []
        for col in ordered:
            val = rec.get(col, "—")
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            elif isinstance(val, dict):
                val = "; ".join(f"{k}: {v}" for k, v in val.items())
            row.append(str(val))
        table.add_row(*row)

    return table
