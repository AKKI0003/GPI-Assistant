"""
retrieval/context_builder.py

Converts search results into a clean, human-readable context block
that is injected into the LLM prompt.

Design
------
- Recursive JSON → readable text (no raw JSON to the LLM)
- Respects a character budget so the LLM context window is not overflowed
- De-duplicates sections when the same content appears under different records
- Degrades gracefully under budget pressure: a block that doesn't fit is
  SKIPPED (not a hard stop), so later, shorter, equally-relevant records
  still make it into the context instead of being silently discarded.
  This matters most for table/list/comparison requests, which retrieve
  many records and need as many of them represented as possible.
"""

from typing import Any, Dict, List, Optional

from core.config import CONTEXT_MAX_CHARS
from core.direct_answer import clean_table_chunk_text


# ─── Recursive flattener ──────────────────────────────────────────────────────

def _flatten_obj(obj: Any, prefix: str = "", indent: int = 0) -> List[str]:
    """Recursively render a JSON value into human-readable lines."""
    pad = "  " * indent
    lines: List[str] = []

    if isinstance(obj, dict):
        for key, val in obj.items():
            label = str(key).replace("_", " ").title()
            if isinstance(val, (dict, list)):
                lines.append(f"{pad}{label}:")
                lines.extend(_flatten_obj(val, prefix, indent + 1))
            else:
                if val not in (None, "", [], {}):
                    lines.append(f"{pad}{label}: {val}")

    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.extend(_flatten_obj(item, prefix, indent))
                lines.append("")  # blank separator between list items
            else:
                if item not in (None, ""):
                    lines.append(f"{pad}• {item}")
    else:
        if obj not in (None, ""):
            lines.append(f"{pad}{obj}")

    return lines


# ─── Single-record formatter ──────────────────────────────────────────────────

def format_result(result: Dict[str, Any]) -> str:
    """
    Format one search result as a readable block.

    Header format:  [CATEGORY | section]
    Body:           key: value pairs, nested structures indented
    """
    category = result.get("category", "unknown").upper()
    section  = result.get("section", "")
    record   = dict(result.get("record", {}))

    # Clean up ambiguous split-header table text (e.g. a literal "Year
    # ended" / "Year ended" header row with the real dates sitting in
    # the NEXT row instead of merged into it) before the model ever
    # sees it. Left raw, the model has to infer which column is which
    # fiscal year purely from position, which is exactly the kind of
    # inference that was producing off-by-one year reads (e.g.
    # reporting FY2024's figure as FY2025's). This only touches
    # `record["text"]`, which is where PDF table chunks carry their
    # own copy of the table -- it's a no-op for records without one.
    if isinstance(record.get("text"), str):
        record["text"] = clean_table_chunk_text(record["text"])

    header = f"[{category}" + (f" | {section.upper()}" if section else "") + "]"

    lines = [header]
    lines.extend(_flatten_obj(record))
    # Explicit "end of record" marker so the model cannot mistake a record
    # boundary for a place to insert its own continuation from training data.
    lines.append("--- end of record ---")
    lines.append("")
    return "\n".join(lines)


# ─── Context builder ──────────────────────────────────────────────────────────

def _source_key(result: Dict[str, Any]) -> str:
    """Group results the same way a person would think about 'a source':
    one document/company/year -- NOT one search result. Falls back to
    the section name for records with no clearer identity, and finally
    to a per-result unique key so ungroupable records never accidentally
    get merged into the same budget bucket as unrelated ones."""
    record = result.get("record", {}) or {}
    source_file = record.get("source_file")
    year = record.get("year")
    if source_file:
        return f"{source_file}::{year}" if year else str(source_file)
    company = record.get("company")
    if company and year:
        return f"{company}::{year}"
    section = result.get("section")
    if section:
        return str(section)
    return f"__unkeyed_{id(result)}"


def build_context(
    results: List[Dict[str, Any]],
    max_chars: Optional[int] = None,
) -> str:
    """
    Combine multiple search results into a single context string,
    respecting a character budget.

    Parameters
    ----------
    results   : Search results from KnowledgeIndex, already ordered by
                relevance (most relevant first).
    max_chars : Optional override of the character budget. Callers that
                retrieve many records for aggregate requests (tables,
                comparisons, full lists) should pass a larger budget —
                see core.config.CONTEXT_MAX_CHARS_TABLE. Defaults to
                core.config.CONTEXT_MAX_CHARS.

    Behavior
    --------
    The budget is first FAIRLY RESERVED across every distinct source
    (see `_source_key` -- effectively one document/company/year) that
    appears anywhere in `results`, at `budget / num_sources` each. This
    matters specifically for multi-year/multi-document questions (e.g.
    "VST's PAT for 2023, 2024, and 2025"): without it, blocks are filled
    strictly in relevance-ranked order, so it's entirely possible for
    2023's and 2024's chunks -- especially once continuation pages get
    stitched onto them -- to consume the ENTIRE budget before 2025's
    chunk is ever reached, silently dropping a whole year rather than
    each year losing a little space. It was also how a real number
    (PBT) could survive for one year while a real, correctly-retrieved
    number a few rows further down the SAME table (PAT) got cut off by
    the old "truncate whichever block happens to be filling when the
    budget runs out" behavior -- one source hogging space starved a
    part of itself, not just other sources. Reservations only apply
    when that source actually needs the space: an unused/partial
    reservation immediately becomes free for every other source to draw
    from, so this never wastes budget when one source is small.

    Within its own reservation, and for any leftover/reclaimed budget,
    blocks are still added in relevance order, and a block that doesn't
    fit is SKIPPED rather than causing an immediate stop, so later,
    shorter, equally-relevant records still get a chance instead of
    being discarded outright.
    """
    budget = max_chars if max_chars is not None else CONTEXT_MAX_CHARS

    sources_in_order: List[str] = []
    seen_sources = set()
    for result in results:
        key = _source_key(result)
        if key not in seen_sources:
            seen_sources.add(key)
            sources_in_order.append(key)
    num_sources = max(len(sources_in_order), 1)
    per_source_reserve = budget / num_sources

    used_by_source: Dict[str, float] = {k: 0.0 for k in sources_in_order}
    blocks: List[str] = []
    total = 0
    omitted_count = 0

    for result in results:
        key = _source_key(result)
        block = format_result(result)
        block_len = len(block)

        # This source's own reservation, PLUS whatever the total budget
        # still has free right now (an unused reservation from a
        # smaller/already-exhausted source, or budget nobody's claimed
        # yet). A source is only capped by its reservation once every
        # other source's reservation is also actually spoken for --
        # until then, unclaimed room is fair game for anyone.
        source_room = per_source_reserve - used_by_source.get(key, 0.0)
        overall_room = budget - total
        room = max(0.0, min(source_room, overall_room)) if source_room > 0 else 0.0
        # If every source's reservation is exhausted but the overall
        # budget still has room (common once most sources turn out to
        # be short), fall back to whatever's globally left rather than
        # refusing space that would otherwise go to waste.
        if room <= 0 and overall_room > 0 and all(
            used_by_source.get(k, 0.0) >= per_source_reserve for k in sources_in_order
        ):
            room = overall_room

        if block_len <= room:
            blocks.append(block)
            total += block_len
            used_by_source[key] = used_by_source.get(key, 0.0) + block_len
            continue

        # Doesn't fit in this source's current room. If there's still
        # meaningful room left FOR THIS SOURCE, include a truncated
        # version so partial information isn't lost entirely; otherwise
        # skip and keep checking subsequent results (a later, shorter
        # block -- for this source or another -- may still fit).
        if room > 200:
            # Cut at the last full line inside the remaining budget rather
            # than at a raw character offset -- a mid-word/mid-row cut
            # (e.g. slicing a table row in half, or a bulleted list of
            # names/sales-centers mid-entry) can read as if that row's
            # data is real but incomplete, when really the whole row
            # just never made it in. Cutting on a line boundary means
            # every line the model sees is a complete, real record.
            truncated = block[:int(room)]
            last_newline = truncated.rfind("\n")
            if last_newline > 0:
                truncated = truncated[:last_newline]
            blocks.append(
                truncated + "\n[...remaining rows/records of this block omitted "
                "due to context size limit — do not assume they don't exist, "
                "and say so if the user needs the full list]"
            )
            total += len(truncated)
            used_by_source[key] = used_by_source.get(key, 0.0) + len(truncated)
        else:
            omitted_count += 1
        # Do not break: keep scanning remaining results in case any are
        # short enough to still squeeze into their own or another
        # source's remaining room.

    if omitted_count:
        blocks.append(
            f"[{omitted_count} additional matching record(s) exist but were "
            "omitted entirely due to context size limit — do not treat this "
            "as the complete set.]"
        )

    return "\n".join(blocks).strip()