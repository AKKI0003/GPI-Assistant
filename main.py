"""
main.py

GPI Knowledge Assistant — Rich terminal interface.

Entry point:   python main.py
"""

import sys
import os
import re

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from datetime import datetime
from typing import Optional

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from core.engine import AssistantEngine, EngineResult
from memory.conversation_memory import ConversationMemory
from storage.chat_storage import ChatManager
from llm.ollama_client import stream_prompt
from llm.prompt_builder import build_prompt
from core.intent_resolver import resolver
from core.direct_answer import (
    try_direct_answer, needs_llm, try_statement_table_answer, try_statement_table_structured,
    _merge_date_row_into_header, _drop_blank_columns,
    try_financial_fact_lookup, get_financial_facts_for_llm,
)
from core.query_normalizer import normalize_question
from core.chart_generator import wants_chart
from core.engine import _is_bare_chart_followup
from retrieval.knowledge_index import index as knowledge_index
from core.config import RETRIEVAL_TOP_K, LLM_MAX_TOKENS, LLM_MAX_TOKENS_TABLE, OLLAMA_TIMEOUT, OLLAMA_TIMEOUT_TABLE, STATEMENT_TABLE_DIRECT_ROUTE

# Actions that use the wider context/output budget (see _token_budget below)
# also need the correspondingly longer OLLAMA_TIMEOUT_TABLE — a detailed
# multi-entity comparison genuinely takes several minutes to prefill and
# generate on a CPU-bound local model, and the default OLLAMA_TIMEOUT
# (300s) was written for ordinary single-fact questions, not this.
_BROAD_ACTIONS = {"table", "compare"}


def _token_budget(action: str) -> int:
    """Tables and comparisons need a much larger output budget than prose."""
    if action in ("table", "compare"):
        return LLM_MAX_TOKENS_TABLE
    return LLM_MAX_TOKENS


console = Console()


# ─── Streaming UI ─────────────────────────────────────────────────────────────

def stream_and_render(prompt_text: str, max_tokens: int = None, timeout: int = OLLAMA_TIMEOUT) -> str:
    """
    Stream LLM output and render it live, then show final answer panel.

    ``timeout``: passed straight through to stream_prompt(). Callers with
    a broad/table/compare action MUST pass OLLAMA_TIMEOUT_TABLE here —
    without it this silently falls back to stream_prompt()'s own default
    (OLLAMA_TIMEOUT, 300s), which is what previously caused detailed
    multi-entity comparisons to abort with a timeout even though the
    rest of the app already had a longer timeout configured for exactly
    this kind of request.
    """
    accumulated = ""

    with Live(
        Panel("Thinking…", title="[bold yellow]Generating[/bold yellow]", border_style="yellow"),
        transient=True,
        refresh_per_second=20,
        console=console,
    ) as live:
        stream_kwargs = {"timeout": timeout}
        if max_tokens is not None:
            stream_kwargs["max_tokens"] = max_tokens
        for chunk in stream_prompt(prompt_text, **stream_kwargs):
            accumulated += chunk
            live.update(
                Panel(
                    accumulated[-3000:],
                    title="[bold yellow]Generating…[/bold yellow]",
                    border_style="yellow",
                )
            )

    console.print()
    console.print(
        Panel(
            _render_final_answer(accumulated),
            title="[bold cyan]Answer[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    return accumulated


def _looks_like_pipe_table(text: str) -> bool:
    """True if *text* contains Markdown pipe-table syntax."""
    return bool(re.search(r"^\s*\|.+\|\s*$", text, re.MULTILINE))


_SEPARATOR_ROW_RE = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")


def _split_pipe_row(line: str) -> list:
    """Split one Markdown table row into its cells."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _explode_multiline_rows(data_rows: list) -> list:
    """
    Occasionally the model crams several line items into ONE table row
    by putting real line breaks INSIDE that row's own cells, instead of
    writing one row per line item -- e.g. a row whose first cell is
    literally "Profit Before Tax\\nIncome Tax Expense\\nProfit After Tax"
    and whose last cell is "78,897.45\\n19,856.91\\n59,040.54". Rich
    happily renders that as a single row with multi-line cell content,
    which reads as a table that's lost almost all its structure -- every
    line item's label and value are still THERE, just stacked into one
    visually confusing row instead of three separate, properly aligned
    ones.

    If every cell in a row splits into the SAME number of lines (more
    than one), that's a strong signal this happened -- explode that one
    row back into that many separate rows, pairing up each cell's Nth
    line across all columns. Rows that don't show this pattern (the
    normal case) pass through completely unchanged.
    """
    exploded = []
    for row in data_rows:
        cell_lines = [c.split("\n") for c in row]
        line_counts = {len(cl) for cl in cell_lines}
        if len(line_counts) == 1 and next(iter(line_counts)) > 1:
            n = next(iter(line_counts))
            for i in range(n):
                exploded.append([cl[i].strip() for cl in cell_lines])
        else:
            exploded.append(row)
    return exploded


def _build_rich_table_from_lines(table_lines: list) -> Table:
    """
    Turn a block of consecutive pipe-table lines (as the LLM wrote them,
    in one self-contained turn) into an actual bordered rich.table.Table
    -- the same look the direct route already uses (box.ROUNDED, cyan
    header) -- instead of leaving it as unstyled monospace text.

    This is a much smaller, safer version of the row-fitting the OLD
    direct route used to do: there, rows being forced into one shape
    came from MANY DIFFERENT chunks/pages that could each have a
    genuinely different column count, which is what caused misaligned
    columns. Here, every row comes from a SINGLE table the model wrote
    in one continuous turn, so padding/truncating a stray uneven row to
    the header's column count is a minor cosmetic safety net, not a
    data-correctness risk.
    """
    rows = [_split_pipe_row(l) for l in table_lines if not _SEPARATOR_ROW_RE.match(l)]
    if not rows:
        raise ValueError("no data rows")
    header_cells, data_rows = rows[0], rows[1:]
    data_rows = _explode_multiline_rows(data_rows)

    # The model sometimes reproduces a source chunk's own messy table
    # structure verbatim -- a header row that's mostly blank leading
    # cells ("| | | | Year ended | Year ended |"), with the REAL column
    # dates sitting in a separate row right below it instead of merged
    # into the header, plus assorted fully-blank columns. Left as-is,
    # that produces a rich table whose header shows only a generic
    # "Year ended" (or nothing) instead of the actual two years, and
    # whose real values end up several columns away from where the
    # (blank) header cells are -- reading as an empty table even though
    # the model's answer did contain every value. Reuse the exact same
    # cleanup the direct route already relies on for this.
    if data_rows:
        header_cells, data_rows = _merge_date_row_into_header(header_cells, data_rows)
        header_cells, data_rows = _drop_blank_columns(header_cells, data_rows)

    t = Table(
        box=box.ROUNDED, border_style="cyan", header_style="bold cyan",
        show_lines=False,
    )
    for i, h in enumerate(header_cells):
        t.add_column(h, overflow="fold", justify="left" if i == 0 else "right",
                     )
    for row in data_rows:
        cells = (row + [""] * len(header_cells))[:len(header_cells)]
        t.add_row(*cells)
    return t


def _render_final_answer(accumulated: str):
    """
    Choose how to render the model's finished answer.

    `Markdown(accumulated)` (Rich's CommonMark/GFM renderer) requires a
    pipe table to be well-formed -- a header row, a `---|---` separator
    row, and every row sharing the same column count, with no stray
    blank lines splitting it. The model doesn't always produce that
    (e.g. a row here or there with one extra/missing `|`). When the
    table doesn't parse as a valid GFM table, Rich's Markdown renderer
    doesn't show an error -- it silently falls back to treating those
    same lines as an ordinary paragraph, and CommonMark's own rule for
    paragraphs is that a single newline is just a "soft break" that
    renders as a SPACE, not a line break. The result: a table that
    displayed fine, line-by-line, while streaming suddenly reads as one
    long run-on sentence the moment the finished answer gets wrapped in
    Markdown() -- even though the model's own output already had the
    right line breaks the whole time.

    Fix: split the answer into alternating prose and pipe-table blocks.
    Prose blocks still render through Markdown (headings, bold, bullets
    all still work normally). Table blocks are parsed directly into a
    real, bordered rich.table.Table -- properly aligned columns and
    borders, matching the direct route's own look -- rather than either
    the fragile Markdown table parser or a flat, unstyled block of raw
    text.
    """
    if not _looks_like_pipe_table(accumulated):
        return Markdown(accumulated)

    lines = accumulated.split("\n")
    blocks = []          # list of ("prose"|"table", [lines])
    for line in lines:
        is_table_line = line.count("|") >= 2
        if blocks and blocks[-1][0] == ("table" if is_table_line else "prose"):
            blocks[-1][1].append(line)
        else:
            blocks.append(("table" if is_table_line else "prose", [line]))

    renderables = []
    for kind, block_lines in blocks:
        if kind == "prose":
            text = "\n".join(block_lines).strip("\n")
            if text.strip():
                renderables.append(Markdown(text))
        else:
            try:
                renderables.append(_build_rich_table_from_lines(block_lines))
            except ValueError:
                # Nothing usable in this block (e.g. a lone stray "|")
                # -- fall back to showing it as plain preformatted text
                # rather than dropping it silently.
                renderables.append(Text("\n".join(block_lines)))

    return Group(*renderables)


# ─── Banner ───────────────────────────────────────────────────────────────────

def print_banner() -> None:
    console.print()
    banner_art = """[bold cyan]
     ██████╗ ██████╗ ██╗     █████╗ ███████╗███████╗██╗███████╗████████╗ █████╗ ███╗   ██╗████████╗
    ██╔════╝ ██╔══██╗██║    ██╔══██╗██╔════╝██╔════╝██║██╔════╝╚══██╔══╝██╔══██╗████╗  ██║╚══██╔══╝
    ██║  ███╗██████╔╝██║    ███████║███████╗███████╗██║███████╗   ██║   ███████║██╔██╗ ██║   ██║
    ██║   ██║██╔═══╝ ██║    ██╔══██║╚════██║╚════██║██║╚════██║   ██║   ██╔══██║██║╚██╗██║   ██║
    ╚██████╔╝██║     ██║    ██║  ██║███████║███████║██║███████║   ██║   ██║  ██║██║ ╚████║   ██║
     ╚═════╝ ╚═╝     ╚═╝    ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝ [/bold cyan]"""

    console.print(
        Panel(
            Align.center(banner_art + "\n\n[bold white]                                       GPI KNOWLEDGE ASSISTANT[/bold white]\n"
                         "[dim]                             Enterprise Knowledge & Intelligence Platform[/dim]"),
            border_style="cyan",
            box=box.DOUBLE,
            padding=(1, 4),
            title="[bold cyan]Godfrey Phillips India[/bold cyan]",
            subtitle="[dim]Powered by RAG + Llama 3.1[/dim]",
        )
    )
    console.print()


def print_system_status() -> None:
    table = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2))
    table.add_column(style="dim cyan")
    table.add_column(style="green")
    table.add_row("Retriever",  f"✓ {len(knowledge_index.items)} sections indexed")
    table.add_row("Memory",     "✓ Active")
    table.add_row("Model",      "Llama 3.1:8b via Ollama")
    table.add_row("Mode",       "Streaming RAG")
    console.print(Panel(table, title="[bold green]System Ready[/bold green]", border_style="green"))
    console.print()


def print_help() -> None:
    t = Table(title="Commands", box=box.ROUNDED, border_style="dim cyan",
              header_style="bold cyan")
    t.add_column("Command", style="cyan", width=20)
    t.add_column("Description")
    rows = [
        ("quit / exit / q",  "Exit the assistant"),
        ("clear",            "Clear conversation memory"),
        ("help",             "Show this panel"),
        ("new chat",         "Start a new chat session"),
        ("chats",            "List saved chats (with last activity time)"),
        ("log",              "Show this chat's full history with timestamps"),
        ("status",           "Show system status"),
    ]
    for cmd, desc in rows:
        t.add_row(cmd, desc)
    console.print()
    console.print(t)
    console.print()


_LOOSE_COLUMN_SPLIT = re.compile(r"\s{2,}")
_MOSTLY_NUMERIC = re.compile(r"^[\d,.\-()%\s]+$")


def _parse_loose_text_into_rows(raw_text: str) -> tuple:
    """
    Best-effort parse of a plain reading-order text block (the rare
    chunk that never extracted as a proper pipe table at ingestion --
    see ingestion/table.py's raw_text_override) into table-ish rows, so
    even this case can still render as a rich.table.Table instead of a
    fenced code block. PDF plain-text extraction preserves column
    alignment as runs of multiple spaces between a label and its
    values, so splitting on 2+ consecutive spaces recovers cells
    reasonably well. Rows are padded to a common column count with
    empty strings; the widest row sets the count.

    If the first parsed row has no numeric-looking cells, it's treated
    as the source's own real header row rather than a data row -- e.g.
    "Particulars   Note No.   As at 31.3.2024   As at 31.3.2023"; a
    generic "Col 1/Col 2/..." header is only synthesized when that
    isn't the case.
    """
    rows = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in _LOOSE_COLUMN_SPLIT.split(line) if c.strip()]
        if cells:
            rows.append(cells)
    if not rows:
        return [], []
    width = max(len(r) for r in rows)

    first_is_header = not any(_MOSTLY_NUMERIC.match(c) for c in rows[0] if c)
    if first_is_header:
        header = rows[0] + [""] * (width - len(rows[0]))
        data_rows = rows[1:]
    else:
        header = ["Particulars"] + [f"Col {i}" for i in range(1, width)] if width > 1 else ["Particulars"]
        data_rows = rows

    padded = [r + [""] * (width - len(r)) for r in data_rows]
    return header, padded


def build_statement_rich_table(data: dict) -> Table:
    """
    Build a clean rich.table.Table from the structured pieces returned
    by core.direct_answer.try_statement_table_structured. Every cell
    passed to Table.add_row is already individually stripped (either by
    _row_cells() at the source, or by _parse_loose_text_into_rows()
    here), so there is no stray/uneven whitespace baked into any cell
    -- rich computes its own consistent column widths and padding from
    clean content, rather than the "some tables render as a real pipe
    grid, others as a fenced plain-text block, both with inconsistent
    spacing" result of routing everything through Markdown().
    """
    if "raw_text" in data:
        header_cells, rows = _parse_loose_text_into_rows(data["raw_text"])
    else:
        header_cells, rows = data["header_cells"], data["data_rows"]

    t = Table(
        title=data.get("title") or None,
        caption=data.get("footer") or None,
        box=box.ROUNDED, border_style="cyan", header_style="bold cyan",
        show_lines=False, title_style="bold white", caption_style="dim italic",
    )
    for i, h in enumerate(header_cells):
        # First column (the line-item label) is left-aligned and never
        # wraps mid-word; value columns are right-aligned, as figures
        # normally are in a financial statement.
        t.add_column(h, overflow="fold", justify="left" if i == 0 else "right",
                     no_wrap=(i == 0))
    for row in rows:
        # Pad/truncate to the header's column count -- a stitched table
        # can very occasionally have a short row if a source row had
        # fewer real cells than the header.
        cells = (row + [""] * len(header_cells))[:len(header_cells)]
        t.add_row(*cells)

    prefix = (data.get("prefix_text") or "").strip()
    if prefix:
        # A short unit-of-measure/context line ("All amounts are in Rs.
        # lakhs...") that isn't itself tabular -- shown as the table's
        # own caption line isn't appropriate since the footer slot is
        # used for the source citation, so it's prepended as its own
        # row-less header line via the table's title instead.
        t.title = f"{data.get('title', '')}\n[dim]{prefix}[/dim]" if data.get("title") else prefix
    return t


def print_answer(result: EngineResult) -> None:
    """Render the answer panel and diagnostics for non-LLM routes."""
    answer = result.answer

    console.print()

    if result.structured_table is not None:
        console.print(build_statement_rich_table(result.structured_table))
    elif isinstance(answer, Table):
        console.print(answer)
    else:
        console.print(
            Panel(
                _render_final_answer(answer),
                title="[bold white]Answer[/bold white]",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    if result.chart_path:
        console.print(
            Panel(
                f"[bold green]Chart saved:[/bold green] {result.chart_path}\n"
                "[dim]Terminal can't display images — open the file above "
                "to view it.[/dim]",
                title="[bold white]Chart[/bold white]",
                border_style="green",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    # Diagnostics row
    route_colors = {
        "FAST":       "green",
        "DIRECT":     "magenta",
        "LLM":        "yellow",
        "CHART":      "green",
        "NO_RESULTS": "red",
        "ERROR":      "red",
    }
    rc = route_colors.get(result.route, "white")

    diag = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    diag.add_column(style="dim")
    diag.add_column()
    diag.add_row("Route",    f"[{rc}]{result.route}[/{rc}]")
    diag.add_row("Retrieval", f"{result.retrieval_time:.3f}s")
    if result.llm_time > 0:
        diag.add_row("LLM", f"{result.llm_time:.2f}s")
    diag.add_row("Total", f"{result.total_time:.2f}s")
    if result.prompt_size:
        diag.add_row("Prompt", f"{result.prompt_size:,} chars")
    console.print(diag)

    # Sources
    if result.results:
        src = Table(box=box.SIMPLE, header_style="bold dim")
        src.add_column("Category", style="dim cyan", width=12)
        src.add_column("Section",  style="dim",      width=16)
        src.add_column("Name",     style="white")
        for r in result.results[:5]:
            src.add_row(
                r.get("category", "—").title(),
                r.get("section", "—"),
                r.get("record", {}).get("name", "—"),
            )
        console.print(Panel(src, title="[dim]Sources[/dim]", border_style="dim", padding=(0, 1)))

    console.print()


def print_error(msg: str) -> None:
    console.print(
        Panel(f"[red]{msg}[/red]", title="[bold red]Error[/bold red]",
              border_style="red", box=box.ROUNDED)
    )
    console.print()


def _format_ts(iso_ts: Optional[str]) -> str:
    """Render a stored ISO timestamp as 'Jul 09 2026, 14:32:05', or a
    dash if the message predates timestamp tracking / has none."""
    if not iso_ts:
        return "—"
    try:
        return datetime.fromisoformat(iso_ts).strftime("%b %d %Y, %H:%M:%S")
    except ValueError:
        return iso_ts


def print_chat_log() -> None:
    """Show every message in the current chat session with the exact
    date/time it was sent — not just when the chat itself was created."""
    messages = _chat_manager.get_messages()
    if not messages:
        print_info("This chat has no messages yet.")
        return

    t = Table(
        title=f"Chat Log — {_chat_manager.get_current()}",
        box=box.ROUNDED, border_style="dim cyan", header_style="bold cyan",
        show_lines=True,
    )
    t.add_column("Time", style="dim", width=20, no_wrap=True)
    t.add_column("Role", style="cyan", width=10)
    t.add_column("Message")

    for msg in messages:
        content = msg.get("content", "")
        preview = content if len(content) <= 300 else content[:300] + "…"
        t.add_row(_format_ts(msg.get("timestamp")), msg.get("role", "—"), preview)

    console.print()
    console.print(t)
    console.print()


def print_info(msg: str) -> None:
    console.print(f"[dim cyan]  {msg}[/dim cyan]")
    console.print()


# ─── Per-session engine (clean memory per chat) ───────────────────────────────

_session_memory = ConversationMemory()
_session_engine = AssistantEngine(memory=_session_memory, stream=False)
_chat_manager   = ChatManager()


def ask_with_streaming(question: str) -> EngineResult:
    """
    Run the full pipeline but handle LLM streaming in the terminal.
    For non-LLM routes, returns immediately.
    """
    import time as _time
    from core.data_analyst import is_analysis_query, answer_analysis_query

    t_start = _time.perf_counter()

    question = normalize_question(question)

    intent  = resolver.resolve(question, _session_memory)
    followup = intent["is_followup"]
    is_chart_request = wants_chart(question)

    # Level 0: Live spreadsheet analysis (fully isolated)
    if is_analysis_query(question):
        analysis = answer_analysis_query(question)
        if analysis.error is None:
            res = EngineResult(
                answer=analysis.text, route="ANALYSIS", intent=intent,
                chart_path=analysis.chart_path, total_time=_time.perf_counter() - t_start,
            )
            _session_memory.add(question, analysis.text, [], intent)
            return res

    # Fast social
    if not followup and not is_chart_request:
        social = try_direct_answer(question, [])
        if social is not None:
            res = EngineResult(
                answer=social, route="FAST", intent=intent,
                total_time=_time.perf_counter() - t_start,
            )
            _session_memory.add(question, social, [], intent)
            return res

    # Chart follow-up: a bare "chart"/"graph it" (or a genuine
    # anaphoric follow-up like "chart based on it") charts the table
    # already shown in the PREVIOUS answer, before touching retrieval
    # at all. This whole chart pipeline was previously missing from
    # this terminal UI's own copy of the routing logic (it only ever
    # DISPLAYED result.chart_path if one happened to be set -- nothing
    # here ever set it), so every chart request here fell all the way
    # through to a normal LLM text answer, which correctly-but-uselessly
    # said it has no way to draw a chart.
    if is_chart_request and (followup or _is_bare_chart_followup(question, intent)):
        # This call can take a few seconds (rendering + file I/O) with
        # zero streamed output of its own — unlike a normal LLM answer,
        # which visibly fills in via stream_and_render()'s "Thinking…"
        # panel, chart generation previously ran completely silently
        # here, which is exactly what made it look frozen/stuck rather
        # than working.
        with console.status("[bold yellow]Generating chart…[/bold yellow]", spinner="dots"):
            chart_path, caption, chart_error = _session_engine._generate_chart_from_last_answer(question, intent)
        if chart_path:
            res = EngineResult(
                answer=caption, route="CHART", intent=intent,
                chart_path=chart_path, total_time=_time.perf_counter() - t_start,
            )
            _session_memory.add(question, caption, [], intent)
            return res

    # Retrieval
    t_ret = _time.perf_counter()
    results = _session_engine._retrieve(question, intent, followup)
    ret_time = _time.perf_counter() - t_ret

    # Chart generation: explicit "chart/graph/plot" requests, using the
    # freshly retrieved context.
    if is_chart_request:
        # Same "silent and stuck-looking" problem as above, but worse
        # here -- this path also runs an LLM call (chart-spec
        # extraction) before rendering, which is the slowest step in
        # the whole chart pipeline and previously had no visible
        # progress indicator at all.
        with console.status("[bold yellow]Generating chart…[/bold yellow]", spinner="dots"):
            chart_path, caption, chart_error = _session_engine._generate_chart(question, results)
        if chart_path:
            res = EngineResult(
                answer=caption, route="CHART", results=results, intent=intent,
                retrieval_time=ret_time, chart_path=chart_path,
                total_time=_time.perf_counter() - t_start,
            )
            _session_memory.add(question, caption, results, intent)
            return res
        res = EngineResult(
            answer=(
                f"I can't build that chart: {chart_error} "
                "I can still answer in text if that helps — just ask "
                "without 'chart'/'graph', or rephrase what you'd like plotted."
            ),
            route="NO_RESULTS", results=results, intent=intent,
            retrieval_time=ret_time, total_time=_time.perf_counter() - t_start,
        )
        return res

    # Direct RAG
    if results and not needs_llm(question) and not followup:
        direct = try_direct_answer(question, results)
        if direct is not None:
            res = EngineResult(
                answer=direct, route="DIRECT", results=results, intent=intent,
                retrieval_time=ret_time, total_time=_time.perf_counter() - t_start,
            )
            _session_memory.add(question, direct, results, intent)
            return res

    # No results
    if not results:
        msg = (
            "I don't have anything on that. "
            "Try asking about leadership, history, products, departments, or financials."
        )
        res = EngineResult(
            answer=msg, route="NO_RESULTS", intent=intent,
            total_time=_time.perf_counter() - t_start,
        )
        return res

    # Direct financial-fact lookup: "PAT for 2023 and 2024"
    if not wants_chart(question):
        fact_answer = try_financial_fact_lookup(question, results)
        if fact_answer is not None:
            # For analysis questions ("why", "how", "analyze"), don't return
            # early -- instead pass the structured facts to the LLM for
            # reasoning/synthesis. For simple lookups, return the FACT answer.
            if needs_llm(question):
                # Store structured facts for LLM prompt injection
                fact_data = get_financial_facts_for_llm(question, results)
                if fact_data:
                    for r in results:
                        if r.get("record"):
                            r["record"]["_injected_financial_facts"] = fact_data
                            break
            else:
                res = EngineResult(
                    answer=fact_answer, route="FACT", results=results, intent=intent,
                    retrieval_time=ret_time, total_time=_time.perf_counter() - t_start,
                )
                _session_memory.add(question, fact_answer, results, intent)
                return res

    # Direct statement table: P&L / balance sheet / cash flow — shows the
    # ingested table verbatim instead of routing through the LLM.
    # Gated behind STATEMENT_TABLE_DIRECT_ROUTE (off by default, see
    # core/config.py) because the grid-stitcher misaligns columns when
    # different pages of the same statement extract with different
    # column counts. With it off, this falls straight through to the
    # LLM route below, unchanged.
    if STATEMENT_TABLE_DIRECT_ROUTE:
        statement_answer = try_statement_table_answer(question, results)
        if statement_answer is not None:
            res = EngineResult(
                answer=statement_answer, route="DIRECT", results=results, intent=intent,
                retrieval_time=ret_time, total_time=_time.perf_counter() - t_start,
                structured_table=try_statement_table_structured(question, results),
            )
            _session_memory.add(question, statement_answer, results, intent)
            return res

    # LLM streaming
    prompt_text = build_prompt(
        question=question,
        results=results,
        history=_session_memory.get_context_string(),
        section_hint=intent.get("section_hint"),
        action=intent.get("action", "overview"),
    )

    t_llm = _time.perf_counter()
    action = intent.get("action", "overview")
    llm_timeout = OLLAMA_TIMEOUT_TABLE if action in _BROAD_ACTIONS else OLLAMA_TIMEOUT
    try:
        answer = stream_and_render(
            prompt_text,
            max_tokens=_token_budget(action),
            timeout=llm_timeout,
        )
    except RuntimeError as exc:
        res = EngineResult(
            answer=f"Error: {exc}", route="ERROR", error=str(exc), intent=intent,
            total_time=_time.perf_counter() - t_start,
        )
        return res

    llm_time = _time.perf_counter() - t_llm
    _session_memory.add(question, answer, results, intent)

    return EngineResult(
        answer=answer, route="LLM", results=results, intent=intent,
        prompt_size=len(prompt_text), retrieval_time=ret_time,
        llm_time=llm_time, total_time=_time.perf_counter() - t_start,
    )


# ─── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    print_banner()
    print_system_status()

    history     = InMemoryHistory()
    input_style = Style.from_dict({"prompt": "cyan bold"})

    while True:
        try:
            question = pt_prompt("  Ask > ", history=history, style=input_style).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim cyan]  Goodbye.[/dim cyan]\n")
            break

        if not question:
            continue

        q = question.lower().strip()

        # ── Commands ──────────────────────────────────────────────────────
        if q in ("quit", "exit", "q"):
            console.print("\n[dim cyan]  Goodbye.[/dim cyan]\n")
            break

        if q == "clear":
            _session_memory.clear()
            print_info("Conversation memory cleared.")
            continue

        if q == "help":
            print_help()
            continue

        if q == "status":
            print_system_status()
            continue

        if q == "new chat":
            name = _chat_manager.create_chat()
            _session_memory.clear()
            print_info(f"New chat created: {name}")
            continue

        if q == "chats":
            names = _chat_manager.get_chat_names()
            chats_table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
            chats_table.add_column("", width=2)
            chats_table.add_column("Chat")
            chats_table.add_column("Last activity", style="dim")
            for n in names:
                last_ts = _chat_manager.get_last_message_time(n)
                chats_table.add_row(
                    "►" if n == _chat_manager.get_current() else "",
                    n,
                    _format_ts(last_ts),
                )
            console.print(Panel(
                chats_table,
                title="[bold cyan]Saved Chats[/bold cyan]",
                border_style="cyan",
            ))
            continue

        if q == "log":
            print_chat_log()
            continue

        # ── Process question ──────────────────────────────────────────────
        try:
            result = ask_with_streaming(question)

            # LLM answer already printed by stream_and_render()
            if result.route != "LLM":
                print_answer(result)
            else:
                # Print diagnostics only (answer was streamed)
                diag = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
                diag.add_column(style="dim")
                diag.add_column()
                diag.add_row("Route",    "[yellow]LLM[/yellow]")
                diag.add_row("Retrieval", f"{result.retrieval_time:.3f}s")
                diag.add_row("LLM",       f"{result.llm_time:.2f}s")
                diag.add_row("Total",     f"{result.total_time:.2f}s")
                if result.prompt_size:
                    diag.add_row("Prompt", f"{result.prompt_size:,} chars")
                console.print(diag)

                if result.results:
                    src = Table(box=box.SIMPLE, header_style="bold dim")
                    src.add_column("Category", style="dim cyan", width=12)
                    src.add_column("Section",  style="dim",      width=16)
                    src.add_column("Name")
                    for r in result.results[:5]:
                        src.add_row(
                            r.get("category", "—").title(),
                            r.get("section", "—"),
                            r.get("record", {}).get("name", "—"),
                        )
                    console.print(Panel(src, title="[dim]Sources[/dim]", border_style="dim", padding=(0, 1)))
                console.print()

            # Save to chat history
            _chat_manager.add_message("user", question)
            _chat_manager.add_message("assistant", result.answer)

        except Exception as exc:
            print_error(str(exc))


if __name__ == "__main__":
    main()