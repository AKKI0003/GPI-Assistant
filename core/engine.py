"""
core/engine.py

Assistant Engine — the single public interface for all UIs.

Architecture
------------
LEVEL 1  Fast social responses (no retrieval, no LLM)
LEVEL 2  Direct RAG          (retrieval, no LLM)
LEVEL 3  LLM reasoning       (retrieval + LLM)

Every UI (terminal, Streamlit, future API) calls:

    result = engine.ask(question)

and receives a standardised EngineResult dataclass.
"""

import os
import time
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

from core.config import RETRIEVAL_TOP_K, LLM_MAX_TOKENS, LLM_MAX_TOKENS_TABLE, OLLAMA_TIMEOUT, OLLAMA_TIMEOUT_TABLE, SELF_REFINE, CONTEXT_MAX_CHARS_TABLE, STATEMENT_TABLE_DIRECT_ROUTE
from core.query_normalizer import normalize_question
from core.intent_resolver import resolver
from core.direct_answer import try_direct_answer, needs_llm, try_statement_table_answer, try_statement_table_structured, _detect_requested_statement, try_financial_fact_lookup, get_financial_facts_for_llm
from core.chart_generator import wants_chart, extract_chart_spec, build_chart_spec_from_markdown_table, try_build_chart_spec_from_facts
from core.data_analyst import is_analysis_query, answer_analysis_query, get_last_analysis_spec
from retrieval.knowledge_index import index as knowledge_index, detect_companies_in_question, detect_years_in_question
from retrieval.embedding_store import embed
from retrieval.context_builder import build_context
from retrieval.pdf_retriever import search as pdf_search
from memory.conversation_memory import ConversationMemory, memory as global_memory
from llm.prompt_builder import build_prompt, build_refine_prompt
from llm.ollama_client import stream_prompt, send_prompt
from ui.chart_renderer import render_chart

logger = logging.getLogger(__name__)


# ─── Result type ──────────────────────────────────────────────────────────────

@dataclass
class EngineResult:
    answer:         str   = ""
    route:          str   = ""          # FAST / DIRECT / LLM / CHART / NO_RESULTS / ERROR
    results:        List  = field(default_factory=list)
    intent:         Dict  = field(default_factory=dict)
    prompt_size:    int   = 0
    retrieval_time: float = 0.0
    llm_time:       float = 0.0
    total_time:     float = 0.0
    is_followup:    bool  = False
    error:          Optional[str] = None
    chart_path:     Optional[str] = None   # PNG path when route == "CHART"
    # Set only for route == "DIRECT" statement-table answers (P&L,
    # balance sheet, cash flow, ...). `answer` stays a plain Markdown
    # string for the API/chat-history/memory paths, which need a
    # string; this carries the SAME data as clean structured cells
    # (see core.direct_answer.try_statement_table_structured) so the
    # terminal UI can render it as a real rich.table.Table instead of
    # relying on Markdown's table renderer, which was producing
    # inconsistent output (some statements as a real pipe table, others
    # -- whichever chunk fell back to plain-text extraction -- as a
    # fenced code block, neither guaranteed clean at every terminal
    # width).
    structured_table: Optional[Dict[str, Any]] = None
    # True when the answer was withheld (or narrowed) because the
    # requesting user's department/identity doesn't have access to some
    # or all of the relevant uploaded file(s) -- see auth/permissions.py.
    # Distinguished from a normal "I don't know" so the frontend can
    # render it differently (and so it's unambiguous in logs/telemetry
    # that this was an access decision, not the assistant failing to
    # find something that was actually open to everyone).
    restricted: bool = False


_NO_RESULTS_MESSAGE = (
    "I don't have anything on that. Try asking about departments, projects, "
    "company leadership, financials, products, or history."
)
_RESTRICTED_MESSAGE = (
    "Restricted access. The document(s) relevant to this question aren't "
    "shared with your department. If you believe this is a mistake, ask "
    "whoever uploaded that file to add your department (or you "
    "specifically) to its access list."
)

# Actions/section-hints that legitimately need a WIDE context window because
# the user is asking to aggregate many records (a table, a comparison, a
# full list, a timeline) rather than a single narrow fact. For these we
# retrieve far more candidates than the default top_k, and we never trim
# down to a single matching section the way a focused lookup would.
_BROAD_ACTIONS = {"table", "compare"}

# Actions worth the cost of the optional SELF_REFINE second pass (see
# core.config.SELF_REFINE) — the more complex/error-prone answer types,
# where a small model is most likely to include an unsupported detail or
# structure the answer poorly. Quick factual lookups never need it, and
# always going through Direct RAG anyway when possible.
_REFINE_ACTIONS = {"table", "compare", "detail"}
_BROAD_QUESTION_SIGNALS = (
    "table", "compare", "comparison", "list", "timeline", "all ",
    # Coverage / aggregation questions ("how much VST data do we have",
    # "how many years", "annual figures", "year on year", "3 years of
    # data") need the same wide net as an explicit compare/table request —
    # they're really asking the index to surface everything it has on a
    # topic, not a single fact. Without these, a query like "how much
    # annual data of VST do we have" was treated as a narrow lookup and
    # got capped down to a couple of chunks even when years of ingested
    # PDF content existed.
    "how much", "how many", "how many years", "annual", "each year",
    "every year", "year on year", "year-on-year", "year wise", "yearly",
    "over the years", "across years", "multiple years",
    "coverage", "how much data", "how much information", "analyze", "analyse",
    "trend", "past",
    # Explicit chart/graph requests need the same wide, per-entity-balanced
    # retrieval as "compare" — a chart comparing GPI and PMI needs both
    # companies' numbers pulled in, not whichever one flat top-k ranking
    # happens to favor.
    "chart", "graph", "plot", "visualize", "visualise", "visualization",
    # Named financial STATEMENTS ("cash flow statement", "P&L", "balance
    # sheet") need the same wide net as "compare"/"table": the statement
    # itself is one numeric table buried among many narrative chunks that
    # merely *mention* the statement's name (accounting-policy notes,
    # MD&A commentary, auditor's report boilerplate) without containing
    # any of its actual figures. At a narrow top_k, those narrative
    # chunks -- which are semantically an excellent match for the
    # statement's name -- routinely crowd out the one chunk that has the
    # real numbers, so a plain "cash flow statement of GPI for 2023" was
    # coming back empty even when the table was fully ingested.
    "cash flow", "profit and loss", "profit & loss", "p&l", "p & l",
    "pnl", "balance sheet", "income statement", "financial statements",
)

# Matches "3 years", "3 year", "three years", "the last 3 years", "past
# three year(s)", etc. A fixed literal-phrase list (just "3 years",
# "three years") previously missed the perfectly natural singular
# phrasing "past 3 year data" — a query worded exactly that way was
# treated as a narrow single-fact lookup instead of the multi-year
# aggregate request it obviously was, and got capped down accordingly.
_MULTI_YEAR_PATTERN = re.compile(
    r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+years?\b",
    re.IGNORECASE,
)

# Categories that are inherently "give me everything you have" requests —
# a user asking about "process" or "project" almost always wants the full
# set of records, not a narrowed-down subset.
_FULL_CATEGORY_REQUESTS = {"process", "project", "kpi", "problem", "department"}

# Company sections that are closely related and should be retrieved
# together rather than discarded, so a query like "history" or "products"
# returns its full neighborhood of context instead of a single thin slice.
_RELATED_SECTIONS: Dict[str, set] = {
    "history":     {"history", "overview", "description", "ownership", "location"},
    "products":    {"products", "overview", "cigarette_segmentation",
                     "pmi_strategic_segments", "segmentation_framework_insights"},
    "financials":  {"financials", "overview", "metrics"},
    "leadership":  {"leadership", "overview"},
    "overview":    {"overview", "description", "history", "leadership", "products"},
}


_TERSE_METRIC_PATTERN = re.compile(
    r"\b(pat|pbt|profit after tax|profit before tax|net profit|"
    r"revenue|ebitda|eps|tax expense|total tax|margin)\b",
    re.IGNORECASE,
)
_TERSE_BLOCKING_SIGNALS = (
    "table", "statement", "breakdown", "detail", "full", "complete",
    "all line", "everything", "explain", "reason", "why", "how did",
    "trend", "analysis", "analyze",
)


def _wants_terse_output(question: str) -> bool:
    """
    True for a plain "just give me the number(s)" figure lookup
    ("PAT for 2023 and 2024", "what's VST's PBT for 2024", "VST's PAT
    for 2023, 2024 and 2025") -- as opposed to an explicit request for
    a table/statement/explanation.

    BUG (reproduced against a live transcript): this used to anchor the
    metric keyword to the very START of the question (optionally after
    a small fixed set of lead-ins like "what's"/"give me"). Real
    phrasing very often puts the company name FIRST instead -- "VST's
    PAT for 2023, 2024 and 2025" -- which never matched that anchored
    pattern at all, so this exact kind of bare figure-lookup question
    got the full table/detail output budget and elaborate prose
    treatment every time, needlessly inflating generation time on a
    slow CPU-bound model (900+ seconds observed for exactly this
    phrasing, almost entirely spent generating output tokens the user
    explicitly didn't want to read). Matching the metric keyword
    ANYWHERE in the question fixes this; the blocking-signal check
    below still protects against over-triggering on a question that
    genuinely wants a full table/statement/explanation.

    This is DELIBERATELY separate from `_wants_brief`/`brief`: brief
    narrows BOTH the input context budget and the output length,
    which is right when the user explicitly asks for a short summary,
    but wrong here -- narrowing input context for a multi-year figure
    lookup is exactly what previously caused a requested year's figure
    to go missing from what the model was given to read in the first
    place. This only shortens the OUTPUT (a small max_tokens cap plus a
    terse-answer instruction), while retrieval/context width is decided
    exactly as before (still wide/broad for a multi-year ask) -- so the
    same amount of source material is available, the model is just told
    not to narrate around it.
    """
    q = question.strip().lower()
    if not _TERSE_METRIC_PATTERN.search(q):
        return False
    return not any(sig in q for sig in _TERSE_BLOCKING_SIGNALS)


def _token_budget(action: str, broad: bool = False, brief: bool = False, terse: bool = False) -> int:
    """
    Tables and structured/detailed answers need a much larger output budget
    than short prose. A table for elaborate, multi-attribute data (e.g.
    segmentation with 8-11 rows across several columns) can easily need
    3000-4500+ tokens — far more than a default conversational answer.
    Aggregate/coverage questions ("how much VST data do we have", "compare
    the last 3 years") need the same larger budget even when the detected
    action isn't literally "table" or "compare".

    ``brief``: overrides everything else with a small hard cap. This is
    the main lever for actually respecting "give me a brief summary" —
    on a slow CPU-bound local model, generation wall-clock time scales
    almost directly with how many tokens it's allowed to produce, and a
    model asked nicely for brevity won't reliably stop itself early
    without an enforced ceiling. Without this, a "brief" request could
    still get the full 3500-4500 token budget and take just as long as a
    detailed one even though the user explicitly asked for something
    short.
    """
    if brief:
        return 500
    if terse:
        return 150
    if action in _BROAD_ACTIONS or action == "detail" or broad:
        return LLM_MAX_TOKENS_TABLE
    return LLM_MAX_TOKENS


_BRIEF_SIGNALS = (
    "brief", "briefly", "short answer", "keep it short", "small summary",
    "small insight", "quick summary", "quick insight", "in short",
    "concise", "summarize briefly", "very small", "don't need details",
    "no need for details", "just a summary", "keep it small",
)


def _wants_brief(question: str) -> bool:
    """
    True if the user explicitly asked for a short answer. This should
    override broad/table-style context and output sizing even if the
    question ALSO contains broad signals ("3 years", "compare", …) —
    "give me a brief summary of the last 3 years" is still a request for
    something short, not a request for an exhaustive multi-year table.
    """
    q = question.lower()
    return any(sig in q for sig in _BRIEF_SIGNALS)


def _wants_broad_context(question: str, intent: Dict[str, Any]) -> bool:
    """True if the question is asking for an aggregate view (table, compare,
    list, timeline, or "all X") that needs many records rather than one."""
    if intent.get("action") in _BROAD_ACTIONS:
        return True
    if intent.get("table_requested"):
        return True
    q = question.lower()
    if _MULTI_YEAR_PATTERN.search(q):
        return True
    # Explicitly naming 2+ distinct years ("PAT for 2023, 2024 and 2025")
    # is just as much a multi-year aggregate request as "past 3 years" or
    # "year on year" -- it should get the exact same wide, per-year
    # balanced retrieval. Without this, a question phrased as an explicit
    # year list (rather than one of the recognized phrases below) fell
    # through to a narrow single top-k search with no per-year balancing
    # or backfill, so only whichever single year's chunks happened to
    # score highest semantically was ever retrieved -- silently dropping
    # the other requested years instead of answering (or honestly saying
    # they're unavailable).
    if len(detect_years_in_question(question)) >= 2:
        return True
    return any(sig in q for sig in _BROAD_QUESTION_SIGNALS)


# ─── Engine ───────────────────────────────────────────────────────────────────

def _is_bare_chart_followup(question: str, intent: Dict[str, Any]) -> bool:
    """
    True for a short chart request that names no fresh topic of its own
    (no company, no statement, no category) -- e.g. a bare "chart",
    "graph it", "make a chart", "can I get a graph or chart". These are
    ALWAYS asking to visualize whatever was just answered, but the
    strict follow-up detector in core/intent_resolver.py requires an
    explicit back-reference pronoun ("it"/"that") -- a bare "chart" or
    "graph" with no pronoun at all was falling through as NOT a
    follow-up, so it went to fresh (empty) retrieval and the LLM
    correctly-but-uselessly reported it had no chart data, instead of
    ever trying to chart the table already on screen.

    Deliberately narrow: only short questions (<=6 words) with no
    detected category/section/company/statement of their own qualify --
    a longer, genuinely fresh multi-part request ("VST's PAT for 3
    years, compare, then chart it") should still go through normal
    retrieval + extraction, not silently reuse an unrelated old table.
    """
    words = question.strip().split()
    if len(words) > 6:
        return False
    if intent.get("category") or intent.get("section_hint"):
        return False
    if detect_companies_in_question(question):
        return False
    if _detect_requested_statement(question):
        return False
    return True


class AssistantEngine:
    """
    Orchestrates intent resolution, retrieval, direct answering, and LLM calls.

    Can use a custom memory instance (useful for isolated sessions / tests).
    """

    def __init__(
        self,
        memory: Optional[ConversationMemory] = None,
        stream: bool = True,
    ) -> None:
        self.memory = memory or global_memory
        self.stream = stream
        self._stream_callback = None  # set by UI to receive streamed tokens

    def set_stream_callback(self, callback) -> None:
        """Register a callback(chunk: str) for streaming output."""
        self._stream_callback = callback

    @staticmethod
    def _chart_caption(spec, path: str) -> str:
        """Build the caption shown alongside a rendered chart. Names the
        output directory/file explicitly, since the chart PNG is saved
        to disk (charts_output/) as well as shown inline."""
        names = ", ".join(s["name"] for s in spec.series)
        span = (
            ", ".join(spec.categories) if len(spec.categories) <= 6
            else f"{spec.categories[0]}\u2013{spec.categories[-1]}"
        )
        filename = os.path.basename(path)
        return (
            f"Here's a {spec.chart_type} chart comparing {names} across {span}. "
            f"Chart generated in charts_output\\{filename}."
        )

    # ── chart generation ────────────────────────────────────────────────────

    def _generate_chart(self, question: str, results: List[Dict[str, Any]]):
        """
        Try to build a chart for an explicit chart/graph/plot request.

        Returns (chart_path, caption, error). chart_path is None (with
        error set) when the context didn't have enough real data — the
        caller should fall back to a normal text answer in that case
        rather than showing a broken/empty chart.
        """
        context = build_context(results, max_chars=CONTEXT_MAX_CHARS_TABLE)

        # Try the deterministic path first -- a plain "chart of <metric>
        # for <year(s)>" request doesn't need an LLM to re-read raw
        # table text for numbers that were already extracted and
        # correctly year-tagged at ingestion time (see
        # ingestion/financial_facts.py). Only falls through to the
        # LLM-based extraction below when this can't confidently
        # identify the metric/years, or none of them have a fact on
        # record — same conservative principle as everywhere else this
        # deterministic-facts path is used.
        spec = try_build_chart_spec_from_facts(question, results)
        if spec is None:
            spec = extract_chart_spec(question, context)

        if spec.insufficient_data:
            return None, None, spec.reason

        try:
            path = render_chart(spec)
        except Exception as exc:
            return None, None, f"Chart rendering failed: {exc}"

        names = ", ".join(s["name"] for s in spec.series)
        caption = self._chart_caption(spec, path)
        return path, caption, None

    def _generate_chart_from_last_answer(self, question: str, intent: Dict[str, Any]):
        """
        Handle a follow-up chart request ("can you draw a chart based on
        it") by charting the Markdown table already shown in the
        PREVIOUS answer, instead of re-retrieving/re-extracting from the
        knowledge base. This is what "it" actually refers to, and the
        previous answer is already clean, on-screen data (whether it
        came from the live spreadsheet analyst or a normal LLM table
        answer) -- re-running retrieval on a nearly content-free
        follow-up sentence like "chart based on it" has no topic words
        of its own to match and can easily latch onto an unrelated
        section instead.

        Returns (chart_path, caption, error) like _generate_chart(), or
        (None, None, None) if this isn't a table-follow-up situation at
        all (caller should fall back to the normal retrieval-based chart
        path in that case) -- e.g. no previous answer exists yet, or it
        didn't contain a table.
        """
        last_answer = self.memory.get_last_answer()
        if not last_answer or "|" not in last_answer:
            return None, None, None

        spec = build_chart_spec_from_markdown_table(question, last_answer)
        if spec.insufficient_data:
            return None, None, None

        try:
            path = render_chart(spec)
        except Exception as exc:
            return None, None, f"Chart rendering failed: {exc}"

        caption = self._chart_caption(spec, path)
        return path, caption, None

    # ── retrieval ─────────────────────────────────────────────────────────────

    def _retrieve(
        self,
        question: str,
        intent: Dict[str, Any],
        is_followup: bool,
    ) -> List[Dict[str, Any]]:
        """
        Public retrieval entry point used by ask()/ask_stream(). Runs the
        existing JSON knowledge-base retrieval (`_retrieve_json`, entirely
        unchanged in behavior) and then merges in results from the
        independent PDF Document Intelligence knowledge source
        (`retrieval/pdf_retriever.py`), per the hybrid architecture:

            JSON Retriever   ─┐
                               ├─ Merge Context ─> LLM
            PDF Retriever    ─┘

        Both retrievers return results in the exact same dict shape
        (`{"category", "section", "record", "score"}`), so merging is a
        plain list concatenation -- `context_builder.build_context()` and
        `llm/prompt_builder.py` need no changes at all to render either
        source. PDF results are appended AFTER JSON results so curated,
        hand-verified JSON knowledge always takes relevance precedence
        when the character budget is tight.

        A failure anywhere in PDF retrieval (missing optional
        dependencies, a malformed PDF, an unreachable embedding backend)
        degrades to simply not adding PDF context for this turn -- it
        never breaks JSON-based answering.
        """
        json_results = self._retrieve_json(question, intent, is_followup)

        pdf_results: List[Dict[str, Any]] = []
        try:
            broad = _wants_broad_context(question, intent)

            # Scope the PDF source the same way the JSON side already
            # scopes itself (see the "Implied company scope" block in
            # _retrieve_json): a plain "GPI finances" question names no
            # company at all, so it should default to GPI; a question
            # naming exactly one competitor should scope to that
            # competitor; a question naming 2+ companies is a genuine
            # comparison and is left unscoped (None) so every named
            # company's chunks come through, matching the JSON side's
            # own multi-entity comparison branch. Without this, PDF
            # retrieval previously had NO company scoping whatsoever,
            # so a broad "GPI finances" query's round-robin-by-source
            # step gave ITC's and VST's annual-report chunks equal
            # weight alongside GPI's own -- diluting (and sometimes
            # outright crowding out) the very figures being asked for.
            # Reuse the SAME company scope _retrieve_json just resolved
            # (intent["companies"]) rather than recomputing it here from
            # scratch — that resolution already accounts for follow-up /
            # "for the same ..." entity carryover (see the "Implied
            # company scope" comment in _retrieve_json). Recomputing
            # independently here from detect_companies_in_question(question)
            # alone would re-lose that carryover for the PDF side even
            # after fixing it for JSON, since a follow-up naming no
            # company of its own has nothing in the raw question text to
            # detect.
            resolved_companies = intent.get("companies") or []
            if not resolved_companies or resolved_companies == ["GPI"]:
                pdf_company_filter = "GPI"
            elif len(resolved_companies) == 1:
                pdf_company_filter = resolved_companies[0]
            else:
                pdf_company_filter = None

            years = intent.get("years") or detect_years_in_question(question)

            # Detect if this is a metric query (PAT, PBT, Revenue, etc.)
            metric_keywords = ("pat", "profit after tax", "profit for the year", "net profit",
                               "pbt", "profit before tax",
                               "revenue", "turnover", "sales",
                               "ebitda", "ebit", "eps", "earnings per share",
                               "tax expense", "total tax", "margin")
            is_metric_query = any(kw in question.lower() for kw in metric_keywords)

            if len(years) >= 2 and len(resolved_companies) >= 2:
                # Genuine multi-company AND multi-year comparison (e.g.
                # "compare my profits to VST and ITC's for 2023, 2024,
                # 2025"): a single per-year round-robin with
                # company_filter=None mixes every named company's
                # chunks together in each year's bucket -- there's no
                # guarantee any specific company actually has real
                # representation for a given year, so one company's
                # (or one year's) chunks can crowd out another's
                # entirely. That's what caused a 3-company x 3-year
                # comparison to end up with GPI's and VST's own 2023
                # and 2024 figures IDENTICAL -- retrieval simply never
                # found each company's OWN distinct 2023 data, so the
                # model filled the gap by duplicating whatever figure
                # it did have (see the reinforced anti-duplication rule
                # in llm/prompt_builder.py, which is the second half of
                # this fix -- retrieval finding the real data and the
                # model being told never to paper over missing data
                # both matter here).
                #
                # Searching every (company, year) PAIR separately and
                # merging round-robin guarantees each pair gets its own
                # dedicated share of the retrieval budget, the same
                # guarantee the single-axis (company-only, or
                # year-only) round-robins below already give along
                # their one axis.
                #
                # "my profits ... VST and ITC's" only ever names the
                # competitors explicitly -- GPI is implied, not listed
                # -- so intent["companies"] can be just ["VST", "ITC"]
                # here. Without adding GPI back in, this cross-product
                # loop would never retrieve GPI's own PDF data at all
                # for exactly this phrasing, mirroring the identical
                # gap already handled on the JSON side (see
                # "named_companies" a bit further down in this file).
                pair_companies = list(resolved_companies)
                if "GPI" not in pair_companies:
                    pair_companies = ["GPI"] + pair_companies

                per_pair_top_k = 12
                pair_results: List[List[Dict[str, Any]]] = []
                for company in pair_companies:
                    for yr in years:
                        pair_results.append(pdf_search(
                            question,
                            broad=broad,
                            company_filter=company,
                            year_filter=yr,
                            top_k=per_pair_top_k,
                        ))

                if is_metric_query:
                    # NOTE: this used to be a hard filter that DELETED every
                    # chunk without financial_facts. That is exactly what
                    # produced "I don't have GPI's PAT for 2023" repeated in
                    # a loop: extract_financial_facts() (ingestion/financial_facts.py)
                    # only succeeds when a chunk's own text has an
                    # unambiguous year-column header in its first few lines
                    # (see _resolve_ordered_years there) -- a single
                    # formatting quirk on one page is enough for it to find
                    # nothing, even though the real PAT figure is sitting
                    # right there in that chunk's raw text. Deleting the
                    # chunk in that case left the LLM with ZERO context for
                    # that year, and at greedy decoding (LLM_TEMPERATURE=0,
                    # LLM_TOP_K=1 in core/config.py) a model with nothing to
                    # say tends to lock onto one short sentence and repeat
                    # it verbatim until it hits the token limit.
                    # A stable sort keeps every chunk (so raw-text grounding
                    # is always available as a fallback) while still moving
                    # the reliable, pre-extracted financial_facts chunks to
                    # the front where the round-robin/top_k trimming below
                    # picks them up first.
                    for bucket in pair_results:
                        bucket.sort(key=lambda item: 0 if (item.get("record", {}).get("financial_facts") or []) else 1)

                max_len = max((len(b) for b in pair_results), default=0)
                seen_pdf_sig: set = set()
                for i in range(max_len):
                    for bucket in pair_results:
                        if i >= len(bucket):
                            continue
                        item = bucket[i]
                        rec = item.get("record", {})
                        sig = (rec.get("source_file"), rec.get("page"))
                        if sig in seen_pdf_sig:
                            continue
                        seen_pdf_sig.add(sig)
                        pdf_results.append(item)
            elif len(years) >= 2:
                # Use a larger top_k for multi-year searches to ensure P&L pages
                # make it into the results (ratio pages often rank higher for
                # specific year queries due to "PAT 2023" appearing in ratio labels)
                per_year_top_k = 20
                per_year_results: List[List[Dict[str, Any]]] = []
                for yr in years:
                    per_year_results.append(pdf_search(
                        question,
                        broad=broad,
                        company_filter=pdf_company_filter,
                        year_filter=yr,
                        top_k=per_year_top_k,
                    ))

                # For metric queries, filter each year's bucket to prioritize
                # chunks that have financial_facts (PAT/PBT/Revenue/etc.)
                # so the round-robin merge picks up actual figures first.
                if is_metric_query:
                    # Same fix as the multi-company branch above: sort
                    # financial_facts chunks to the front instead of
                    # deleting everything else. See the long comment there
                    # -- deleting was what caused a single extraction miss
                    # for one year's page to leave the LLM with no
                    # grounding at all for that year, which is what a
                    # greedy-decoding model (temperature=0, top_k=1) turns
                    # into a verbatim repeated-sentence loop rather than a
                    # normal "I don't have that" reply.
                    for bucket in per_year_results:
                        bucket.sort(key=lambda item: 0 if (item.get("record", {}).get("financial_facts") or []) else 1)

                max_len = max((len(b) for b in per_year_results), default=0)
                seen_pdf_sig: set = set()
                for i in range(max_len):
                    for bucket in per_year_results:
                        if i >= len(bucket):
                            continue
                        item = bucket[i]
                        rec = item.get("record", {})
                        sig = (rec.get("source_file"), rec.get("page"))
                        if sig in seen_pdf_sig:
                            continue
                        seen_pdf_sig.add(sig)
                        pdf_results.append(item)
            else:
                pdf_year_filter = years[0] if len(years) == 1 else None

                pdf_results = pdf_search(
                    question, broad=broad,
                    company_filter=pdf_company_filter,
                    year_filter=pdf_year_filter,
                )
        except Exception:
            logger.exception("PDF knowledge-source retrieval failed; continuing with JSON results only.")

        if not pdf_results:
            return json_results

        # Merge JSON and PDF results by INTERLEAVING them rather than
        # concatenating two large contiguous blocks. `build_context()`
        # fills its character budget strictly in list order and only
        # skips a block once the remaining room drops below ~200 chars —
        # it never reshuffles for fairness. A broad financial query can
        # easily pull in 15 PDF chunks (multi-year, per-company
        # round-robin — see pdf_retriever.py), each a couple thousand
        # characters, which can consume the ENTIRE context budget
        # (CONTEXT_MAX_CHARS_TABLE) on its own before a single JSON block
        # is ever considered. That's how a small, cheap fact (e.g.
        # headquarters, mission, a single curated financial snapshot)
        # could go missing from context the moment the PDF side had more
        # to say — not because it was irrelevant, but purely because of
        # its position after a wall of PDF text. Interleaving guarantees
        # every JSON block gets a turn near the front of the list
        # regardless of how many PDF chunks follow.
        #
        # For financial questions PDF data leads each pair (it's the
        # authoritative, per-year source — see the note below); for
        # everything else JSON leads, matching the original priority.
        financial_keywords = (
            "financial", "revenue", "profit", "loss", "income",
            "balance sheet", "cash flow", "turnover", "sales", "ebitda",
        )
        pdf_leads = any(kw in question.lower() for kw in financial_keywords)

        merged: List[Dict[str, Any]] = []
        max_len = max(len(json_results), len(pdf_results))
        for i in range(max_len):
            first, second = (pdf_results, json_results) if pdf_leads else (json_results, pdf_results)
            if i < len(first):
                merged.append(first[i])
            if i < len(second):
                merged.append(second[i])
        return merged

    def _retrieve_json(
        self,
        question: str,
        intent: Dict[str, Any],
        is_followup: bool,
    ) -> List[Dict[str, Any]]:
        """
        Select and run the appropriate JSON-knowledge-base retrieval
        strategy. This is the ORIGINAL, unmodified retrieval logic (only
        renamed from `_retrieve` -- see the new `_retrieve` wrapper above
        for where PDF knowledge-source results get merged in).

        Guiding principle: never starve the LLM of relevant context. It is
        far better to hand the model a few extra, loosely-related sections
        than to hand it one isolated fragment and force it to guess (which
        is what causes hallucination). Filtering here only ever narrows
        among ALREADY-relevant results returned by the index; it never
        substitutes for them.
        """
        category     = intent.get("category")
        section_hint = intent.get("section_hint")
        broad        = _wants_broad_context(question, intent)

        # ── Year detection ───────────────────────────────────────────────
        # A query naming one specific year (e.g. "GPI financials for
        # 2023") previously had no way to stop retrieval from returning a
        # DIFFERENT year's data that happened to score higher
        # semantically — there was no signal distinguishing a 2023 chunk
        # from a 2024/2025 chunk in the same section. "the same year" /
        # "that year" is a backward reference to whatever year the
        # previous turn resolved, tracked via intent["years"] persisted
        # into conversation memory.
        years = detect_years_in_question(question)
        # "for the same [company]", "same period", etc. often carry no
        # year of their own -- they refer back to whatever year the
        # previous turn resolved. Previously this only fired on the
        # literal phrase "same year" (e.g. missed "for the same"),
        # which is exactly the kind of follow-up that silently reset to
        # a default/most-recent year instead of the year actually being
        # asked about.
        mentions_same = bool(re.search(r"\bsame\b", question.lower()))
        if not years and mentions_same:
            prev_years = (self.memory.get_last_intent() or {}).get("years")
            if prev_years:
                years = prev_years
        intent["years"] = years  # persisted via memory.add() for the next turn

        single_year = years[0] if len(years) == 1 else None

        # ── Implied company scope ───────────────────────────────────────────
        # "our company" / "the company" / a bare "what departments do we
        # have" always means GPI (see the TERMINOLOGY NOTE in the system
        # prompt) — but nothing downstream of here enforced that. A plain
        # question naming no company at all was searched/category-scanned
        # with NO company filter whatsoever, so a competitor's ingested
        # annual-report page that happened to mention, say, an "HR
        # department" in passing (and which _detect_company failed to tag,
        # or which the search ranked highly on pure semantic similarity)
        # could sit right next to GPI's own curated data with nothing
        # distinguishing them — which is exactly what let the model
        # "discover" phantom departments and hedge with "the company likely
        # has these departments" instead of sticking to departments.json.
        # If exactly one company is named and it isn't GPI, scope narrowly
        # to that company (e.g. "what is ITC's revenue"). If nothing is
        # named, or GPI itself is named, scope to GPI. If 2+ DIFFERENT
        # companies are named, leave this unset (None = no filter) — that
        # case is a comparison and is handled by the per-company branch
        # below, which does its own explicit filtering per company.
        #
        # EXCEPT: when a follow-up (or a "same"-worded continuation, e.g.
        # "give me the P&L for the same") names no company of its own, it
        # means the SAME company as the previous turn -- not GPI by
        # default. Without this, "balance sheet of VST 2024" correctly
        # scoped to VST, but a follow-up "statement of profit and loss
        # for the same" silently reset the scope to GPI, since nothing
        # here ever looked at what company the previous turn was about.
        detected_companies = detect_companies_in_question(question)
        if detected_companies:
            if detected_companies == ["GPI"]:
                implied_company = "GPI"
            elif len(detected_companies) == 1:
                implied_company = detected_companies[0]
            else:
                implied_company = None
        elif is_followup or mentions_same:
            prev_companies = (self.memory.get_last_intent() or {}).get("companies")
            if prev_companies and len(prev_companies) == 1:
                implied_company = prev_companies[0]
            elif prev_companies and len(prev_companies) > 1:
                implied_company = None
            else:
                implied_company = "GPI"
        else:
            implied_company = "GPI"

        intent["companies"] = detected_companies or ([implied_company] if implied_company else [])

        # ── Follow-up: reuse previous turn's results ────────────────────────
        if is_followup and self.memory.has_last_results():
            return self._scope_followup_results(intent, broad)

        # ── Category-wide requests: "process", "project", "kpi", … ─────────
        # These mean "show me everything", so bypass relevance filtering
        # entirely and return the full category rather than a top_k subset.
        if category in _FULL_CATEGORY_REQUESTS and (
            broad or category in question.lower() or section_hint is None
        ):
            full = knowledge_index.get_all_by_category(
                category, company_filter=implied_company,
            )
            if full:
                return full

        top_k = RETRIEVAL_TOP_K * 3 if broad else RETRIEVAL_TOP_K

        # ── Multi-entity comparison: retrieve each named company SEPARATELY ──
        # A single flat top-k ranked search has no concept of "give me
        # roughly equal coverage of VST, ITC, and GPI" — it just returns
        # whatever scores highest overall, and since every competitor's
        # document chunks land in the same generic section (e.g.
        # "competitor_finances"), one competitor's chunks can easily crowd
        # out another's, or crowd out GPI's own data, even though the
        # per-section cap has already been raised for broad queries. When
        # 2+ companies are named in a broad/compare question, search each
        # one independently (with a query nudged toward that company) and
        # interleave the results round-robin, so every named company is
        # guaranteed real representation in what gets sent to the LLM —
        # this is what "compare VST, ITC, and GPI" actually needs.
        named_companies = detect_companies_in_question(question) if broad else []
        if named_companies:
            # The app is fundamentally "GPI vs named competitors" — include
            # GPI's own data even if the question said "my company" rather
            # than spelling out "GPI" literally.
            if "GPI" not in named_companies:
                named_companies = ["GPI"] + named_companies

            # ── Which years to balance across ───────────────────────────
            # Mirrors the reasoning for per-company balancing below: a
            # flat top-k search has no concept of "give me roughly equal
            # coverage of 2023, 2024, AND 2025" either, and one year's
            # report chunks (whichever score highest semantically) can
            # just as easily crowd out the other years within a single
            # company's own results as one competitor used to crowd out
            # another's. If the question names 2+ years explicitly, use
            # exactly those. If it names one, that single year already
            # comes through as `single_year`. If it names none but is
            # clearly a multi-year aggregate ask ("past 3 years", "trend",
            # "year on year", …), look up which years actually exist in
            # the index for these companies and balance across ALL of
            # them, so a "past 3 years" question can't silently collapse
            # onto whichever one or two years happened to rank highest.
            if len(years) >= 2:
                target_years: List[Optional[str]] = list(years)
            elif single_year:
                target_years = [single_year]
            elif _MULTI_YEAR_PATTERN.search(question.lower()) or any(
                sig in question.lower() for sig in (
                    "past", "trend", "year on year", "year-on-year",
                    "over the years", "across years", "yearly",
                    "each year", "every year", "year wise",
                )
            ):
                discovered: set = set()
                for company in named_companies:
                    discovered.update(knowledge_index.available_years(company))
                target_years = sorted(discovered, reverse=True) if discovered else [None]
            else:
                target_years = [None]

            per_bucket_k = max(
                3, top_k // max(1, len(named_companies) * len(target_years))
            )
            # Compute this once — every bucket below searches a near-
            # identical variant of the same question, so there's no need
            # to pay for a separate Ollama embedding round-trip per
            # (company, year) combination. See search()'s query_embedding
            # docstring for why reusing one embedding doesn't change which
            # items get matched.
            base_embedding = embed(question)
            per_company_results: Dict[str, List[Dict[str, Any]]] = {}
            for company in named_companies:
                year_buckets: List[List[Dict[str, Any]]] = []
                for yr in target_years:
                    year_hint = f", year {yr}" if yr else ""
                    company_query = f"{question} ({company} specifically{year_hint})"
                    year_buckets.append(knowledge_index.search(
                        company_query, top_k=per_bucket_k, section_hint=section_hint,
                        max_document_per_section=per_bucket_k,
                        company_filter=company,
                        year_filter=yr,
                        query_embedding=base_embedding,
                    ))

                # Round-robin merge across years first (so no single year
                # crowds out the others within this company's own slice),
                # then de-duplicate — an undetected-year chunk (year=None)
                # passes every year_filter bucket above and would
                # otherwise appear once per bucket.
                merged_company: List[Dict[str, Any]] = []
                seen_sig: set = set()
                max_bucket_len = max((len(b) for b in year_buckets), default=0)
                for i in range(max_bucket_len):
                    for bucket in year_buckets:
                        if i >= len(bucket):
                            continue
                        item = bucket[i]
                        rec = item.get("record", {})
                        sig = (rec.get("source_file"), rec.get("page_or_slide"),
                               item.get("section"), item.get("category"))
                        if sig in seen_sig:
                            continue
                        seen_sig.add(sig)
                        merged_company.append(item)

                per_company_results[company] = merged_company

            # Round-robin interleave so build_context's char-budget cutoff
            # (which fills in list order) can't exhaust the entire budget
            # on the first company before later companies get a look-in.
            merged: List[Dict[str, Any]] = []
            max_len = max((len(v) for v in per_company_results.values()), default=0)
            for i in range(max_len):
                for company in named_companies:
                    bucket = per_company_results[company]
                    if i < len(bucket):
                        merged.append(bucket[i])

            if merged:
                return self._shape_results(merged, category, section_hint, broad)
            # Fall through to the normal single-search path if per-entity
            # retrieval somehow found nothing (e.g. these companies were
            # never actually ingested) — better to try the generic search
            # than return empty outright.

        # On broad/aggregate queries, also lift the per-section document
        # cap so e.g. three years of a "VST" competitor PDF (all tagged
        # into the same "competitor_finances" section) can all surface,
        # instead of being capped down to a couple of chunks regardless
        # of how much was actually ingested.
        results = knowledge_index.search(
            question, top_k=top_k, section_hint=section_hint,
            max_document_per_section=top_k if broad else None,
            year_filter=single_year,
            company_filter=implied_company,
        )

        if not results:
            return []

        return self._shape_results(results, category, section_hint, broad)

    def _scope_followup_results(
        self, intent: Dict[str, Any], broad: bool,
    ) -> List[Dict[str, Any]]:
        """
        Reuse the previous turn's retrieval for a follow-up question.

        IMPORTANT: "broad" (table/compare/list) means "give me a complete
        view of THIS topic," not "ignore the topic and dump everything
        from the previous turn." The old code treated broad follow-ups as
        an instant bypass of topic scoping — so "segment it in tabular"
        after a cigarette-brands question would grab the ENTIRE previous
        result set (whatever mix of sections that happened to include)
        instead of just the segmentation data, producing a table of the
        whole company instead of cigarette segments. Topic scoping now
        always runs first, broad or not; "broad" only affects how much of
        that scoped topic we keep (all matching records, not a top-1 slice).
        """
        prev_results = self.memory.get_last_results()

        section_hint = intent.get("section_hint")
        category     = intent.get("category")

        if section_hint:
            related = _RELATED_SECTIONS.get(section_hint, {section_hint})
            scoped = [r for r in prev_results if r.get("section") in related]
            if scoped:
                # Put exact-topic matches first, related-but-not-exact
                # matches after. build_context() fills its char budget in
                # list order, so without this a widened "related section"
                # match (e.g. "overview" pulled in alongside "history")
                # could crowd out the actual topic the follow-up is about
                # — which is exactly backwards for a "give me more detail
                # on this" follow-up, whose whole point is to surface MORE
                # of the primary topic, not a broader mix.
                scoped.sort(key=lambda r: 0 if r.get("section") == section_hint else 1)
                return scoped

            # The requested section wasn't part of what got retrieved last
            # turn — e.g. the previous answer pulled the "products" section
            # for cigarette brands, but "cigarette_segmentation" is a
            # distinct KB section that was never fetched. Do a fresh,
            # section-scoped lookup rather than silently falling back to
            # the previous turn's unrelated mix, which is what caused a
            # narrow follow-up to answer with a broad, wrong-topic table.
            fresh_top_k = RETRIEVAL_TOP_K * 3 if broad else RETRIEVAL_TOP_K
            fresh = knowledge_index.search(
                intent.get("original_question") or section_hint,
                top_k=fresh_top_k,
                section_hint=section_hint,
                max_document_per_section=fresh_top_k if broad else None,
            )
            fresh_scoped = [r for r in fresh if r.get("section") in related]
            if fresh_scoped:
                return fresh_scoped
            if fresh:
                return fresh

        if category:
            scoped = [r for r in prev_results if r.get("category") == category]
            if scoped:
                return scoped

        # Nothing matched a specific topic — only now fall back to
        # whatever the previous turn had, broad or not.
        return prev_results

    def _shape_results(
        self,
        results: List[Dict[str, Any]],
        category: Optional[str],
        section_hint: Optional[str],
        broad: bool,
    ) -> List[Dict[str, Any]]:
        """
        Narrow a fresh search result set toward the resolved intent, while
        always preserving the section's related neighborhood so genuinely
        useful context (e.g. "overview" alongside "history") is never
        thrown away.
        """
        # A resolved section_hint still narrows results even in broad
        # mode -- "broad" only means "don't cap it down to a top-1
        # slice," not "ignore the topic entirely." Previously `broad`
        # returned every result completely unfiltered, so a question
        # like "detailed GPI finances from the annual report" (which
        # trips the broad path via the "annual" keyword) let a
        # high-scoring but off-topic "competitor_finances" record sit
        # right alongside GPI's own "financials" section with nothing
        # to exclude it -- exactly the wrong-section leak this filter
        # exists to prevent for narrow queries. If nothing matches the
        # related-section set, fall back to the unfiltered wide net
        # (broad or not) rather than returning nothing.
        if section_hint:
            related = _RELATED_SECTIONS.get(section_hint, {section_hint})
            scoped = [r for r in results if r.get("section") in related]
            if scoped:
                return scoped
            if broad:
                return results
            # Fall through if nothing matched the related-section set —
            # better to return the unfiltered results than nothing at all.

        if broad:
            # Aggregate requests (tables, comparisons, lists, timelines)
            # with no specific section_hint: keep the wide net the
            # search already cast.
            return results

        if category:
            preferred = [r for r in results if r["category"] == category]
            if preferred:
                others = [r for r in results if r not in preferred]
                return preferred + others

        return results

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(
        self,
        question: str,
        results: List[Dict[str, Any]],
        intent: Dict[str, Any],
    ) -> str:
        """Build prompt, call LLM (streaming or blocking), return full answer.

        When SELF_REFINE is on and the action is complex enough to warrant
        it (see _REFINE_ACTIONS), this runs two passes: a silent draft,
        then a review pass that checks the draft against the same context
        and tightens it. Only the refined answer is streamed to the user —
        streaming the rough draft first and then replacing it would be a
        confusing UX, so the draft pass always runs blocking regardless of
        self.stream.
        """
        broad = _wants_broad_context(question, intent)
        brief = _wants_brief(question)
        action = intent.get("action", "overview")
        terse = _wants_terse_output(question)
        max_tokens = _token_budget(action, broad, brief, terse)
        # A brief/terse request should also get a shorter timeout ceiling
        # to match its shorter output budget — no reason to wait up to
        # 900s for something that should only ever need a few hundred
        # tokens (or, for terse, a couple dozen).
        if brief or terse:
            timeout = OLLAMA_TIMEOUT
        elif action in _BROAD_ACTIONS or action == "detail" or broad:
            timeout = OLLAMA_TIMEOUT_TABLE
        else:
            timeout = OLLAMA_TIMEOUT

        # A brief/terse request never needs the extra cost of a self-
        # review second pass — that's a straight doubling of generation
        # time for a request that should be fast. Bare `broad` alone
        # (retrieval width, e.g. a multi-year lookup) does NOT trigger
        # refine on its own either — see the note on `_wants_broad_context`
        # call sites: a broad multi-year context already pulls a much
        # larger prompt, so doubling generation on top of THAT is what
        # previously pushed ordinary multi-year questions past their
        # timeout with nothing ever shown to the user. Refine is only
        # worth it for genuinely complex, non-broad table/compare/detail
        # answers, where a second pass is still cheap.
        do_refine = SELF_REFINE and not brief and not terse and (action in _REFINE_ACTIONS) and not broad

        prompt, context = build_prompt(
            question       = question,
            results        = results,
            history        = self.memory.get_context_string(),
            section_hint   = intent.get("section_hint"),
            action         = action,
            broad          = broad,
            brief          = brief,
            terse          = terse,
            return_context = True,
        )

        if not do_refine:
            if self.stream and self._stream_callback:
                accumulated = ""
                for chunk in stream_prompt(prompt, max_tokens=max_tokens, timeout=timeout):
                    accumulated += chunk
                    self._stream_callback(chunk)
                return accumulated
            return send_prompt(prompt, max_tokens=max_tokens, timeout=timeout)

        # ── Two-pass: silent draft, then a streamed/returned refined answer ──
        draft = send_prompt(prompt, max_tokens=max_tokens, timeout=timeout)
        refine_prompt = build_refine_prompt(question, context, draft)
        # Reuse the same generous timeout/budget — the review pass reads a
        # similar amount of context and can produce a similarly long answer.
        if self.stream and self._stream_callback:
            accumulated = ""
            for chunk in stream_prompt(refine_prompt, max_tokens=max_tokens, timeout=timeout):
                accumulated += chunk
                self._stream_callback(chunk)
            return accumulated or draft
        refined = send_prompt(refine_prompt, max_tokens=max_tokens, timeout=timeout)
        return refined or draft

    # ── main ask ──────────────────────────────────────────────────────────────

    def ask(self, question: str, user: Optional[Dict[str, Any]] = None) -> EngineResult:
        """
        Process a question and return an EngineResult.

        This is the primary entry point for all UIs.
        Streaming output (if enabled) is delivered via the callback registered
        with set_stream_callback().

        `user` (a plain dict: email/department/is_admin -- see
        auth/store.py's User.public_dict()) is optional so every
        existing caller (terminal CLI, tests) keeps working unchanged
        with full access to everything, exactly as before this feature
        existed. Only api/server.py's authenticated endpoints pass a
        real user, which is what makes per-file department access
        control actually apply.
        """
        t_start = time.perf_counter()
        result  = EngineResult()

        # Fix known domain word-confusions (e.g. "branch" -> "brand" --
        # this knowledge base has no branch/location data, only brand
        # data) BEFORE anything else touches the question, so intent
        # resolution, retrieval, and the live analyst all see the same,
        # corrected wording. See core/query_normalizer.py.
        question = normalize_question(question)

        # ── Intent resolution ─────────────────────────────────────────────
        intent = resolver.resolve(question, self.memory)
        result.intent      = intent
        result.is_followup = intent["is_followup"]

        # ── Level 0: Live spreadsheet analysis (fully isolated) ────────────
        # is_analysis_query() is cheap and returns False immediately when
        # no spreadsheet has been ingested or the question doesn't
        # reference one -- see core/data_analyst.py's module docstring.
        # This means the PDF retrieval/RAG path below is completely
        # unaffected unless both conditions are met, and even then a
        # failure here (any `error`) falls through to that same path
        # rather than ever leaving the user with nothing.
        #
        # A genuine follow-up to a PRIOR analysis answer ("tell me the
        # state for the above", "now break that down by segment") is a
        # second, separate way in: is_analysis_query() alone can't catch
        # these, since a bare "tell me the state for the above" has no
        # analysis verb (no "total"/"outlier"/"compare"/etc.) and no
        # column name of its own to match against -- it only makes
        # sense in light of what was just asked. get_last_analysis_spec()
        # returns the previous spec (set by answer_analysis_query() the
        # last time it ran) so the LLM can inherit that context instead
        # of trying to build a full spec from a few underspecified words.
        # A pure chart follow-up ("show that on a line chart", "chart
        # it", "graph that") is NOT a request to compute anything new --
        # it's a request to re-render whatever was already just
        # answered. Routing it through analysis-spec regeneration would
        # ask the LLM to reconstruct the ENTIRE previous spec (filters,
        # group_by, outlier detection, ...) from context AND correctly
        # add a chart block, purely from a handful of words with no
        # topic of their own -- exactly the kind of compound ask a small
        # local model is most likely to get subtly wrong (e.g. silently
        # dropping the outlier detection while keeping the chart). The
        # deterministic "chart the last answer" path a little further
        # down (see _generate_chart_from_last_answer) re-renders the
        # SAME table that was already correctly computed, with zero
        # regeneration risk -- so a pure chart follow-up defers to that
        # path instead of this one. A follow-up that ALSO asks for new
        # analysis ("now show top 5 outliers on a chart") still comes
        # through here as before, since is_analysis_query() catches that.
        is_chart_request = wants_chart(question)
        pure_chart_followup = (
            is_chart_request
            and (intent["is_followup"] or _is_bare_chart_followup(question, intent))
            and not is_analysis_query(question)
        )

        last_spec = get_last_analysis_spec() if intent["is_followup"] else None
        if not pure_chart_followup and (is_analysis_query(question) or last_spec is not None):
            analysis = answer_analysis_query(question, previous_spec=last_spec, user=user)
            if analysis.error is None:
                result.answer = analysis.text
                result.route = "ANALYSIS"
                result.chart_path = analysis.chart_path
                self.memory.add(question, analysis.text, [], intent)
                result.total_time = time.perf_counter() - t_start
                return result
            if analysis.error == "restricted":
                result.answer = _RESTRICTED_MESSAGE
                result.route = "RESTRICTED"
                result.restricted = True
                result.total_time = time.perf_counter() - t_start
                return result

        # ── Level 1: Fast social responses ────────────────────────────────
        # Only skip for genuine follow-ups that need retrieval context, and
        # for explicit chart requests — try_direct_answer has no concept of
        # "the user wants an image", so e.g. "chart of GPI revenue" would
        # otherwise get silently answered as a plain-text revenue lookup
        # before the chart branch below ever got a chance to run. Also
        # skip when this question was already identified as a spreadsheet
        # analysis question that failed to execute — this fast path only
        # knows generic company-background facts (vision statements, HQ
        # address, etc.), and confidently answering a failed data question
        # with unrelated company trivia is worse than falling through to
        # the real RAG/LLM pipeline below, which can at least reason about
        # the actual question instead of pattern-matching the wrong thing.
        if not intent["is_followup"] and not is_chart_request and not is_analysis_query(question):
            direct_social = try_direct_answer(question, [])
            if direct_social is not None:
                result.answer = direct_social
                result.route  = "FAST"
                self.memory.add(question, direct_social, [], intent)
                result.total_time = time.perf_counter() - t_start
                return result

        # ── Chart follow-up: "chart based on it" charts the table the
        # user is already looking at, before touching retrieval at all.
        if is_chart_request and (intent["is_followup"] or _is_bare_chart_followup(question, intent)):
            # Chart generation emits none of its own tokens the way a
            # normal LLM answer does via stream_prompt() -- without this,
            # a UI wired to the stream callback (e.g. web/app.py) shows
            # a completely blank bubble for the whole duration, which
            # looks identical to "stuck"/frozen rather than working.
            if self.stream and self._stream_callback:
                self._stream_callback("Generating chart…")
            chart_path, caption, chart_error = self._generate_chart_from_last_answer(question, intent)
            if chart_path:
                result.answer     = caption
                result.chart_path = chart_path
                result.route      = "CHART"
                self.memory.add(question, caption, [], intent)
                result.total_time = time.perf_counter() - t_start
                return result

        # ── Retrieval ──────────────────────────────────────────────────────
        t_ret = time.perf_counter()
        results = self._retrieve(question, intent, intent["is_followup"])
        result.retrieval_time = time.perf_counter() - t_ret

        # ── Access control ──────────────────────────────────────────────
        # Applied here, right after retrieval and before ANYTHING else
        # (context building, the LLM call, memory) ever sees the results
        # -- a restricted chunk is removed from the candidate pool
        # entirely, not merely hidden in the final text, so there's no
        # path by which its content could leak into an answer even
        # indirectly (e.g. via a summary that paraphrases it). See
        # auth/permissions.py for the actual allow/deny rule.
        from auth.permissions import filter_allowed
        results, restricted_hit = filter_allowed(results, user)
        result.results = results

        if not results:
            result.answer = _RESTRICTED_MESSAGE if restricted_hit else _NO_RESULTS_MESSAGE
            result.route  = "RESTRICTED" if restricted_hit else "NO_RESULTS"
            result.restricted = restricted_hit
            result.total_time = time.perf_counter() - t_start
            return result

        # ── Fast-fail: named entity has genuinely no matching data ───────
        # If the question names specific companies (e.g. "VST"), and NONE
        # of the retrieved results actually belong to any of them, the LLM
        # is not going to conjure the answer from context that plainly
        # isn't about the thing asked — it will (correctly) eventually
        # say "not found", but only after spending however long a full
        # generation call takes on this hardware, which on a slow
        # CPU-bound local model can be minutes. Checking this cheaply
        # up front, using the SAME company tags retrieval itself already
        # computed, avoids ever burning that time on a call that was
        # always going to end in the same "I don't have that" answer.
        named = detect_companies_in_question(question)
        if named and not intent["is_followup"]:
            present = {r.get("record", {}).get("company") for r in results}
            present |= {"GPI"} if any(
                r.get("record", {}).get("company") is None for r in results
            ) else set()
            if not (present & set(named)):
                missing = ", ".join(named)
                result.answer = (
                    f"I couldn't find any ingested data for {missing} in the "
                    "knowledge base. Double-check that a document for "
                    f"{missing} has actually been imported (see "
                    "documents/processed/) — if it has, try rephrasing the "
                    "question, since the current phrasing didn't match it."
                )
                result.route = "NO_RESULTS"
                result.total_time = time.perf_counter() - t_start
                return result

        # ── Direct financial-fact lookup: "PAT for 2023 and 2024" ──────────
        # See the matching block in ask() for the full rationale.
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
                        # Inject facts into results so prompt_builder can use them
                        for r in results:
                            if r.get("record"):
                                r["record"]["_injected_financial_facts"] = fact_data
                                break
                else:
                    result.answer = fact_answer
                    result.route  = "FACT"
                    self.memory.add(question, fact_answer, results, intent)
                    result.total_time = time.perf_counter() - t_start
                    return result

        # ── Direct statement table: P&L/balance sheet/cash flow ────────────
        # Bypasses the LLM entirely for a confidently-matched named
        # statement -- see core/direct_answer.try_statement_table_answer
        # for why this is more reliable than asking a small local model
        # to reproduce a large table verbatim.
        if STATEMENT_TABLE_DIRECT_ROUTE and not wants_chart(question):
            statement_answer = try_statement_table_answer(question, results)
            if statement_answer is not None:
                result.answer = statement_answer
                result.structured_table = try_statement_table_structured(question, results)
                result.route  = "DIRECT"
                self.memory.add(question, statement_answer, results, intent)
                result.total_time = time.perf_counter() - t_start
                return result

        # ── Chart generation: explicit "chart/graph/plot" requests ─────────
        # Runs before Direct RAG / normal LLM answering — a chart request
        # is a distinct output type, not a phrasing of a text question.
        # Falls through to the normal text-answer path below if the
        # context genuinely doesn't have enough real numbers, rather than
        # ever rendering a chart from invented data.
        if wants_chart(question):
            # This path additionally runs an LLM call (chart-spec
            # extraction) before rendering -- the slowest step in the
            # whole chart pipeline -- so the blank-UI problem is worse
            # here than in the from-last-answer path above.
            if self.stream and self._stream_callback:
                self._stream_callback("Generating chart…")
            chart_path, caption, chart_error = self._generate_chart(question, results)
            if chart_path:
                result.answer     = caption
                result.chart_path = chart_path
                result.route      = "CHART"
                self.memory.add(question, caption, results, intent)
                result.total_time = time.perf_counter() - t_start
                return result
            # No usable data for a chart — say so plainly instead of
            # silently answering as if nothing was asked for a chart.
            result.answer = (
                f"I can't build that chart: {chart_error} "
                "I can still answer in text if that helps — just ask "
                "without 'chart'/'graph', or rephrase what you'd like plotted."
            )
            result.route = "NO_RESULTS"
            result.total_time = time.perf_counter() - t_start
            return result

        # ── Level 2: Direct RAG ───────────────────────────────────────────
        if not needs_llm(question) and not intent["is_followup"]:
            direct = try_direct_answer(question, results)
            if direct is not None:
                result.answer = direct
                result.route  = "DIRECT"
                self.memory.add(question, direct, results, intent)
                result.total_time = time.perf_counter() - t_start
                return result

        # ── Level 3: LLM reasoning ────────────────────────────────────────
        try:
            answer = self._call_llm(question, results, intent)
        except RuntimeError as exc:
            result.error  = str(exc)
            result.answer = f"Error: {exc}"
            result.route  = "ERROR"
            result.total_time = time.perf_counter() - t_start
            return result

        result.answer = answer
        result.route  = "LLM"
        self.memory.add(question, answer, results, intent)
        result.total_time = time.perf_counter() - t_start
        return result

    # ── streaming variant (generator-based) ───────────────────────────────────

    def ask_stream(self, question: str, user: Optional[Dict[str, Any]] = None) -> Generator:
        """
        Generator-based streaming entry point.

        Yields (event_type, data) tuples:
          ("intent",    intent_dict)
          ("retrieval", results_list)
          ("chunk",     token_string)
          ("done",      EngineResult)
        """
        t_start = time.perf_counter()

        question = normalize_question(question)

        intent = resolver.resolve(question, self.memory)
        yield ("intent", intent)

        is_chart_request = wants_chart(question)

        # ── Level 0: Live spreadsheet analysis (fully isolated) ────────────
        # See the identical block in ask() above and core/data_analyst.py's
        # module docstring for why this can never affect PDF retrieval.
        # A pure chart follow-up ("show that on a line chart") defers to
        # the deterministic re-chart-the-last-answer path further down
        # instead of regenerating a whole analysis spec from scratch --
        # see the matching block in ask() for the full rationale.
        pure_chart_followup = (
            is_chart_request
            and (intent["is_followup"] or _is_bare_chart_followup(question, intent))
            and not is_analysis_query(question)
        )
        last_spec = get_last_analysis_spec() if intent["is_followup"] else None
        if not pure_chart_followup and (is_analysis_query(question) or last_spec is not None):
            analysis = answer_analysis_query(question, previous_spec=last_spec, user=user)
            if analysis.error is None:
                res = EngineResult(
                    answer=analysis.text, route="ANALYSIS",
                    intent=intent, chart_path=analysis.chart_path,
                    total_time=time.perf_counter() - t_start,
                )
                self.memory.add(question, analysis.text, [], intent)
                yield ("chunk", analysis.text)
                if analysis.chart_path:
                    yield ("chart", analysis.chart_path)
                yield ("done", res)
                return
            if analysis.error == "restricted":
                res = EngineResult(
                    answer=_RESTRICTED_MESSAGE, route="RESTRICTED", restricted=True,
                    intent=intent, total_time=time.perf_counter() - t_start,
                )
                yield ("chunk", _RESTRICTED_MESSAGE)
                yield ("done", res)
                return

        # ── Social fast path ────────────────────────────────────────────────
        if not intent["is_followup"] and not is_chart_request and not is_analysis_query(question):
            social = try_direct_answer(question, [])
            if social is not None:
                res = EngineResult(
                    answer=social, route="FAST",
                    intent=intent, total_time=time.perf_counter() - t_start,
                )
                self.memory.add(question, social, [], intent)
                yield ("chunk", social)
                yield ("done", res)
                return

        # ── Chart follow-up: "chart based on it" charts the table the
        # user is already looking at, before touching retrieval at all.
        if is_chart_request and (intent["is_followup"] or _is_bare_chart_followup(question, intent)):
            chart_path, caption, chart_error = self._generate_chart_from_last_answer(question, intent)
            if chart_path:
                res = EngineResult(
                    answer=caption, route="CHART",
                    intent=intent, chart_path=chart_path,
                    total_time=time.perf_counter() - t_start,
                )
                self.memory.add(question, caption, [], intent)
                yield ("chunk", caption)
                yield ("chart", chart_path)
                yield ("done", res)
                return

        # ── Retrieval ────────────────────────────────────────────────────────
        t_ret = time.perf_counter()
        results = self._retrieve(question, intent, intent["is_followup"])
        ret_time = time.perf_counter() - t_ret

        from auth.permissions import filter_allowed
        results, restricted_hit = filter_allowed(results, user)
        yield ("retrieval", results)

        # ── No results ───────────────────────────────────────────────────────
        if not results:
            msg = _RESTRICTED_MESSAGE if restricted_hit else _NO_RESULTS_MESSAGE
            res = EngineResult(
                answer=msg, route="RESTRICTED" if restricted_hit else "NO_RESULTS",
                restricted=restricted_hit,
                intent=intent, total_time=time.perf_counter() - t_start,
            )
            yield ("chunk", msg)
            yield ("done", res)
            return

        # ── Direct financial-fact lookup: "PAT for 2023 and 2024" ──────────
        # See the matching block in ask() for the full rationale.
        if not wants_chart(question):
            fact_answer = try_financial_fact_lookup(question, results)
            if fact_answer is not None:
                res = EngineResult(
                    answer=fact_answer, route="FACT", results=results,
                    intent=intent, retrieval_time=ret_time,
                    total_time=time.perf_counter() - t_start,
                )
                self.memory.add(question, fact_answer, results, intent)
                yield ("chunk", fact_answer)
                yield ("done", res)
                return

        # ── Direct statement table: P&L/balance sheet/cash flow ────────────
        if STATEMENT_TABLE_DIRECT_ROUTE and not wants_chart(question):
            statement_answer = try_statement_table_answer(question, results)
            if statement_answer is not None:
                res = EngineResult(
                    answer=statement_answer, route="DIRECT", results=results,
                    intent=intent, retrieval_time=ret_time,
                    total_time=time.perf_counter() - t_start,
                    structured_table=try_statement_table_structured(question, results),
                )
                self.memory.add(question, statement_answer, results, intent)
                yield ("chunk", statement_answer)
                yield ("done", res)
                return

        # ── Chart generation: explicit "chart/graph/plot" requests ─────────
        if wants_chart(question):
            chart_path, caption, chart_error = self._generate_chart(question, results)
            if chart_path:
                res = EngineResult(
                    answer=caption, route="CHART", results=results,
                    intent=intent, retrieval_time=ret_time, chart_path=chart_path,
                    total_time=time.perf_counter() - t_start,
                )
                self.memory.add(question, caption, results, intent)
                yield ("chunk", caption)
                yield ("chart", chart_path)
                yield ("done", res)
                return
            fallback = (
                f"I can't build that chart: {chart_error} "
                "I can still answer in text if that helps — just ask "
                "without 'chart'/'graph', or rephrase what you'd like plotted."
            )
            res = EngineResult(
                answer=fallback, route="NO_RESULTS", results=results,
                intent=intent, retrieval_time=ret_time,
                total_time=time.perf_counter() - t_start,
            )
            yield ("chunk", fallback)
            yield ("done", res)
            return

        # ── Direct RAG ───────────────────────────────────────────────────────
        if not needs_llm(question) and not intent["is_followup"]:
            direct = try_direct_answer(question, results)
            if direct is not None:
                res = EngineResult(
                    answer=direct, route="DIRECT", results=results,
                    intent=intent, retrieval_time=ret_time,
                    total_time=time.perf_counter() - t_start,
                )
                self.memory.add(question, direct, results, intent)
                yield ("chunk", direct)
                yield ("done", res)
                return

        # ── LLM streaming ───────────────────────────────────────────────────
        broad = _wants_broad_context(question, intent)
        brief = _wants_brief(question)
        terse = _wants_terse_output(question)
        action = intent.get("action", "overview")
        max_tokens = _token_budget(action, broad, brief, terse)
        if brief or terse:
            timeout = OLLAMA_TIMEOUT
        else:
            timeout = OLLAMA_TIMEOUT_TABLE if (action in _BROAD_ACTIONS or action == "detail" or broad) else OLLAMA_TIMEOUT
        # See the matching note in _call_llm: bare `broad` no longer
        # triggers refine on its own, and brief/terse always skip it.
        do_refine = SELF_REFINE and not brief and not terse and (action in _REFINE_ACTIONS) and not broad

        prompt, context = build_prompt(
            question=question,
            results=results,
            history=self.memory.get_context_string(),
            section_hint=intent.get("section_hint"),
            action=action,
            broad=broad,
            brief=brief,
            terse=terse,
            return_context=True,
        )

        accumulated = ""
        t_llm = time.perf_counter()
        try:
            if do_refine:
                # Silent draft pass — not streamed, so the user only ever
                # sees the reviewed/tightened final answer.
                draft = send_prompt(prompt, max_tokens=max_tokens, timeout=timeout)
                refine_prompt = build_refine_prompt(question, context, draft)
                stream_source = stream_prompt(refine_prompt, max_tokens=max_tokens, timeout=timeout)
            else:
                stream_source = stream_prompt(prompt, max_tokens=max_tokens, timeout=timeout)

            for chunk in stream_source:
                accumulated += chunk
                yield ("chunk", chunk)

            if do_refine and not accumulated:
                # Refine pass produced nothing usable — fall back to the
                # draft rather than returning an empty answer.
                accumulated = draft
                yield ("chunk", draft)
        except RuntimeError as exc:
            error_msg = f"Error: {exc}"
            res = EngineResult(
                answer=error_msg, route="ERROR",
                error=str(exc), intent=intent,
                total_time=time.perf_counter() - t_start,
            )
            yield ("chunk", error_msg)
            yield ("done", res)
            return

        llm_time = time.perf_counter() - t_llm
        self.memory.add(question, accumulated, results, intent)

        res = EngineResult(
            answer=accumulated,
            route="LLM",
            results=results,
            intent=intent,
            prompt_size=len(prompt),
            retrieval_time=ret_time,
            llm_time=llm_time,
            total_time=time.perf_counter() - t_start,
        )
        yield ("done", res)


# ─── Module-level singleton ───────────────────────────────────────────────────

engine = AssistantEngine()