"""
core/direct_answer.py

Direct RAG Layer — answers questions WITHOUT calling the LLM.

When to use Direct RAG
-----------------------
- Greetings / social phrases
- Simple lookup: "What is the CEO name?"  →  direct string from data
- List requests: "List all departments"
- Owner / status queries
- Table requests for structured data
- Simple company facts (headquarters, mission, vision, …)

The layer returns None if it cannot answer confidently,
which tells the router to escalate to the LLM.
"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from retrieval.knowledge_index import index as knowledge_index, detect_years_in_question, detect_companies_in_question
from utils.data_loader import load_json
from core.config import COMPANY_FILE


# ─── Social response pools ────────────────────────────────────────────────────

_GREETINGS = {
    "hello", "hi", "hey", "yo", "greetings",
    "good morning", "good afternoon", "good evening",
}

_THANKS = {"thanks", "thank you", "thx", "appreciate it", "thank you very much"}

_GOODBYES = {"bye", "goodbye", "see you", "see ya", "exit", "quit", "take care"}

_GREETING_REPLIES = [
    "Hello! I'm your GPI Knowledge Assistant. Ask me anything about Godfrey Phillips India.",
    "Hi! What would you like to know about GPI today?",
    "Hey! I'm here to answer your questions about Godfrey Phillips India.",
    "Good to see you. What can I help you with?",
]

_THANKS_REPLIES = [
    "You're welcome!", "Happy to help.", "Glad I could assist.",
    "Anytime! Let me know if you have more questions.",
]

_GOODBYE_REPLIES = [
    "Goodbye! Have a great day.", "See you later.", "Take care!",
    "Thanks for using the GPI Knowledge Assistant.",
]


# ─── Simple field lookups ─────────────────────────────────────────────────────

# Maps question keywords → company record field path. "label" is a short,
# human-readable name for the field, used to phrase the answer as a full
# sentence (varied across a few templates) instead of returning the bare
# value with no context — e.g. "Sharad Aggarwal" alone doesn't tell the
# user what role that answers, and always phrasing it identically felt
# robotic on repeated questions.
_SIMPLE_LOOKUPS: List[Dict[str, Any]] = [
    {"keywords": {"headquarters", "hq", "head office", "where is gpi", "location"},
     "field_path": ["additional_info", "gpi_details", "headquarters"], "label": "headquarters"},
    {"keywords": {"mission"},
     "field_path": ["additional_info", "mission"], "label": "mission"},
    {"keywords": {"vision"},
     "field_path": ["additional_info", "vision"], "label": "vision"},
    {"keywords": {"incorporation date", "when was gpi founded", "founded", "founding date"},
     "field_path": ["additional_info", "gpi_details", "incorporation_date"], "label": "incorporation date"},
    {"keywords": {"company name", "name of company", "what is gpi?", "what is gpi's name"},
     "field_path": ["name"], "label": "name"},
    {"keywords": {"ceo", "who is the ceo", "chief executive"},
     "field_path": ["additional_info", "gpi_details", "leadership", "ceo"], "label": "CEO"},
    {"keywords": {"cfo", "who is the cfo", "chief financial"},
     "field_path": ["additional_info", "gpi_details", "leadership", "cfo"], "label": "CFO"},
    {"keywords": {"coo", "who is the coo", "chief operating"},
     "field_path": ["additional_info", "gpi_details", "leadership", "coo"], "label": "COO"},
    {"keywords": {"chairperson", "chairwoman", "chairman", "md", "managing director"},
     "field_path": ["additional_info", "gpi_details", "leadership", "chairperson_md"], "label": "Chairperson & MD"},
]

# A few interchangeable phrasings so repeated simple-lookup questions don't
# all come back in the exact same sentence shape.
_ROLE_TEMPLATES = [
    "{val} is the {label} of {company}.",
    "The {label} of {company} is {val}.",
    "{company}'s {label} is {val}.",
]
_FACT_TEMPLATES = [
    "{company}'s {label} is {val}.",
    "The {label} of {company}: {val}.",
    "{val} — that's {company}'s {label}.",
]
_ROLE_LABELS = {"CEO", "CFO", "COO", "Chairperson & MD"}


def _phrase(label: str, val: str, company_name: str) -> str:
    templates = _ROLE_TEMPLATES if label in _ROLE_LABELS else _FACT_TEMPLATES
    return random.choice(templates).format(label=label, val=val, company=company_name or "the company")


def _traverse(record: Dict[str, Any], path: List[str]) -> Optional[str]:
    """Walk a key path into a nested dict; return string value or None."""
    obj: Any = record
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
        if obj is None:
            return None
    if isinstance(obj, (str, int, float)):
        return str(obj)
    return None


# ─── Entity scoping (GPI vs PMI) ───────────────────────────────────────────
# The underlying company_overview.json record is a single JSON object that
# covers BOTH Godfrey Phillips India (GPI) and its affiliate Philip Morris
# International (PMI) — two entirely different companies with different
# owners, leadership, and financials. Direct RAG reads straight from that
# raw record (see _get_company_record below) and previously had no concept
# of which of the two companies a question was actually about, so "who is
# the owner of GPI" and "who is the owner of PMI" both silently returned
# the SAME top-level "owner" field (Dr. Bina Modi) — correct for GPI, but
# wrong for PMI, which is a separate, independently owned, publicly traded
# multinational (PMI in fact holds a 25.1% stake IN GPI, not the reverse).
_PMI_WORDS = {"pmi", "philip morris", "philip morris international"}
_GPI_WORDS = {"gpi", "godfrey phillips", "godfrey phillips india"}


def _detect_entity(q: str) -> str:
    """
    Decide whether a question is scoped to GPI, PMI, or is ambiguous/general.
    Returns "pmi" only when PMI is named and GPI is not — otherwise defaults
    to "gpi", since GPI is the primary company this assistant covers and
    most unscoped questions ("who owns the company") mean GPI.
    """
    has_pmi = any(w in q for w in _PMI_WORDS)
    has_gpi = any(w in q for w in _GPI_WORDS)
    if has_pmi and not has_gpi:
        return "pmi"
    return "gpi"


_PMI_OWNER_ANSWER = (
    "Philip Morris International (PMI) is a separate, publicly traded, "
    "independently owned multinational company (NYSE: PM) — it isn't owned "
    "by an individual or promoter group. In fact the relationship runs the "
    "other way: PMI itself holds a 25.1% minority equity stake in Godfrey "
    "Phillips India (GPI). GPI's own owner/promoter is Dr. Bina Modi, but "
    "that ownership applies to GPI only, not to PMI."
)


def _get_company_record() -> Optional[Dict[str, Any]]:
    """
    Return the first RAW company record (straight from company_overview.json),
    not the flattened "overview" section KnowledgeIndex builds for retrieval.

    The "overview" IndexItem is deliberately a lean summary (see
    retrieval/knowledge_index.py) — it doesn't carry the
    additional_info.gpi_details.* / metrics.* nesting that _SIMPLE_LOOKUPS'
    field paths expect, so using it here silently broke every direct
    lookup (CEO, CFO, mission, vision, revenue, …), forcing them all to
    fall through to the slower LLM path even though these are simple,
    single-fact answers Direct RAG is meant to handle instantly.
    """
    try:
        data = load_json(COMPANY_FILE)
        records = data.get("company_profile", {}).get("records", [])
        return records[0] if records else None
    except Exception:
        return None


# ─── Public API ───────────────────────────────────────────────────────────────

_METRIC_QUESTION_SYNONYMS = {
    "PAT": [r"\bpat\b", r"profit\s+after\s+tax", r"profit\s+for\s+the\s+year",
            r"net\s+profit"],
    "PBT": [r"\bpbt\b", r"profit\s+before\s+tax"],
    "Tax Expense": [r"tax\s+expense", r"\btax\b(?!.*profit)"],
    "Revenue": [r"revenue", r"turnover", r"total\s+sales\b"],
    "Total Income": [r"total\s+income"],
    "EBITDA": [r"\bebitda\b"],
    "EPS (Basic)": [r"\beps\b(?!.*dilut)", r"basic\s+earnings\s+per\s+share"],
    "EPS (Diluted)": [r"diluted\s+earnings\s+per\s+share"],
}


def _detect_requested_metric(question: str) -> Optional[str]:
    q = question.lower()
    for metric, patterns in _METRIC_QUESTION_SYNONYMS.items():
        if any(re.search(p, q) for p in patterns):
            return metric
    return None


def collect_financial_facts(
    metric: str, years: List[int], companies: Optional[List[str]],
    results: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, Tuple[Optional[str], Optional[int]]]]]:
    """
    Shared collection logic behind both `try_financial_fact_lookup` (a
    text answer) and chart generation's deterministic path
    (`core.chart_generator.try_build_chart_spec_from_facts`) -- one
    dictionary lookup against ingestion-time-extracted facts, with a
    full-index fallback scan for any year that didn't surface in the
    already-retrieved chunks. Kept in one place so both callers can
    never silently diverge in which value counts as "the" answer for a
    given metric/year/company.
    
    Now supports multiple companies. Returns facts grouped by company.
    """
    def _fact_selection_rank(raw_label: str, is_table: Any, page: Any) -> Tuple[int, int, int]:
        """
        Rank competing facts for the SAME (metric, year) so the correct
        one wins deterministically. This is the core fix for values like
        GPI's 2025 PAT, where the standalone P&L page legitimately prints
        BOTH a "Profit for the year from continuing operations" subtotal
        (112,378.13) AND the true aggregate "Profit for the year"
        (104,320.37) -- first-match-wins was returning whichever chunk the
        retriever happened to surface first (the subtotal), not the total.

        Higher tuple wins (we take the max), by three ordered preferences:

          1. table-sourced over prose-sourced -- a figure lifted from an
             actual statement grid is more trustworthy than one scraped
             from a surrounding sentence.
          2. aggregate TOTAL over a "from continuing/discontinued
             operations" / "before ..." SUBTOTAL -- the headline figure a
             user means by "PAT"/"PBT" is the whole-company total line, not
             one operating-segment component of it. A year with NO
             discontinued operations (2023, 2024 here) has only the total
             line, so this preference is a no-op there and can't regress
             the years that were already correct.
          3. EARLIER page over later -- in every one of these annual
             reports the STANDALONE statements are printed before the
             CONSOLIDATED ones, and the standalone figure is the primary
             one these questions expect (and the one 2023/2024 already
             return). Page order is the only reliable standalone-vs-
             consolidated signal available: the chunk text itself carries
             no "standalone"/"consolidated" marker to key on.
        """
        label = (raw_label or "").lower()
        is_subtotal = any(m in label for m in (
            "from continuing operations", "from discontinued operations",
            "before share of", "before exceptional",
        ))
        return (
            1 if is_table else 0,
            0 if is_subtotal else 1,
            -(page if isinstance(page, int) else 1_000_000),
        )

    def _is_ratio_label(label: str, metric: str) -> bool:
        """True if a label is a margin/ratio/percentage row rather than an
        absolute figure -- matched as WHOLE WORDS only.

        The whole-word match is essential, not cosmetic. "ratio" is a
        substring of "ope-ratio-ns", so a plain `"ratio" in label` test
        silently discards EVERY "Revenue from operations" and "... from
        continuing operations" row -- which is exactly how Revenue (all
        years) and 2025's PBT came back empty until this was fixed. This
        is the very same trap the ingestion-side blocklist already had to
        solve with `\\bword\\b`; the selection side has to solve it too.
        Whole-word matching still catches genuine "PAT Margin (%)" /
        "Current Ratio" labels without eating "operations".
        """
        if not label:
            return False
        # PAT, PBT, Revenue, etc. are absolute figures; a label carrying
        # one of these as a WHOLE WORD is a margin/ratio/percentage row
        # whose value must never be served as the real absolute figure.
        if metric not in ("PAT", "PBT", "Revenue", "Total Income", "EBITDA"):
            return False
        label_lower = label.lower()
        return any(
            re.search(rf"\b{w}\b", label_lower)
            for w in ("ratio", "margin", "percentage", "percent", "rate")
        )

    def _company_mismatch(company: Optional[str], comp_val: Any) -> bool:
        # Preserved naive substring check: correctly keeps GPI's own
        # gpi_*.pdf chunks (tagged "GPI") while excluding e.g.
        # tobacco_*.pdf chunks tagged "Godfrey Phillips India" from a GPI
        # lookup, and is harmless when no company was requested.
        return bool(company and comp_val and company.lower() not in str(comp_val).lower())

    all_facts: Dict[str, Dict[int, float]] = {}
    all_sources: Dict[str, Dict[int, Tuple[Optional[str], Optional[int]]]] = {}

    # If no companies specified, use None as a single default
    target_companies = companies if companies else [None]

    for company in target_companies:
        # year -> (best_rank_tuple, value, source_file, page). We gather
        # EVERY candidate for each requested year from all three sources
        # below and keep only the highest-ranked one, instead of taking
        # the first fact the retriever happened to return.
        best: Dict[int, Tuple[Tuple[int, int, int], float, Optional[str], Optional[int]]] = {}

        def _consider(f: Dict[str, Any], is_table: Any, page: Any, source_file: Optional[str]):
            if f.get("metric") != metric or f.get("year") not in years:
                return
            val = f.get("value")
            if val is None:
                return
            raw_label = f.get("raw_label", "")
            if _is_ratio_label(raw_label, metric):
                return
            y = f["year"]
            rank = _fact_selection_rank(raw_label, is_table, page)
            if y not in best or rank > best[y][0]:
                best[y] = (rank, val, source_file, page)

        # 1) already-retrieved result records
        for r in results:
            record = r.get("record") or {}
            if _company_mismatch(company, record.get("company")):
                continue
            for f in record.get("financial_facts") or []:
                _consider(f, record.get("is_table"), record.get("page"),
                          record.get("source_file"))

        # 2) full PDF-index sweep -- so the best-ranked fact wins even when
        #    the retriever surfaced only a subtotal / consolidated chunk
        #    and not the standalone-total one. Always run (not just for
        #    missing years): the standalone total may sit in a chunk that
        #    lost retrieval to the subtotal chunk for a year we already
        #    "have", and ranked selection must see both to choose right.
        try:
            from retrieval.pdf_retriever import pdf_retriever
            pdf_retriever.ensure_index_built()
            for cached in pdf_retriever._chunks:
                cd = cached.chunk_dict
                if _company_mismatch(company, cd.get("company")):
                    continue
                for f in cd.get("financial_facts") or []:
                    _consider(f, cd.get("is_table"), cd.get("page"),
                              cd.get("source"))
        except Exception:
            pass

        # 3) curated JSON knowledge index
        try:
            from retrieval.knowledge_index import index as knowledge_index
            for item in knowledge_index.items:
                rec = item.record
                if _company_mismatch(company, rec.get("company")):
                    continue
                for f in rec.get("financial_facts") or []:
                    _consider(f, rec.get("is_table"), rec.get("page"),
                              rec.get("source_file"))
        except Exception:
            pass

        if best:
            all_facts[company or "default"] = {y: b[1] for y, b in best.items()}
            all_sources[company or "default"] = {y: (b[2], b[3]) for y, b in best.items()}

    return all_facts, all_sources


def try_financial_fact_lookup(question: str, results: List[Dict[str, Any]]) -> Optional[str]:
    """
    Deterministic, non-LLM answer for a "give me <metric> for <year(s)>"
    question (PAT/PBT/Revenue/EBITDA/etc), using facts extracted at
    ingestion time (ingestion/financial_facts.py) -- each one already
    tagged to the exact year of the column it came from. No LLM
    reasoning or column-position inference happens here at all: this is
    a direct dictionary lookup, so it can't reproduce the "gives the
    right year, then recomputes or loses track of the next one" failure
    mode an LLM re-deriving the same mapping from raw table text was
    prone to.

    Only fires for a clean single-metric question with at least one
    explicit year and no request for a full statement/table/comparison-
    with-reasoning (those still go through the normal LLM path, which
    is better suited to explaining WHY something changed, not just what
    the number was).

    Returns None (falls through to normal retrieval/LLM) if the metric,
    company, or year can't be confidently identified, or if none of the
    requested years have a fact on record -- this only ever answers
    when it can do so with full confidence, same principle as the rest
    of this module's direct-answer functions.
    """
    if any(kw in question.lower() for kw in (
        "statement", "table", "breakdown", "full", "complete", "why",
        "reason", "explain", "trend", "all line",
    )):
        return None

    metric = _detect_requested_metric(question)
    if metric is None:
        return None

    years = [int(y) for y in detect_years_in_question(question) if y.isdigit()]
    if not years:
        return None

    companies = detect_companies_in_question(question)

    all_facts, all_sources = collect_financial_facts(metric, years, companies, results)

    # Check if any company has facts
    has_facts = any(facts for facts in all_facts.values())
    if not has_facts:
        return None

    lines = [f"**{metric}**"]
    for company in companies:
        facts_by_year = all_facts.get(company, {})
        if not facts_by_year:
            continue
        lines.append(f"\n**{company}**")
        for y in sorted(years):
            if y in facts_by_year:
                lines.append(f"- {y}: {facts_by_year[y]:,.2f}")
            else:
                lines.append(f"- {y}: not found in the ingested documents")

    return "\n".join(lines)


def get_financial_facts_for_llm(question: str, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Same fact lookup as try_financial_fact_lookup, but returns structured
    data (metric, company, facts_by_year, sources_by_year) for the LLM
    to use in analysis questions, instead of a pre-formatted string.
    
    Returns None if no facts found or question isn't a clean metric lookup.
    Now supports multi-company queries.
    """
    if any(kw in question.lower() for kw in (
        "statement", "table", "breakdown", "full", "complete", "all line",
    )):
        return None

    metric = _detect_requested_metric(question)
    if metric is None:
        return None

    years = [int(y) for y in detect_years_in_question(question) if y.isdigit()]
    if not years:
        return None

    companies = detect_companies_in_question(question)
    if not companies:
        return None

    all_facts, all_sources = collect_financial_facts(metric, years, companies, results)

    if not all_facts:
        return None

    if not all_facts:
        return None

    return {
        "metric": metric,
        "companies": companies,
        "years": years,
        "facts_by_company": all_facts,
        "sources_by_company": all_sources,
    }


def try_direct_answer(
    question: str,
    results: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Attempt to answer *question* directly from retrieved results.

    Returns a string answer if confidence is high, or None to fall back to LLM.
    """
    q = question.lower().strip()

    # ── Social ──────────────────────────────────────────────────────────────
    if q in _GREETINGS:
        return random.choice(_GREETING_REPLIES)
    if q in _THANKS:
        return random.choice(_THANKS_REPLIES)
    if q in _GOODBYES:
        return random.choice(_GOODBYE_REPLIES)

    # ── Company owner lookup (kept separate from _SIMPLE_LOOKUPS below) ──────
    # "owner" is a genuinely ambiguous word — "who is the owner of the
    # company" should answer from the record's top-level "owner" field,
    # but "who is the owner of the IT department" is asking about a
    # completely different entity and must NOT be shortcut to the
    # company's owner. A plain keyword-table entry can't express that
    # distinction (it just checks substring containment), so this check
    # runs first and only fires when the question isn't clearly scoped to
    # a department/project/process/kpi/problem instead.
    _SUB_ENTITY_WORDS = {
        "department", "departments", "team", "division", "project",
        "projects", "process", "processes", "kpi", "kpis", "problem",
        "problems", "initiative",
    }
    _OWNER_PHRASES = [
        "The owner of {name} is {owner}.",
        "{name} is owned by {owner}.",
        "{owner} owns {name}.",
    ]
    company = _get_company_record()
    entity = _detect_entity(q)
    if company and ("owner" in q or "who owns" in q) and not any(w in q for w in _SUB_ENTITY_WORDS):
        if entity == "pmi":
            return _PMI_OWNER_ANSWER
        owner = company.get("owner")
        if owner:
            # Use GPI's own name here, not the raw record's combined
            # "GPI & PMI" name — the owner field is GPI-specific.
            return random.choice(_OWNER_PHRASES).format(
                name="Godfrey Phillips India (GPI)", owner=owner
            )

    # ── Simple field lookups on company record ───────────────────────────────
    # All of _SIMPLE_LOOKUPS (CEO/CFO/COO/chairperson/revenue/etc.) are GPI
    # figures specifically — PMI has its own separate leadership and
    # financials that this dataset doesn't carry, so a PMI-scoped question
    # ("who is the CFO of PMI") must not be shortcut to GPI's answer.
    if company and entity != "pmi":
        for entry in _SIMPLE_LOOKUPS:
            if any(kw in q for kw in entry["keywords"]):
                val = _traverse(company, entry["field_path"])
                if val:
                    return _phrase(entry["label"], val, "Godfrey Phillips India (GPI)")
    elif company and entity == "pmi":
        for entry in _SIMPLE_LOOKUPS:
            if any(kw in q for kw in entry["keywords"]):
                return (
                    f"I don't have PMI's {entry['label']} in this knowledge "
                    "base — the figures here (CEO, CFO, COO, chairperson, "
                    "revenue, etc.) are Godfrey Phillips India's, which is a "
                    "separate company from Philip Morris International. I do "
                    "have PMI's global strategy, market presence, and product "
                    "portfolio if that helps."
                )

    # ── Named entity owner / status lookup ───────────────────────────────────
    if results:
        for r in results:
            rec = r.get("record", {})
            name = str(rec.get("name", "")).lower()
            if name and name in q:
                if "owner" in q:
                    owner = rec.get("owner")
                    if owner:
                        return f"{rec.get('name')} is owned by {owner}."
                if "status" in q:
                    status = rec.get("status")
                    if status:
                        return f"{rec.get('name')} status: {status}."
                if "head" in q or "who leads" in q:
                    head = rec.get("head") or rec.get("owner")
                    if head:
                        return f"{rec.get('name')} is headed by {head}."

    return None  # escalate to LLM


def needs_llm(question: str) -> bool:
    """
    Return True if the question requires reasoning / synthesis
    and should be handled by the LLM, not Direct RAG.
    """
    q = question.lower()
    llm_signals = [
        "why", "how", "explain", "analyze", "analysis",
        "compare", "difference", "strategy", "impact",
        "elaborate", "tell me about", "describe",
        "what does", "what are the implications",
        "pros and cons", "advantages", "disadvantages",
        "recommendation", "suggest",
    ]
    return any(w in q for w in llm_signals)


# ─── Financial-statement direct table answer ──────────────────────────────────
#
# A full P&L / balance sheet / cash-flow statement is exactly the kind of
# large, dense table a small local LLM reliably mangles when asked to
# "reproduce" it: it can lose row boundaries, collapse the whole table
# into one run-on line, or add a hedge like "the complete statement is
# not available" even when the real table is sitting right there in
# context. ingestion/table.py's Table.to_markdown() ALREADY produces a
# clean, well-formed Markdown table straight from the PDF at ingestion
# time -- so once a query is clearly asking for ONE specific named
# statement, and a genuine table chunk for that exact statement was
# retrieved, the most reliable answer is to show that chunk's own
# Markdown verbatim instead of routing it through the LLM at all. No
# LLM call here means no chance of a hallucinated caveat, a garbled
# row, or an inconsistent answer between two different phrasings of the
# same question.
_STATEMENT_KEYWORDS = (
    ("cash flow", "Cash Flow Statement"),
    ("cashflow", "Cash Flow Statement"),
    ("balance sheet", "Balance Sheet"),
    ("changes in equity", "Statement of Changes in Equity"),
    ("profit and loss", "Statement of Profit and Loss"),
    ("profit & loss", "Statement of Profit and Loss"),
    ("p&l", "Statement of Profit and Loss"),
    ("p & l", "Statement of Profit and Loss"),
    ("pnl", "Statement of Profit and Loss"),
    ("income statement", "Statement of Profit and Loss"),
)


def _detect_requested_statement(question: str) -> Optional[str]:
    q = question.lower()
    for kw, canonical in _STATEMENT_KEYWORDS:
        if kw in q:
            return canonical
    return None


def clean_table_chunk_text(text: str) -> str:
    """
    Clean up a SINGLE chunk's own table text for display -- merges a
    split "Year ended" / actual-date header pair into one real header
    row, and drops fully-blank columns. This is the same cleanup the
    direct route already applies while stitching multiple chunks
    together, but here it's applied independently to one chunk at a
    time so ANY consumer (not just the direct route) can show a table
    with real year labels instead of raw, ambiguous placeholder headers.

    This matters most for the LLM route: when a chunk's raw text still
    has an un-merged header (literal repeated "Year ended" / "Year
    ended" cells, with the real dates sitting in the very next row),
    the model has to INFER which column is which fiscal year purely
    from position -- across a table it didn't structure itself, that's
    exactly the kind of task a smaller model gets wrong, producing
    off-by-one column reads (e.g. reporting FY2025's requested figure
    as what is actually FY2024's number). Giving the model an
    already-merged, explicitly-dated header removes that inference
    step entirely instead of relying on the model to get it right.

    No-op (returns text unchanged) if the text doesn't look like a
    pipe-table chunk, or if the header isn't the ambiguous placeholder
    pattern this specifically targets.
    """
    if "|" not in text:
        return text
    prefix, table_lines = _split_prefix_and_table(text)
    if len(table_lines) < 3:
        return text
    header = _row_cells(table_lines[0])
    # table_lines[1] is normally the "---|---|..." Markdown separator row
    data_rows = [_row_cells(l) for l in table_lines[2:]]
    if not data_rows:
        return text

    new_header, new_rows = _merge_date_row_into_header(header, data_rows)
    new_header, new_rows = _drop_blank_columns(new_header, new_rows)
    if new_header == header and new_rows == data_rows:
        return text  # nothing changed, avoid needless rebuild

    rebuilt = "\n".join(prefix + _rows_to_pipe_lines(new_header, new_rows))
    return rebuilt


def _split_prefix_and_table(text: str) -> Tuple[List[str], List[str]]:
    """Split a chunk's Markdown text into its non-table prefix lines
    (heading, unit-of-measure line, ...) and its pipe-table lines."""
    prefix: List[str] = []
    table: List[str] = []
    for line in text.split("\n"):
        if line.strip().startswith("|"):
            table.append(line)
        elif not table:
            prefix.append(line)
        # A non-pipe line appearing AFTER the table has started is
        # trailing prose (rare) -- dropped rather than interleaved.
    return prefix, table


def _is_header_pair(lines: List[str], header_lines: List[str]) -> bool:
    """True if the first two lines of *lines* are the same header +
    separator row as *header_lines* -- used to avoid repeating the
    header in the middle of a stitched multi-page table."""
    if len(lines) < 2 or len(header_lines) < 2:
        return False
    return lines[0].strip() == header_lines[0].strip()


def _row_cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


_YEAR_PLACEHOLDER_RE = re.compile(r"^(year ended|for the year ended|as at)$", re.IGNORECASE)
_DATE_CELL_RE = re.compile(
    r"^\(?\d{1,2}[./]\d{1,2}[./]\d{2,4}\)?$"     # 31.3.2023 / 31/03/2023
    r"|^\(?(19|20)\d{2}\)?$"                       # bare 2023
    r"|^\d{1,2}(st|nd|rd|th)?\s+\w+\s+(19|20)\d{2}$",  # 31st March 2023
    re.IGNORECASE,
)


def _merge_date_row_into_header(header: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """
    Some source PDFs print a statement's column headers on TWO physical
    lines -- "Year ended" / "Year ended" on one line, the actual dates
    ("31.3.2023" / "31.3.2022") on the next -- and pdfplumber's
    text-based column detection can pull those in as two separate table
    rows instead of one merged header row. Left alone, this shows a
    header of literal, repeated "Year ended" placeholders with the real
    year never appearing anywhere in the rendered table.

    If the header is entirely made of that placeholder wording, and the
    very next row's cells all look like dates/years in the SAME
    columns, merge that row's actual values into the header (and drop
    it from the data rows) so the table shows the real year instead of
    a repeated generic label.
    """
    if not header or not rows:
        return header, rows
    if not all(_YEAR_PLACEHOLDER_RE.match(h.strip()) or not h.strip() for h in header):
        return header, rows

    first_row = rows[0]
    if len(first_row) != len(header):
        # A date row that isn't the same width as the header can't be
        # safely position-matched to it -- merging it in anyway would
        # shift every date into the wrong column instead of just leaving
        # the (already imperfect) "Year ended" placeholder in place.
        return header, rows
    date_like = [c for c in first_row if c.strip()]
    if not date_like or not all(_DATE_CELL_RE.match(c.strip()) for c in date_like):
        return header, rows

    new_header = [
        (first_row[i].strip() if i < len(first_row) and first_row[i].strip() else header[i])
        for i in range(len(header))
    ]
    return new_header, rows[1:]


def _drop_blank_columns(header: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """
    Drop any column that's blank across the header AND every row. This
    is the SAME fix ingestion/table.py's to_markdown() now applies for
    newly-ingested PDFs (pdfplumber routinely invents extra, entirely
    empty columns from faint/ambiguous ruling lines) -- but chunks that
    were already ingested BEFORE that fix still have the old,
    blank-column-riddled text baked into the cache, and reingesting was
    explicitly ruled out. Re-applying the same cleanup here, at answer
    time, on the already-cached text gets the same result without
    touching ingestion again.
    """
    num_cols = max([len(header)] + [len(r) for r in rows], default=0)
    if not num_cols:
        return header, rows

    def cell(row: List[str], i: int) -> str:
        return row[i] if i < len(row) else ""

    keep = [
        i for i in range(num_cols)
        if cell(header, i).strip() or any(cell(r, i).strip() for r in rows)
    ]
    new_header = [cell(header, i) for i in keep]
    new_rows = [[cell(r, i) for i in keep] for r in rows]
    return new_header, new_rows


def _rows_to_pipe_lines(header: List[str], rows: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return lines


def _rows_to_plain_block(header: List[str], rows: List[List[str]]) -> str:
    """
    Render as a compact, aligned plain-text block (inside a fenced code
    block by the caller) instead of a Markdown pipe table. Used when a
    table has too many columns or too high a blank-cell ratio to render
    as a clean table even after dropping fully-blank columns -- forcing
    it into a Markdown table at that point tends to either overflow a
    narrow terminal into a wrapped, garbled mess, or (if a renderer
    fails to parse it as valid GFM) fall back to showing the raw pipe
    characters as one run-on paragraph. Each row here is instead one
    guaranteed single line, with only its non-blank cells shown -- still
    unambiguous (each value stays paired with its column header), just
    not drawn as a bordered grid.
    """
    out_lines = []
    for r in rows:
        pairs = [f"{h}: {v}" for h, v in zip(header, r) if v.strip()]
        if pairs:
            out_lines.append("  |  ".join(pairs))
    return "\n".join(out_lines)


def _looks_cleanly_renderable(header: List[str], rows: List[List[str]]) -> bool:
    if not header or not rows:
        return False
    if len(header) > 10:
        return False
    total = sum(len(r) for r in rows)
    blank = sum(1 for r in rows for v in r if not v.strip())
    return total == 0 or (blank / total) <= 0.5


def _build_statement_table_data(question: str, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Core logic shared by try_statement_table_answer (Markdown string, for
    the API/engine/chat-history callers) and try_statement_table_structured
    (clean structured cells, for a real rich.table.Table in the terminal
    UI -- see main.py). Returns a dict with:
        title, prefix_text, footer            -- plain strings
        header_cells, data_rows                -- List[str] / List[List[str]]
                                                   (already stripped/clean)
        raw_text                               -- set INSTEAD of the above
                                                   when this particular
                                                   chunk never extracted as
                                                   a proper pipe table at
                                                   ingestion time (see the
                                                   raw_text_override path
                                                   in ingestion/table.py);
                                                   callers that can only
                                                   show real tables should
                                                   attempt to parse this
                                                   with _parse_loose_text_
                                                   into_rows rather than
                                                   showing it verbatim.

    Returns None under the exact same conditions the old function did
    (no statement detected in the question, or nothing matching was
    actually retrieved) -- this function makes no behavior change to
    WHAT gets matched/stitched, only how the result is handed back.

    A real statement is often split across MULTIPLE extracted chunks --
    a page break mid-table, or pdfplumber finding it as two adjacent
    table regions. Showing only the single best-scoring chunk (the
    original version of this function) meant the user only ever saw
    whichever page happened to extract with the most rows, silently
    missing the rest of the statement. This now groups every matching
    chunk from the SAME source document together, orders them by page
    number, and stitches their table rows into one combined table
    (de-duplicating a repeated header/separator row and any exact
    duplicate data row along the way) -- so the answer is the full
    statement, not a fragment of it.
    """
    canonical = _detect_requested_statement(question)
    if canonical is None:
        return None

    # Collect every candidate chunk, grouped by source document + company
    # + section (as a cheap extra guard against two different companies'
    # reports -- or two different tables in the same report -- both
    # being tagged with the same generic statement name).
    groups: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]] = {}
    for r in results:
        if r.get("category") != "document_pdf":
            continue
        record = r.get("record") or {}
        if not record.get("is_table"):
            continue
        text = record.get("text") or ""
        section = (r.get("section") or "").strip().lower()

        # Match on the chunk's OWN heading (the "### ..." line
        # ingestion actually detected for this table) or its `section`
        # tag -- both are set once, at ingestion, from the table's real
        # caption. A loose "does the canonical name appear anywhere in
        # the first N characters of the body" check used to also match
        # things like the Income-Tax note ("... recognised in the
        # statement of profit and loss ...") purely because it MENTIONS
        # the statement's name in a sentence, which is a different table
        # entirely -- that's what was pulling an unrelated tax
        # reconciliation footnote into the middle of the P&L answer.
        heading_line = next(
            (l.strip("# ").strip().lower() for l in text.split("\n") if l.strip().startswith("###")),
            "",
        )
        heading_matches = heading_line == canonical.lower() or heading_line.startswith(canonical.lower())
        section_matches = section == canonical.lower()
        if not heading_matches and not section_matches:
            continue

        key = (record.get("source_file"), record.get("company"), section or heading_line)
        groups.setdefault(key, []).append({**record, "_score": r.get("score", 0)})

    if not groups:
        return None

    # A single (source_file, company, section) group can span the ENTIRE
    # document -- a 400-page consolidated annual report routinely has
    # dozens of DIFFERENT tables (the parent's own statement, plus one
    # per subsidiary/segment note) all tagged with the same generic
    # section name (e.g. every one of them detected as "cash flow
    # statement"). Treating the whole group as one table means chunks
    # from totally unrelated pages -- 50, 100, 200 pages apart -- get
    # stitched together, and the "keep only rows matching the first
    # chunk's column count" rule then throws away nearly everything
    # except whichever handful of chunks happen to share that one
    # chunk's exact shape. This is what produced a "Cash Flow Statement"
    # answer citing pages 160-255 that was really just one lone section
    # header with no data rows: the real, contiguous statement table got
    # diluted across ~100 pages of unrelated same-tagged notes.
    #
    # Fix: split each group into clusters of chunks whose pages are
    # close together (a real multi-page statement is a handful of
    # CONSECUTIVE or near-consecutive pages, never a 50+ page gap), then
    # treat each cluster as its own candidate table. This keeps the
    # existing "pick whichever has the most real rows" behavior, just
    # scoped to a single genuine statement instead of the whole document.
    _MAX_PAGE_GAP = 3

    def _cluster_by_page(chunk_list: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        ordered = sorted(chunk_list, key=lambda c: (c.get("page") or 0))
        clusters: List[List[Dict[str, Any]]] = []
        for c in ordered:
            page = c.get("page") or 0
            if clusters and (page - (clusters[-1][-1].get("page") or 0)) <= _MAX_PAGE_GAP:
                clusters[-1].append(c)
            else:
                clusters.append([c])
        return clusters

    def _row_count(chunk_list: List[Dict[str, Any]]) -> int:
        # A chunk that never extracted as a proper pipe table (the
        # raw_text_override path in ingestion/table.py) has ZERO "\n|"
        # occurrences no matter how much real data it contains -- which
        # previously meant a cluster made entirely of such chunks always
        # lost to any other cluster with even one tiny, near-empty pipe
        # table, discarding a large genuine table in favor of a small
        # irrelevant one. Falling back to counting real content lines
        # for a pipe-less chunk means it's judged on how much data it
        # actually has, not on which extraction path it happened to take.
        total = 0
        for c in chunk_list:
            text = c.get("text") or ""
            pipe_rows = text.count("\n|")
            if pipe_rows:
                total += pipe_rows
            else:
                total += sum(
                    1 for line in text.split("\n")
                    if line.strip() and not line.strip().startswith("#")
                    and not line.strip().startswith("(")
                )
        return total

    all_clusters: List[List[Dict[str, Any]]] = []
    for group_chunks in groups.values():
        all_clusters.extend(_cluster_by_page(group_chunks))

    if not all_clusters:
        return None

    chunks = max(all_clusters, key=_row_count)

    # De-duplicate: the same physical chunk can occasionally show up
    # twice in `results` (e.g. via the JSON/PDF interleaving in
    # core/engine.py._retrieve). Keep the first occurrence only.
    seen_pages_text = set()
    deduped = []
    for c in chunks:
        sig = (c.get("page"), (c.get("text") or "")[:120])
        if sig in seen_pages_text:
            continue
        seen_pages_text.add(sig)
        deduped.append(c)
    chunks = deduped

    # Order the winning cluster so a chunk that DID extract as a proper
    # pipe table comes first, ahead of any raw-text-fallback chunk that
    # merely happens to be an earlier page. Previously only chunks[0]
    # (by page order) was ever checked for a pipe table -- if THAT
    # specific chunk happened to be the rare raw-text fallback, the
    # entire cluster was rendered as a plain-text block even when a
    # later chunk in the very same cluster had a perfectly good,
    # structured table. This is exactly what could turn a real,
    # complete statement into an apparently "empty"/sparse answer: the
    # good data was sitting right there, one chunk later, and never
    # looked at.
    def _has_pipe_table(c: Dict[str, Any]) -> bool:
        _, tlines = _split_prefix_and_table(c.get("text") or "")
        return bool(tlines)

    pipe_chunks = [c for c in chunks if _has_pipe_table(c)]
    if pipe_chunks:
        chunks = pipe_chunks + [c for c in chunks if c not in pipe_chunks]

    # Stitch: first chunk contributes its prefix (heading/unit line) and
    # its full table (header + separator + rows). Every subsequent
    # chunk contributes only its rows -- its own header/separator row is
    # dropped if it repeats the first chunk's, and any row that's an
    # exact duplicate of one already included is skipped.
    prefix_lines, first_table_lines = _split_prefix_and_table(chunks[0].get("text") or "")
    if not first_table_lines:
        # NONE of the chunks in this cluster ever extracted as a proper
        # pipe table (the check above already promoted any chunk that
        # did to the front) -- this particular statement genuinely only
        # exists as plain reading-order text at ingestion time (see
        # ingestion/table.py's `raw_text_override` path). Rendering
        # that as a Markdown TABLE isn't possible without re-extracting
        # it (a reingest) -- but rendering it as-is also isn't safe:
        # plain text with single newlines collapses into one run-on
        # paragraph under Markdown's line-break rules, which is exactly
        # the "one continuous block of text" symptom. Wrapping it in a
        # fenced code block preserves every line break and the original
        # column-ish spacing exactly as extracted, without needing to
        # touch ingestion at all.
        raw_text = (chunks[0].get("text") or "").strip()
        label_bits = []
        if chunks[0].get("company"):
            label_bits.append(str(chunks[0]["company"]))
        if chunks[0].get("year"):
            label_bits.append(str(chunks[0]["year"]))
        title = " — ".join(label_bits + [canonical]) if label_bits else canonical
        page = chunks[0].get("page")
        source = chunks[0].get("document_title") or chunks[0].get("source_file") or "the ingested document"
        footer = f"Source: {source}" + (f", page {page}" if page else "")
        return {"title": title, "prefix_text": "", "raw_text": raw_text, "footer": footer}

    header_cells = _row_cells(first_table_lines[0])
    header_len = len(header_cells)

    def _fit_row(cells: List[str], n: int) -> List[str]:
        # Pad a short row, or truncate a long one, to the header's
        # column count -- rather than dropping the row outright the way
        # a strict "exact column-count match" check used to. Different
        # PAGES of the same physical statement routinely extract with a
        # slightly different column count (one extra/missing spurious
        # blank column on one page but not another), and dropping every
        # row that didn't exactly match the FIRST page's shape is
        # exactly what could reduce a genuine multi-page statement down
        # to almost nothing -- in the worst case, an apparently "empty"
        # table. A padded/truncated row can still be imperfect, but a
        # present, mostly-correct row is always better than a silently
        # missing one, and _drop_blank_columns() below cleans up any
        # resulting sparse columns afterward.
        if len(cells) > n:
            # A too-long row is very often just ONE spurious blank
            # leading column (a stray ruling-line artifact) shifting
            # every real value one slot to the right -- stripping
            # leading blanks first re-aligns the row instead of just
            # chopping its LAST (often a real number) cell off.
            while len(cells) > n and cells and not cells[0].strip():
                cells = cells[1:]
        if len(cells) < n:
            return cells + [""] * (n - len(cells))
        return cells[:n]

    data_rows: List[List[str]] = []
    seen_rows = set()
    for line in first_table_lines[2:]:
        cells = _fit_row(_row_cells(line), header_len)
        sig = tuple(cells)
        if sig in seen_rows:
            continue
        seen_rows.add(sig)
        data_rows.append(cells)

    for chunk in chunks[1:]:
        _, table_lines = _split_prefix_and_table(chunk.get("text") or "")
        if not table_lines:
            continue
        if _is_header_pair(table_lines, first_table_lines):
            table_lines = table_lines[2:]
        for line in table_lines:
            cells = _fit_row(_row_cells(line), header_len)
            if not any(c.strip() for c in cells):
                continue  # a fully-blank fitted row adds nothing
            sig = tuple(cells)
            if sig in seen_rows:
                continue
            seen_rows.add(sig)
            data_rows.append(cells)

    if not data_rows:
        # Nothing usable came out of stitching at all -- rather than
        # returning a table with a header and zero rows (the exact
        # "empty markdown box" symptom), escalate to the LLM path
        # instead, same as if no statement had matched in the first
        # place. A missing answer that can still be tried another way
        # is better than a confidently-empty one.
        return None

    header_cells, data_rows = _merge_date_row_into_header(header_cells, data_rows)
    header_cells, data_rows = _drop_blank_columns(header_cells, data_rows)
    if not header_cells or not data_rows:
        return None

    def _clean_year_label(value: Any) -> str:
        # A handful of older-ingested chunks have a garbled multi-year
        # string in their `year` metadata field (e.g.
        # "2024/2025/2023/2020/2021") from an early, looser year-detection
        # pass, rather than the single year that table actually covers.
        # Showing that verbatim in the title is confusing/wrong-looking,
        # so only the first (most specific) year is used for display.
        s = str(value)
        return s.split("/")[0].strip() if "/" in s else s

    label_bits = []
    if chunks[0].get("company"):
        label_bits.append(str(chunks[0]["company"]))
    if chunks[0].get("year"):
        label_bits.append(_clean_year_label(chunks[0]["year"]))
    elif chunks[0].get("years_covered"):
        label_bits.append("/".join(str(y) for y in chunks[0]["years_covered"]))
    title = " — ".join(label_bits + [canonical]) if label_bits else canonical

    pages = sorted({c.get("page") for c in chunks if c.get("page")})
    source = chunks[0].get("document_title") or chunks[0].get("source_file") or "the ingested document"
    footer_bits = [f"Source: {source}"]
    if pages:
        footer_bits.append(
            f"page {pages[0]}" if len(pages) == 1
            else f"pages {pages[0]}\u2013{pages[-1]}"
        )
    footer = ", ".join(footer_bits)

    prefix_text = "\n".join(prefix_lines).strip()
    return {
        "title": title, "prefix_text": prefix_text, "footer": footer,
        "header_cells": header_cells, "data_rows": data_rows,
    }


def try_statement_table_structured(question: str, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Same matching/stitching as try_statement_table_answer, but returns
    the clean structured pieces (title/prefix_text/footer plus either
    header_cells+data_rows, or raw_text for the rare chunk that never
    extracted as a proper pipe table) instead of a pre-rendered Markdown
    string.

    This exists for the terminal UI (main.py), which builds a real
    rich.table.Table directly from header_cells/data_rows -- every cell
    is already individually .strip()'d by _row_cells() upstream, so the
    resulting table has no stray/uneven whitespace and rich handles all
    column-width sizing itself, rather than depending on whatever a
    Markdown-table renderer happens to do with hand-formatted pipe
    strings (which was producing inconsistent results: some statements
    rendering as a real pipe table, others -- whichever chunk fell back
    to the plain-text path at ingestion -- rendering as a fenced code
    block instead, with neither one guaranteed to look clean in every
    terminal width).
    """
    return _build_statement_table_data(question, results)


def try_statement_table_answer(question: str, results: List[Dict[str, Any]]) -> Optional[str]:
    """
    Same as try_statement_table_structured, but pre-rendered into a
    single Markdown string -- kept for the API/engine/chat-history
    callers that need a plain string rather than structured cells.
    """
    data = _build_statement_table_data(question, results)
    if data is None:
        return None

    if "raw_text" in data:
        # This particular chunk's table never came out as a proper pipe
        # table at ingestion (e.g. label-column extraction failed, so
        # ingestion fell back to plain reading-order text -- see
        # ingestion/table.py's `raw_text_override` path). Rendering that
        # as a Markdown TABLE isn't possible without re-extracting it
        # (a reingest) -- but rendering it as-is also isn't safe: plain
        # text with single newlines collapses into one run-on paragraph
        # under Markdown's line-break rules. Wrapping it in a fenced
        # code block preserves every line break and the original
        # column-ish spacing exactly as extracted.
        return f"### {data['title']}\n\n```\n{data['raw_text']}\n```\n\n*{data['footer']}*"

    header_cells, data_rows = data["header_cells"], data["data_rows"]
    body_table = "\n".join(_rows_to_pipe_lines(header_cells, data_rows))

    body = "\n\n".join(p for p in (data["prefix_text"], body_table) if p)
    return f"### {data['title']}\n\n{body}\n\n*{data['footer']}*"