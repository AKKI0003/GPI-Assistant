"""
retrieval/pdf_retriever.py

The "PDF Retriever" half of the hybrid architecture:

    User Question
        |
    Intent Detection
        |
    JSON Retriever  (retrieval/knowledge_index.py -- UNCHANGED)
        |                                       \
        |                                        >-- Merge Context --> LLM
    PDF Retriever   (this file)                 /

This module is completely independent of `retrieval/knowledge_index.py`:
it has its own document source (`core.config.PDF_KNOWLEDGE_RAW_DIR`), its
own ingestion pipeline (the `ingestion/` package's `PDFIngestor`, which
preserves section/table/image structure instead of flattening a PDF into
raw text), and its own on-disk cache. Nothing here reads or writes the
JSON knowledge base.

Integration point: `core/engine.py`'s `AssistantEngine._retrieve()` calls
`search()` here and appends the results to the JSON retriever's results
list before context/prompt building. Because each result dict returned
by `search()` uses the exact same shape KnowledgeIndex results already
use (`{"category", "section", "record", "score"}`), `retrieval/
context_builder.py` and `llm/prompt_builder.py` render PDF-derived
context with ZERO changes -- both files stay completely untouched.

Design
------
- PDFs dropped into `PDF_KNOWLEDGE_RAW_DIR` are picked up automatically.
  Each file's SHA-256 checksum (already computed by the ingestion
  package's `MetadataExtractor`) is used to detect new/changed files, so
  an unchanged PDF is never re-ingested or re-embedded across restarts.
- Chunk embeddings reuse `retrieval.embedding_store.embed()` -- the same
  Ollama embedding model and on-disk cache the JSON retriever already
  uses -- rather than introducing a second embedding backend
  (sentence-transformers/faiss are available as *optional* extras inside
  `ingestion.utils` for a caller that wants them, but this retriever
  deliberately reuses the model already running for JSON retrieval so
  there is only ever one embedding backend to keep warm/consistent).
- If the embedding backend is unreachable, search degrades to a simple
  lexical (token-overlap) fallback rather than returning nothing --
  matching the JSON retriever's own graceful-degradation philosophy.
- The index is built lazily on first use and rebuilt incrementally
  (only new/changed files) on every subsequent call, so a long-running
  process picks up newly dropped-in PDFs without needing a restart.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.config import (
    PDF_KNOWLEDGE_RAW_DIR,
    PDF_KNOWLEDGE_IMAGES_DIR,
    PDF_KNOWLEDGE_CHROMA_DIR,
    PDF_KNOWLEDGE_TOP_K,
    PDF_KNOWLEDGE_MIN_SIMILARITY,
    RERANKER_ENABLED,
    RERANKER_TOP_N,
    RERANKER_WEIGHT,
)
from retrieval.embedding_store import embed, embed_many, cosine_similarity
from retrieval.chroma_store import ChromaStore, _CHROMA_AVAILABLE
from retrieval.reranker import rerank as _rerank_candidates

logger = logging.getLogger(__name__)

# The ingestion package is a heavy-ish optional dependency (PyMuPDF /
# pdfplumber / camelot). Import defensively so that a machine which
# hasn't installed those yet still runs the rest of the assistant
# (JSON retrieval + LLM) with the PDF path simply reporting empty
# results and a clear one-time log message, rather than crashing on
# import at process startup.
try:
    from ingestion import PDFIngestor
    from ingestion.utils import file_checksum
    from ingestion.document import INGESTION_LOGIC_VERSION
    _INGESTION_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - environment-dependent
    _INGESTION_AVAILABLE = False
    _INGESTION_IMPORT_ERROR = exc
    INGESTION_LOGIC_VERSION = 0


@dataclass
class _CachedChunk:
    chunk_dict: Dict[str, Any]
    embedding: Optional[List[float]]


# How many chunks' worth of embedding work happens between progress log
# lines during ingestion. Independent of embedding_store.py's internal
# HTTP batch size (_EMBED_BATCH_SIZE) -- this only controls log
# granularity, so progress is still visible every ~500 chunks even
# though the actual network calls underneath now cover 64 chunks each.
_EMBED_LOG_BATCH = 500


# ─── Financial vs. ESG/CSR topic awareness ────────────────────────────────────
#
# Annual reports devote far more PAGES to ESG/CSR/sustainability content
# than to the handful of pages that actually carry Revenue/Profit/Balance
# Sheet figures. Because those ESG pages are also full of prose that
# semantically "sounds financial" (money, targets, percentages), plain
# cosine similarity regularly ranks them ahead of the actual financial-
# statement chunks for a query like "GPI financials for past 3 years" or
# "VST financials" -- which is exactly why real numbers were missing from
# GPI's answer and VST's answer surfaced nothing but emissions/production
# figures. These two keyword sets let `search()` recognize (a) that the
# *query* is asking for hard financial figures, and (b) whether a given
# *chunk* actually looks like a financial-statement chunk or an ESG/CSR
# chunk, so the ranking can be nudged accordingly -- without touching
# chunk size/ingestion at all.
_FINANCIAL_QUERY_KEYWORDS = (
    "financial", "financials", "revenue", "profit", "loss", "income",
    "balance sheet", "cash flow", "turnover", "sales", "ebitda",
    "net worth", "expenditure", "expense", "margin", "earnings",
    "profit before tax", "profit after tax", "pbt", "pat", "gsv",
)

_FINANCIAL_SECTION_KEYWORDS = (
    "financial statements", "financial highlights", "performance overview",
    "revenue", "profit", "profit and loss", "p&l", "cash flow",
    "balance sheet", "notes to accounts", "auditor", "income statement",
    "turnover", "highlights",
)

# Headline-metric words that show up on a report's "FY Highlights" page
# (revenue/net-profit/EPS/EBIT-margin/RoCE callouts) even when that page
# is laid out as icons/pull-quotes rather than a clean prose table, so
# pdfplumber's table extraction may miss it entirely and it only survives
# as loose running text. Boost on these keywords lets that page still
# outrank ESG/CSR content for a financial query even without a table.
_FINANCIAL_HEADLINE_KEYWORDS = (
    "net profit", "revenue from operations", "earnings per share", "eps",
    "ebit margin", "roce", "dividend per share", "net assets employed",
    "total income", "profit before tax", "profit after tax",
)

_ESG_SECTION_KEYWORDS = (
    "esg", "environmental social governance", "csr",
    "corporate social responsibility", "sustainability",
    "greenhouse gas", "emissions", "human resources", "awards",
)

# A "strategy"/"strategies"/"policy"/"policies" question, asked WITHOUT
# any ESG/sustainability/CSR wording of its own, means business/sales/
# operations/financial strategy -- but annual reports devote so much
# prose to "our sustainability strategy" / "ESG strategy" / "climate
# strategy" that plain semantic similarity regularly surfaces nothing
# but ESG/CSR content for a bare "strategy" query, even though the
# user never asked about sustainability at all. This mirrors the
# existing financial-vs-ESG mechanism above: down-weight ESG/CSR
# sections and up-weight business/operations sections, but ONLY for a
# strategy-flavoured query that didn't itself ask about ESG/CSR/
# sustainability -- a query that DOES name those explicitly is left
# completely untouched, same as any other ESG/CSR question.
_STRATEGY_QUERY_WORDS = ("strategy", "strategies", "policy", "policies")
_ESG_QUERY_WORDS = (
    "esg", "sustainability", "csr", "corporate social responsibility",
    "environment", "environmental", "governance", "emission", "carbon",
)
_BUSINESS_STRATEGY_SECTION_KEYWORDS = (
    "management discussion and analysis", "md&a", "business overview",
    "business review", "operations review", "growth strategy",
    "market outlook", "strategic priorities", "our strategy",
    "sales", "distribution", "marketing", "operations", "financial review",
    "business model", "competitive advantage", "market position",
    "key strengths", "risk management", "governance", "board report",
    "director", "chairman", "ceo message", "md message", "managing director",
    "outlook", "future prospects", "expansion", "investment", "capex",
    "capacity", "new product", "innovation", "r&d", "research",
    "technology", "digital", "transformation", "cost optimization",
    "margin improvement", "pricing", "volume growth", "market share",
    "segment", "brand", "portfolio", "consumer", "customer", "channel",
    "export", "international", "subsidiary", "joint venture", "acquisition",
)


def _is_business_strategy_query(query: str) -> bool:
    q = query.lower()
    if not any(w in q for w in _STRATEGY_QUERY_WORDS):
        return False
    return not any(w in q for w in _ESG_QUERY_WORDS)


def _strategy_topic_adjustment(query_is_business_strategy: bool, chunk_dict: Dict[str, Any]) -> float:
    """Score delta for a business-strategy-flavoured query: nudges
    business/operations/sales strategy content up and ESG/CSR content
    down. A no-op (0.0) for any query that isn't asking about strategy,
    or that already named ESG/sustainability/CSR itself."""
    if not query_is_business_strategy:
        return 0.0

    section = (chunk_dict.get("section") or "").lower()
    subsection = (chunk_dict.get("subsection") or "").lower()
    combined_section = f"{section} {subsection}"

    adjustment = 0.0
    if any(kw in combined_section for kw in _BUSINESS_STRATEGY_SECTION_KEYWORDS):
        adjustment += 0.25
    if any(kw in combined_section for kw in _ESG_SECTION_KEYWORDS):
        adjustment -= 0.35
    return adjustment

# CSR-SPEND ANNEXURE boilerplate. India's Companies Act mandates a
# standard "Annexure to the Directors' Report" disclosing CSR spend --
# these tables are financial-shaped (rupee figures, "financial year",
# multi-year columns) and were getting mistaken for real P&L/balance-
# sheet content (see the "Financial Statements" default-label bug fixed
# in ingestion/document.py). Even after that fix, section detection can
# still mislabel a page, so this is a second, content-based line of
# defense: if a chunk's own text is clearly this CSR-spend boilerplate,
# it is penalized regardless of what section it got tagged with.
_CSR_ANNEXURE_KEYWORDS = (
    "average net profit", "prescribed csr expenditure", "amount spent",
    "unspent csr account", "due date of payment", "csr expenditure",
    "ongoing project", "csr committee",
)

# Employee-benefit actuarial disclosure notes (Ind AS 19: Gratuity and
# Pension obligation schedules) are a THIRD kind of note that reads as
# financial-shaped -- dense with rupee figures across multiple years --
# but answers a completely different question than "what was PAT/
# revenue/profit". Confirmed: a "VST's PAT for the past 3 years"
# question returned the "Present Value of Defined Benefit Obligation"
# figures for gratuity and pension instead, with years misattributed
# across the different sub-tables (gratuity vs. pension each have their
# own opening/closing DBO figures per year, and stitching them together
# under one "3 years" umbrella scrambled which figure belongs to which
# year). These notes should never win over a real P&L table for a PAT/
# profit/revenue question.
_ACTUARIAL_NOTE_KEYWORDS = (
    "defined benefit obligation", "present value of obligation",
    "present value of defined benefit", "actuarial valuation",
    "actuarial (gain)", "actuarial gain", "actuarial loss",
    "discount rate", "attrition rate", "gratuity", "pension",
)

# A condensed multi-year "Operating Results" / "N-Year Summary" table
# (e.g. VST's own report includes one titled "Operating Results 2015 -
# 2024", cramming 10 years of Revenue/PBT/PAT into ONE row each) is a
# lower-fidelity DUPLICATE of the real Statement of Profit and Loss --
# same figures, but confirmed to be a much more error-prone extraction
# target (10 year-columns crammed into a single row is far easier for
# table detection to misalign than the real statement's clean 1-2
# column comparative format). Confirmed directly: the real PDF text for
# this exact table has correct year alignment (2023 -> 32698,
# 2024 -> 30157), but a "VST PAT for 2023/2024/2025" answer showed
# those same two figures shifted forward one year AND a fabricated
# 2025 value that doesn't exist in this table at all. Since the real
# Statement of Profit and Loss table already reliably has the same
# data, correctly aligned, this summary table is penalized rather than
# excluded outright -- it's still useful supporting context, just never
# the preferred source for a specific year's PAT/PBT/revenue figure.
_MULTI_YEAR_SUMMARY_KEYWORDS = (
    "operating results", "n-year summary", "5 year summary",
    "10 year summary", "financial summary", "year summary",
    "years financial highlights",
)

# A chunk that actually carries real financial figures tends to be dense
# with currency/number markers (₹, Rs., "crore", "lakh", "%", multi-digit
# figures) -- this is a cheap, ingestion-agnostic signal that a chunk is
# a genuine financial-figure chunk even when section detection mis-tags
# it (e.g. a table embedded inside a general "Corporate Overview"
# section rather than a cleanly-headed "Financial Statements" section).
_CURRENCY_NUMBER_PATTERN = re.compile(
    r"(₹|rs\.?\s?\d|crore|lakh|\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+\.\d+%)",
    re.IGNORECASE,
)


def _is_financial_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _FINANCIAL_QUERY_KEYWORDS)


def _is_analysis_query(query: str) -> bool:
    """Detect queries asking for deep analysis of strategy, business model,
    competitive position, policies, etc. -- these need WIDE retrieval to
    capture intricate details across many sections of the annual report."""
    q = query.lower()
    analysis_signals = (
        "analyze", "analysis", "deep dive", "detailed analysis",
        "strategy", "strategies", "policy", "policies",
        "business model", "competitive", "competition",
        "market position", "strengths", "weaknesses", "swot",
        "outlook", "future", "prospects", "growth drivers",
        "key risks", "risk factors", "challenges", "opportunities",
        "investment thesis", "value creation", "moat",
        "management discussion", "md&a", "director report",
        "chairman", "ceo", "leadership", "governance",
        "capital allocation", "capex", "r&d", "innovation",
        "digital transformation", "cost structure", "margin",
        "pricing power", "volume", "market share", "segment",
    )
    return any(sig in q for sig in analysis_signals)


# Maps a query's own wording to the canonical statement label that
# ingestion/document.py now attaches to each table chunk's `section`
# field (see the detected_heading fix there). Checked in this specific
# order because "cash flow" must win over the generic "profit"/"loss"
# checks when a query says something like "cash flow and profit
# statement" -- unlikely in practice, but keeps this deterministic.
_STATEMENT_QUERY_PATTERNS = (
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
    # PAT ("profit after tax") is a LINE ITEM inside the P&L, not a
    # statement name of its own -- but a query asking for it (e.g.
    # "VST's PAT for 3 years") means exactly the same table as an
    # explicit P&L request. Without this, a PAT-only query got no
    # statement-specific boost at all, so a generic "financial
    # highlights" page (which also mentions a PAT-like figure, but not
    # necessarily the right one for that year) could outrank the actual
    # P&L table for a given document -- which is exactly how a
    # multi-year PAT comparison ended up quietly wrong for one year
    # while correct for the others.
    ("profit after tax", "Statement of Profit and Loss"),
    ("pat", "Statement of Profit and Loss"),
)


def _detect_requested_statement(query: str) -> Optional[str]:
    """If the query is asking for ONE specific financial statement (as
    opposed to a generic "financials"/"revenue" question that's happy to
    see any of them), return its canonical label so search() can boost
    exactly that statement's chunks. Returns None for generic financial
    questions, which are left to the existing broad financial-vs-ESG
    boost untouched."""
    q = query.lower()
    for keyword, canonical_label in _STATEMENT_QUERY_PATTERNS:
        if len(keyword) <= 4:
            # Short keywords ("pat", "p&l") need a word boundary, or
            # they false-positive inside unrelated words -- "pat" alone
            # would otherwise match inside "compatibility", "expatriate",
            # "patent", etc.
            if re.search(rf"\b{re.escape(keyword)}\b", q):
                return canonical_label
        elif keyword in q:
            return canonical_label
    return None


def _topic_adjustment(query: str, query_is_financial: bool, chunk_dict: Dict[str, Any]) -> float:
    """Returns a score delta (added to the base similarity/lexical score)
    that nudges genuinely financial chunks up, and ESG/CSR chunks down,
    ONLY when the query itself is asking for financial figures. For any
    non-financial query this always returns 0.0 -- ESG/CSR content is
    never penalized when someone actually asks about ESG/CSR."""
    if not query_is_financial:
        return 0.0

    section = (chunk_dict.get("section") or "").lower()
    subsection = (chunk_dict.get("subsection") or "").lower()
    combined_section = f"{section} {subsection}"
    text = chunk_dict.get("text") or ""
    text_lower = text.lower()

    adjustment = 0.0

    if any(kw in combined_section for kw in _FINANCIAL_SECTION_KEYWORDS):
        adjustment += 0.18
    if any(kw in text_lower for kw in _FINANCIAL_HEADLINE_KEYWORDS):
        adjustment += 0.12
    if _CURRENCY_NUMBER_PATTERN.search(text):
        adjustment += 0.10
    if chunk_dict.get("is_table"):
        # Table chunks already get +0.08 for being tables; a financial
        # query asking for numbers benefits from an extra nudge on top
        # of that so a real figures-table clearly wins over ESG prose.
        adjustment += 0.06

    if any(kw in combined_section for kw in _ESG_SECTION_KEYWORDS):
        adjustment -= 0.20

    # Statement-name narration vs. the actual statement table -- a chunk
    # can mention "cash flow statement"/"balance sheet" purely as prose
    # (accounting-policy notes, MD&A commentary) and still be a near-
    # perfect semantic match for a query asking for that statement, even
    # though it carries NO real figures. Left unchecked, this routinely
    # outranks the genuine numeric table (whose own section label is
    # often just "Table" post-extraction, or -- worse -- inherits a
    # stale/garbage section string from ingestion; see ingestion/
    # section.py). A chunk that talks ABOUT a financial statement but is
    # sparse in actual currency/number tokens is demoted so the real
    # table wins; a table chunk that's also dense in numbers is boosted
    # further on top of the existing is_table bonus.
    if any(kw in text_lower for kw in _FINANCIAL_SECTION_KEYWORDS):
        number_hits = len(_CURRENCY_NUMBER_PATTERN.findall(text))
        if chunk_dict.get("is_table") and number_hits >= 3:
            adjustment += 0.15
        elif number_hits < 2:
            adjustment -= 0.15

    if chunk_dict.get("is_table") and _CURRENCY_NUMBER_PATTERN.search(text):
        # Independent of section-keyword matching (which is unreliable
        # for these chunks -- see above), a table cell-dense in currency
        # figures is a strong standalone signal it IS a statement table.
        adjustment += 0.05

    # Content-based CSR-spend-annexure penalty -- applies regardless of
    # how the chunk got labeled, since this boilerplate is easy to
    # mistake for real financial-statement content by section title
    # alone (see module docstring above).
    csr_hits = sum(1 for kw in _CSR_ANNEXURE_KEYWORDS if kw in text_lower)
    if csr_hits >= 2:
        adjustment -= 0.30

    # Same reasoning for actuarial/employee-benefit obligation notes
    # (Gratuity, Pension DBO schedules) -- see _ACTUARIAL_NOTE_KEYWORDS'
    # comment. Only applied when the QUERY itself is financial (PAT,
    # revenue, profit, etc.) -- a genuine question actually about
    # gratuity/pension is a different, non-financial-keyword query and
    # is completely unaffected by this penalty.
    actuarial_hits = sum(1 for kw in _ACTUARIAL_NOTE_KEYWORDS if kw in text_lower)
    if actuarial_hits >= 2:
        adjustment -= 0.30

    # A "does this chunk look like a condensed multi-year summary
    # table" penalty. Checked against the chunk's own TEXT (not the
    # section label): table.caption is never populated by either
    # extraction backend in this pipeline (neither looks at text
    # surrounding a table's own bounding box), so `section` is always a
    # generic/canonical label here, never a table's own page heading --
    # but that heading text ("OPERATING RESULTS NOTES / Operating
    # Results 2015 - 2024") does appear at the top of the table's own
    # extracted body text, since its bounding box starts right where
    # that heading line sits on the page.
    if any(kw in text_lower for kw in _MULTI_YEAR_SUMMARY_KEYWORDS):
        adjustment -= 0.25

# P&L vs Balance Sheet adjustment for financial metric queries.
    # When the query asks for PAT/PBT/Revenue/EBITDA, the actual
    # Statement of Profit and Loss (P&L) contains the authoritative
    # figures, while Balance Sheet sections often contain ratio analysis
    # (e.g., "Net profit ratio") that mention the metric name with a
    # completely different value. Boost P&L sections and penalize
    # Balance Sheet sections for metric queries.
    metric_query = any(kw in query for kw in (
        "pat", "profit after tax", "profit for the year", "net profit",
        "pbt", "profit before tax",
        "revenue", "turnover", "sales",
        "ebitda", "ebit", "eps", "earnings per share",
    ))
    if metric_query:
        # PAT facts are in the actual P&L table chunks, which are labeled
        # as "Financial Statements" (the table extraction label), not
        # "Statement of Profit and Loss" (which are narrative tax/PBT notes).
        if "financial statements" in section:
            adjustment += 0.35
        elif "statement of profit and loss" in section:
            adjustment += 0.25
        elif "profit and loss" in section and "statement" in section:
            adjustment += 0.20
        if "balance sheet" in section:
            adjustment -= 0.25

        # Additional boost for chunks that actually have PAT/PBT/Revenue facts
        # when the query asks for those specific metrics.
        query_lower = query.lower()
        facts = chunk_dict.get("financial_facts") or []
        if "pat" in query_lower or "profit after tax" in query_lower or "profit for the year" in query_lower or "net profit" in query_lower:
            if any(f.get("metric") == "PAT" for f in facts):
                adjustment += 0.50
        if "pbt" in query_lower or "profit before tax" in query_lower:
            if any(f.get("metric") == "PBT" for f in facts):
                adjustment += 0.50
        if "revenue" in query_lower or "turnover" in query_lower or "sales" in query_lower:
            if any(f.get("metric") == "Revenue" for f in facts):
                adjustment += 0.50

    return adjustment


class PDFRetriever:
    """Independent PDF knowledge source. One process-wide instance
    (`pdf_retriever`, below) is shared across requests, matching how
    `retrieval.knowledge_index.index` is a module-level singleton."""

    def __init__(
        self,
        raw_dir: str = PDF_KNOWLEDGE_RAW_DIR,
        images_dir: str = PDF_KNOWLEDGE_IMAGES_DIR,
        chroma_dir: str = PDF_KNOWLEDGE_CHROMA_DIR,
        top_k: int = PDF_KNOWLEDGE_TOP_K,
        min_similarity: float = PDF_KNOWLEDGE_MIN_SIMILARITY,
        caption_images: bool = False,
    ) -> None:
        self.raw_dir = raw_dir
        self.images_dir = images_dir
        self.chroma_dir = chroma_dir
        # Kept as an alias for backward compatibility with anything
        # (e.g. import_pdf_documents.py) that prints/reads `.cache_file`
        # as a human-facing "where is this persisted" path.
        self.cache_file = chroma_dir
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.caption_images = caption_images

        self._lock = threading.Lock()
        self._chunks: List[_CachedChunk] = []
        self._known_checksums: Dict[str, str] = {}  # file_path -> checksum
        self._index_built = False
        self._ingestor = None  # constructed lazily, only if ingestion is available
        self._store: Optional[ChromaStore] = None  # constructed lazily, only if chromadb is available

    # -- public API ---------------------------------------------------------

    # Maps the JSON retriever's canonical short company tags (see
    # retrieval.knowledge_index.KnowledgeIndex._COMPANY_ALIASES /
    # detect_companies_in_question) to the substrings that actually show
    # up in a PDF chunk's `company` field, e.g. "ITC Limited",
    # "VST Industries", "GPI". These are DIFFERENT strings from the
    # canonical "ITC"/"VST"/"GPI" tags, so a naive `==` check between the
    # two would never match anything.
    _COMPANY_TAG_SUBSTRINGS: Dict[str, List[str]] = {
        "GPI": ["gpi", "godfrey phillips"],
        "ITC": ["itc"],
        "VST": ["vst"],
    }

    @classmethod
    def _chunk_matches_company(cls, chunk_company: Optional[str], company_filter: str) -> bool:
        if not chunk_company:
            return False
        chunk_lower = chunk_company.lower()
        substrings = cls._COMPANY_TAG_SUBSTRINGS.get(company_filter, [company_filter.lower()])
        return any(s in chunk_lower for s in substrings)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        broad: bool = False,
        company_filter: Optional[str] = None,
        year_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the PDF knowledge base and return results in the SAME
        dict shape KnowledgeIndex.search() returns, so callers can
        simply concatenate the two result lists. Never raises: any
        failure degrades to an empty result list plus a logged warning,
        so a PDF-ingestion problem never breaks JSON-based answering.

        Parameters
        ----------
        top_k : override for the number of chunks returned. Ignored when
                `broad` is True (see below).
        broad : True for aggregate-style questions ("summarize 3 years
                of VST", "compare VST across years", "all VST reports").
                A flat top-k ranked search has no concept of "give me
                roughly equal coverage of each source document" -- if
                three years of VST reports are ingested, whichever single
                file's chunks happen to score highest semantically can
                crowd out the other two entirely, so "summarize 3 years"
                would silently only ever see one year's data. When broad
                is True, this pulls a much larger candidate pool and then
                round-robins across distinct source files (see
                `_round_robin_by_source`) so every ingested document
                relevant to the query gets a fair share of the result
                budget, mirroring how the JSON retriever already handles
                multi-company comparisons in core/engine.py.
        company_filter : restrict results to chunks tagged with one
                company ("GPI", "ITC", or "VST" -- the same canonical
                tags `detect_companies_in_question` returns). Without
                this, a plain "GPI finances" question had NO company
                scoping at all on the PDF side (unlike the JSON side,
                which always scopes to `implied_company`) -- so a broad
                query's round-robin-by-source step gave ITC's and VST's
                annual-report chunks equal weight alongside GPI's own,
                diluting the very GPI figures the question asked for.
                Pass None for genuinely multi-company questions
                (comparisons), where every company's chunks are wanted.
        year_filter : restrict document chunks to one detected fiscal
                year, mirroring KnowledgeIndex.search()'s year_filter.
                Chunks with no detected year always pass, so this only
                ever narrows among already-dated content.
        """
        if not _INGESTION_AVAILABLE:
            logger.debug(
                "PDF retrieval skipped: ingestion package unavailable (%s).",
                _INGESTION_IMPORT_ERROR,
            )
            return []

        try:
            self.ensure_index_built()
        except Exception:
            logger.exception("Failed to build/update the PDF knowledge index.")
            return []

        if not self._chunks:
            return []

        k = top_k if top_k is not None else self.top_k
        analysis_query = _is_analysis_query(query)
        # Analysis queries need a MUCH wider net to capture intricate details
        # across many sections of annual reports. They also often don't name
        # specific keywords that would match via semantic similarity alone,
        # so we pull a very large candidate pool and let round-robin ensure
        # coverage across sources AND sections.
        candidate_pool = k * 12 if (broad or analysis_query) else k
        query_embedding = embed(query)

        pool = self._chunks
        if company_filter:
            scoped_pool = [
                c for c in pool
                if self._chunk_matches_company(c.chunk_dict.get("company"), company_filter)
            ]
            # Only actually narrow the pool if it leaves something to
            # search -- an over-eager filter should never make a
            # question that clearly has ingested data come back empty.
            if scoped_pool:
                pool = scoped_pool
        if year_filter:
            year_scoped = [
                c for c in pool
                if (
                    (not c.chunk_dict.get("year") and not c.chunk_dict.get("years_mentioned"))
                    or str(c.chunk_dict.get("year")) == str(year_filter)
                    or str(year_filter) in [str(y) for y in (c.chunk_dict.get("years_mentioned") or [])]
                    # A chunk's coarse "year" / "years_mentioned" tags are a
                    # single best-effort guess made at ingestion time (see
                    # ingestion/document.py's _detect_year). A 3-year
                    # comparative financial statement chunk can easily have
                    # that guess land on only ONE of its years (e.g. the
                    # report's cover year), even though the table itself
                    # plainly has columns for all three -- which is exactly
                    # why ingestion/financial_facts.py does its OWN,
                    # separate, per-COLUMN year resolution for that same
                    # chunk (see extract_financial_facts() there). A chunk
                    # whose financial_facts already know about this year is
                    # unambiguous proof it belongs here even when the
                    # coarser tag missed it -- this is what let a real
                    # 2023 PAT figure, sitting in a chunk tagged only
                    # "years_mentioned: [2024]", get silently excluded from
                    # every 2023 search before the model ever saw it.
                    or str(year_filter) in [
                        str(f.get("year")) for f in (c.chunk_dict.get("financial_facts") or [])
                    ]
                )
            ]
            if year_scoped:
                pool = year_scoped

        financial_query = _is_financial_query(query)
        strategy_query = _is_business_strategy_query(query)
        requested_statement = _detect_requested_statement(query)
        # A query naming ONE specific statement ("cash flow statement")
        # needs a wider net than the default top_k=8: that table is one
        # chunk among possibly several years x several statement types,
        # and a flat top_k previously let the P&L/balance-sheet chunks
        # (which usually extract more cleanly/score higher generically
        # on a "financial"-flavoured query) fill the whole budget before
        # the specifically-requested statement's own chunk was ever
        # considered. Widening here costs nothing when that chunk was
        # already going to rank first; it only helps the case where it
        # wasn't.
        if requested_statement and top_k is None and not broad:
            k = max(k, 20)

        scored: List[tuple] = []
        if query_embedding is not None:
            for cached in pool:
                if cached.embedding is None:
                    continue
                score = cosine_similarity(query_embedding, cached.embedding)
                # A table chunk (real rows/columns from TableExtractor) is
                # far more likely to contain the actual number someone
                # asked for than a narrative prose chunk that merely
                # discusses the topic -- boost it so it doesn't lose out
                # to a longer, more "semantically on-topic"-sounding
                # section of running text.
                if cached.chunk_dict.get("is_table"):
                    score += 0.08
                # Financial-vs-ESG topic nudge -- see _topic_adjustment().
                # This is a no-op (0.0) for any non-financial query, so
                # ESG/CSR/general questions are completely unaffected.
                score += _topic_adjustment(query, financial_query, cached.chunk_dict)
                # Business-strategy-vs-ESG topic nudge -- see
                # _strategy_topic_adjustment(). No-op for any query that
                # isn't strategy-flavoured, or that named ESG/CSR/
                # sustainability itself.
                score += _strategy_topic_adjustment(strategy_query, cached.chunk_dict)
                # Statement-specific nudge: when the query names ONE
                # statement, prefer chunks whose own detected section
                # matches it. Deliberately additive-only (no penalty for
                # other statement types): a demotion here means any
                # mislabeled chunk gets ACTIVELY suppressed instead of
                # just not-boosted, which is worse than doing nothing --
                # a query still needs a chance to surface a table via
                # plain semantic similarity even if section detection
                # missed or mislabeled it.
                if requested_statement:
                    section = (cached.chunk_dict.get("section") or "")
                    if section == requested_statement:
                        score += 0.35
                if score >= self.min_similarity:
                    scored.append((score, cached))
        else:
            # Lexical fallback: plain token-overlap scoring, mirroring
            # the JSON retriever's own degraded path when the embedding
            # backend is unreachable.
            scored = self._lexical_fallback_scores(query, pool, financial_query=financial_query)

        scored.sort(key=lambda pair: pair[0], reverse=True)

        if RERANKER_ENABLED:
            # Rerank only the top candidates (the reranker is a
            # comparatively expensive per-pair model call; anything
            # already far down the embedding-similarity ranking is very
            # unlikely to matter, so there's no value in scoring it).
            # Falls back to the untouched `scored` list automatically if
            # the reranker isn't installed/available.
            scored = _rerank_candidates(
                query, scored,
                text_fn=lambda cached: cached.chunk_dict.get("text", ""),
                top_n=RERANKER_TOP_N, weight=RERANKER_WEIGHT,
            )

        if broad or analysis_query:
            candidates = scored[:candidate_pool]
            final_k = max(k, 20)  # analysis needs even more room for diverse sections
            selected = self._round_robin_by_source_and_section(candidates, final_k)
        else:
            selected = scored[:k]

        if financial_query or requested_statement:
            selected = self._stitch_statement_continuations(selected, pool)

        if (broad or analysis_query) and company_filter:
            selected = self._ensure_all_sources_represented(
                selected, pool, query_embedding, requested_statement,
                min_similarity=self.min_similarity,
            )

        return [self._to_result_dict(cached, score) for score, cached in selected]

    @staticmethod
    def _ensure_all_sources_represented(
        selected: List[tuple],
        pool: List["_CachedChunk"],
        query_embedding,
        requested_statement: Optional[str],
        min_similarity: float,
    ) -> List[tuple]:
        """
        For a broad, company-scoped question ("3 years of VST profit"),
        make sure EVERY distinct ingested source document for that
        company contributes at least one relevant chunk -- not just
        whichever document(s) happened to score inside the top
        candidate pool. Two separate annual-report PDFs (e.g. an
        FY23-24 report and an FY24-25 report) can both mention "profit"
        prominently, but if the older report's chunk scores even
        slightly lower semantically than the newer one's (a few extra
        distinctive words, a slightly different phrasing), it can miss
        the candidate_pool cutoff entirely -- silently dropping that
        whole year from a "3 year" answer instead of it ever being
        considered, rather than the model correctly saying that year
        genuinely isn't available.

        IMPORTANT: this must never force in a chunk that ISN'T actually
        relevant. An earlier version of this had no floor at all --
        "pick whatever scores highest for this source, however low
        that is" -- which meant a source with nothing genuinely on
        topic still contributed its single best (but still irrelevant)
        chunk. That chunk then sat in the context alongside the
        genuinely correct chunks for the other years, and confused the
        chart-extraction step enough to pull WRONG numbers even for
        years that were already being answered correctly before this
        backfill ran. Requiring the RAW (pre-boost) similarity to clear
        the same min_similarity floor normal retrieval uses means a
        source with nothing relevant is correctly left out -- and the
        model reports that year as unavailable, which is the honest,
        correct answer -- rather than silently injecting noise.
        """
        if query_embedding is None:
            return selected
        selected_sources = {c.chunk_dict.get("source") for _, c in selected}
        all_sources = {c.chunk_dict.get("source") for c in pool}
        missing_sources = all_sources - selected_sources
        if not missing_sources:
            return selected

        extra: List[tuple] = []
        # Sorted rather than iterated directly off the set: Python
        # randomizes string hash seeds per process by default, so a
        # plain `for source in missing_sources` iterates in a DIFFERENT
        # order every time the process restarts. That doesn't change
        # WHICH sources get backfilled, but it can change tie-breaking
        # in list ordering, which in turn changes where each source's
        # chunk lands in the final context block handed to the LLM --
        # enough to make the exact same question, against the exact
        # same files, produce different answers across restarts. Same
        # data in, same order out, every time.
        for source in sorted(missing_sources, key=lambda s: str(s)):
            best = None
            best_score = -1.0
            best_raw = -1.0
            for c in pool:
                if c.chunk_dict.get("source") != source or c.embedding is None:
                    continue
                raw_score = cosine_similarity(query_embedding, c.embedding)
                score = raw_score
                if c.chunk_dict.get("is_table"):
                    score += 0.08
                if requested_statement and (c.chunk_dict.get("section") or "") == requested_statement:
                    score += 0.35
                if score > best_score:
                    best_score, best_raw, best = score, raw_score, c
            # Gate on the RAW similarity, not the boosted score -- the
            # boosts exist to break ties among genuinely-relevant
            # candidates, not to manufacture relevance a chunk doesn't
            # actually have.
            if best is not None and best_raw >= min_similarity:
                extra.append((best_score, best))
        return selected + extra

    @staticmethod
    def _stitch_statement_continuations(
        selected: List[tuple], pool: List["_CachedChunk"],
    ) -> List[tuple]:
        """A financial statement table in these annual reports routinely
        spans several consecutive pages with NO repeated header ("P&L"
        might start on page 109 and run through 111) -- each page is a
        separate chunk, and a later page's chunk (fewer distinctive
        words, no restated header) often doesn't score high enough on
        its own to independently make top_k, even though it's the rest
        of the exact same statement the top-ranked chunk belongs to.
        This was the direct cause of a "give me the complete P&L"
        request coming back self-admittedly truncated ("[Remaining rows
        omitted]") despite the full statement being fully ingested.

        For each selected `is_table` chunk, pull in any other `is_table`
        chunk from the SAME source file on the immediately following
        1-3 pages, even if it wasn't independently selected -- capped
        modestly so this can't balloon an unrelated broad/compare answer.
        """
        selected_ids = {id(c) for _, c in selected}
        by_source_page: Dict[tuple, "_CachedChunk"] = {
            (c.chunk_dict.get("source"), c.chunk_dict.get("page")): c
            for c in pool if c.chunk_dict.get("is_table")
        }
        extra: List[tuple] = []
        added = 0
        MAX_EXTRA = 6
        for score, cached in list(selected):
            if not cached.chunk_dict.get("is_table"):
                continue
            source = cached.chunk_dict.get("source")
            page = cached.chunk_dict.get("page")
            if source is None or page is None:
                continue
            for offset in (1, 2, 3):
                if added >= MAX_EXTRA:
                    break
                sibling = by_source_page.get((source, page + offset))
                if sibling is None or id(sibling) in selected_ids:
                    continue
                # Slightly below the parent's own score so it still
                # sorts after the pages that actually matched the query,
                # but is included in what gets sent to the LLM.
                extra.append((max(score - 0.01, 0.0), sibling))
                selected_ids.add(id(sibling))
                added += 1
        return selected + extra

    @staticmethod
    def _round_robin_by_source(
        scored: List[tuple], limit: int
    ) -> List[tuple]:
        """Given already-relevance-sorted (score, chunk) pairs, pick
        results round-robin across distinct source files (each pass
        takes the next-best still-unused chunk from each source in turn)
        rather than just taking the global top-N. This guarantees a
        3-year, 3-document question actually sees all 3 documents
        instead of whichever one happened to embed closest to the
        query."""
        by_source: Dict[str, List[tuple]] = {}
        order: List[str] = []
        for pair in scored:
            source = pair[1].chunk_dict.get("source", "unknown")
            if source not in by_source:
                by_source[source] = []
                order.append(source)
            by_source[source].append(pair)

        selected: List[tuple] = []
        round_index = 0
        while len(selected) < limit and any(by_source[src] for src in order):
            for src in order:
                if len(selected) >= limit:
                    break
                bucket = by_source[src]
                if round_index < len(bucket):
                    selected.append(bucket[round_index])
            round_index += 1
            if round_index > limit:  # safety valve against pathological input
                break
        return selected

    @staticmethod
    def _round_robin_by_source_and_section(
        scored: List[tuple], limit: int
    ) -> List[tuple]:
        """Like _round_robin_by_source but ALSO round-robins across
        distinct sections within each source. This ensures analysis
        queries get diverse coverage: business overview, MD&A, risk
        factors, corporate governance, financial review, operations,
        strategy, outlook, etc. -- not just the top-scoring section
        from each annual report."""
        by_source_section: Dict[tuple, List[tuple]] = {}
        order: List[tuple] = []
        for pair in scored:
            cd = pair[1].chunk_dict
            source = cd.get("source", "unknown")
            section = (cd.get("section") or "").lower()
            # Normalize section names to group similar ones
            key = (source, section)
            if key not in by_source_section:
                by_source_section[key] = []
                order.append(key)
            by_source_section[key].append(pair)

        selected: List[tuple] = []
        round_index = 0
        while len(selected) < limit and any(by_source_section[k] for k in order):
            for key in order:
                if len(selected) >= limit:
                    break
                bucket = by_source_section[key]
                if round_index < len(bucket):
                    selected.append(bucket[round_index])
            round_index += 1
            if round_index > limit:
                break
        return selected

    def ensure_index_built(self) -> None:
        """Build the index on first call; on every subsequent call,
        incrementally pick up any new or changed PDFs in `raw_dir`
        without re-processing files that have not changed."""
        with self._lock:
            if not os.path.isdir(self.raw_dir):
                os.makedirs(self.raw_dir, exist_ok=True)
                self._index_built = True
                return

            if not self._index_built:
                self._load_cache()

            pdf_paths = self._discover_pdfs()
            for path in pdf_paths:
                checksum = file_checksum(path)
                if self._known_checksums.get(path) == checksum:
                    continue  # unchanged since last ingestion; skip
                self._ingest_and_index(path, checksum)

            # Prune ORPHANED cache entries: a source path recorded in
            # `_known_checksums`/`_chunks` that no longer matches any
            # file `_discover_pdfs()` currently finds. This happens
            # whenever the project directory is moved, renamed, or
            # reorganized (a refactor, a copy to a new machine) -- the
            # checksum dict is keyed by absolute path, so the OLD path's
            # entry never gets touched by the loop above (it simply
            # isn't in `pdf_paths` any more) and its chunks -- however
            # old, mislabeled, or produced by a since-fixed ingestion
            # bug -- silently live in the cache forever, sitting right
            # alongside the fresh, correctly re-ingested chunks for what
            # is otherwise the same file. This is what let chunks tagged
            # with a section label the current code can no longer even
            # produce keep polluting retrieval indefinitely. Without
            # this, the only way to clear them out was a full `--force`
            # wipe of the entire cache.
            current_paths = set(pdf_paths)
            stale_paths = set(self._known_checksums) - current_paths
            if stale_paths:
                removed_chunks = sum(
                    1 for c in self._chunks if c.chunk_dict.get("source") in stale_paths
                )
                self._chunks = [
                    c for c in self._chunks if c.chunk_dict.get("source") not in stale_paths
                ]
                store = self._get_store()
                for stale in stale_paths:
                    del self._known_checksums[stale]
                    if store is not None:
                        store.delete_chunks_for_source(stale)
                        store.delete_checksum(stale)
                logger.warning(
                    "PDF knowledge source: pruned %d orphaned chunk(s) from %d stale "
                    "source path(s) no longer found under %s: %s",
                    removed_chunks, len(stale_paths), self.raw_dir, sorted(stale_paths),
                )

            self._index_built = True

    def rebuild(self, force: bool = False, caption_images: Optional[bool] = None) -> int:
        """Force a full rebuild (or incremental update if force=False).
        Returns the number of files (re)ingested. Intended for a manual
        CLI trigger (see `import_pdf_documents.py`) after dropping new
        annual reports into `PDF_KNOWLEDGE_RAW_DIR`.

        ``caption_images``: override the instance's default for this
        call only (e.g. the CLI's --caption-images flag). Note this only
        affects files actually re-ingested this run -- an unchanged file
        that's skipped by the checksum check keeps whatever captions (or
        lack of captions) it already has, so pair this with --force to
        recaption images in already-ingested PDFs."""
        if caption_images is not None:
            self.caption_images = caption_images
            self._ingestor = None  # rebuilt lazily with the new setting
        if force:
            with self._lock:
                self._chunks = []
                self._known_checksums = {}
                self._index_built = False
        before = len(self._known_checksums)
        self.ensure_index_built()
        return len(self._known_checksums) - before if not force else len(self._known_checksums)

    # -- internals ------------------------------------------------------------

    def _discover_pdfs(self) -> List[str]:
        paths = []
        for root, _dirs, files in os.walk(self.raw_dir):
            for name in files:
                if name.lower().endswith(".pdf"):
                    paths.append(os.path.join(root, name))
        return sorted(paths)

    def _get_ingestor(self):
        if self._ingestor is None:
            self._ingestor = PDFIngestor(
                image_output_dir=self.images_dir,
                caption_images=self.caption_images,
                use_camelot_fallback=False,
            )
        return self._ingestor

    def _get_store(self) -> Optional[ChromaStore]:
        if not _CHROMA_AVAILABLE:
            logger.warning(
                "chromadb is not installed; PDF knowledge base cannot be "
                "persisted or loaded. Install it with `pip install chromadb`."
            )
            return None
        if self._store is None:
            self._store = ChromaStore(self.chroma_dir)
        return self._store

    def _ingest_and_index(self, path: str, checksum: str) -> None:
        logger.info("PDF knowledge source: ingesting %s", path)
        document = self._get_ingestor().ingest_with_timing(path)

        if document.quality_score < 0.15:
            logger.warning(
                "PDF %s produced a very low quality score (%.2f); indexing it "
                "anyway, but its extracted content may be unreliable. "
                "Warnings: %s",
                path, document.quality_score, document.extraction_warnings,
            )

        # Drop any previously-cached chunks for this exact path (covers
        # the "file changed" case -- old chunks must not linger).
        self._chunks = [
            c for c in self._chunks if c.chunk_dict.get("source") != path
        ]

        new_chunks = document.all_chunks()
        total_new = len(new_chunks)
        new_chunk_dicts = [c.to_dict() for c in new_chunks]
        # Genuinely batched: one HTTP request covers up to
        # _EMBED_BATCH_SIZE chunks instead of one request per chunk --
        # see retrieval/embedding_store.py's embed_many() docstring.
        # This is what actually fixes ingestion appearing to hang for
        # tens of minutes on a large/table-dense PDF with zero visible
        # progress: the previous per-chunk embed() loop made one
        # sequential HTTP round-trip per chunk (7,847 chunks = 7,847
        # requests for a large annual report). Quality/what gets
        # embedded is identical either way -- this only changes how many
        # network requests it takes, with the same on-disk cache reuse
        # and the same automatic fallback to one-at-a-time calls if the
        # batch endpoint isn't available.
        new_embeddings: List[Optional[List[float]]] = []
        chunk_texts = [c.text for c in new_chunks]
        for start in range(0, total_new, _EMBED_LOG_BATCH):
            batch = chunk_texts[start:start + _EMBED_LOG_BATCH]
            new_embeddings.extend(embed_many(batch))
            done = min(start + _EMBED_LOG_BATCH, total_new)
            logger.info(
                "  embedded %d/%d chunks for %s",
                done, total_new, os.path.basename(path),
            )

        for chunk_dict, embedding in zip(new_chunk_dicts, new_embeddings):
            self._chunks.append(_CachedChunk(chunk_dict=chunk_dict, embedding=embedding))

        store = self._get_store()
        if store is not None:
            store.delete_chunks_for_source(path)
            store.upsert_chunks(new_chunk_dicts, new_embeddings)
            store.set_checksum(path, checksum)
            store.set_logic_version(INGESTION_LOGIC_VERSION)

        self._known_checksums[path] = checksum
        logger.info(
            "PDF knowledge source: indexed %d chunks from %s (quality=%.2f)",
            len(new_chunks), path, document.quality_score,
        )

    @staticmethod
    def _to_result_dict(cached: "_CachedChunk", score: float) -> Dict[str, Any]:
        chunk = cached.chunk_dict
        section_label = chunk.get("section") or "document"
        if chunk.get("subsection"):
            section_label = f"{section_label} / {chunk['subsection']}"
        years_mentioned = chunk.get("years_mentioned") or []
        record = {
            "document_title": chunk.get("document_title"),
            "company": chunk.get("company"),
            "page": chunk.get("page"),
            "text": chunk.get("text"),
            "source_file": chunk.get("source"),
            "is_table": chunk.get("is_table", False),
            "financial_facts": chunk.get("financial_facts") or [],
        }
        # A chunk with a single confidently-detected year gets a plain
        # "Year" field, same as before. A chunk spanning several years
        # (a 3-year comparative table) gets an explicit "Years Covered"
        # list instead of one arbitrary year -- showing a single "Year:
        # 2022" field next to a table whose rows actually cover 2022,
        # 2023, AND 2024 is what caused the model to misattribute every
        # row in the table to 2022 and drop 2024 entirely.
        if chunk.get("year") is not None:
            record["year"] = chunk.get("year")
        elif years_mentioned:
            record["years_covered"] = years_mentioned
        return {
            "category": "document_pdf",
            "section": section_label,
            "record": record,
            "score": round(float(score) * 1000, 2),  # same 0-1000-ish scale as knowledge_index
        }

    def _lexical_fallback_scores(
        self,
        query: str,
        pool: Optional[List["_CachedChunk"]] = None,
        financial_query: Optional[bool] = None,
    ) -> List[tuple]:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not query_tokens:
            return []
        if financial_query is None:
            financial_query = _is_financial_query(query)
        strategy_query = _is_business_strategy_query(query)
        scored = []
        for cached in (pool if pool is not None else self._chunks):
            text = cached.chunk_dict.get("text", "")
            tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
            overlap = len(query_tokens & tokens)
            if overlap > 0:
                # Normalize into roughly the same [0, 1] range cosine
                # similarity would occupy, so the min_similarity
                # threshold still means something in fallback mode.
                score = min(overlap / max(len(query_tokens), 1), 1.0)
                score += _topic_adjustment(query, financial_query, cached.chunk_dict)
                score += _strategy_topic_adjustment(strategy_query, cached.chunk_dict)
                if score >= self.min_similarity:
                    scored.append((score, cached))
        return scored

    # -- persistent storage (Chroma, keyed by source file) -----------------

    def _load_cache(self) -> None:
        store = self._get_store()
        if store is None:
            self._known_checksums = {}
            self._chunks = []
            return

        cached_logic_version = store.get_logic_version()
        if cached_logic_version != INGESTION_LOGIC_VERSION:
            # The detection/labeling logic in ingestion/document.py has
            # changed since this store was written -- every stored chunk
            # may carry a section/heading/text a fresh ingestion run
            # would no longer produce (this is exactly what let an old,
            # already-fixed mislabeling bug keep serving wrong answers
            # indefinitely: the source PDF's checksum never changed, so
            # it was never re-ingested even after the bug was fixed in
            # code). Wipe the store so every file falls through the
            # normal "no known checksum" path and gets ingested fresh
            # with the CURRENT logic.
            logger.warning(
                "PDF knowledge store was built with ingestion logic v%s; "
                "current logic is v%s. Discarding stored chunks and "
                "re-ingesting all PDFs so they reflect the latest "
                "detection/labeling fixes.",
                cached_logic_version, INGESTION_LOGIC_VERSION,
            )
            store.clear_all()
            store.set_logic_version(INGESTION_LOGIC_VERSION)
            self._known_checksums = {}
            self._chunks = []
            return

        self._known_checksums = store.get_checksums()
        self._chunks = [
            _CachedChunk(chunk_dict=chunk_dict, embedding=embedding)
            for chunk_dict, embedding in store.load_all_chunks()
        ]
        logger.info(
            "PDF knowledge source: loaded %d chunks from Chroma (%d file(s)).",
            len(self._chunks), len(self._known_checksums),
        )


# ─── Module-level singleton, mirroring retrieval.knowledge_index's `index` ────

pdf_retriever = PDFRetriever()


def search(
    query: str,
    top_k: Optional[int] = None,
    broad: bool = False,
    company_filter: Optional[str] = None,
    year_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Module-level convenience wrapper around the shared singleton --
    mirrors the calling convention of `retrieval.knowledge_index`."""
    return pdf_retriever.search(
        query, top_k=top_k, broad=broad,
        company_filter=company_filter, year_filter=year_filter,
    )