"""
retrieval/knowledge_index.py

Global Knowledge Index — the heart of the RAG retrieval system.

Design principles
-----------------
1. Schema-agnostic: uses recursive JSON flattening so any future
   data structure works automatically without hardcoding field names.
2. SEMANTIC retrieval is the primary signal: each indexed record and each
   incoming query are embedded (via retrieval.embedding_store, backed by
   a local Ollama embedding model) and ranked by cosine similarity. This
   is what lets a query like "the firm's beginnings" match a "history"
   record even though the two share almost no literal words — something
   keyword/fuzzy matching cannot reliably do, and the retrieval gap that
   forces an LLM to hallucinate when it can't find what it needs.
3. Lexical scoring (token overlap + fuzzy partial ratio + synonym
   expansion) is kept ONLY as a fallback for when the embedding model is
   unreachable, so the assistant never goes fully blind if Ollama is down.
4. Company record is split into semantic "sections" (overview, history,
   leadership, products, financials, competitors, …) so that large
   records don't drown out specific answers.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz

from core.config import (
    COMPANY_FILE, DEPARTMENTS_FILE, PROJECTS_FILE,
    PROCESSES_FILE, PROBLEMS_FILE, KPIS_FILE,
    RETRIEVAL_TOP_K, RETRIEVAL_MIN_SCORE, EMBEDDING_MIN_SIMILARITY,
    DOCUMENTS_CHUNKS_FILE, DOCUMENTS_PROCESSED_DIR, MAX_PER_SECTION, MAX_DOCUMENT_PER_SECTION,
    RERANKER_ENABLED, RERANKER_TOP_N, RERANKER_WEIGHT,
)

logger = logging.getLogger(__name__)
from utils.data_loader import load_json
from utils.text import tokenize
from retrieval.embedding_store import embed, embed_many, cosine_similarity, flush_disk_cache
from retrieval.reranker import rerank as _rerank_candidates

# ─── Synonym Map (lexical-fallback path ONLY) ────────────────────────────────
# This is no longer the primary retrieval mechanism. It exists solely to
# improve recall on the degraded lexical scorer used when the embedding
# model can't be reached — semantic search does not need it, because
# embeddings already capture "products" ~ "brands" ~ "portfolio" style
# relationships without an explicit mapping.

SYNONYMS: Dict[str, List[str]] = {
    "ceo":                      ["chairperson", "chairwoman", "managing director", "leadership", "executive"],
    "cfo":                      ["finance head", "chief financial officer", "leadership"],
    "coo":                      ["operations head", "chief operating officer", "leadership"],
    "leadership":               ["ceo", "cfo", "coo", "management", "executives", "directors", "board"],
    "owner":                    ["ownership", "shareholding", "promoter", "stake", "owns", "proprietor"],
    "owns":                     ["owner", "ownership"],
    "history":                  ["founded", "incorporation", "established", "origin", "founding", "heritage"],
    "overview":                 ["summary", "about", "introduction", "profile", "description"],
    "competitor":               ["competitor_finances", "rival", "competition", "market player", "itc", "vst"],
    "competitors":              ["competitor_finances", "rival", "competition", "market player"],
    "rival":                    ["competitor_finances"],
    "competition":              ["competitor_finances"],
    "product":                  ["products_services", "brands", "portfolio", "cigarette", "tobacco", "confectionery"],
    "products":                 ["products_services", "cigarette_segmentation", "confectionery_segmentation"],
    "segmentation":             ["cigarette_segmentation", "pmi_strategic_segments", "confectionery_segmentation",
                                 "segmentation_framework_insights", "segment", "segments", "portfolio breakdown"],
    "segment":                  ["cigarette_segmentation", "pmi_strategic_segments", "segmentation_framework_insights"],
    "portfolio":                ["products_services", "cigarette_segmentation", "pmi_strategic_segments"],
    "financials":               ["revenue", "profit", "turnover", "earnings", "financial", "results"],
    "manufacturing":            ["operations", "factory", "plant", "production", "facilities"],
    "operations":               ["operations"],
    "mission":                  ["mission_vision", "purpose", "goal", "objective"],
    "vision":                   ["mission_vision", "aspiration", "future goal"],
    "department":               ["team", "division", "function", "unit"],
    "project":                  ["initiative", "program", "engagement"],
    "process":                  ["workflow", "procedure", "steps"],
    "kpi":                      ["metric", "metrics", "target", "targets", "measurement"],
    "problem":                  ["issue", "risk", "challenge", "difficulty"],
    "location":                 ["headquarters", "hq", "office", "city", "address"],
    "csr":                      ["corporate social responsibility", "sustainability", "esg", "environment"],
    "award":                    ["recognition", "achievement", "accolade", "honor"],
    "pmi":                      ["pmi_global_strategy", "pmi_strategic_segments", "philip morris"],
    "indian market":            ["indian_market_overview"],
    "market":                   ["indian_market_overview", "competitor_finances"],
    "hr":                       ["department", "human resources"],
    "human resources":          ["department"],
}


# ─── Index Item ───────────────────────────────────────────────────────────────

class IndexItem:
    """
    A single indexed knowledge unit.

    Each item carries:
    - category:  source data category (company, department, …)
    - section:   semantic sub-section label for company records
    - record:    original Python dict
    - text:      flat searchable string derived from the record
    - tokens:    pre-tokenized set, used only by the lexical fallback scorer
    - entities:  important name strings for precise exact-match bonuses
    - embedding: semantic vector for this item's text (None until computed,
                 and remains None permanently if the embedding model is
                 unreachable — callers must handle that gracefully)
    """

    __slots__ = ("category", "section", "record", "text", "tokens", "entities", "embedding")

    def __init__(
        self,
        category: str,
        record: Dict[str, Any],
        section: str = "",
    ) -> None:
        self.category  = category
        self.section   = section
        self.record    = record
        self.text      = self._build_text(category, section, record)
        self.tokens    = tokenize(self.text)
        self.entities  = self._extract_entities(record)
        self.embedding = None  # populated by KnowledgeIndex._embed_items()

    # ── text builder ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_text(
        category: str,
        section: str,
        record: Dict[str, Any],
    ) -> str:
        """Flatten record to a searchable string, prepending category / section."""
        parts = [f"category:{category}"]
        if section:
            parts.append(f"section:{section}")

        def _flatten(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    next_prefix = f"{prefix}.{k}" if prefix else k
                    _flatten(v, next_prefix)
            elif isinstance(obj, list):
                for item in obj:
                    _flatten(item, prefix)
            else:
                if obj is not None and str(obj).strip():
                    parts.append(f"{prefix}:{obj}" if prefix else str(obj))

        _flatten(record)
        return " ".join(parts).lower()

    @staticmethod
    def _extract_entities(record: Dict[str, Any]) -> List[str]:
        """Pull out high-value name strings for precise exact-match bonus scoring."""
        entities: List[str] = []
        for key in ("name", "owner", "head", "manager"):
            val = record.get(key)
            if isinstance(val, str) and val.strip():
                entities.append(val.lower())
        return entities


# ─── Knowledge Index ──────────────────────────────────────────────────────────

class KnowledgeIndex:
    """
    Loads all JSON knowledge files, embeds every record, and provides
    semantic (embedding-first) search with a lexical fallback.
    """

    def __init__(self) -> None:
        self.items: List[IndexItem] = []
        self._build()
        self._embed_items()

    # ── loading ───────────────────────────────────────────────────────────────

    def _build(self) -> None:
        self.items.clear()
        self._load_company()
        self._load_entity_file(DEPARTMENTS_FILE, "department")
        self._load_entity_file(PROJECTS_FILE,    "project")
        self._load_entity_file(PROCESSES_FILE,   "process")
        self._load_entity_file(PROBLEMS_FILE,    "problem")
        self._load_entity_file(KPIS_FILE,        "kpi")
        self._load_documents()

    def _load_entity_file(self, path: str, category: str) -> None:
        """Load a standard entity file (departments, projects, …)."""
        try:
            data = load_json(path)
            records = data.get(category, {}).get("records", [])
            for rec in records:
                self.items.append(IndexItem(category=category, record=rec))
        except Exception:
            # Same reasoning as _load_company's exception handling above:
            # a silently-swallowed parse error here used to make an entire
            # entity file (departments/projects/processes/problems/kpis)
            # vanish from the index with no visible indication why.
            logger.exception(
                "Failed to load/parse %s (category=%s) — this file's "
                "records will be missing from the index until fixed.",
                path, category,
            )

    # Known company aliases → canonical tag. Extend this if you ingest
    # other competitors. Checked against both the source filename and the
    # chunk's own text, since a scanned/exported PDF's filename is often
    # more reliable than trying to infer the company from body text alone
    # (e.g. "VST_Annual_Report_2022.pdf" vs. a body page that just says
    # "the Company" throughout).
    _COMPANY_ALIASES = {
        "VST":  ["vst industries", "vst", "vazir sultan"],
        "ITC":  ["itc limited", "itc ltd", "itc"],
        "GPI":  ["godfrey phillips", "gpi"],
    }

    # Matches a plain 4-digit year (2015-2029) or common Indian-style
    # fiscal-year notations: "FY23", "FY 2023", "FY2022-23", "2022-23".
    # Fiscal-year ranges are normalized to their END year (e.g. "2022-23"
    # -> "2023") since that's what people usually mean by "the 2023
    # report" for an April-March fiscal year.
    _YEAR_PATTERN = re.compile(
        r"\bFY\s?(\d{2})(?:-(\d{2}))?\b|\b(20[1-2]\d)(?:-(\d{2}))?\b",
        re.IGNORECASE,
    )

    @classmethod
    def _normalize_year_match(cls, m: "re.Match") -> Optional[str]:
        fy2, fy2_end, y4, y4_end = m.groups()
        if y4:
            if y4_end:
                return "20" + y4_end
            return y4
        if fy2:
            end = fy2_end or fy2
            return "20" + end
        return None

    @classmethod
    def _detect_year(cls, filename: str, text: str) -> Optional[str]:
        """
        Best-effort tag for which fiscal/reporting year a document chunk
        covers. Without this, a query naming one specific year (e.g.
        "GPI financials for 2023") has no way to tell a 2023 chunk apart
        from a 2024 or 2025 chunk in the same section — retrieval just
        returns whatever ranks highest semantically, which in practice
        meant a query about 2023 could silently come back with 2024-25
        data instead, with no signal to the user that the year was wrong
        until they noticed the numbers didn't match. Checked filename
        first (usually the most reliable signal — e.g.
        "GPI_Annual_Report_2023.pdf"), falling back to a text scan.
        """
        fname_match = cls._YEAR_PATTERN.search(cls._normalize_for_matching(filename))
        if fname_match:
            year = cls._normalize_year_match(fname_match)
            if year:
                return year

        text_match = cls._YEAR_PATTERN.search(cls._normalize_for_matching((text or "")[:500]))
        if text_match:
            return cls._normalize_year_match(text_match)

        return None

    @classmethod
    def _normalize_for_matching(cls, text: str) -> str:
        """
        Prepare filename/text for word-boundary alias and year matching.
        Underscores are the most common filename word-separator (e.g.
        "VST_2023.pdf", "GPI_Annual_Report_2023.pdf") but regex `\\b`
        does NOT treat "_" as a boundary — it's a \\w character, so
        "vst_2023" is one unbroken token to `\b`, and `\bvst\b` fails to
        match "vst" inside it. Replacing underscores with spaces first
        fixes this for the (very common) underscore-separated filename
        case without touching hyphens, which fiscal-year ranges like
        "2022-23" rely on being adjacent (hyphen is already a proper
        \\b boundary on its own).
        """
        return (text or "").lower().replace("_", " ")

    @classmethod
    def _alias_matches(cls, text: str, alias: str) -> bool:
        """
        Word-boundary match for a company alias — NOT a naive substring
        check. Short aliases like "itc", "vst", "gpi" are substrings of
        completely unrelated ordinary English words: "itc" alone matches
        inside "kitchen", "pitch", "witch", "hitch", "stitch", "ditch".
        A naive `"itc" in text` check previously mistagged any GPI or VST
        chunk that happened to contain one of those words (e.g. "sales
        pitch", a facility's "kitchen") as ITC's data — which is exactly
        what caused an unrelated GPI/VST chunk to show up mislabeled as
        "context related to ITC" in a real conversation. `\b` boundaries
        require the alias to appear as a whole word, not embedded inside
        another one.
        """
        return re.search(r"\b" + re.escape(alias) + r"\b", text) is not None

    @classmethod
    def _detect_company(cls, filename: str, text: str) -> Optional[str]:
        """
        Best-effort tag for which company a document chunk is actually
        about. Without this, chunks from DIFFERENT competitors ingested
        into the SAME semantic section (e.g. both VST's and ITC's annual
        reports tag as "competitor_finances") are indistinguishable to
        retrieval — a flat top-k ranked list can let one competitor's
        chunks crowd out the other's entirely, which is exactly what made
        a 3-way "compare VST, ITC, and GPI" query silently drop one or
        two of the three companies from context. Checked filename first
        (cheap, usually reliable — e.g. "VST_2022.pdf"), falling back to
        a text scan only if the filename doesn't match anything.
        """
        fname_lower = cls._normalize_for_matching(filename)
        for company, aliases in cls._COMPANY_ALIASES.items():
            if any(cls._alias_matches(fname_lower, alias) for alias in aliases):
                return company

        text_lower = cls._normalize_for_matching((text or "")[:500])
        for company, aliases in cls._COMPANY_ALIASES.items():
            if any(cls._alias_matches(text_lower, alias) for alias in aliases):
                return company

        return None

    def _load_documents(self) -> None:
        """
        Load pre-processed PDF/PPT/chart/image content written by
        import_documents.py.

        Storage: a Chroma collection (documents_knowledge_chunks, in the
        same on-disk database as the PDF-intelligence store) holding one
        record per chunk, replacing the old documents/processed/*.json
        per-file layout and the legacy single documents/processed_chunks.json.
        The per-file JSON layout already fixed the "one giant file
        rewritten on every ingest" problem; moving to Chroma removes the
        remaining downsides — a full directory listing + JSON parse of
        every document on every index build/reload, and no indexed way
        to delete/replace just one document's chunks (import_documents.py
        had to know and rewrite that exact file). Chroma gives incremental
        per-source upsert/delete and a single indexed read.

        On first run after upgrading, if the Chroma collection is empty
        but legacy files exist on disk, they're migrated in automatically
        (see _migrate_legacy_documents_to_chroma) so nothing already
        ingested is lost — you don't need to re-run import_documents.py
        just to pick up this storage change.

        Each chunk becomes a normal IndexItem, tagged with the category the
        ingester guessed (company/department) and the semantic section
        (financials/leadership/history/products/…) it detected — so
        document content participates in the exact same section-hint
        boosting and category filtering as the curated JSON data. A
        financials question can pull in both the curated numbers AND a
        supporting chart/slide, clearly labeled as coming from a document
        via record["source_type"] == "document".

        Also tags each chunk with a best-effort record["company"] and
        record["year"] (see _detect_company / _detect_year) so
        multi-entity, multi-year comparisons can retrieve each named
        company/year's data separately instead of competing in one
        undifferentiated pool.

        Note: only the STORAGE of these records moved to Chroma. Their
        embeddings are still computed uniformly for every IndexItem
        (curated JSON and document-sourced alike) by _embed_items(),
        which still uses the untouched, unchanged JSON-file-backed
        embedding_store disk cache -- that part was deliberately left
        exactly as it was.
        """
        chunk_records: List[Dict[str, Any]] = self._migrate_legacy_documents_to_chroma()

        try:
            store = self._get_documents_store()
            if store is not None:
                chunk_records = store.load_all_records()
        except Exception as exc:
            logger.warning(
                "Failed to load document chunks from Chroma; falling back to "
                "any legacy files on disk for this run: %s", exc,
            )
            chunk_records = self._load_legacy_document_files()

        for entry in chunk_records:
            source_file = str(entry.get("source_file") or "")
            # BUG (reproduced live): a "pat of gpi 2023, 2024 and 2025"
            # query's Sources panel showed real PDF financial-statement
            # chunks retrieved alongside THREE unrelated spreadsheet
            # row-batches ("gpi_sales_data_feb2024 -- rows 3901-4000",
            # etc.), all tagged category "Company" -- filling result
            # slots with content that can never contain a PAT figure,
            # crowding out the real financial statement pages, and (since
            # _retrieve() in core/engine.py deliberately keeps JSON/
            # document-index results FIRST, ahead of PDF chunks, on the
            # theory that curated data should outrank raw PDF text) doing
            # so with PRIORITY over the genuine PDF data.
            # Spreadsheet files already have their own dedicated, far
            # more capable path -- core/data_analyst.py loads them
            # directly as live pandas DataFrames for real filter/group/
            # aggregate queries. They were never meant to also be chunked
            # into this JSON/document knowledge index (that's for curated
            # facts and PDF-derived content); this only happens when the
            # legacy ingest/ingest.py pipeline (run via import_documents.py)
            # was pointed at a directory that also contained spreadsheet
            # files. Skipping spreadsheet-sourced entries here removes the
            # duplication at its root rather than trying to out-rank it.
            if source_file.lower().endswith((".xlsx", ".xls", ".xlsm", ".csv")):
                continue
            category = entry.get("category") or "company"
            section  = entry.get("section") or "overview"
            record   = dict(entry.get("record") or {})
            record["source_type"] = "document"
            record.setdefault("source_file", entry.get("source_file"))
            record["company"] = self._detect_company(
                record.get("source_file", ""), entry.get("text", "")
            )
            record["year"] = self._detect_year(
                record.get("source_file", ""), entry.get("text", "")
            )
            self.items.append(IndexItem(category, record, section=section))

    @staticmethod
    def _load_legacy_document_files() -> List[Dict[str, Any]]:
        """Read the old file-based storage (documents/processed/*.json
        and the legacy documents/processed_chunks.json), if present.
        Used only for one-time migration into Chroma and as an
        emergency fallback if Chroma itself can't be read."""
        chunk_records: List[Dict[str, Any]] = []

        if os.path.isdir(DOCUMENTS_PROCESSED_DIR):
            for fname in sorted(os.listdir(DOCUMENTS_PROCESSED_DIR)):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(DOCUMENTS_PROCESSED_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        chunk_records.extend(json.load(f))
                except (json.JSONDecodeError, IOError) as exc:
                    logger.warning("Could not read %s: %s", fpath, exc)

        if os.path.exists(DOCUMENTS_CHUNKS_FILE):
            try:
                with open(DOCUMENTS_CHUNKS_FILE, "r", encoding="utf-8") as f:
                    chunk_records.extend(json.load(f))
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning("Could not read %s: %s", DOCUMENTS_CHUNKS_FILE, exc)

        return chunk_records

    def _get_documents_store(self):
        """Lazily construct the Chroma store for ingest/ pipeline document
        chunks. Returns None (never raises) if chromadb isn't installed,
        so the app degrades to file-based/empty document content instead
        of failing to start."""
        if getattr(self, "_documents_store", None) is not None:
            return self._documents_store
        try:
            from retrieval.chroma_store import ChromaStore
            from core.config import (
                PDF_KNOWLEDGE_CHROMA_DIR,
                DOCUMENTS_CHROMA_CHUNKS_COLLECTION,
                DOCUMENTS_CHROMA_META_COLLECTION,
            )
            self._documents_store = ChromaStore(
                PDF_KNOWLEDGE_CHROMA_DIR,
                chunks_collection=DOCUMENTS_CHROMA_CHUNKS_COLLECTION,
                meta_collection=DOCUMENTS_CHROMA_META_COLLECTION,
            )
        except Exception as exc:
            logger.warning(
                "Chroma unavailable for document-chunk storage (%s); "
                "falling back to reading legacy files on disk if present.",
                exc,
            )
            self._documents_store = None
        return self._documents_store

    def _migrate_legacy_documents_to_chroma(self) -> List[Dict[str, Any]]:
        """
        One-time upgrade path: if the Chroma document-chunks collection
        is empty but the old documents/processed/*.json (or legacy
        documents/processed_chunks.json) files exist, copy their records
        into Chroma so nothing already ingested is lost just because the
        storage backend changed. Returns [] once migration has happened
        (or wasn't needed) -- the caller then loads from Chroma as the
        actual source of truth.
        """
        store = self._get_documents_store()
        if store is None:
            return self._load_legacy_document_files()

        if store.count() > 0:
            return []  # already on Chroma, nothing to migrate

        legacy_records = self._load_legacy_document_files()
        if not legacy_records:
            return []

        logger.info(
            "Migrating %d legacy document chunk record(s) into Chroma "
            "(one-time upgrade; documents/processed/*.json and "
            "documents/processed_chunks.json are no longer read after "
            "this point and can be deleted once you've verified "
            "everything came through).",
            len(legacy_records),
        )
        store.upsert_records(legacy_records)
        return []


    def _load_company(self) -> None:
        """
        Load the company profile and split it into semantic sections.

        Each section becomes a separate IndexItem so that a query about
        leadership retrieves the leadership block instead of the entire
        (very large) company record.
        """
        try:
            data = load_json(COMPANY_FILE)
            records = data.get("company_profile", {}).get("records", [])
            if not records:
                logger.warning(
                    "%s loaded successfully but contains no company_profile.records — "
                    "0 curated company sections will be indexed. Check the file's structure.",
                    COMPANY_FILE,
                )
            for rec in records:
                try:
                    self._index_company_record(rec)
                except Exception:
                    logger.exception(
                        "Failed to index one company_profile record from %s (id=%s) — "
                        "that record's sections will be missing from the index, but other "
                        "records/files are unaffected.",
                        COMPANY_FILE, rec.get("id", "?"),
                    )
        except Exception:
            # This used to be a bare `except Exception: pass` — meaning a
            # SINGLE typo while hand-editing company_overview.json (a
            # trailing comma, a mismatched brace, anything that makes the
            # file fail to parse) silently discarded ALL ~18 curated
            # company sections with zero warning anywhere. That is exactly
            # what happened when editing out the "metrics" and
            # "competitor_finances" keys previously took the indexed
            # section count from 28 down to 10 instead of the expected 26
            # — the file broke, and the failure was invisible. Logging the
            # real exception here means a broken edit now shows up
            # immediately in the terminal/logs instead of silently
            # vanishing 18 sections' worth of company knowledge.
            logger.exception(
                "Failed to load/parse %s — ALL curated company sections "
                "(overview, financials, leadership, etc.) will be missing "
                "from this index build until the file is fixed.",
                COMPANY_FILE,
            )

    def _index_company_record(self, rec: Dict[str, Any]) -> None:
        """
        Decompose a large company record into focused semantic sections.

        The "overview" section stays LEAN — a curated, high-level summary
        (identity, scale, headline metrics, leadership names, top product
        lines) rather than the raw record, so retrieving "overview" doesn't
        pull in thousands of characters that duplicate every other section.
        Full depth is available on demand via the dedicated sections below.
        """
        ai = rec.get("additional_info", {})
        gpi = ai.get("gpi_details", {})
        metrics = rec.get("metrics", {})
        leadership = gpi.get("leadership", {})
        products = ai.get("products_services", {})
        # The record's top-level "owner" field (e.g. "Dr. Bina Modi") is the
        # actual answer to "who owns the company" — distinct from the
        # shareholding percentages under gpi_details.ownership, and from
        # the CEO/CFO/COO titles under leadership. It previously wasn't
        # copied into ANY section, so no query — however phrased — could
        # ever surface it; a "leadership" or "ownership" section query
        # would only ever find titles or percentages instead. Stamping it
        # into overview/leadership/ownership/history below fixes that
        # regardless of which section a query happens to resolve to, and
        # also lets IndexItem._extract_entities() pick it up as a
        # high-confidence exact-match entity (it looks for an "owner" key).
        owner_name = rec.get("owner")

        # ── Section: overview (LEAN summary, not the raw record) ───────────
        overview_data: Dict[str, Any] = {
            "name": rec.get("name"),
            "description": rec.get("description"),
            "status": rec.get("status"),
            "owner": owner_name,
            "owner_scope_note": (
                f"{owner_name} owns/promotes Godfrey Phillips India (GPI) only. "
                "Philip Morris International (PMI) is a separate, independently "
                "owned, publicly traded company (PMI holds a 25.1% stake IN GPI, "
                "not the reverse)."
            ) if owner_name else None,
            "headquarters": gpi.get("headquarters"),
            "incorporation_date": gpi.get("incorporation_date"),
        }
        if leadership:
            overview_data["leadership"] = leadership
        if metrics:
            overview_data["key_metrics"] = {
                k: v for k, v in metrics.items()
                if k in ("gpi_revenue", "gpi_profit_after_tax", "gpi_currency", "pmi_revenue_usd")
            }
        if products:
            overview_data["product_categories"] = list(products.keys())
        overview_data = {k: v for k, v in overview_data.items() if v not in (None, "", [], {})}
        self.items.append(IndexItem("company", overview_data, section="overview"))

        # ── Section: description ──────────────────────────────────────────
        if rec.get("description"):
            self.items.append(IndexItem(
                "company",
                {"name": rec.get("name"), "description": rec["description"]},
                section="description",
            ))

        # ── Section: history / founding ───────────────────────────────────
        history_data: Dict[str, Any] = {}
        for key in ("history", "origin", "founding", "background"):
            if ai.get(key):
                history_data[key] = ai[key]
        if gpi.get("incorporation_date"):
            history_data["incorporation_date"] = gpi["incorporation_date"]
        if gpi.get("headquarters"):
            history_data["headquarters"] = gpi["headquarters"]
        if gpi.get("ownership"):
            history_data["ownership"] = gpi["ownership"]
        if owner_name:
            history_data["owner"] = owner_name
        if history_data:
            # This history/incorporation/ownership data is GPI's specifically
            # (PMI is a separate, much older multinational with its own
            # history) — name it explicitly rather than the combined
            # "GPI & PMI" record name to avoid the two getting conflated.
            history_data["name"] = "Godfrey Phillips India (GPI)"
            history_data["description"] = rec.get("description")
            self.items.append(IndexItem("company", history_data, section="history"))

        # ── Section: leadership ───────────────────────────────────────────
        # This leadership roster (CEO/CFO/COO/chairperson) is GPI's, not
        # PMI's — PMI has its own separate leadership not captured in this
        # dataset. Naming it explicitly "GPI" (instead of the combined
        # "GPI & PMI" record name) stops the LLM from presenting these
        # titles, or GPI's owner, as also applying to PMI.
        if leadership:
            leadership_data = {"name": "Godfrey Phillips India (GPI)", "leadership": leadership,
                                **{k: v for k, v in leadership.items()}}
            if owner_name:
                # Kept distinct from chairperson_md/ceo/etc: the person who
                # OWNS the company is not automatically the same as whoever
                # holds an executive title, and the model should be able to
                # tell them apart instead of defaulting to a title.
                leadership_data["owner"] = owner_name
                leadership_data["note"] = (
                    f"{owner_name} is the owner/promoter of GPI specifically. "
                    "Philip Morris International (PMI) is a separate, "
                    "independently owned company, not owned by this person."
                )
            self.items.append(IndexItem("company", leadership_data, section="leadership"))

        # ── Section: financials ───────────────────────────────────────────
        if metrics:
            self.items.append(IndexItem(
                "company",
                {"name": rec.get("name"), **metrics},
                section="financials",
            ))

        # ── Section: products ─────────────────────────────────────────────
        if products:
            self.items.append(IndexItem(
                "company",
                {"name": rec.get("name"), "products_services": products},
                section="products",
            ))

        # ── Section: competitors ──────────────────────────────────────────
        competitors = ai.get("competitors", [])
        if competitors:
            self.items.append(IndexItem(
                "company",
                {"name": rec.get("name"), "competitors": competitors},
                section="competitors",
            ))

        # ── Section: mission / vision ─────────────────────────────────────
        mv: Dict[str, Any] = {"name": rec.get("name")}
        for key in ("mission", "vision", "values", "purpose"):
            val = ai.get(key) or gpi.get(key) or rec.get(key)
            if val:
                mv[key] = val
        if len(mv) > 1:
            self.items.append(IndexItem("company", mv, section="mission_vision"))

        # ── Section: headquarters / location ──────────────────────────────
        hq = gpi.get("headquarters")
        if hq:
            self.items.append(IndexItem(
                "company",
                {"name": rec.get("name"), "headquarters": hq},
                section="location",
            ))

        # ── Section: ownership ────────────────────────────────────────────
        # "owner" (Dr. Bina Modi) and the shareholding breakdown both apply
        # to GPI specifically, never to PMI — PMI is a separate, publicly
        # traded company and is one of GPI's shareholders (25.1%), not the
        # other way round. Scoping "name" to GPI explicitly (instead of the
        # combined "GPI & PMI" record name) and carrying the owner_note
        # keeps the LLM from attributing GPI's owner to PMI as well.
        ownership = gpi.get("ownership", {})
        if ownership or owner_name:
            ownership_data: Dict[str, Any] = {"name": "Godfrey Phillips India (GPI)"}
            if owner_name:
                ownership_data["owner"] = owner_name
            if ownership:
                ownership_data["shareholding"] = ownership
            if gpi.get("owner_note"):
                ownership_data["note"] = gpi["owner_note"]
            self.items.append(IndexItem("company", ownership_data, section="ownership"))

        # ── Section: metadata (additional fields) ─────────────────────────
        meta_fields = ai.get("metadata_fields", [])
        if meta_fields:
            meta_dict = {mf["field"]: mf["value"] for mf in meta_fields if "field" in mf and "value" in mf}
            if meta_dict:
                meta_dict["name"] = rec.get("name")
                self.items.append(IndexItem("company", meta_dict, section="metadata"))

        # ── Dynamic: index any ai sub-dict not yet covered ────────────────
        covered = {
            "gpi_details", "products_services", "competitors",
            "mission", "vision", "values", "history", "metadata_fields",
            "origin", "founding", "background",
        }
        # Sections that describe PMI specifically, not the combined record —
        # tagging them with PMI's own name (instead of the ambiguous
        # "GPI & PMI" combined name) stops the LLM from reading GPI-only
        # facts (like the owner) across into a PMI-scoped section.
        _pmi_only_keys = {"pmi_global_strategy", "pmi_strategic_segments"}
        for key, val in ai.items():
            if key in covered:
                continue
            if isinstance(val, (dict, list)) and val:
                entity_name = (
                    "Philip Morris International (PMI)"
                    if key in _pmi_only_keys
                    else rec.get("name")
                )
                self.items.append(IndexItem(
                    "company",
                    {"name": entity_name, key: val},
                    section=key,
                ))

    # ── embedding ──────────────────────────────────────────────────────────────

    def _embed_items(self) -> None:
        """
        Compute (or load from cache) the semantic embedding for every
        indexed item. Safe to call repeatedly: embed() is itself cached,
        so re-running this after a reindex only embeds genuinely new or
        changed records. Uses embed_many() so any items not already
        cached are embedded via as few batched HTTP requests as possible
        rather than one request per item.
        """
        pending = [i for i, item in enumerate(self.items) if item.embedding is None]
        if pending:
            vectors = embed_many([self.items[i].text for i in pending])
            for i, vector in zip(pending, vectors):
                self.items[i].embedding = vector
        flush_disk_cache()

    def reindex(self) -> None:
        """Rebuild the index and recompute embeddings from scratch."""
        self._build()
        self._embed_items()

    def reload(self) -> None:
        """Alias for reindex() — import_documents.py calls this by this name."""
        self.reindex()

    # ── lexical fallback (used only when embeddings are unavailable) ──────────

    @staticmethod
    def _expand(query: str) -> str:
        words = query.lower().split()
        expanded = list(words)
        for w in words:
            if w in SYNONYMS:
                expanded.extend(SYNONYMS[w])
        return " ".join(expanded)

    def _lexical_score(
        self, query_tokens: set, query_lower: str, item: IndexItem,
    ) -> float:
        """Token overlap + fuzzy partial ratio. Degraded-mode scorer only."""
        score: float = 0.0
        overlap = len(query_tokens & item.tokens)
        score += overlap * 120
        score += fuzz.partial_ratio(query_lower, item.text[:500]) * 0.8
        return score

    # ── scoring ───────────────────────────────────────────────────────────────

    def _score(
        self,
        query: str,
        query_embedding: Optional[List[float]],
        query_tokens: set,
        query_lower: str,
        item: IndexItem,
        section_hint: Optional[str],
    ) -> float:
        """
        Primary signal: cosine similarity between the query embedding and
        the item's embedding, scaled to roughly the same range the rest of
        the scoring system expects (0-1000+). Falls back to lexical scoring
        only when either embedding is unavailable.
        """
        used_semantic = query_embedding is not None and item.embedding is not None

        if used_semantic:
            similarity = cosine_similarity(query_embedding, item.embedding)
            score = similarity * 1000.0
        else:
            score = self._lexical_score(query_tokens, query_lower, item)

        # Precise exact-name match bonus. This is identifying a specific
        # named entity the user typed, not a keyword category lookup, so
        # it remains useful even in a semantic-first system.
        for ent in item.entities:
            if ent in query_lower:
                score += 250

        # Hard section_hint match: boost if item matches the resolved hint.
        # Not restricted to item.category == "company" — imported document
        # chunks can carry a category of "department" (or others) with a
        # matching section label too, and they deserve the same boost.
        if section_hint and item.section == section_hint:
            score += 400

        # Curated, hand-maintained JSON facts should win over a lower-
        # confidence document extraction (OCR/vision) when both match the
        # same section — otherwise a noisy slide description can outscore
        # and bump out the authoritative company/department record.
        if item.record.get("source_type") == "document":
            score *= 0.85

        # Down-weight "overview" only when a different, more specific
        # section_hint is requested, so a focused question doesn't get
        # swamped by the broad summary section.
        if (
            section_hint
            and section_hint != "overview"
            and item.category == "company"
            and item.section == "overview"
        ):
            score *= 0.6

        # Financial-statement pages actually containing a real, structured
        # figures table (P&L / balance sheet / cash flow) vs. a page that
        # merely mentions "revenue" or "financial" in passing -- see
        # fix_processed_data.py, which sets this flag. Restricted to
        # records with a specific, named company: a company=None chunk
        # (e.g. a general market-research compilation citing secondary
        # sources like news trackers) can ALSO get flagged by the same
        # heading/number heuristic, but it isn't a primary filing and its
        # figures are frequently rounded/approximate/re-quoted -- it
        # should never be able to outrank or crowd out a company's own
        # actual annual-report numbers for a company-specific question.
        if (
            item.record.get("is_highlight")
            and item.record.get("company") is not None
            and section_hint in ("financials", "competitor_finances", None)
        ):
            score += 350

        # Generic literal-token boost: if a capitalized, name-shaped word
        # from the ORIGINAL (non-lowercased) query -- e.g. "Delhi" in
        # "talk about the Delhi branch" -- appears verbatim in this
        # item's own text, boost it regardless of how well the query
        # embeds semantically overall. This matters most for structured
        # data ingested from spreadsheets (sales registers, distributor
        # lists, ...): a row-batch chunk is 100 rows of otherwise
        # unrelated entries, so its OVERALL semantic similarity to a
        # question naming one specific row's value can be weak even
        # though the literal answer is sitting right there in the text.
        # Deliberately generic (no company/field names hardcoded) so it
        # applies equally to any proper noun in any ingested table.
        for word in re.findall(r"[A-Z][a-zA-Z]{2,}", query):
            if word in item.text:
                score += 300
                break

        return score

    # ── search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        category_filter: Optional[str] = None,
        section_hint: Optional[str] = None,
        max_per_section: Optional[int] = None,
        max_document_per_section: Optional[int] = None,
        company_filter: Optional[str] = None,
        year_filter: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the top_k most relevant IndexItems for *query*, ranked
        primarily by semantic similarity (embedding cosine similarity).

        If the embedding model is unreachable for this query, every item
        degrades to the lexical fallback scorer automatically — retrieval
        keeps working, just with the older, less semantically-aware logic,
        and never raises or returns nothing purely due to that outage.

        ``max_per_section`` / ``max_document_per_section``: override the
        per-(category, section) result cap. Kept as two separate knobs
        (see core.config.MAX_PER_SECTION / MAX_DOCUMENT_PER_SECTION)
        because curated JSON records and imported document chunks behave
        very differently — a single ingested PDF can legitimately produce
        many chunks that all land in the same section (e.g. three years
        of a competitor report all tagged "competitor_finances"), and a
        cap sized for "one curated record per section" would silently
        discard almost all of that multi-year content. Callers doing a
        broad/aggregate query (compare, table, timeline, "how much data
        do we have") should pass a much larger value for both.

        ``company_filter``: restrict results to one detected company (see
        _detect_company). "GPI" matches curated company-profile records
        (category=="company" with no explicit company tag, i.e. the
        assistant's own data) PLUS any document chunk explicitly detected
        as GPI's own report. Any other value matches only document chunks
        tagged with that exact company. This exists so a multi-entity
        comparison (e.g. "compare VST, ITC, and GPI") can retrieve each
        company's data as its own separately-ranked pool instead of one
        flat ranked list where one competitor's chunks can crowd out
        another's entirely.

        ``year_filter``: restrict document chunks to one detected fiscal
        year (see _detect_year), e.g. "2023". Items with NO detected year
        (curated JSON records, or a document chunk where no year could be
        confidently identified) always pass this filter regardless of its
        value — a year filter should never hide undated-but-relevant
        content, only prefer/isolate content that's confidently dated to
        something else. Without this, a query naming one specific year
        has no way to avoid retrieval returning a different year's data
        that simply scored higher semantically.

        ``query_embedding``: reuse an already-computed embedding instead
        of calling the (network-bound) embedding model again for *query*.
        A caller doing several near-duplicate searches per turn — e.g.
        one per (company, year) bucket for a balanced multi-entity/
        multi-year comparison — previously triggered one Ollama round-trip
        PER bucket even though those variant strings only differ by a
        short "(ITC specifically, year 2024)" suffix and rank almost
        identically. Computing the embedding once from the plain question
        and passing it into every bucket's search() call cuts N round-
        trips down to 1 without changing which items get returned, since
        company/year precision already comes from company_filter/
        year_filter and the exact-entity-name score bonus, not from the
        embedding text itself.
        """
        query_embedding = query_embedding if query_embedding is not None else embed(query)

        # Lexical fallback path still benefits from synonym expansion,
        # since it has no semantic understanding of its own.
        expanded   = self._expand(query)
        q_tokens   = tokenize(expanded)
        q_lower    = expanded.lower()

        scored = []
        for item in self.items:
            if category_filter and item.category != category_filter:
                continue
            if company_filter:
                item_company = item.record.get("company")
                if company_filter == "GPI":
                    if item_company not in (None, "GPI"):
                        continue
                elif item_company != company_filter:
                    continue
            if year_filter:
                item_year = item.record.get("year")
                if item_year is not None and str(item_year) != str(year_filter):
                    continue
            sc = self._score(query, query_embedding, q_tokens, q_lower, item, section_hint)

            threshold = (
                EMBEDDING_MIN_SIMILARITY * 1000.0
                if (query_embedding is not None and item.embedding is not None)
                else RETRIEVAL_MIN_SCORE
            )
            if sc >= threshold:
                scored.append((sc, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        if RERANKER_ENABLED:
            # Same local cross-encoder reranker used by the PDF
            # retriever (see retrieval/reranker.py). Scores here run on
            # a ~0-1000+ scale (see _score()'s `similarity * 1000.0`),
            # not pdf_retriever's ~0-1 cosine scale, so the blend weight
            # is scaled up to match -- otherwise a 0-1 nudge would be
            # numerically invisible against these scores. Falls back to
            # the untouched `scored` list automatically if the reranker
            # isn't installed/available.
            scored = _rerank_candidates(
                query, scored,
                text_fn=lambda item: item.text,
                top_n=RERANKER_TOP_N, weight=RERANKER_WEIGHT * 1000.0,
            )

        results = []
        seen_curated: Dict[str, int] = {}
        seen_document: Dict[str, int] = {}
        # Curated JSON: normally one authoritative record per section, so a
        # small cap just guards against near-duplicates. Document chunks
        # get their own, much larger cap so multi-page/multi-year PDF
        # content sharing a section isn't starved down to a couple of
        # pages — see the docstring above and core.config for the why.
        cap_curated  = MAX_PER_SECTION if max_per_section is None else max_per_section
        cap_document = MAX_DOCUMENT_PER_SECTION if max_document_per_section is None else max_document_per_section

        for sc, item in scored[:top_k * 3]:
            sig = f"{item.category}:{item.section}"
            is_document = item.record.get("source_type") == "document"
            seen = seen_document if is_document else seen_curated
            cap  = cap_document if is_document else cap_curated
            if seen.get(sig, 0) >= cap:
                continue
            seen[sig] = seen.get(sig, 0) + 1
            results.append({
                "score":    sc,
                "category": item.category,
                "section":  item.section,
                "record":   item.record,
                "text":     item.text,
            })
            if len(results) >= top_k:
                break

        return results

    # ── convenience helpers ───────────────────────────────────────────────────

    def available_years(self, company_filter: Optional[str] = None) -> List[str]:
        """
        Return the distinct fiscal years actually present in the index
        (as detected by _detect_year), optionally scoped to one company.

        Used by callers building a multi-year aggregate request ("past 3
        years", "trend", "year on year") that doesn't name specific years
        explicitly — without this, the caller has no way to know WHICH
        years exist to balance retrieval across, and a flat top-k search
        can silently let one year's chunks (e.g. a single annual report
        with the most semantically-similar wording) crowd out every other
        year, even though several other years were actually ingested.
        """
        years: set = set()
        for item in self.items:
            year = item.record.get("year")
            if not year:
                continue
            if company_filter:
                item_company = item.record.get("company")
                if company_filter == "GPI":
                    if item_company not in (None, "GPI"):
                        continue
                elif item_company != company_filter:
                    continue
            years.add(year)
        return sorted(years, reverse=True)

    def get_all_by_category(
        self, category: str, company_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return all indexed records for a given category.

        ``company_filter``: same semantics as KnowledgeIndex.search() —
        "GPI" matches curated records (no company tag) plus any chunk
        explicitly detected as GPI's; any other value matches only chunks
        tagged with that exact company. Without this, a plain "what
        departments do we have" question — which bypasses relevance
        ranking entirely to return the FULL department category — had no
        way to exclude a competitor's annual-report page that happened to
        get guessed into the "department" category (e.g. it mentioned an
        "HR department" or "finance division" in passing), so a stray
        ingested competitor chunk could sit right next to the curated
        departments.json records with nothing to tell them apart.
        """
        results = []
        for item in self.items:
            if item.category != category:
                continue
            if company_filter:
                item_company = item.record.get("company")
                if company_filter == "GPI":
                    if item_company not in (None, "GPI"):
                        continue
                elif item_company != company_filter:
                    continue
            results.append({
                "score": 999, "category": item.category, "section": item.section,
                "record": item.record, "text": item.text,
            })
        return results

    @property
    def records(self) -> List[Dict[str, Any]]:
        """Legacy-compatible property: returns all items as result dicts."""
        return [
            {"score": 0, "category": item.category, "section": item.section,
             "record": item.record, "text": item.text}
            for item in self.items
        ]


def detect_years_in_question(question: str) -> List[str]:
    """
    Return distinct fiscal/reporting years explicitly named in a
    question (e.g. "2023", "FY24", "2022-23"), normalized the same way
    as document year-tagging (see KnowledgeIndex._detect_year), in the
    order they appear. Deliberately does NOT try to resolve relative
    phrases like "the same year" or "last year" — that needs the
    previous turn's resolved year from conversation memory, which is
    handled by the caller, not this pure text-scanning function.

    Also supports relative phrases like "past 4 years", "last 3 years",
    "past N years", "last N years" which are resolved relative to the
    current year.
    """
    years = []
    for m in KnowledgeIndex._YEAR_PATTERN.finditer(question):
        year = KnowledgeIndex._normalize_year_match(m)
        if year and year not in years:
            years.append(year)

    # Handle relative year phrases like "past 4 years", "last 3 years", etc.
    if not years:
        import re
        from datetime import datetime
        current_year = datetime.now().year
        # Match patterns like "past 4 years", "last 3 years", "past 1 year", "last 1 year"
        relative_patterns = [
            r"\bpast\s+(\d+)\s+years?\b",
            r"\blast\s+(\d+)\s+years?\b",
            r"\bprevious\s+(\d+)\s+years?\b",
        ]
        for pattern in relative_patterns:
            m = re.search(pattern, question.lower())
            if m:
                n = int(m.group(1))
                years.extend([str(current_year - i) for i in range(1, n + 1)])
                break

    return years


def detect_companies_in_question(question: str) -> List[str]:
    """
    Return the canonical company tags (see KnowledgeIndex._COMPANY_ALIASES)
    named in a question, in the order they were matched. Used to decide
    whether a query needs per-entity balanced retrieval (e.g. "compare
    VST, ITC, and GPI") rather than one flat ranked search.
    """
    q_lower = question.lower()
    found = []
    for company, aliases in KnowledgeIndex._COMPANY_ALIASES.items():
        if any(KnowledgeIndex._alias_matches(q_lower, alias) for alias in aliases):
            found.append(company)
    return found


# ─── Module-level singleton ───────────────────────────────────────────────────

index = KnowledgeIndex()