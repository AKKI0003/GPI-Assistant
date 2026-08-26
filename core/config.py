"""
core/config.py

Central configuration for GPI Assistant.
All paths, model settings, and tuning parameters live here.
"""

import os

# ─── Base Paths ──────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ─── Signup department list (auth/store.py, static/register.html) ─────────
# THIS is "where to change or add" the department dropdown shown at
# registration -- a plain list, edit it directly. It's intentionally
# separate from data/departments.json (that file holds KNOWLEDGE-BASE
# content -- descriptions/projects the chatbot can answer questions
# about) -- this list is only ever used for the signup dropdown and for
# matching a file's allowed_departments during access-control checks
# (see auth/permissions.py). Renaming/removing an entry here does NOT
# retroactively change any existing user's stored department or any
# file's existing permissions -- both are stored as plain strings, so
# keep names consistent if you rename one after people have already
# registered/uploaded against it.
SIGNUP_DEPARTMENTS = [
    "Sales",
    "Marketing",
    "Finance",
    "HR",
    "Operations",
    "IT",
    "Legal",
    "Procurement",
    "R&D",
    "Executive",
]

# ─── Data Files ──────────────────────────────────────────────────────────────

COMPANY_FILE     = os.path.join(DATA_DIR, "company_overview.json")
DEPARTMENTS_FILE = os.path.join(DATA_DIR, "departments.json")
PROJECTS_FILE    = os.path.join(DATA_DIR, "projects.json")
PROCESSES_FILE   = os.path.join(DATA_DIR, "processes.json")
PROBLEMS_FILE    = os.path.join(DATA_DIR, "problems.json")
KPIS_FILE        = os.path.join(DATA_DIR, "kpis.json")
INTENT_DICT_FILE = os.path.join(DATA_DIR, "intent_dictionary.json")

CHAT_DIR         = os.path.join(DATA_DIR, "chats")
CHAT_FILE        = os.path.join(CHAT_DIR, "chat_sessions.json")
CURRENT_CHAT_FILE = os.path.join(CHAT_DIR, "current_chat.txt")

# ─── Document Ingestion (PPT / PDF / chart / image imports) ───────────────────
# Populated by import_documents.py, consumed by retrieval/knowledge_index.py's
# _load_documents(). These MUST both point at the same file, or ingested
# documents will be written but never actually retrievable.
DOCUMENTS_DIR         = os.path.join(BASE_DIR, "documents")
# Legacy: the original single combined file. Still supported for reading
# (if it exists) so nobody's existing ingested data disappears, but
# import_documents.py no longer writes to it — see DOCUMENTS_PROCESSED_DIR.
DOCUMENTS_CHUNKS_FILE = os.path.join(DOCUMENTS_DIR, "processed_chunks.json")
# One JSON file per ingested source document (named after it), e.g.
# documents/processed/VST_Annual_Report_2023.json. Replaces the single
# ever-growing combined file: with dozens of PDFs, that file grew past
# 40,000+ lines, which made it slow and unwieldy to inspect, and meant
# every single ingestion run had to rewrite the ENTIRE thing. Per-file
# storage means re-ingesting one document only touches its own small
# file, any new document dropped into documents/raw/ is picked up
# automatically on the next `python import_documents.py` run with zero
# code changes (nothing here hardcodes a filename), and you can open one
# file to check exactly what was extracted from one specific PDF instead
# of searching through everything at once.
DOCUMENTS_PROCESSED_DIR = os.path.join(DOCUMENTS_DIR, "processed")

# Chroma collection names for the ingest/ingest.py -> knowledge_index
# pipeline's document chunks. This replaces documents/processed/*.json
# and the legacy documents/processed_chunks.json as the source of truth
# -- storage is now one incrementally-updatable record per chunk instead
# of one JSON file rewritten whole per document, and it lives in the SAME
# on-disk Chroma database as the PDF-intelligence store below (one
# PersistentClient, two collections) rather than a second separate
# database directory. DOCUMENTS_PROCESSED_DIR/DOCUMENTS_CHUNKS_FILE above
# are kept only as a one-time migration source (see
# knowledge_index._migrate_legacy_documents_to_chroma) and can be deleted
# once you've confirmed the app is reading correctly from Chroma.
DOCUMENTS_CHROMA_CHUNKS_COLLECTION = "documents_knowledge_chunks"
DOCUMENTS_CHROMA_META_COLLECTION  = "documents_knowledge_meta"

# ─── PDF Document Intelligence knowledge source (ingestion/ package) ──────────
# This is a SECOND, fully independent knowledge source that sits alongside
# (never replaces) the JSON-based KnowledgeIndex above and the simpler
# ingest/ingest.py -> DOCUMENTS_PROCESSED_DIR pipeline. It is powered by the
# ingestion/ package (ingestion.PDFIngestor), which preserves document
# structure (sections, tables, images) instead of flattening a PDF into raw
# text. Drop PDFs into PDF_KNOWLEDGE_RAW_DIR; retrieval/pdf_retriever.py picks
# them up automatically (checksummed, so an unchanged file is never
# re-ingested) and merges its results into the same prompt context the JSON
# retriever builds -- see AssistantEngine._retrieve() in core/engine.py.
PDF_KNOWLEDGE_RAW_DIR      = os.path.join(DOCUMENTS_DIR, "pdf_knowledge", "raw")
PDF_KNOWLEDGE_IMAGES_DIR   = os.path.join(DOCUMENTS_DIR, "pdf_knowledge", "extracted_images")
PDF_KNOWLEDGE_CHROMA_DIR   = os.path.join(DOCUMENTS_DIR, "pdf_knowledge", "chroma_db")
PDF_KNOWLEDGE_TOP_K        = int(os.getenv("PDF_KNOWLEDGE_TOP_K", "8"))
PDF_KNOWLEDGE_MIN_SIMILARITY = float(os.getenv("PDF_KNOWLEDGE_MIN_SIMILARITY", "0.30"))

# ─── Ollama Settings ──────────────────────────────────────────────────────────

OLLAMA_ENDPOINT  = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TIMEOUT   = int(os.getenv("OLLAMA_TIMEOUT", "900"))
# Reserved for a future LLM-ASSISTED ingestion pass (e.g. an extra check
# on ambiguous statement-heading/table labeling that regex alone can't
# resolve confidently). Ingestion today is entirely deterministic/regex-
# based (see ingestion/document.py) and does NOT currently call this --
# nothing reads this constant yet, so it's independent of OLLAMA_MODEL
# above: changing that one has ZERO effect on ingestion quality/labeling.
INGEST_OLLAMA_MODEL = os.getenv("INGEST_OLLAMA_MODEL", "nemotron-3-nano:4b")

# ─── Model choice, now that this runs on a real GPU (RTX 3050, 6GB VRAM) ───
# Nemotron 3 Nano 4B was sized for a CPU-only company laptop -- with an
# actual GPU now in the picture, a 6GB card comfortably fits a real 7-8B
# model fully in VRAM at Q4_K_M (~4.5-5GB weights), which is a genuine
# reasoning/structured-output step up, not just a bigger number: this
# app's own JSON spec-generation step (core/data_analyst.py's outlier/
# aggregate spec, the chart-extraction spec) leans hard on the model
# reliably producing strict, well-formed JSON, which is exactly where a
# 4B model is weakest and a 7-8B model is noticeably more reliable.
#
#   ollama pull qwen3:8b        (~5.2GB at Q4_K_M) -- current default.
#   Dense 8B with an explicit "thinking" mode (already forced off by
#   default here -- see OLLAMA_THINK below -- specifically because
#   thinking mode burns extra generation time for no visible benefit on
#   most of this app's questions), strong instruction-following and
#   structured-JSON reliability, comfortably GPU-resident on 6GB.
#
# One real constraint at 6GB specifically: LLM_CTX_WINDOW below is 32768
# tokens by default, and the KV cache for a long context competes with
# the model's own weights for the SAME 6GB. If you notice generation
# slow down a lot on long, table-heavy answers (a sign Ollama is
# spilling some layers to CPU RAM to make room), lower LLM_CTX_WINDOW --
# e.g. LLM_CTX_WINDOW=12288 -- to keep everything GPU-resident; check
# with `nvidia-smi` while a query is running to see whether VRAM is
# actually maxed out before assuming this is needed.
#
# `ollama pull <model>` it once, then set OLLAMA_MODEL to switch --
# nothing else in the app needs to change.

# ─── Answer quality tuning ─────────────────────────────────────────────────
# Two-pass self-review: after the first draft, ask the model to check its
# own answer against the context and tighten it (catches unsupported
# claims, redundancy, disorganized structure) before returning it. This is
# the single biggest quality lever available in software here — but it
# roughly doubles latency, so it's OFF by default and only applies to
# detail/compare/table answers (the ones already budgeted for a long
# OLLAMA_TIMEOUT_TABLE wait), never to quick factual lookups.
SELF_REFINE = os.getenv("SELF_REFINE", "0") == "1"

# The "direct" statement-table route (core/direct_answer.py:
# try_statement_table_answer) tries to stitch multiple PDF pages of the
# same statement into one rigid Markdown grid by padding/truncating every
# row to the FIRST chunk's column count. Different pages of the physically
# same statement routinely extract with a genuinely different column count
# (pdfplumber's own per-page table detection isn't perfectly consistent),
# so that padding/truncation misaligns cells across a page boundary --
# random-looking column lines and mid-cell text cut-offs. This was
# confirmed to be specific to this route: the exact same underlying data
# rendered cleanly when it went through the normal LLM route instead
# (the LLM reconstructs the table in its own words rather than forcing a
# literal cell-by-cell grid merge). Default OFF until the grid-stitcher is
# made robust to per-page column-count drift; statement questions fall
# through to the LLM route below, which already receives the same broad,
# multi-page retrieval context (see _wants_broad_context / table_requested)
# so it has the same level of size/detail to work from.
STATEMENT_TABLE_DIRECT_ROUTE = os.getenv("STATEMENT_TABLE_DIRECT_ROUTE", "0") == "1"

# Table/comparison requests deliberately use a much larger context
# (CONTEXT_MAX_CHARS_TABLE) and a much larger output budget
# (LLM_MAX_TOKENS_TABLE) below. On a small CPU-bound local model, prefill
# time scales with input size and generation time scales with output
# size — a detailed multi-year comparison can legitimately take several
# minutes end-to-end. OLLAMA_TIMEOUT (above) is tuned for normal
# questions; without a separate, longer timeout here, the HTTP client can
# give up and raise a read-timeout WHILE Ollama is still genuinely
# working, which looks identical to a hang from the user's side. Increase
# this further if comparisons still time out on slower hardware.
OLLAMA_TIMEOUT_TABLE = int(os.getenv("OLLAMA_TIMEOUT_TABLE", "900"))

# Embedding model used for semantic retrieval (see retrieval/embedding_store.py).
# Must be pulled once on the host running Ollama: `ollama pull nomic-embed-text`.
OLLAMA_EMBED_MODEL  = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_EMBED_TIMEOUT = int(os.getenv("OLLAMA_EMBED_TIMEOUT", "60"))

LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.0"))   # 0 = greedy/deterministic, eliminates creative drift
LLM_MAX_TOKENS   = int(os.getenv("LLM_MAX_TOKENS", "3500"))
# Generation time on a CPU-bound local model scales almost linearly with
# this number — it is usually the single biggest lever over wall-clock
# answer time, bigger than context size. 4500 was sized for a worst-case
# 8-11 row multi-attribute table, but most table/compare answers finish
# well under 3000 tokens; a request that genuinely needs more can still
# get it (this is just the ceiling, not a target the model is pushed
# toward). Raise back up via the env var if answers are getting cut off.
LLM_MAX_TOKENS_TABLE = int(os.getenv("LLM_MAX_TOKENS_TABLE", "6000"))
# Raised from 16384: full financial statements (P&L, balance sheet) plus a
# wide CONTEXT_MAX_CHARS_TABLE retrieval budget were together pushing right
# up against — and sometimes past — the old window, which is exactly what
# silently truncated statements to a "portion" of their real line items.
# 32768 leaves real headroom for a large retrieved context AND a full
# LLM_MAX_TOKENS_TABLE-sized output within the same shared window.
LLM_CTX_WINDOW   = int(os.getenv("LLM_CTX_WINDOW", "32768"))  # input + output share this window
LLM_TOP_K        = int(os.getenv("LLM_TOP_K", "1"))            # top-1 = greedy, no sampling randomness
LLM_TOP_P        = float(os.getenv("LLM_TOP_P", "1.0"))        # disabled (irrelevant when top_k=1)
LLM_REPEAT_PENALTY = float(os.getenv("LLM_REPEAT_PENALTY", "1.05"))
# Ollama's own thread auto-detection is sometimes conservative on
# Windows/WSL. An i5-11th-gen is typically 4 cores / 8 threads -- pin
# this explicitly so CPU inference actually uses them. Override with the
# env var if you've confirmed a different value works better for your
# exact CPU.
LLM_NUM_THREAD     = int(os.getenv("LLM_NUM_THREAD", "8"))
# qwen3 (the new OLLAMA_MODEL default) and a few other recent models
# support an explicit "thinking" mode: before writing its real answer,
# the model first generates a separate, often LONG chain-of-thought
# block that never gets shown to the user but still costs real
# generation time on a CPU -- exactly the cost this switch to a smaller
# model was meant to cut. Ollama's /api/generate accepts a top-level
# "think" flag to turn this off outright for models that support it (a
# no-op, harmless flag for models that don't). Default OFF for speed;
# set OLLAMA_THINK=1 if you'd rather trade some of that speed back for
# the model reasoning more carefully step-by-step on harder questions.
OLLAMA_THINK       = os.getenv("OLLAMA_THINK", "0") == "1"

# ─── Retrieval Settings ───────────────────────────────────────────────────────

RETRIEVAL_TOP_K           = 8   # Max results returned from search
RETRIEVAL_MIN_SCORE       = 50  # Minimum score threshold (lexical fallback path only)

# How many items sharing the same (category, section) are allowed into a
# single result set. Kept separate for curated JSON facts vs. imported
# document chunks, because they behave very differently:
#   - Curated JSON: normally exactly one authoritative record per section,
#     so a small cap is fine and mostly just guards against duplicates.
#   - Imported documents (PDF/PPTX pages): a single multi-year report can
#     legitimately produce many chunks that all land in the SAME section
#     (e.g. three years of a "VST" competitor PDF all tag as
#     "competitor_finances"). A small cap here silently throws away
#     everything past the first couple of pages/years, which is why
#     "how much VST data do we have" or "compare VST over 3 years" used
#     to fail even though all the data had been ingested. This default is
#     intentionally generous; broad queries (compare/table/timeline/"how
#     much"/"all") raise it further via the `max_document_per_section`
#     argument to KnowledgeIndex.search().
MAX_PER_SECTION            = 2
MAX_DOCUMENT_PER_SECTION   = 6
CONTEXT_MAX_CHARS         = 8000   # Max context chars sent to LLM (default)
# NOTE: lowered from 20000 -> 12000. On CPU-only hardware, prefill time
# scales close to linearly with this number, and it was the single
# biggest contributor to 500-900s answers on broad/table queries (a
# "compare 3 companies" question was round-robining in chunks from ALL
# ingested PDFs up to a 20k-char budget before generation even started).
# 12000 is still generous relative to RETRIEVAL_TOP_K*3 / round-robin
# coverage; raise back up only if answers start getting visibly
# truncated on legitimately wide comparisons.
CONTEXT_MAX_CHARS_TABLE   = 22000  # Wider budget for table/compare/full-list requests,
                                    # which legitimately need more records to be
                                    # complete — e.g. a balanced 3-way company
                                    # comparison needs real room for each entity's
                                    # data, not just one. Trimmed down from an
                                    # earlier 32000: prefill time on a CPU-bound
                                    # local model scales with this size, and the
                                    # per-company/per-year round-robin retrieval
                                    # (see core/engine.py) already guarantees fair
                                    # coverage across entities/years well within
                                    # this budget, so the extra headroom was mostly
                                    # just added prefill latency, not more complete
                                    # answers.
MEMORY_MAX_HISTORY        = 10
MEMORY_CONTEXT_TURNS      = 3   # How many prior turns to include in prompt

# ─── Semantic Embedding Settings ──────────────────────────────────────────────
# Primary retrieval signal: cosine similarity between the query embedding and
# each knowledge record's embedding (computed via OLLAMA_EMBED_MODEL). This is
# what lets a query like "the firm's beginnings" match a "history" record even
# though the two share almost no literal words — keyword/fuzzy matching alone
# cannot do this, and gaps in keyword coverage are what let the LLM fall back
# on hallucinated facts. Lexical scoring (token overlap + fuzzy string ratio)
# is retained only as a fallback for when the embedding model is unreachable.

EMBEDDING_MIN_SIMILARITY  = float(os.getenv("EMBEDDING_MIN_SIMILARITY", "0.30"))

# ─── Local cross-encoder reranker (retrieval/reranker.py) ─────────────────────
# Optional precision boost applied to the top candidates AFTER the usual
# embedding-similarity ranking, for both the JSON knowledge index and the
# PDF retriever. Fully local (sentence-transformers, runs on-device) --
# disable entirely with RERANKER_ENABLED=false if you'd rather skip the
# extra ~90MB model download / inference cost. No effect on results if
# sentence-transformers isn't installed; retrieval just silently keeps
# using pure embedding-similarity ranking.
#
# On a machine with no Hugging Face access (e.g. a locked-down corporate
# laptop), set RERANKER_MODEL_NAME to the absolute local folder path you
# copied the model files into instead of the Hugging Face repo id, e.g.:
#   RERANKER_MODEL_NAME=C:\Users\you\models\ms-marco-MiniLM-L-6-v2
# sentence-transformers accepts a local folder path exactly like it does
# a repo id -- no other code changes needed. retrieval/reranker.py already
# forces offline mode (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE) before that
# folder is ever loaded, so it won't attempt to contact huggingface.co
# either way.
RERANKER_ENABLED     = os.getenv("RERANKER_ENABLED", "true").lower() != "false"
RERANKER_MODEL_NAME  = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_TOP_N        = int(os.getenv("RERANKER_TOP_N", "40"))   # how many top candidates get reranked
RERANKER_WEIGHT        = float(os.getenv("RERANKER_WEIGHT", "0.5"))  # blend weight vs. the existing score
EMBEDDING_CACHE_FILE      = os.path.join(DATA_DIR, "embeddings_cache.json")

# ─── Company Identity ─────────────────────────────────────────────────────────

COMPANY_SHORT_NAME = "GPI"
COMPANY_FULL_NAME  = "Godfrey Phillips India"
ASSISTANT_NAME     = "GPI Knowledge Assistant"

# ─── Chart generation ──────────────────────────────────────────────────────
# When the user explicitly asks for a chart/graph/plot, the assistant does
# NOT ask the LLM to "draw" anything — small local text models can't
# generate images, and asking one to hand-draw an SVG/ASCII chart from
# memory is exactly the kind of task that invites invented numbers. Instead:
# an LLM call extracts the real numbers from the SAME retrieved knowledge
# context into strict JSON (chart_type/categories/series), then matplotlib
# (deterministic, no model involved) renders that JSON into an actual PNG.
# This is also why chart quality is a data-extraction problem, not an
# image-generation one — the model your other settings are tuned for is
# fine for this. CHART_MODEL defaults to the SAME model as OLLAMA_MODEL
# now (Nemotron 3 Nano, itself trained with structured-output/agentic use
# in mind) rather than a separate, larger 7-8B model — one fewer model to
# download, and chart extraction reuses whatever's already loaded instead
# of a cold-loading a second, bigger model on every chart request, which
# was the main hidden cost behind chart generation feeling slow. Override
# with a different model here if you ever find one that follows the
# strict JSON schema more reliably on your own real questions.
CHART_MODEL       = os.getenv("CHART_MODEL", OLLAMA_MODEL)
CHART_TIMEOUT     = int(os.getenv("CHART_TIMEOUT", "300"))
CHART_MAX_TOKENS  = int(os.getenv("CHART_MAX_TOKENS", "1500"))
CHARTS_OUTPUT_DIR = os.path.join(BASE_DIR, "charts_output")