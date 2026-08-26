# GPI Knowledge Assistant — Production Refactor

Enterprise RAG-powered knowledge assistant for Godfrey Phillips India.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama with Llama 3.1:8b
ollama serve
ollama pull llama3.1:8b-instruct-q4_K_M

# 3. Run the terminal assistant
python main.py
```

## Project Structure

```
gpi_assistant/
├── main.py                   ← Terminal UI entry point
├── requirements.txt
│
├── core/
│   ├── config.py             ← All settings (paths, model params)
│   ├── engine.py             ← Central orchestrator (ask / ask_stream)
│   ├── intent_resolver.py    ← Intent + section detection
│   └── direct_answer.py      ← No-LLM fast answers
│
├── retrieval/
│   ├── knowledge_index.py    ← Schema-agnostic index + search
│   └── context_builder.py    ← Formats results for LLM prompt
│
├── memory/
│   └── conversation_memory.py ← Rolling Q&A + entity state
│
├── llm/
│   ├── prompt_builder.py     ← Grounding-focused prompt constructor
│   └── ollama_client.py      ← Streaming + blocking Ollama wrapper
│
├── storage/
│   └── chat_storage.py       ← Persistent chat session manager
│
├── ui/
│   └── table_renderer.py     ← Rich table builder
│
├── utils/
│   ├── data_loader.py        ← Cached JSON loader
│   └── text.py               ← Tokenizer, normalizer, flattener
│
└── data/
    ├── company_overview.json
    ├── departments.json
    ├── projects.json
    ├── processes.json
    ├── problems.json
    ├── kpis.json
    ├── intent_dictionary.json
    └── chats/                ← Auto-created chat storage
```

## Configuration

All settings live in `core/config.py`. Override via environment variables:

| Variable          | Default                   | Description                     |
|-------------------|---------------------------|---------------------------------|
| OLLAMA_ENDPOINT   | http://localhost:11434    | Ollama server URL               |
| OLLAMA_MODEL      | llama3.1:8b-instruct-q4_K_M | Model name                    |
| LLM_TEMPERATURE   | 0.15                      | Lower = less hallucination      |
| LLM_MAX_TOKENS    | 1800                      | Max response length             |

## Routing Architecture

```
Question
   │
   ▼
Intent Resolver  ──→  category, section_hint, action, is_followup
   │
   ▼
LEVEL 1: Fast social (greetings/thanks/goodbye) ──→ instant reply
   │
   ▼
Retrieval (AssistantEngine._retrieve)
   │
   ├──→ JSON Retriever   (KnowledgeIndex.search — curated company data, unchanged)
   │
   └──→ PDF Retriever    (retrieval/pdf_retriever.py — ingested annual reports / financial PDFs)
   │
   ▼
Merge Context  (plain list concatenation — both retrievers return the same
                {category, section, record, score} shape, so context_builder.py
                and prompt_builder.py need no changes to render either source)
   │
   ▼
LEVEL 2: Direct RAG (simple field lookups, no LLM) ──→ instant reply
   │
   ▼
LEVEL 3: LLM (streaming, grounded prompt over the merged context) ──→ streamed answer
```

### The PDF Document Intelligence knowledge source

A second, fully independent knowledge source sits alongside the JSON
knowledge base (which is never modified by it):

```
documents/pdf_knowledge/raw/        ← drop annual reports / financial PDFs here
documents/pdf_knowledge/chunk_cache.json   ← checksum-keyed chunk + embedding cache
```

- **Ingestion** (`ingestion/` package): structure-preserving PDF parsing —
  semantic section/subsection hierarchy (not page-based chunking),
  structured tables (rows/columns/header, never flattened to prose),
  extracted images/figures, and rich metadata (company/year/document
  type), via `ingestion.PDFIngestor`.
- **Retrieval** (`retrieval/pdf_retriever.py`): embeds each chunk with
  the *same* Ollama embedding model/cache the JSON retriever already
  uses, and exposes `search(query) -> List[dict]` in the exact result
  shape `KnowledgeIndex.search()` returns.
- **Merge**: `AssistantEngine._retrieve()` (in `core/engine.py`) calls
  both retrievers and concatenates their results before context/prompt
  building — JSON results are kept first so curated, hand-verified data
  always wins the character budget over ingested PDF text when both are
  relevant.

**Usage:**

```bash
# 1. Drop PDFs into documents/pdf_knowledge/raw/
# 2. (optional — the index also builds itself lazily on first question)
python import_pdf_documents.py
# 3. Ask questions as normal — PDF-derived context is merged in automatically.
```

If `pdfplumber`/`camelot`/PyMuPDF aren't installed, PDF retrieval simply
returns no results (logged, not raised) — JSON-based answering is
completely unaffected either way.

## Extending the Knowledge Base

Simply add new JSON files. The `KnowledgeIndex` uses recursive flattening —
no code changes needed for new fields or nesting levels.

To add a new entity file:
1. Create `data/yourfile.json` with structure `{"your_category": {"records": [...]}}`
2. Add one line to `retrieval/knowledge_index.py` `_build()`:
   ```python
   self._load_entity_file("data/yourfile.json", "your_category")
   ```

To extend intent detection, edit `data/intent_dictionary.json` — no Python changes needed.
