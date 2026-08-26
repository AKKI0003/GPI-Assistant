#!/usr/bin/env python3
"""
import_documents.py

Pre-processes PDFs, PPTXs, and images into the GPI knowledge base.

Run this after adding new documents to the documents/raw/ folder. The app
loads every JSON file in documents/processed/ automatically at startup —
one file per source document, named after it — so dropping a new PDF into
documents/raw/ and re-running this script is the entire workflow. Nothing
in the app hardcodes a filename; any file that shows up in
documents/processed/ gets picked up.

Workflow
--------
1. Drop your files into  documents/raw/
2. Run: python import_documents.py
3. Restart (or reload) the app
   The new documents are now part of the knowledge base.

Folder structure created automatically
---------------------------------------
documents/
├── raw/                          ← put your PDFs, PPTXs, images here
│   ├── VST_Annual_Report_2023.pdf
│   ├── ITC_2024.pptx
│   ├── competitor_chart.png
│   └── ...
└── processed/                    ← written by this script, read by knowledge_index
    ├── VST_Annual_Report_2023.json
    ├── ITC_2024.json
    └── competitor_chart.json

Re-ingesting one document only rewrites its own small JSON file — every
other document's file (and the embedding-cache cost of re-processing it)
is left untouched. This replaces an earlier design that merged everything
into one ever-growing documents/processed_chunks.json — with a few dozen
PDFs that file grew past 40,000+ lines, made every single ingestion run
rewrite the ENTIRE combined file, and made it hard to check what came
from any one specific document. That legacy file is still read if it
exists (nothing you've already ingested is lost), but this script no
longer writes to it.

Usage examples
--------------
    python import_documents.py                      # process everything in documents/raw/
    python import_documents.py --dir path/to/docs   # custom folder
    python import_documents.py --file report.pdf    # single file
    python import_documents.py --dry-run            # preview without writing
    python import_documents.py --clear              # wipe documents/processed/ and re-ingest

Required one-time Ollama setup
-------------------------------
    ollama pull nomic-embed-text    # semantic embeddings (already needed)
    ollama pull llava:7b            # vision model for charts/infographics/photos
                                     # alternatives: minicpm-v, llava-phi3

Large documents (100s of pages)
--------------------------------
PDFs are processed page by page (see ingest/ingest.py), so a 300-page
annual report is handled the same way as a 3-page brief — memory use
stays flat, every extracted chunk keeps its real page number, and
progress is logged every 25 pages/slides so a long-running ingest of a
big batch of documents doesn't look stuck. There's no hard page-count
limit; ingestion time scales with page count and how many images/charts
need a vision-model call.
"""

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict
from collections import defaultdict

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

try:
    from rich.logging import RichHandler
    from ingest.ingest import console as _rich_console
    # Bind the root logger to the SAME console instance the progress bar
    # uses. This is the actual fix for a bar that looks frozen/stuck:
    # every logger.info() call in the app now prints through the bar's
    # own console, so rich draws it above the live bar instead of the two
    # colliding and corrupting the terminal. Without this, the bar's
    # numbers can look stuck at 0/N even while ingestion is genuinely
    # progressing underneath.
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=_rich_console, show_path=False, markup=False)],
    )
except ImportError:
    # rich (or the console it exports) isn't available — plain logging,
    # no bar, but nothing breaks.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

logger = logging.getLogger(__name__)


def _ensure_dirs():
    """Create documents/raw/ if it doesn't exist yet."""
    from core.config import DOCUMENTS_DIR
    raw_dir = os.path.join(DOCUMENTS_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    return raw_dir


def _chunks_to_json_records(chunks) -> list:
    """
    Convert IngestedChunk objects to the flat JSON format that
    knowledge_index._load_documents() expects.

    Chunks from the same (source_file, category, section, chunk_type,
    page_or_slide) are merged into one record so a 300-page PDF doesn't
    turn into thousands of tiny disconnected entries, while still keeping
    text chunks and vision chunks from DIFFERENT pages as separate records
    — merging every page together would throw away the per-page
    attribution that makes ingestion of large documents actually useful.
    """
    groups = defaultdict(list)
    for chunk in chunks:
        key = (chunk.source_file, chunk.category, chunk.section,
               chunk.chunk_type, chunk.page_or_slide)
        groups[key].append(chunk)

    records = []
    for (source_file, category, section, chunk_type, page_or_slide), group in groups.items():
        combined = "\n\n".join(c.text for c in group)
        page_label = f" p{page_or_slide}" if page_or_slide else ""
        name = f"{Path(source_file).stem}{page_label} [{section}]"
        records.append({
            "category":    category,
            "section":     section,
            "source_file": source_file,
            "chunk_type":  chunk_type,
            "record": {
                "name":          name,
                "source_file":   source_file,
                "page_or_slide": page_or_slide,
                "content":       combined,
                "chunk_count":   len(group),
                "confidence":    round(
                    sum(c.confidence for c in group) / len(group), 2
                ),
            }
        })
    return records


def _sanitize_filename(name: str) -> str:
    """Turn a source filename into a safe JSON filename stem."""
    stem = os.path.splitext(name)[0]
    return re.sub(r"[^A-Za-z0-9_.-]", "_", stem)


def _save_to_chroma(records: list) -> Dict[str, int]:
    """
    Write chunks into the documents_knowledge_chunks Chroma collection,
    grouped by source file. Each source file's OWN prior chunks are
    deleted before its new ones are upserted, so re-ingesting one
    document only touches its own records -- every other document's
    chunks (and the embedding-cache cost of re-processing them, handled
    separately by knowledge_index._embed_items()) are left untouched,
    same guarantee the old one-JSON-file-per-document layout gave.

    Returns {source_file: record_count} for the files actually written.
    """
    from core.config import (
        PDF_KNOWLEDGE_CHROMA_DIR,
        DOCUMENTS_CHROMA_CHUNKS_COLLECTION,
        DOCUMENTS_CHROMA_META_COLLECTION,
    )
    from retrieval.chroma_store import ChromaStore

    store = ChromaStore(
        PDF_KNOWLEDGE_CHROMA_DIR,
        chunks_collection=DOCUMENTS_CHROMA_CHUNKS_COLLECTION,
        meta_collection=DOCUMENTS_CHROMA_META_COLLECTION,
    )

    by_source: Dict[str, list] = defaultdict(list)
    for r in records:
        by_source[r.get("source_file", "unknown")].append(r)

    written = {}
    for source_file, recs in by_source.items():
        store.delete_records_for_source(source_file)
        store.upsert_records(recs)
        written[source_file] = len(recs)
        logger.info("  Wrote %d records → Chroma (source=%s)", len(recs), source_file)

    return written


def _preflight_ollama_check() -> None:
    """
    Verify Ollama is reachable and the embedding + vision models are
    actually pulled, BEFORE processing any files. Without this, a missing
    vision model degrades silently — every image/chart/infographic in a
    PDF or PPTX gets skipped with only a log line, and there's no
    indication that charts were never actually read.
    """
    import requests
    from ingest.ingest import OLLAMA_ENDPOINT, VISION_MODEL

    try:
        resp = requests.get(f"{OLLAMA_ENDPOINT}/api/tags", timeout=5)
        resp.raise_for_status()
        pulled = {m["name"] for m in resp.json().get("models", [])}
    except Exception as exc:
        print(f"\n⚠️  Cannot reach Ollama at {OLLAMA_ENDPOINT} ({exc}).")
        print("   Run `ollama serve` first, or ingestion will silently skip")
        print("   image/chart descriptions and may fail on embeddings too.\n")
        return

    def _pulled(model: str) -> bool:
        base = model.split(":")[0]
        return any(p == model or p.startswith(base + ":") or p == base for p in pulled)

    missing = []
    if not _pulled("nomic-embed-text"):
        missing.append("nomic-embed-text")
    if not _pulled(VISION_MODEL):
        missing.append(VISION_MODEL)

    if missing:
        print(f"\n⚠️  Missing Ollama model(s): {', '.join(missing)}")
        print("   Charts, infographics, and photos embedded in your PDFs/PPTs")
        print("   will be SILENTLY skipped (no error, just no description)")
        print("   until you pull the vision model:\n")
        for m in missing:
            print(f"       ollama pull {m}")
        print()
        return

    print(f"✓ Ollama reachable — embedding + vision model ({VISION_MODEL}) both pulled.\n")

    if os.environ.get("INGEST_SKIP_VISION") == "1":
        return

    # Warm up the vision model with a trivial call BEFORE timed ingestion
    # starts. A model's first-ever call often includes loading it into
    # RAM/VRAM — on CPU this alone can take 1-3 minutes — and if that cold
    # load happens silently during page 1's first image, it looks
    # indistinguishable from being stuck. Separating it out here means the
    # per-page timings you see during ingestion are real inference time,
    # not a one-time hidden load spike.
    print(f"⏳ Warming up vision model ({VISION_MODEL})... "
          f"first load can take 1-3 minutes on CPU, faster on GPU.")
    warmup_start = time.time()
    try:
        tiny_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
        requests.post(
            f"{OLLAMA_ENDPOINT}/api/generate",
            json={"model": VISION_MODEL, "prompt": "describe", "images": [base64.b64encode(tiny_png).decode()],
                  "stream": False, "options": {"num_predict": 5}},
            timeout=300,
        )
        print(f"✓ Vision model warm and ready ({time.time() - warmup_start:.0f}s).\n")
    except Exception as exc:
        print(f"⚠️  Vision model warm-up failed ({exc}). Ingestion will still try, "
              f"but the first image may be slow or time out.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-process documents into the GPI knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file", "-f",
        help="Ingest a single file.")
    parser.add_argument("--dir", "-d",
        help="Ingest all supported files in this directory (recursive). "
             "Defaults to documents/raw/")
    parser.add_argument("--ext", "-e", action="append",
        help="Limit to extension(s), e.g. --ext .pdf --ext .pptx")
    parser.add_argument("--dry-run", action="store_true",
        help="Show what would be ingested without writing anything.")
    parser.add_argument("--clear", action="store_true",
        help="Wipe all previously processed documents from the Chroma "
             "documents_knowledge_chunks collection and re-ingest from scratch.")
    parser.add_argument("--no-vision", action="store_true",
        help="Skip image/chart/infographic description entirely (much faster; "
             "useful for a quick test ingest of large PDFs, or if the vision "
             "model isn't pulled). Text extraction is unaffected.")
    args = parser.parse_args()

    if args.no_vision:
        os.environ["INGEST_SKIP_VISION"] = "1"
        logger.info("--no-vision: image/chart descriptions will be skipped this run.")

    _preflight_ollama_check()

    from core.config import DOCUMENTS_DIR
    raw_dir       = _ensure_dirs()

    source_dir = args.dir or raw_dir
    if not args.file and not os.path.isdir(source_dir):
        logger.error("Source directory not found: %s", source_dir)
        sys.exit(1)

    from ingest.ingest import ingest_file, ingest_directory

    all_chunks = []

    if args.file:
        logger.info("Ingesting: %s", args.file)
        chunks = ingest_file(args.file)
        logger.info("  → %d chunks", len(chunks))
        all_chunks.extend(chunks)

    if not args.file or args.dir:
        logger.info("Scanning: %s", source_dir)
        chunks = ingest_directory(source_dir, extensions=args.ext)
        logger.info("  → %d chunks from directory", len(chunks))
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning(
            "No content extracted. Supported formats: PDF, PPTX, PNG, JPG, JPEG, WEBP, XLSX, XLS.\n"
            "Put files in:  %s", source_dir
        )
        sys.exit(0)

    text_n   = sum(1 for c in all_chunks if c.chunk_type == "text")
    vision_n = sum(1 for c in all_chunks if c.chunk_type == "vision")
    sections = sorted({c.section for c in all_chunks})
    sources  = sorted({c.source_file for c in all_chunks})

    logger.info(
        "Extracted %d chunks total  (%d text, %d vision)",
        len(all_chunks), text_n, vision_n
    )
    logger.info("Sections detected: %s", ", ".join(sections))
    logger.info("Source files: %s", ", ".join(sources))

    if args.dry_run:
        logger.info("--- DRY RUN — nothing written ---")
        for i, chunk in enumerate(all_chunks[:15]):
            print(f"\n[{i+1}] {chunk.source_file}  "
                  f"p{chunk.page_or_slide}  "
                  f"{chunk.category}/{chunk.section}  "
                  f"({chunk.chunk_type})")
            print("     " + chunk.text[:200].replace("\n", " "))
        if len(all_chunks) > 15:
            print(f"\n  … and {len(all_chunks) - 15} more chunks.")
        return

    new_records = _chunks_to_json_records(all_chunks)

    if args.clear:
        from core.config import (
            PDF_KNOWLEDGE_CHROMA_DIR,
            DOCUMENTS_CHROMA_CHUNKS_COLLECTION,
            DOCUMENTS_CHROMA_META_COLLECTION,
        )
        from retrieval.chroma_store import ChromaStore
        ChromaStore(
            PDF_KNOWLEDGE_CHROMA_DIR,
            chunks_collection=DOCUMENTS_CHROMA_CHUNKS_COLLECTION,
            meta_collection=DOCUMENTS_CHROMA_META_COLLECTION,
        ).clear_all()
        logger.info("--clear: wiped the documents_knowledge_chunks Chroma collection")

    written = _save_to_chroma(new_records)

    try:
        from retrieval.knowledge_index import index as knowledge_index
        knowledge_index.reload()
        logger.info(
            "Knowledge index reloaded live: %d items total",
            len(knowledge_index.items)
        )
    except Exception:
        pass   # Normal when running as a standalone script

    print("\n✓ Done. Reload the app to pick up new documents.")
    print(f"  Stored in: Chroma (documents_knowledge_chunks collection)")
    print(f"  Documents written this run: {len(written)}")
    for source_file, count in written.items():
        print(f"    {source_file}: {count} records")


if __name__ == "__main__":
    main()
