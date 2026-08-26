#!/usr/bin/env python3
"""
import_pdf_documents.py

Builds/refreshes the INDEPENDENT PDF Document Intelligence knowledge
source (see retrieval/pdf_retriever.py and the ingestion/ package). This
is a SEPARATE pipeline from import_documents.py / ingest/ingest.py --
it does not touch documents/processed/ or the JSON knowledge base at
all.

Workflow
--------
1. Drop annual reports / financial PDFs into:
       documents/pdf_knowledge/raw/
2. Run:
       python import_pdf_documents.py
3. Ask a question that needs report/financial detail -- AssistantEngine
   automatically merges PDF-derived context alongside the JSON
   knowledge base's context for every question (see
   AssistantEngine._retrieve() in core/engine.py). No restart is
   strictly required (the index also updates itself lazily on first
   use), but running this script after adding new files gives you an
   explicit, inspectable summary of what was ingested before you start
   asking questions.

This script only ever ingests NEW or CHANGED files (tracked by SHA-256
checksum in the Chroma store under documents/pdf_knowledge/chroma_db/) --
pass --force to wipe the store and re-ingest everything from scratch.
"""

from __future__ import annotations

import argparse
import sys

from ingestion.utils import setup_logging
from retrieval.pdf_retriever import pdf_retriever


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Wipe the existing PDF knowledge cache and re-ingest every file from scratch.",
    )
    parser.add_argument(
        "--caption-images", action="store_true",
        help=(
            "Run vision captioning on extracted images (charts, infographics, "
            "logos) during ingestion, so image content becomes searchable "
            "(e.g. 'describe the ITC logo'). This is the slow step -- expect "
            "several extra minutes per PDF depending on how many images it "
            "has. Only applies to files actually (re-)ingested this run: an "
            "already-ingested, unchanged file is skipped by the checksum "
            "check regardless of this flag, so pair with --force to add "
            "captions to PDFs you've already ingested without --caption-images."
        ),
    )
    args = parser.parse_args()

    setup_logging()

    print("=" * 78)
    print("PDF Document Intelligence knowledge source -- build/refresh")
    print(f"Source folder: {pdf_retriever.raw_dir}")
    print(f"Chroma store:  {pdf_retriever.chroma_dir}")
    if args.caption_images:
        print("Image captioning: ON (slower -- vision model runs per new/changed image)")
    print("=" * 78)

    before_files = len(pdf_retriever._known_checksums)
    pdf_retriever.rebuild(force=args.force, caption_images=args.caption_images)
    after_files = len(pdf_retriever._known_checksums)
    after_chunks = len(pdf_retriever._chunks)

    print()
    if args.force:
        print(f"Rebuilt from scratch: {after_files} file(s), {after_chunks} chunk(s) indexed.")
    else:
        new_or_changed = after_files - before_files
        print(f"Indexed {after_files} file(s) total ({after_chunks} chunk(s)); "
              f"{new_or_changed} were new or changed this run.")

    if after_files == 0:
        print()
        print("No PDFs found. Drop annual reports / financial PDFs into:")
        print(f"  {pdf_retriever.raw_dir}")
        print("...then re-run this script.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
