"""
ingestion/metadata_extractor.py

Populates a `DocumentMetadata` instance from a combination of:
  - filesystem facts (size, created/modified dates) -- always available.
  - the PDF's own embedded metadata dictionary (title, author, producer,
    creator, embedded creation/mod dates) -- available whenever the PDF
    library in use (PyMuPDF preferred, pypdf fallback) can read it.
  - lightweight text-based heuristics (company, year, document type) run
    over a small text sample (first couple of pages), since these fields
    are essentially never present in a PDF's technical metadata but
    almost always appear on a cover/title page.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .metadata import (
    DocumentMetadata,
    compute_checksum,
    file_system_dates,
    guess_company,
    guess_document_type,
    guess_year,
)

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FITZ_AVAILABLE = False

try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYPDF_AVAILABLE = False


class MetadataExtractor:
    """Extracts and normalizes all document-level metadata for one PDF."""

    def extract(
        self,
        file_path: str,
        text_sample: str,
        page_count: int,
        language: str = "unknown",
    ) -> DocumentMetadata:
        metadata = DocumentMetadata()
        metadata.source_file = file_path
        metadata.page_count = page_count
        metadata.language = language

        try:
            metadata.file_size_bytes = os.path.getsize(file_path)
        except OSError as exc:
            logger.warning("Could not stat file size for %s: %s", file_path, exc)

        try:
            metadata.checksum_sha256 = compute_checksum(file_path)
        except OSError as exc:
            logger.warning("Could not compute checksum for %s: %s", file_path, exc)

        try:
            dates = file_system_dates(file_path)
            metadata.created_date = dates["created_date"]
            metadata.modified_date = dates["modified_date"]
        except OSError as exc:
            logger.warning("Could not read filesystem dates for %s: %s", file_path, exc)

        pdf_meta = self._extract_pdf_technical_metadata(file_path)
        metadata.title = pdf_meta.get("title") or self._filename_as_title(file_path)
        metadata.author = pdf_meta.get("author")
        metadata.pdf_producer = pdf_meta.get("producer")
        metadata.pdf_creator = pdf_meta.get("creator")
        metadata.pdf_created_date = pdf_meta.get("creation_date")
        metadata.pdf_modified_date = pdf_meta.get("mod_date")

        filename = os.path.basename(file_path)
        metadata.company = guess_company(text_sample, filename)
        metadata.year = guess_year(text_sample, filename)
        metadata.document_type = guess_document_type(text_sample, filename)

        return metadata

    @staticmethod
    def _filename_as_title(file_path: str) -> str:
        name = os.path.splitext(os.path.basename(file_path))[0]
        return name.replace("_", " ").replace("-", " ").strip()

    def _extract_pdf_technical_metadata(self, file_path: str) -> dict:
        if _FITZ_AVAILABLE:
            return self._extract_with_fitz(file_path)
        if _PYPDF_AVAILABLE:
            return self._extract_with_pypdf(file_path)
        logger.warning(
            "Neither PyMuPDF nor pypdf installed; PDF technical metadata "
            "(title/author/producer/etc.) will be unavailable for %s.",
            file_path,
        )
        return {}

    @staticmethod
    def _extract_with_fitz(file_path: str) -> dict:
        try:
            doc = fitz.open(file_path)
            try:
                raw = doc.metadata or {}
                return {
                    "title": raw.get("title") or None,
                    "author": raw.get("author") or None,
                    "producer": raw.get("producer") or None,
                    "creator": raw.get("creator") or None,
                    "creation_date": raw.get("creationDate") or None,
                    "mod_date": raw.get("modDate") or None,
                }
            finally:
                doc.close()
        except Exception as exc:
            logger.warning("PyMuPDF failed to read metadata for %s: %s", file_path, exc)
            return {}

    @staticmethod
    def _extract_with_pypdf(file_path: str) -> dict:
        try:
            reader = PdfReader(file_path)
            raw = reader.metadata or {}
            return {
                "title": getattr(raw, "title", None),
                "author": getattr(raw, "author", None),
                "producer": getattr(raw, "producer", None),
                "creator": getattr(raw, "creator", None),
                "creation_date": str(getattr(raw, "creation_date", "") or "") or None,
                "mod_date": str(getattr(raw, "modification_date", "") or "") or None,
            }
        except Exception as exc:
            logger.warning("pypdf failed to read metadata for %s: %s", file_path, exc)
            return {}
