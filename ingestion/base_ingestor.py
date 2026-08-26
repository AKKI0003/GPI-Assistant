"""
ingestion/base_ingestor.py

Abstract base class every format-specific ingestor must implement, plus
a tiny extension-keyed registry so adding support for a new format
(Word, PowerPoint, Excel, HTML, Markdown, a web page, SharePoint) is
purely additive: write a new `BaseIngestor` subclass, register it for
its extension(s), and nothing else in this package (or in any caller of
`get_ingestor_for`) needs to change.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Type

from .document import Document

logger = logging.getLogger(__name__)


class BaseIngestor(ABC):
    """Common interface for every document ingestor in this package."""

    #: File extensions (including the leading dot, lowercase) this
    #: ingestor handles. Subclasses must set this.
    supported_extensions: tuple = ()

    def ingest_with_timing(self, file_path: str) -> Document:
        """Wraps `ingest()` to record extraction duration on the
        resulting Document's statistics, and to guarantee a Document is
        always returned (never a bare exception) for a malformed input
        file -- callers processing hundreds of files in a batch should
        not have one bad PDF crash the whole run."""
        start = time.monotonic()
        try:
            document = self.ingest(file_path)
        except Exception as exc:
            logger.exception("Ingestion failed for %s", file_path)
            document = Document(source_file=file_path)
            document.extraction_warnings.append(
                f"Ingestion failed entirely: {type(exc).__name__}: {exc}"
            )
            document.quality_score = 0.0
        document.statistics.extraction_duration_seconds = round(time.monotonic() - start, 3)
        return document

    @abstractmethod
    def ingest(self, file_path: str) -> Document:
        """Parse `file_path` and return a fully populated `Document`.
        Must be implemented by every concrete ingestor."""
        raise NotImplementedError

    @staticmethod
    def validate_file_exists(file_path: str) -> None:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")


# -- registry -----------------------------------------------------------------

_INGESTOR_REGISTRY: Dict[str, Type[BaseIngestor]] = {}


def register_ingestor(ingestor_cls: Type[BaseIngestor]) -> Type[BaseIngestor]:
    """Class decorator: register an ingestor for every extension it
    declares in `supported_extensions`. Usage::

        @register_ingestor
        class PDFIngestor(BaseIngestor):
            supported_extensions = (".pdf",)
            ...
    """
    for extension in ingestor_cls.supported_extensions:
        normalized = extension.lower()
        if normalized in _INGESTOR_REGISTRY and _INGESTOR_REGISTRY[normalized] is not ingestor_cls:
            logger.warning(
                "Overwriting ingestor for extension %s: %s -> %s",
                normalized,
                _INGESTOR_REGISTRY[normalized].__name__,
                ingestor_cls.__name__,
            )
        _INGESTOR_REGISTRY[normalized] = ingestor_cls
    return ingestor_cls


def get_ingestor_for(file_path: str) -> BaseIngestor:
    """Look up (and instantiate) the correct ingestor for a file purely
    from its extension. Raises `ValueError` for an unregistered
    extension rather than silently guessing."""
    extension = os.path.splitext(file_path)[1].lower()
    ingestor_cls = _INGESTOR_REGISTRY.get(extension)
    if ingestor_cls is None:
        raise ValueError(
            f"No ingestor registered for extension '{extension}'. "
            f"Registered extensions: {sorted(_INGESTOR_REGISTRY.keys())}"
        )
    return ingestor_cls()


def registered_extensions() -> tuple:
    return tuple(sorted(_INGESTOR_REGISTRY.keys()))
