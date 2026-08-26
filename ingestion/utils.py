"""
ingestion/utils.py

Small, dependency-light utilities shared across the ingestion package:
logging setup, lightweight language detection, a simple on-disk
extraction cache (so re-running ingestion over an unchanged file never
re-parses it), a lazy page generator helper for memory-safe processing
of very large PDFs, and optional embedding helpers for callers that want
to vectorize the `Chunk` objects a `Document` produces.

Nothing in this module is required for the core ingestion pipeline to
run -- the embedding helpers in particular degrade to a clear, logged
no-op if `sentence-transformers` / `faiss` are not installed, since
embedding is a downstream concern (retrieval-time merging with the
existing JSON knowledge base), not a requirement for structural PDF
ingestion itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from functools import lru_cache
from typing import Callable, Dict, Iterator, List, Optional, TypeVar

T = TypeVar("T")

_LOGGING_CONFIGURED = False

# Matches bare 4-digit years (2019-2049) as well as Indian-fiscal-year
# notation ("FY2022-23", "FY 2022-23", "2022-23") which is how every
# comparative financial-statement table header in these annual reports
# actually labels its columns. For "2022-23" style spans, only the
# starting/ending 4-digit year is captured by the outer group (the
# 2-digit suffix is matched but not separately returned) -- callers that
# want the closing year of a fiscal span should add 1 when the string
# also contains a "-\d{2}" suffix; `detect_years_in_text` below already
# does this so downstream code can just use its output directly.
_YEAR_TOKEN_PATTERN = re.compile(
    r"\bFY\s?(\d{4})-(\d{2})\b|\b(\d{4})-(\d{2})\b|\b(19[5-9]\d|20[0-4]\d)\b",
    re.IGNORECASE,
)


def detect_years_in_text(text: str) -> List[int]:
    """Return every distinct calendar year mentioned in `text`, in
    document order, deduplicated. Handles bare years ("2023"), Indian
    fiscal-year spans ("FY2022-23", "2022-23" -- returns BOTH 2022 and
    2023, since a table column headed "2022-23" reports that fiscal
    year's closing figures and is commonly queried by either label),
    and plain "FY2023" / "FY 2023".

    This exists because a `Document`'s single `metadata.year` (usually
    derived from the filename or the report's cover page) is NOT a
    reliable stand-in for which year(s) a specific TABLE or SECTION
    inside that document actually covers -- annual reports routinely
    show 3 years of comparative figures in one table, and CSR/BRSR
    annexures elsewhere in the same file reference entirely different
    fiscal years than the report's own headline year. Tagging every
    chunk with only the document-level year is what caused 2024 figures
    to be mislabeled as 2022 and 2024 rows to disappear from answers
    entirely.
    """
    years: List[int] = []
    for match in _YEAR_TOKEN_PATTERN.finditer(text):
        if match.group(1):  # "FY2022-23"
            start = int(match.group(1))
            end_suffix = int(match.group(2))
            years.append(start)
            century = start - (start % 100)
            years.append(century + end_suffix if end_suffix >= (start % 100) else century + 100 + end_suffix)
        elif match.group(3):  # "2022-23"
            start = int(match.group(3))
            end_suffix = int(match.group(4))
            years.append(start)
            century = start - (start % 100)
            years.append(century + end_suffix if end_suffix >= (start % 100) else century + 100 + end_suffix)
        elif match.group(5):  # bare "2023"
            years.append(int(match.group(5)))

    seen = set()
    ordered_unique = []
    for y in years:
        if 1950 <= y <= 2049 and y not in seen:
            seen.add(y)
            ordered_unique.append(y)
    return ordered_unique


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Idempotent logging setup for the ingestion package -- safe to call
    from every entry point (the test script, a future batch-ingestion
    CLI, etc.) without producing duplicate handlers."""
    global _LOGGING_CONFIGURED
    logger = logging.getLogger("ingestion")
    if not _LOGGING_CONFIGURED:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        _LOGGING_CONFIGURED = True
    return logger


# -- language detection -------------------------------------------------------

# Deliberately not a hard dependency: `langdetect` is used when present
# for real statistical detection; otherwise a tiny stopword-overlap
# heuristic covering the languages this pipeline is most likely to meet
# (English plus a handful of others) keeps `Document.language` populated
# rather than always falling back to "unknown".
try:
    from langdetect import detect as _langdetect_detect
    _LANGDETECT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _LANGDETECT_AVAILABLE = False

_STOPWORDS_BY_LANGUAGE: Dict[str, set] = {
    "en": {"the", "and", "of", "to", "in", "is", "for", "on", "with", "as", "by", "at"},
    "hi": {"aur", "hai", "ke", "ki", "ka", "mein", "yah", "ko", "se"},
    "es": {"el", "la", "de", "y", "en", "los", "que", "con", "por"},
    "fr": {"le", "la", "de", "et", "les", "des", "en", "un", "une"},
}


def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "unknown"
    if _LANGDETECT_AVAILABLE:
        try:
            return _langdetect_detect(text[:2000])
        except Exception:
            pass  # fall through to the heuristic below

    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    if not words:
        return "unknown"
    best_lang, best_overlap = "unknown", 0
    for lang, stopwords in _STOPWORDS_BY_LANGUAGE.items():
        overlap = len(words & stopwords)
        if overlap > best_overlap:
            best_lang, best_overlap = lang, overlap
    return best_lang if best_overlap >= 2 else "unknown"


# -- checksumming / caching ---------------------------------------------------

def file_checksum(file_path: str, chunk_size: int = 1 << 20) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            sha256.update(block)
    return sha256.hexdigest()


class ExtractionCache:
    """A minimal on-disk cache keyed by file checksum, so re-ingesting an
    unchanged PDF (a common occurrence when re-running a batch job over a
    `documents/raw/` folder that only had one or two new files added)
    never re-parses files that have not changed since last time.

    Deliberately file-based and dependency-free (no sqlite/redis) since
    ingestion runs are typically a batch CLI process, not a long-lived
    service -- a directory of small JSON files is simplest to inspect,
    debug, and delete/reset by hand.
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, checksum: str) -> str:
        return os.path.join(self.cache_dir, f"{checksum}.json")

    def get(self, checksum: str) -> Optional[dict]:
        path = self._cache_path(checksum)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logging.getLogger("ingestion").warning(
                "Failed to read cache entry %s: %s", path, exc
            )
            return None

    def set(self, checksum: str, document_dict: dict) -> None:
        path = self._cache_path(checksum)
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(document_dict, handle, ensure_ascii=False)
        except OSError as exc:
            logging.getLogger("ingestion").warning(
                "Failed to write cache entry %s: %s", path, exc
            )


# -- lazy iteration helpers ----------------------------------------------------

def iter_files_lazily(directory: str, extension: str = ".pdf") -> Iterator[str]:
    """Generator over every file with the given extension in a
    directory tree. Used by a future batch-ingestion CLI so that a
    `documents/raw/` folder containing thousands of PDFs is walked
    lazily rather than materializing a full file list up front."""
    extension = extension.lower()
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if name.lower().endswith(extension):
                yield os.path.join(root, name)


def batched(iterable, batch_size: int):
    """Yield successive `batch_size`-sized lists from an iterable, without
    ever holding the entire iterable in memory at once."""
    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# -- optional embedding helpers (used only by a future retrieval-merge step) --

_EMBEDDING_MODEL = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazily load a sentence-transformers model, memoized for the
    process lifetime. Returns None (with a clear log message, not a
    crash) if sentence-transformers is not installed -- embedding chunks
    is a downstream/optional concern for this ingestion package, not a
    requirement for structural PDF extraction to succeed."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logging.getLogger("ingestion").info(
            "sentence-transformers not installed; embedding helpers are disabled. "
            "This does not affect PDF structural ingestion."
        )
        return None
    _EMBEDDING_MODEL = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL


def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> Optional[List[List[float]]]:
    model = get_embedding_model(model_name)
    if model is None:
        return None
    return model.encode(texts, show_progress_bar=False).tolist()


def build_faiss_index(vectors: List[List[float]]):
    """Build a flat L2 FAISS index over a list of vectors. Returns None
    (logged, not raised) if faiss is not installed -- vector search is an
    optional acceleration for large chunk sets, not a hard requirement."""
    try:
        import faiss  # type: ignore
        import numpy as np
    except ImportError:
        logging.getLogger("ingestion").info(
            "faiss not installed; vector-index helpers are disabled. "
            "This does not affect PDF structural ingestion."
        )
        return None
    if not vectors:
        return None
    matrix = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index
