"""
retrieval/embedding_store.py

Semantic embedding layer for the knowledge index.

This replaces keyword / token-overlap / fuzzy-string matching as the
PRIMARY retrieval signal with true semantic similarity, computed via a
local Ollama embedding model (default: "nomic-embed-text"). This is what
lets a query like "tell me about the firm's beginnings" match a "history"
record even though the two share almost no literal words — a case keyword
matching and synonym dictionaries cannot reliably cover, and exactly the
kind of retrieval gap that forces the LLM to guess (hallucinate) instead
of answering from real context.

Design
------
- Embeddings are computed once per unique piece of text and cached to
  disk (core.config.EMBEDDING_CACHE_FILE), keyed by a content hash, so
  restarts don't re-embed unchanged knowledge records. Only genuinely
  new or changed text triggers a new call to Ollama.
- If the embedding model / Ollama is unreachable, every call returns
  None and callers fall back to lexical scoring. The system never
  crashes and never blocks startup waiting on a model that may not be
  pulled yet — semantic search is a strict upgrade, not a hard
  dependency.
- No third-party numerical library is required: vectors are plain lists
  of floats and cosine similarity is computed in pure Python, which is
  more than fast enough for a knowledge base of this size.
"""

import hashlib
import json
import os
import threading
from typing import Dict, List, Optional

import requests

from core.config import (
    OLLAMA_ENDPOINT,
    OLLAMA_EMBED_MODEL,
    OLLAMA_EMBED_TIMEOUT,
    EMBEDDING_CACHE_FILE,
)

_lock = threading.Lock()
_disk_cache: Dict[str, List[float]] = {}
_disk_cache_loaded = False

# Once an embedding call fails, avoid hammering an unreachable endpoint
# with one HTTP request per knowledge item / query. Reset on next process
# start; this is a soft circuit-breaker, not a permanent disable.
_embedding_unavailable = False

# Batches disk writes instead of rewriting the entire cache file after
# EVERY single new embedding. With hundreds/thousands of new records (a
# few large PDF/PPTX reports easily produce this many chunks), saving on
# every item makes total ingestion time scale roughly with the SQUARE of
# the record count (write 1 record, then 2, then 3, ... then N — an
# increasingly large file, N times over) instead of linearly. Batching
# saves once every N new embeddings, plus one guaranteed final flush.
_SAVE_EVERY = 20
_dirty_count = 0


# ─── Disk cache ───────────────────────────────────────────────────────────────

def _load_disk_cache() -> None:
    global _disk_cache_loaded
    if _disk_cache_loaded:
        return
    with _lock:
        if _disk_cache_loaded:
            return
        try:
            if os.path.exists(EMBEDDING_CACHE_FILE):
                with open(EMBEDDING_CACHE_FILE, "r", encoding="utf-8") as fh:
                    _disk_cache.update(json.load(fh))
        except Exception:
            pass
        _disk_cache_loaded = True


def _save_disk_cache() -> None:
    try:
        os.makedirs(os.path.dirname(EMBEDDING_CACHE_FILE), exist_ok=True)
        with open(EMBEDDING_CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(_disk_cache, fh)
    except Exception:
        # Caching is a performance optimization, never a correctness
        # requirement — failure to persist must not break retrieval.
        pass


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ─── Public API ───────────────────────────────────────────────────────────────

def embed(text: str) -> Optional[List[float]]:
    """
    Return the embedding vector for *text*.

    Uses the on-disk cache when available. Returns None if the embedding
    model cannot be reached or the text is empty — callers MUST handle
    this by falling back to lexical scoring rather than treating it as
    an error.
    """
    if not text or not text.strip():
        return None

    global _embedding_unavailable

    _load_disk_cache()
    key = _hash(text)

    with _lock:
        cached = _disk_cache.get(key)
        if cached is not None:
            return cached
        if _embedding_unavailable:
            return None

    try:
        resp = requests.post(
            f"{OLLAMA_ENDPOINT}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=OLLAMA_EMBED_TIMEOUT,
        )
        resp.raise_for_status()
        vector = resp.json().get("embedding")
        if not vector:
            return None
    except Exception:
        with _lock:
            _embedding_unavailable = True
        return None

    with _lock:
        _disk_cache[key] = vector
        global _dirty_count
        _dirty_count += 1
        if _dirty_count >= _SAVE_EVERY:
            _save_disk_cache()
            _dirty_count = 0

    return vector


def flush_disk_cache() -> None:
    """
    Force-save any embeddings computed since the last batched write.
    Call this once after a bulk operation (e.g. KnowledgeIndex rebuilding
    embeddings for every record) to guarantee nothing computed in the
    final partial batch is lost if the process exits right after.
    """
    global _dirty_count
    with _lock:
        if _dirty_count > 0:
            _save_disk_cache()
            _dirty_count = 0


def reset_availability() -> None:
    """
    Clear the soft circuit-breaker so the next embed() call retries the
    endpoint. Call this after the host confirms Ollama / the embedding
    model is back up (e.g. from an admin "reindex" action).
    """
    global _embedding_unavailable
    with _lock:
        _embedding_unavailable = False


# Once the batch embedding endpoint (/api/embed) is confirmed to not
# exist or not behave as expected (older Ollama versions only have the
# singular /api/embeddings, one input at a time), stop trying it and
# fall back to the original one-request-per-text path for the rest of
# this process's lifetime. Soft circuit-breaker, same pattern as
# _embedding_unavailable above -- reset on next process start.
_batch_embed_unavailable = False

# How many texts go into one /api/embed request. Large enough to cut
# HTTP round-trip count by orders of magnitude (7,847 chunks / 64 ≈ 123
# requests instead of 7,847), small enough that one request's JSON
# payload/response stays reasonable and a single failed batch only
# costs a bounded amount of re-work when falling back per-item.
_EMBED_BATCH_SIZE = 64


def _embed_batch_via_api(texts: List[str]) -> Optional[List[Optional[List[float]]]]:
    """
    One HTTP call to Ollama's batch embedding endpoint for multiple
    texts at once. Returns a list of vectors (same order/length as
    `texts`, with None for any entry Ollama returned nothing for), or
    None if the batch endpoint itself failed/is unavailable -- callers
    should fall back to per-item embed() in that case, not treat it as
    "every text failed."
    """
    global _batch_embed_unavailable
    if _batch_embed_unavailable or not texts:
        return None
    try:
        resp = requests.post(
            f"{OLLAMA_ENDPOINT}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": texts},
            timeout=max(OLLAMA_EMBED_TIMEOUT, OLLAMA_EMBED_TIMEOUT * len(texts) // 8),
        )
        resp.raise_for_status()
        data = resp.json()
        vectors = data.get("embeddings")
        if not vectors or not isinstance(vectors, list):
            # Endpoint responded but not in the expected shape -- treat
            # as "batch mode doesn't work here" rather than guessing.
            _batch_embed_unavailable = True
            return None
        # Defensive: don't silently misalign texts<->vectors if the
        # response came back a different length than requested.
        if len(vectors) != len(texts):
            _batch_embed_unavailable = True
            return None
        return [v if v else None for v in vectors]
    except Exception:
        _batch_embed_unavailable = True
        return None


def embed_many(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Embed a batch of texts, reusing the on-disk cache per item and
    making as few HTTP calls to Ollama as possible for whatever isn't
    already cached.

    This is the function ingestion (PDF chunk embedding, KnowledgeIndex
    rebuilding) should call for a list of texts -- NOT a Python-level
    loop calling embed() once per item, which is what previously made
    embedding a few thousand PDF chunks take one sequential HTTP request
    per chunk (tens of minutes with zero visible progress). Order of the
    returned list always matches `texts`, cached or not.
    """
    if not texts:
        return []

    _load_disk_cache()
    results: List[Optional[List[float]]] = [None] * len(texts)
    to_fetch_indices: List[int] = []
    to_fetch_texts: List[str] = []

    with _lock:
        already_unavailable = _embedding_unavailable
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            key = _hash(text)
            cached = _disk_cache.get(key)
            if cached is not None:
                results[i] = cached
            elif not already_unavailable:
                to_fetch_indices.append(i)
                to_fetch_texts.append(text)

    if not to_fetch_texts:
        return results

    global _dirty_count
    for start in range(0, len(to_fetch_texts), _EMBED_BATCH_SIZE):
        batch_texts = to_fetch_texts[start:start + _EMBED_BATCH_SIZE]
        batch_indices = to_fetch_indices[start:start + _EMBED_BATCH_SIZE]

        batch_vectors = _embed_batch_via_api(batch_texts)
        if batch_vectors is None:
            # Batch endpoint unavailable/failed -- fall back to the
            # original, proven one-at-a-time path for this batch (and,
            # since _batch_embed_unavailable is now set, every batch
            # after it too). Quality/correctness is identical either
            # way; this only affects request count, never what gets
            # returned or cached.
            for idx, text in zip(batch_indices, batch_texts):
                results[idx] = embed(text)
            continue

        with _lock:
            for idx, text, vector in zip(batch_indices, batch_texts, batch_vectors):
                results[idx] = vector
                if vector is not None:
                    _disk_cache[_hash(text)] = vector
                    _dirty_count += 1
            if _dirty_count >= _SAVE_EVERY:
                _save_disk_cache()
                _dirty_count = 0

    return results


def cosine_similarity(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    """
    Cosine similarity in [-1, 1]. Returns 0.0 if either vector is missing
    or the vectors are mismatched/empty — never raises.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)