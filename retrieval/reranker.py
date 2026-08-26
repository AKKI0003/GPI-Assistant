"""
retrieval/reranker.py

Local, fully-offline cross-encoder reranker.

Cosine similarity between two INDEPENDENTLY computed embeddings (the
query's and a chunk's) is a decent first-pass filter but can't capture
fine-grained query/document interaction -- e.g. it can't easily tell
"VST's net profit" apart from "VST's net revenue" the way a model that
reads query and candidate TOGETHER can. A cross-encoder re-scores each
(query, candidate) pair jointly, which is exactly what improves
precision among the top handful of already-plausible candidates.

Fully local: runs via sentence-transformers/torch on-device. The model
weights are downloaded once from Hugging Face on first use and cached
locally (~90MB); after that, zero network calls, nothing ever leaves
the machine. If sentence-transformers or the model isn't available,
every function here degrades to a no-op (returns None) rather than
raising, so retrieval simply falls back to pure embedding-similarity
ranking -- reranking is a pure quality add-on, never a hard dependency.
"""

from __future__ import annotations

import os

# Force fully-offline mode for Hugging Face / transformers BEFORE
# sentence_transformers is ever imported below. Without this, CrossEncoder(...)
# still tries to reach huggingface.co first (even when pointed at a local
# folder) to check for updates -- on a locked-down corporate network that
# hits a TLS failure (SSL: CERTIFICATE_VERIFY_FAILED) instead of just
# failing fast, and burns several retries before giving up. This also
# matches the project's own goal of nothing ever leaving the device.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import logging
import math
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

logger = logging.getLogger(__name__)

# Small (~90MB), CPU-friendly, well-established for exactly this task
# (MS MARCO passage reranking). Swap via RERANKER_MODEL_NAME in
# core/config.py if you want a different local model -- this can be
# either a Hugging Face repo id (e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2",
# downloaded once and cached) OR an absolute local folder path (e.g. a
# copy of that same model transferred onto a machine with no Hugging
# Face access), which sentence-transformers accepts natively.
_DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None
_model_load_failed = False


def _get_model():
    global _model, _model_load_failed
    if _model is not None or _model_load_failed:
        return _model
    try:
        from core.config import RERANKER_MODEL_NAME
        model_name = RERANKER_MODEL_NAME or _DEFAULT_MODEL_NAME
    except ImportError:
        model_name = _DEFAULT_MODEL_NAME
    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(model_name)
        logger.info("Local reranker loaded (%s).", model_name)
    except Exception as exc:
        logger.warning(
            "Local reranker unavailable (%s); continuing with pure "
            "embedding-similarity ranking. Install with "
            "`pip install sentence-transformers` to enable it.", exc,
        )
        _model_load_failed = True
        _model = None
    return _model


def is_available() -> bool:
    """True if the reranker model is loaded and usable. Triggers a lazy
    load on first call."""
    return _get_model() is not None


def score_pairs(query: str, texts: Sequence[str]) -> Optional[List[float]]:
    """Raw cross-encoder relevance logits for each (query, text) pair,
    or None if the reranker isn't available. Never raises."""
    model = _get_model()
    if model is None or not texts:
        return None
    try:
        pairs = [[query, t or ""] for t in texts]
        scores = model.predict(pairs)
        return [float(s) for s in scores]
    except Exception as exc:
        logger.warning("Reranker scoring failed, falling back: %s", exc)
        return None


_T = TypeVar("_T")


def rerank(
    query: str,
    scored: List[Tuple[float, _T]],
    text_fn: Callable[[_T], str],
    top_n: int = 40,
    weight: float = 0.5,
) -> List[Tuple[float, _T]]:
    """
    Rerank the top `top_n` of an already-sorted (score, item) list using
    the local cross-encoder, blended additively with the existing score
    so upstream boosts (table-chunk preference, financial/ESG topic
    nudges, statement-match bonus, etc.) aren't thrown away -- the
    reranker adjusts within that existing signal rather than replacing
    it outright. Items beyond `top_n` are left untouched and appended
    unchanged (reranking the full pool isn't worth the extra latency
    once candidates are already clearly out of contention).

    Falls back to returning `scored` completely unchanged if the
    reranker isn't available or scoring fails -- callers never need
    their own fallback branch.
    """
    if not scored:
        return scored

    head = scored[:top_n]
    tail = scored[top_n:]

    texts = [text_fn(item) for _, item in head]
    ce_scores = score_pairs(query, texts)
    if ce_scores is None or len(ce_scores) != len(head):
        return scored

    blended: List[Tuple[float, _T]] = []
    for (orig_score, item), ce in zip(head, ce_scores):
        # Cross-encoder logits are unbounded; squash to (0, 1) with a
        # sigmoid so they blend sanely with cosine-similarity-based
        # scores (roughly 0-1, plus small additive boosts) instead of
        # one signal swamping the other.
        ce_norm = 1.0 / (1.0 + math.exp(-ce))
        blended.append((orig_score + weight * ce_norm, item))

    blended.sort(key=lambda pair: pair[0], reverse=True)
    return blended + tail
