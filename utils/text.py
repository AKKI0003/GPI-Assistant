"""
utils/text.py

Shared text processing utilities used across the project.
"""

import re
from typing import Set

STOPWORDS: Set[str] = {
    "what", "is", "are", "the", "a", "an",
    "tell", "me", "about", "show", "give",
    "please", "describe", "explain",
    "of", "for", "to", "in", "on", "at", "by",
    "our", "we", "all", "list", "get", "can",
    "how", "who", "where", "when", "which",
    "do", "does", "did", "have", "has",
}


def _stem(word: str) -> str:
    """Very lightweight plural/singular normalization -- NOT a real
    stemmer, just enough to collapse the most common English plural
    patterns so lexical token-overlap scoring isn't defeated purely by
    a query saying "states" when the source data says "state" (or vice
    versa). Deliberately conservative: only strips a trailing -s/-es
    when the word is long enough that this can't turn one short,
    unrelated word into another (e.g. leaves "gas", "bus", "this"
    alone)."""
    if len(word) <= 4 or not word.isalpha():
        return word
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"           # companies -> company
    if word.endswith(("xes", "zes", "ches", "shes", "sses")):
        return word[:-2]                 # boxes -> box, matches -> match
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]                 # states -> state, sales -> sale
    return word


def tokenize(text: str) -> Set[str]:
    """Lowercase word-tokenize, removing stopwords. Both the exact word
    and its lightly-stemmed form are kept, so a query token and an item
    token that only differ by plural/singular still overlap."""
    if not text:
        return set()
    words = re.findall(r"\w+", str(text).lower())
    kept = {w for w in words if w not in STOPWORDS}
    stemmed = {_stem(w) for w in kept}
    return kept | stemmed


def normalize(text: str) -> str:
    """Lowercase, strip possessives and punctuation."""
    text = text.lower().strip()
    text = re.sub(r"['']s\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def truncate(text: str, max_chars: int) -> str:
    """Hard-truncate text to max_chars."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def flatten_value(value) -> str:
    """Recursively flatten any JSON value to a plain string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(flatten_value(v) for v in value if v is not None)
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            flat = flatten_value(v)
            if flat:
                parts.append(f"{k}: {flat}")
        return "; ".join(parts)
    return str(value)
