"""
ingestion/ner.py

Local, fully-offline named-entity recognition (spaCy), used to sharpen
company detection in ingestion/metadata.py's guess_company().

guess_company()'s text-frequency fallback originally counted every raw
substring match of a known company name ANYWHERE in the sampled text --
a table caption, a footnote, an unrelated boilerplate mention all count
equally toward "how many times is this name mentioned". Restricting
those counts to spans spaCy actually recognizes as an organization name
(ORG entities) filters out that noise and sharpens the "which company is
this document actually about" signal, on top of the filename-first +
frequency fix already in guess_company().

Fully local: spaCy's small English model runs entirely on-device. The
model is downloaded once (`python -m spacy download en_core_web_sm`,
~15MB) and cached locally; after that, zero network calls, nothing ever
leaves the machine. If spaCy or the model isn't installed, every
function here degrades to returning None, and guess_company() falls
back automatically to its original whole-text regex scan -- NER is a
pure precision add-on, never a hard dependency.
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "en_core_web_sm"

_nlp = None
_load_failed = False


def _get_nlp():
    global _nlp, _load_failed
    if _nlp is not None or _load_failed:
        return _nlp
    try:
        import spacy
    except Exception as exc:
        logger.warning(
            "spaCy unavailable (%s); continuing without NER-assisted "
            "company detection. Install with `pip install spacy` and "
            "`python -m spacy download en_core_web_sm` to enable it.", exc,
        )
        _load_failed = True
        return None
    try:
        # Only the NER component is needed here; disabling the parser
        # and lemmatizer noticeably speeds up loading and per-call
        # inference for a use case that only ever reads ent.label_.
        _nlp = spacy.load(_DEFAULT_MODEL_NAME, disable=["parser", "lemmatizer"])
        logger.info("Local NER model loaded (%s).", _DEFAULT_MODEL_NAME)
    except OSError:
        logger.warning(
            "spaCy model '%s' not downloaded; run "
            "`python -m spacy download %s` to enable NER-assisted "
            "company detection. Falling back to plain text scanning.",
            _DEFAULT_MODEL_NAME, _DEFAULT_MODEL_NAME,
        )
        _load_failed = True
        _nlp = None
    except Exception as exc:
        logger.warning("Failed to load spaCy model: %s", exc)
        _load_failed = True
        _nlp = None
    return _nlp


def is_available() -> bool:
    """True if the NER model is loaded and usable. Triggers a lazy load
    on first call."""
    return _get_nlp() is not None


def extract_organizations(text: str, max_chars: int = 20000) -> Optional[List[str]]:
    """
    Return every text span spaCy tags as an ORG entity in `text`, or
    None if NER isn't available -- callers should fall back to their
    own logic in that case, NOT treat None the same as "found nothing".
    `text` is capped at max_chars for speed; a metadata text_sample is
    already just the first page or two, so this is a safety net rather
    than a real limitation in practice. Never raises.
    """
    nlp = _get_nlp()
    if nlp is None or not text:
        return None
    try:
        doc = nlp(text[:max_chars])
        return [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    except Exception as exc:
        logger.warning("NER extraction failed, falling back: %s", exc)
        return None
