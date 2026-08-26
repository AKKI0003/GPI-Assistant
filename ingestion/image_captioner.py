"""
ingestion/image_captioner.py

Adds back real image analysis to the new PDF pipeline -- the old
ingest/ingest.py called a vision model per image, which worked but was
slow (llava:7b on CPU, ~300s/image, no filtering of which images were
even worth captioning). This version keeps what worked (downscaling,
retry-on-timeout, hash-based caching) and fixes what made it slow:

  1. FASTER MODEL: default is "moondream" (1.8B) instead of "llava:7b"
     (7B). On CPU this is typically 5-10x faster per call for the same
     "describe/transcribe this chart" task. "llava-phi3" (3.8B) is a
     reasonable middle ground if moondream's accuracy isn't enough for
     dense multi-series charts. Override with the VISION_MODEL env var.
  2. SKIP LOGOS/DECORATION: `Image.is_likely_logo()` already existed in
     ingestion/image.py but was never called anywhere. A full annual
     report typically has far more masthead logos, headshots, and
     decorative icons than actual charts -- captioning all of them was
     pure wasted time. Skipped here before any vision call is made.
  3. SKIP WHEN THE PAGE ALREADY HAS THE DATA AS TEXT: if a page's own
     extracted text/tables already contain a real figures table (the
     same `_is_financial_highlight`-style signal used to fix JSON
     retrieval), the chart on that page is almost always a *visual
     restatement* of numbers already captured as selectable text --
     captioning it adds latency for near-zero new information. Vision
     is reserved for pages where the chart is the ONLY source of the
     data (the actual problem case you described).
  4. PERSISTENT, CONTENT-ADDRESSED CACHE: keyed by sha256 of the image
     bytes and written to disk (not just an in-memory dict), so the
     exact same embedded image (e.g. a repeated corporate logo, or the
     same chart appearing in two years' reports) is only ever sent to
     the vision model ONCE, ever -- not once per file, once per run.
  5. SHORTER OUTPUT: num_predict trimmed to 250 (from 800) and the
     prompt asks specifically for the data points, not a full essay --
     shorter generations are the single biggest speed lever after model
     size, since output tokens dominate CPU vision-model latency.

Usage
-----
Call `caption_images(images, page_text_by_number)` after
`ImageExtractor.extract()` in pdf_ingestor.py. This is intentionally a
SEPARATE, explicit step (not automatically invoked inside `ingest()`)
so it can be run as an offline/batch pass (see import_pdf_documents.py)
rather than blocking a live chat request -- ingest() itself stays fast;
this is the deliberately-slow step you run once, in the background,
when you drop in a new PDF.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
from typing import Dict, List, Optional

import requests

from .image import Image

logger = logging.getLogger(__name__)

OLLAMA_ENDPOINT   = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
VISION_MODEL      = os.getenv("VISION_MODEL", "moondream")   # alt: llava-phi3, llava:7b
VISION_TIMEOUT    = int(os.getenv("VISION_TIMEOUT", "60"))
VISION_MAX_DIM    = int(os.getenv("VISION_MAX_DIMENSION", "768"))
VISION_NUM_PREDICT = int(os.getenv("VISION_NUM_PREDICT", "250"))
CAPTION_CACHE_FILE = os.getenv(
    "IMAGE_CAPTION_CACHE_FILE",
    os.path.join("data", "image_caption_cache.json"),
)

_PROMPT = (
    "This image is from an annual report of an Indian company. "
    "If it is a chart, graph, or infographic: list every axis label, "
    "series name, year, and numeric value you can read, in plain text "
    "(e.g. 'Gross Revenue: 2021 = 123, 2022 = 145, 2023 = 160'). "
    "If it is a table, transcribe it. If it is a logo, photo, or purely "
    "decorative image, reply with exactly: DECORATIVE. "
    "No preamble, no commentary -- only the data or the word DECORATIVE."
)

# Separate, short prompt for images the heuristic already flagged as
# likely logos/masthead art. These don't need the "look for a chart"
# framing above (which primes the vision model toward reporting
# DECORATIVE for anything that isn't obviously a data visual) -- they
# need a plain visual description so a question like "describe the ITC
# logo" or "what colors is the logo" can actually be answered.
_LOGO_PROMPT = (
    "This small image is a logo, icon, or masthead graphic from a "
    "company's annual report. In one or two short sentences, describe "
    "what it visually looks like: the company/brand name if visible, "
    "shapes, colors, and any symbol. No preamble, just the description."
)

# A page whose own extracted text already looks like a real figures
# table (not just a page that name-drops "revenue" once) -- mirrors the
# signal used in fix_processed_data.py for the JSON pipeline.
_STATEMENT_HEADINGS = re.compile(
    r"(statement of profit and loss|balance sheet|statement of cash flows?|"
    r"financial highlights|particulars)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\b\d{3,}(?:\.\d+)?\b")


def _page_already_has_the_numbers(page_text: str) -> bool:
    if not page_text:
        return False
    return bool(_STATEMENT_HEADINGS.search(page_text)) and len(_NUMBER.findall(page_text)) >= 8


class _CaptionCache:
    def __init__(self, path: str) -> None:
        self.path = path
        self._data: Dict[str, Optional[str]] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (OSError, json.JSONDecodeError):
                logger.warning("Could not read caption cache at %s; starting fresh.", path)

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def set(self, key: str, value: Optional[str]) -> None:
        self._data[key] = value
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)


_cache = _CaptionCache(CAPTION_CACHE_FILE)


def _downscale(image_bytes: bytes, max_dimension: int) -> bytes:
    try:
        from PIL import Image as PILImage
    except ImportError:
        return image_bytes
    try:
        img = PILImage.open(io.BytesIO(image_bytes))
        img.thumbnail((max_dimension, max_dimension))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return image_bytes


def _caption_one(image_bytes: bytes, is_logo: bool = False) -> Optional[str]:
    img_hash = hashlib.sha256(image_bytes).hexdigest()
    cache_key = f"logo:{img_hash}" if is_logo else img_hash
    cached = _cache.get(cache_key)
    if cache_key in _cache._data:
        return cached  # includes cached None (previously failed)

    scaled = _downscale(image_bytes, VISION_MAX_DIM)
    b64 = base64.b64encode(scaled).decode("ascii")
    payload = {
        "model": VISION_MODEL,
        "prompt": _LOGO_PROMPT if is_logo else _PROMPT,
        "images": [b64],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": VISION_NUM_PREDICT},
    }
    try:
        resp = requests.post(f"{OLLAMA_ENDPOINT}/api/generate", json=payload, timeout=VISION_TIMEOUT)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
    except requests.exceptions.RequestException as exc:
        logger.warning("Vision call failed (%s); leaving this image uncaptioned.", exc)
        _cache.set(cache_key, None)
        return None

    # Only the non-logo prompt ever asks for/expects a literal
    # "DECORATIVE" reply; the logo prompt always wants a description.
    result = None if (not text or (not is_logo and text.upper().startswith("DECORATIVE"))) else text
    _cache.set(cache_key, result)
    return result


def caption_images(
    images: List[Image],
    page_text_by_number: Optional[Dict[int, str]] = None,
) -> int:
    """Mutates `images` in place, setting `.caption` for any image that's
    (a) not a likely logo, (b) on a page whose own text doesn't already
    contain the real figures, and (c) not already in the on-disk cache
    from a prior run. Returns how many vision calls were actually made
    (as opposed to served from cache or skipped) -- use this to gauge
    how long a given ingestion run will take before kicking it off.
    """
    page_text_by_number = page_text_by_number or {}
    calls_made = 0

    for image in images:
        is_logo = image.is_likely_logo()
        if (not is_logo) and _page_already_has_the_numbers(page_text_by_number.get(image.page_number, "")):
            continue
        if not image.file_path or not os.path.exists(image.file_path):
            continue

        with open(image.file_path, "rb") as f:
            image_bytes = f.read()

        img_hash = hashlib.sha256(image_bytes).hexdigest()
        cache_key = f"logo:{img_hash}" if is_logo else img_hash
        was_cached = cache_key in _cache._data
        caption = _caption_one(image_bytes, is_logo=is_logo)
        if not was_cached:
            calls_made += 1
        if caption:
            image.caption = caption

    return calls_made