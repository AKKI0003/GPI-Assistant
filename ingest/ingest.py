"""
ingest/ingest.py

Multimodal document ingestion pipeline for GPI Assistant.

Supports:
  - PDF (text-heavy, scanned/image-heavy, mixed, and multi-hundred-page
    documents) via PyMuPDF + pymupdf4llm, processed PAGE BY PAGE so large
    files stay memory-safe and every chunk keeps an accurate page number.
  - PPTX (slides, speaker notes, embedded images) via python-pptx.
  - Images / charts / infographics / photos via Ollama vision (llava,
    minicpm-v, or whichever vision model is pulled on the local Ollama
    instance) — this is what lets the assistant "read" a bar chart, a
    table baked into a slide image, or an infographic, not just the
    selectable text around it.

Output: a list of IngestedChunk objects, each carrying the text content,
source metadata, and a suggested knowledge-index category/section so the
caller (import_documents.py) can slot it straight into KnowledgeIndex.

Usage
-----
    from ingest.ingest import ingest_file, IngestedChunk
    chunks = ingest_file("reports/annual_report_2024.pdf")
    for chunk in chunks:
        print(chunk.category, chunk.section, chunk.page_or_slide, chunk.text[:120])
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests

try:
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
    )
    from rich.console import Console
    _RICH_AVAILABLE = True
    # Single shared Console. Anything printed/logged through THIS console
    # while a Progress (built with the same console) is live gets drawn
    # above the bar correctly. Printing through plain stdlib logging
    # instead — writing straight to the terminal — is what corrupts a
    # live-rendering bar and makes it look frozen even while work is
    # actually progressing. import_documents.py binds its logging handler
    # to this same console to avoid that.
    console = Console()
except ImportError:
    _RICH_AVAILABLE = False
    console = None

logger = logging.getLogger(__name__)

_NON_TTY_NOTICE_SHOWN = False


def _log(msg: str, *args) -> None:
    """
    Emit a log line safely whether or not a rich Progress bar is
    currently live. Always goes through the shared console (via the
    logger, which is bound to that console by import_documents.py) so
    it never corrupts an active progress bar.
    """
    logger.info(msg, *args)


def _make_progress() -> Optional["Progress"]:
    """
    Build a rich progress bar (spinner + bar + n/total + elapsed + ETA).

    Returns None — meaning "no bar, rely on the per-page log lines
    instead" — in two cases:
      1. rich isn't installed.
      2. The console isn't a real interactive terminal capable of
         in-place ANSI redraws (common with some Windows consoles,
         certain IDE-embedded terminals, or when output is piped/
         redirected to a file). In that situation rich would otherwise
         keep re-printing over a stale first frame — visually "stuck"
         at 0/N — while the per-page log lines (which don't rely on
         cursor control) keep updating correctly underneath. Skipping
         the bar there avoids that misleading frozen-looking display;
         the log lines already carry the same page/vision-call/timing
         info.
    """
    if not _RICH_AVAILABLE:
        return None
    if not console.is_terminal:
        global _NON_TTY_NOTICE_SHOWN
        if not _NON_TTY_NOTICE_SHOWN:
            logger.info(
                "  (Terminal doesn't support a live-updating progress bar — "
                "showing per-page log lines instead, which is just as accurate.)"
            )
            _NON_TTY_NOTICE_SHOWN = True
        return None
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("elapsed •"),
        TimeRemainingColumn(),
        TextColumn("remaining"),
        TextColumn("• {task.fields[extra]}"),
        console=console,
    )

# ─── Config (reads from env or uses sensible defaults) ────────────────────────

OLLAMA_ENDPOINT   = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
VISION_MODEL      = os.getenv("VISION_MODEL", "llava:7b")   # alt: minicpm-v, llava-phi3

# 120s was too short — on CPU inference a single large image (a big
# infographic, a high-DPI screenshot, a dense multi-panel chart) can
# legitimately take longer than that just for prefill, before the model
# has produced a single token, which showed up as "vision model timing
# out on large images". Bumped the default to 300s. Still configurable
# via env if the host is slower. VISION_TIMEOUT is the preferred name;
# OLLAMA_TIMEOUT is kept as a fallback for anyone with it already set.
OLLAMA_TIMEOUT    = int(os.getenv("VISION_TIMEOUT", os.getenv("OLLAMA_TIMEOUT", "300")))

# Deliberately-provided standalone images (see _ingest_image) get an even
# longer ceiling — the user chose this one file, so it's worth waiting for.
VISION_TIMEOUT_STANDALONE = int(os.getenv("VISION_TIMEOUT_STANDALONE", "420"))

# Longest side (px) a first attempt is downscaled to. If that attempt
# times out, we retry once at VISION_MAX_DIMENSION_RETRY — a smaller,
# faster-to-encode image — rather than giving up outright. This is what
# actually fixes "cannot read large images": previously a single huge
# image had exactly one (slow) shot at succeeding.
VISION_MAX_DIMENSION       = int(os.getenv("VISION_MAX_DIMENSION", "1568"))
VISION_MAX_DIMENSION_RETRY = int(os.getenv("VISION_MAX_DIMENSION_RETRY", "900"))

# Above this raw byte size, shrink harder on the FIRST attempt too —
# a multi-megabyte screenshot/photo doesn't need to pay for a slow first
# failure before we admit it's too big; go straight to a size we know
# CPU inference can chew through in reasonable time.
VISION_LARGE_IMAGE_BYTES = int(os.getenv("VISION_LARGE_IMAGE_BYTES", str(3 * 1024 * 1024)))

# Maximum characters per chunk before we split (avoids overflowing context)
MAX_CHUNK_CHARS   = 3000
# Minimum characters — tiny slivers of text are noise, skip them
MIN_CHUNK_CHARS   = 60
# Skip describing images smaller than this many bytes (icons/logos/bullets)
MIN_IMAGE_BYTES   = 2000
# Log a progress line every N pages/slides. Kept low (not 25!) so a slow
# vision-heavy ingest still gives visible per-page feedback instead of
# looking hung for many minutes between checkpoints.
PROGRESS_EVERY    = 1

# When set, image/vision description is skipped entirely — huge speedup
# for text-only test ingests of large PDFs. Toggle with
# INGEST_SKIP_VISION=1 or the --no-vision flag on import_documents.py.
#
# IMPORTANT: this is a FUNCTION, not a frozen constant, and must stay
# that way. import_documents.py imports `console` from this module at
# the very top of the file (for logging setup) — that import happens
# BEFORE argparse runs, so BEFORE `--no-vision` ever gets a chance to set
# os.environ["INGEST_SKIP_VISION"]. If SKIP_VISION were computed once at
# import time (`SKIP_VISION = os.getenv(...) == "1"`), it would have
# already locked in "False" during that early import, and Python caches
# modules — the later `from ingest.ingest import ingest_file` wouldn't
# re-run this line — so --no-vision would silently have zero effect for
# the rest of the run. Checking os.environ fresh on every call avoids
# that entirely.
def SKIP_VISION() -> bool:
    return os.environ.get("INGEST_SKIP_VISION", "0") == "1"

# Cache of image-hash -> description, so an identical image (e.g. a
# header/footer logo repeated on every page of an 85-page report) is only
# ever sent to the vision model ONCE per ingest run instead of once per
# page. This is typically the single biggest speed win for corporate PDFs.
_vision_cache: dict = {}


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class IngestedChunk:
    """One logical unit of knowledge extracted from a document."""
    text:          str
    source_file:   str
    page_or_slide: int   = 0
    section:       str   = "overview"
    category:      str   = "company"     # matches KnowledgeIndex categories
    record:        dict  = field(default_factory=dict)
    chunk_type:    str   = "text"        # text | vision
    confidence:    float = 1.0           # 0-1; vision outputs default to 0.75


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Remove null bytes, collapse excessive whitespace."""
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{4,}", "  ", text)
    return text.strip()


def _split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """
    Split a long text into chunks at paragraph boundaries, staying under
    max_chars. Falls back to sentence boundaries if a single paragraph
    exceeds max_chars.
    """
    paragraphs = re.split(r"\n{2,}", text)
    chunks, current = [], ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                buf = ""
                for sent in sentences:
                    if len(buf) + len(sent) + 1 <= max_chars:
                        buf = (buf + " " + sent).strip()
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = sent
                current = buf if buf else ""
            else:
                current = para
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def _guess_section(text: str, filename: str) -> str:
    """
    Heuristically map text content to a GPI knowledge section name so the
    chunk integrates with the existing retrieval section-hint system.
    Checked in priority order — more specific signals first — so e.g. a
    page that mentions both "revenue" and "cigarette" isn't randomly
    assigned to whichever happens to be scanned first.
    """
    t = (text + " " + filename).lower()
    if any(w in t for w in ["revenue", "profit", "ebitda", "crore", "turnover", "financial", "balance sheet", "p&l"]):
        return "financials"
    if any(w in t for w in ["market share", "competitor", "itc", "vst", "rival", "competition"]):
        return "competitor_finances"
    if any(w in t for w in ["kpi", "key performance", "target", "metric", "indicator"]):
        return "kpi"
    if any(w in t for w in ["segment", "segmentation"]):
        return "cigarette_segmentation"
    if any(w in t for w in ["product", "brand", "cigarette", "tobacco", "portfolio", "iqos", "heated"]):
        return "products"
    if any(w in t for w in ["leadership", "ceo", "cfo", "coo", "director", "board", "management"]):
        return "leadership"
    if any(w in t for w in ["history", "founded", "milestone", "origin", "established"]):
        return "history"
    if any(w in t for w in ["department", "team", "division", "hr", "finance dept", "sales dept"]):
        return "department"
    if any(w in t for w in ["project", "initiative", "launch", "rollout", "programme"]):
        return "project"
    if any(w in t for w in ["market size", "industry", "volume", "india market", "category size"]):
        return "indian_market_overview"
    if any(w in t for w in ["strategy", "strategic", "vision", "mission", "objective"]):
        return "mission_vision"
    return "overview"


def _guess_category(section: str) -> str:
    dept_sections = {"department", "project", "process", "kpi", "problem"}
    return "department" if section in dept_sections else "company"


# ─── Vision: describe images/charts via Ollama ────────────────────────────────

def _image_bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _downscale_image(image_bytes: bytes, max_dimension: int = VISION_MAX_DIMENSION) -> bytes:
    """
    Shrink an image so its longest side is at most `max_dimension` pixels,
    re-encoding as JPEG (quality 85) to also cut payload size.

    Standalone images the user drops in directly — full-resolution
    screenshots, exported dashboard PNGs, high-DPI photos — are routinely
    much larger than the images embedded inside a PDF/PPTX (which are
    usually already compressed for print/display). A large raw PNG can be
    several megabytes; base64-encoded into a JSON payload and sent to a
    CPU-bound vision model, that's both slow and a common cause of the
    request timing out or Ollama running out of memory mid-decode.
    Vision models don't benefit from resolution far beyond ~1500px on the
    long side anyway, so this is a pure speed/reliability win with no
    real loss of description quality.

    Falls back to the original bytes unchanged if Pillow isn't installed
    or the image can't be parsed (e.g. an unsupported format) — the
    caller will simply send the original bytes as before.
    """
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=85)
        return out.getvalue()
    except Exception:
        return image_bytes


def _describe_image(
    image_bytes: bytes,
    context_hint: str = "",
    min_bytes: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Optional[str]:
    """
    Send an image to the Ollama vision model and get a rich textual
    description — this is what lets the assistant genuinely "understand"
    a chart, table-as-image, or infographic rather than just skip it.
    Returns None if the vision model is unavailable or the image is too
    small (icons/logos/bullet graphics are not worth a vision call).

    ``min_bytes``: override MIN_IMAGE_BYTES. Pass 0 for images the user
    deliberately provided as a standalone file — unlike a PDF/PPTX page
    that may contain many incidental decorative images, a file the user
    explicitly chose to ingest should never be silently skipped just for
    being small (e.g. a simple, efficiently-compressed chart).

    ``timeout``: override OLLAMA_TIMEOUT. A deliberately-provided single
    image is worth waiting longer for than one of dozens of images
    embedded across a multi-hundred-page batch ingest.
    """
    if SKIP_VISION():
        return None

    effective_min_bytes = MIN_IMAGE_BYTES if min_bytes is None else min_bytes
    if len(image_bytes) < effective_min_bytes:
        return None

    # Dedup identical images (e.g. a logo repeated on every page) — only
    # ever pay the vision-model cost once per unique image per run. Hash
    # the ORIGINAL bytes (before any downscaling) so this still catches
    # true duplicates regardless of downscale rounding differences.
    img_hash = hashlib.sha256(image_bytes).hexdigest()
    if img_hash in _vision_cache:
        return _vision_cache[img_hash]

    effective_timeout = timeout if timeout is not None else OLLAMA_TIMEOUT

    # Large raw images get shrunk harder on the very first attempt instead
    # of paying for a near-guaranteed slow failure first.
    first_pass_dimension = (
        VISION_MAX_DIMENSION_RETRY
        if len(image_bytes) >= VISION_LARGE_IMAGE_BYTES
        else VISION_MAX_DIMENSION
    )

    prompt = (
        "You are an expert business analyst reviewing a document image for "
        "Godfrey Phillips India (GPI), a tobacco and consumer goods company.\n\n"
        f"{('Context hint: ' + context_hint + chr(10)) if context_hint else ''}"
        "Describe this image in full detail:\n"
        "- If it is a chart or graph: state the chart type, all axis labels, "
        "all data series names, key values, trends, and your interpretation.\n"
        "- If it is a table: transcribe every row and column into plain text.\n"
        "- If it is an infographic: list all text, numbers, and visual elements.\n"
        "- If it is a photograph: describe what is shown and its business relevance.\n"
        "Respond with ONLY the description — no preamble, no meta-commentary."
    )

    # Two attempts: first at `first_pass_dimension`, and — only if that one
    # times out (not on other errors, which won't be fixed by resizing) —
    # a second, smaller/faster attempt with extra time budget. This is what
    # actually lets large images succeed instead of failing outright once.
    attempts = [
        (first_pass_dimension, effective_timeout),
        (VISION_MAX_DIMENSION_RETRY if first_pass_dimension != VISION_MAX_DIMENSION_RETRY
         else int(VISION_MAX_DIMENSION_RETRY * 0.6),
         int(effective_timeout * 1.5)),
    ]

    last_exc_kind = None
    for attempt_num, (dimension, attempt_timeout) in enumerate(attempts, start=1):
        scaled = _downscale_image(image_bytes, max_dimension=dimension)
        b64 = _image_bytes_to_base64(scaled)
        payload = {
            "model":   VISION_MODEL,
            "prompt":  prompt,
            "images":  [b64],
            "stream":  False,
            "options": {"temperature": 0.0, "num_predict": 800},
        }
        try:
            resp = requests.post(
                f"{OLLAMA_ENDPOINT}/api/generate",
                json=payload,
                timeout=attempt_timeout,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            result = text if len(text) > 30 else None
            _vision_cache[img_hash] = result
            return result
        except requests.exceptions.ConnectTimeout:
            logger.warning(
                "Vision model unreachable: could not connect to Ollama at %s. "
                "Is `ollama serve` running?", OLLAMA_ENDPOINT
            )
            _vision_cache[img_hash] = None
            return None
        except requests.exceptions.ReadTimeout:
            last_exc_kind = "timeout"
            if attempt_num < len(attempts):
                logger.warning(
                    "Vision model timed out after %ss at %spx (model: %s) — "
                    "retrying once at a smaller size (%spx, %ss timeout) "
                    "instead of giving up.",
                    attempt_timeout, dimension, VISION_MODEL,
                    attempts[attempt_num][0], attempts[attempt_num][1],
                )
                continue
            logger.warning(
                "Vision model timed out on both attempts (model: %s). This "
                "image may just be too complex for CPU inference right now — "
                "try VISION_TIMEOUT / VISION_MAX_DIMENSION_RETRY env vars, "
                "or re-run later.", VISION_MODEL,
            )
            _vision_cache[img_hash] = None
            return None
        except requests.exceptions.ConnectionError as exc:
            logger.warning(
                "Cannot reach Ollama at %s (%s). Run `ollama serve` first.",
                OLLAMA_ENDPOINT, exc,
            )
            _vision_cache[img_hash] = None
            return None
        except Exception as exc:
            logger.warning("Vision model call failed (%s). Skipping image.", exc)
            _vision_cache[img_hash] = None
            return None

    _vision_cache[img_hash] = None
    return None


# ─── PDF ingestion ─────────────────────────────────────────────────────────────

def _ingest_pdf(path: str, progress: Optional["Progress"] = None) -> List[IngestedChunk]:
    """
    Extract text + images from a PDF, PAGE BY PAGE.

    Strategy (per page, so this scales to multi-hundred-page documents
    without ever holding the whole document's text in memory at once, and
    so every chunk keeps a real, citable page number instead of a single
    "whole-doc" placeholder):

    1. pymupdf4llm's page-chunked Markdown extraction (handles tables,
       columns, headers far better than raw text extraction) for each
       page's selectable text.
    2. Every embedded image on the page is separately vision-described —
       charts, graphs, and tables-as-images carry data plain text
       extraction cannot see, so these run even on text-rich pages, not
       just image-only/scanned ones.

    ``progress``: an optional shared rich.progress.Progress instance (used
    when ingesting a whole directory, so one Progress renders all bars
    together). If omitted, a standalone bar is created and shown just for
    this file — either way you get live terminal progress.
    """
    try:
        import fitz          # PyMuPDF
        import pymupdf4llm
    except ImportError:
        raise ImportError("Install pymupdf and pymupdf4llm: pip install pymupdf pymupdf4llm")

    chunks: List[IngestedChunk] = []
    fname = os.path.basename(path)

    doc = fitz.open(path)
    n_pages = len(doc)
    logger.info("Ingesting PDF: %s (%d pages)", fname, n_pages)

    # ── Per-page Markdown text extraction ──────────────────────────────────
    # page_chunks=True returns one dict per page (with real page numbers),
    # instead of one giant concatenated string for the whole document —
    # essential for accurate citations and for not losing track of where
    # in a 300-page report a fact came from.
    page_texts: dict = {}
    try:
        page_data = pymupdf4llm.to_markdown(path, page_chunks=True)
        for i, pg in enumerate(page_data):
            page_texts[i] = pg.get("text", "") if isinstance(pg, dict) else str(pg)
    except TypeError:
        # Older pymupdf4llm without page_chunks support — fall back to a
        # single whole-document string and approximate page numbers by
        # splitting on the form-feed characters PyMuPDF inserts per page.
        try:
            md_text = pymupdf4llm.to_markdown(path)
            parts = md_text.split("\f") if "\f" in md_text else [md_text]
            for i, part in enumerate(parts):
                page_texts[i] = part
        except Exception as exc:
            logger.warning("pymupdf4llm failed for %s: %s. Text extraction skipped.", fname, exc)
    except Exception as exc:
        logger.warning("pymupdf4llm failed for %s: %s. Text extraction skipped.", fname, exc)

    if SKIP_VISION():
        logger.info("  (INGEST_SKIP_VISION=1 — image/chart descriptions will be skipped)")

    # Own the Progress lifecycle only if the caller didn't hand us a shared
    # one (directory-level ingest shares a single Progress across files).
    owns_progress = progress is None
    if owns_progress:
        progress = _make_progress()

    task_id = None
    if progress is not None:
        if owns_progress:
            progress.start()
        task_id = progress.add_task(f"[cyan]{fname}", total=n_pages, extra="starting…")

    try:
        for page_num in range(n_pages):
            page_start = time.time()
            n_images_described = 0
            text = _clean(page_texts.get(page_num, ""))
            if len(text) >= MIN_CHUNK_CHARS:
                for i, chunk_text in enumerate(_split_into_chunks(text)):
                    section = _guess_section(chunk_text, fname)
                    chunks.append(IngestedChunk(
                        text          = chunk_text,
                        source_file   = fname,
                        page_or_slide = page_num + 1,
                        section       = section,
                        category      = _guess_category(section),
                        record        = {"name": f"{fname} p{page_num+1}", "chunk_index": i,
                                         "source": "pdf_text"},
                        chunk_type    = "text",
                        confidence    = 1.0,
                    ))

            # ── Images on this page (charts, graphs, tables-as-images, photos) ─
            # Skip touching images at all when vision is off — not just
            # skipping the description call, but never even calling
            # page.get_images()/extract_image() for this page. With
            # --no-vision this makes text-only ingestion of an
            # image-heavy PDF meaningfully faster, and guarantees no
            # image bytes are read or decoded during a text-only run.
            page_images = [] if SKIP_VISION() else doc[page_num].get_images(full=True)
            for img_index, img_info in enumerate(page_images):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes  = base_image["image"]

                    # Live sub-page feedback: this shows up on the bar the
                    # instant a vision call starts, so a slow 120s image
                    # description doesn't look like the whole run is dead —
                    # you can see exactly which image is being processed.
                    if progress is not None and task_id is not None \
                            and len(img_bytes) >= MIN_IMAGE_BYTES:
                        progress.update(
                            task_id,
                            extra=(f"page {page_num+1}/{n_pages} • "
                                   f"describing image {img_index+1}/{len(page_images)} "
                                   f"(can take up to {OLLAMA_TIMEOUT}s)…"),
                        )

                    context    = f"Page {page_num+1} of {fname}. " + text[:200]
                    description = _describe_image(img_bytes, context)
                    if len(img_bytes) >= MIN_IMAGE_BYTES:
                        n_images_described += 1
                    if description:
                        section = _guess_section(description + " " + text, fname)
                        chunks.append(IngestedChunk(
                            text          = f"[Visual content — page {page_num+1}, image {img_index+1}]\n{description}",
                            source_file   = fname,
                            page_or_slide = page_num + 1,
                            section       = section,
                            category      = _guess_category(section),
                            record        = {"name": f"{fname} p{page_num+1} img{img_index+1}",
                                             "source": "pdf_vision"},
                            chunk_type    = "vision",
                            confidence    = 0.75,
                        ))
                except Exception:
                    continue

            page_elapsed = time.time() - page_start
            extra = f"page {page_num+1}/{n_pages} • {n_images_described} vision call{'s' if n_images_described != 1 else ''} • {page_elapsed:.1f}s"
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1, extra=extra)
            if (page_num + 1) % PROGRESS_EVERY == 0:
                logger.info(
                    "  … %s: %s%s",
                    fname, extra,
                    "  ⚠ slow page" if page_elapsed > 15 else "",
                )
    finally:
        if progress is not None and task_id is not None:
            if owns_progress:
                progress.stop()
            else:
                progress.remove_task(task_id)

    doc.close()
    return chunks


# ─── PPTX ingestion ────────────────────────────────────────────────────────────

def _ingest_pptx(path: str, progress: Optional["Progress"] = None) -> List[IngestedChunk]:
    """
    Extract content from PowerPoint:
    - All text boxes and titles per slide
    - Speaker notes
    - Embedded images (charts, diagrams, photos) via vision model
    """
    try:
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE
    except ImportError:
        raise ImportError("Install python-pptx: pip install python-pptx")

    chunks: List[IngestedChunk] = []
    fname = os.path.basename(path)
    prs   = Presentation(path)
    n_slides = len(prs.slides)
    logger.info("Ingesting PPTX: %s (%d slides)", fname, n_slides)

    owns_progress = progress is None
    if owns_progress:
        progress = _make_progress()

    task_id = None
    if progress is not None:
        if owns_progress:
            progress.start()
        task_id = progress.add_task(f"[cyan]{fname}", total=n_slides, extra="starting…")

    try:
        for slide_num, slide in enumerate(prs.slides, start=1):
            slide_start = time.time()
            slide_texts: List[str] = []

            # ── Text from shapes ────────────────────────────────────────────
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        line = " ".join(run.text for run in para.runs).strip()
                        if line:
                            slide_texts.append(line)

                # ── Tables (native PPTX tables, not images) ──────────────────
                if shape.has_table:
                    table = shape.table
                    rows = []
                    for row in table.rows:
                        cells = [c.text.strip() for c in row.cells]
                        rows.append(" | ".join(cells))
                    if rows:
                        slide_texts.append("[Table]\n" + "\n".join(rows))

                # ── Embedded images / charts ─────────────────────────────
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    try:
                        img_bytes = shape.image.blob
                        context   = f"Slide {slide_num} of {fname}. " + " ".join(slide_texts[:5])
                        description = _describe_image(img_bytes, context)
                        if description:
                            section = _guess_section(description, fname)
                            chunks.append(IngestedChunk(
                                text          = f"[Slide {slide_num} visual]\n{description}",
                                source_file   = fname,
                                page_or_slide = slide_num,
                                section       = section,
                                category      = _guess_category(section),
                                record        = {"name": f"{fname} slide{slide_num} img",
                                                 "source": "pptx_vision"},
                                chunk_type    = "vision",
                                confidence    = 0.75,
                            ))
                    except Exception:
                        continue

            # ── Speaker notes ────────────────────────────────────────────────
            if slide.has_notes_slide:
                notes_text = slide.notes_slide.notes_text_frame.text.strip()
                if notes_text:
                    slide_texts.append(f"[Speaker notes] {notes_text}")

            # ── Combine slide text into chunk(s) ──────────────────────────────
            combined = _clean("\n".join(slide_texts))
            if len(combined) >= MIN_CHUNK_CHARS:
                for i, chunk_text in enumerate(_split_into_chunks(combined)):
                    section = _guess_section(chunk_text, fname)
                    chunks.append(IngestedChunk(
                        text          = chunk_text,
                        source_file   = fname,
                        page_or_slide = slide_num,
                        section       = section,
                        category      = _guess_category(section),
                        record        = {"name": f"{fname} slide{slide_num}", "chunk_index": i,
                                         "source": "pptx_text"},
                        chunk_type    = "text",
                        confidence    = 1.0,
                    ))

            slide_elapsed = time.time() - slide_start
            extra = f"slide {slide_num}/{n_slides} • {slide_elapsed:.1f}s"
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1, extra=extra)
            if slide_num % PROGRESS_EVERY == 0:
                logger.info("  … %s: %s", fname, extra)
    finally:
        if progress is not None and task_id is not None:
            if owns_progress:
                progress.stop()
            else:
                progress.remove_task(task_id)

    return chunks


# ─── Standalone image ingestion ────────────────────────────────────────────────

def _ingest_image(path: str) -> List[IngestedChunk]:
    """
    Describe a standalone image file (PNG, JPG, JPEG, WEBP, BMP) via the
    Ollama vision model — for infographics, exported charts, or photos
    dropped in on their own rather than embedded in a PDF/PPT.
    """
    fname = os.path.basename(path)
    with open(path, "rb") as f:
        img_bytes = f.read()

    logger.info("Ingesting image: %s (%.1f KB)", fname, len(img_bytes) / 1024)

    # min_bytes=0: the user deliberately chose this exact file, so it
    # should never be silently skipped just for being small/efficiently
    # compressed the way an incidental embedded logo would be. Give it a
    # generous timeout too — this is a single deliberate image, not one
    # of dozens in a batch, so it's worth waiting longer for.
    description = _describe_image(
        img_bytes, context_hint=fname,
        min_bytes=0, timeout=max(OLLAMA_TIMEOUT, VISION_TIMEOUT_STANDALONE),
    )
    if not description:
        logger.warning(
            "  → No description produced for %s. Check the log line above "
            "for the specific reason (Ollama unreachable, vision model not "
            "pulled, or the request timed out).", fname
        )
        return []

    section = _guess_section(description, fname)
    return [IngestedChunk(
        text          = f"[Image: {fname}]\n{description}",
        source_file   = fname,
        page_or_slide = 0,
        section       = section,
        category      = _guess_category(section),
        record        = {"name": fname, "source": "standalone_image"},
        chunk_type    = "vision",
        confidence    = 0.75,
    )]


# ─── Public entry point ────────────────────────────────────────────────────────

def ingest_file(path: str, progress: Optional["Progress"] = None) -> List[IngestedChunk]:
    """
    Ingest a single file and return a list of IngestedChunk objects.

    Supported extensions: .pdf, .pptx, .ppt, .png, .jpg, .jpeg, .webp, .bmp,
    .xlsx, .xls, .csv

    ``progress``: optional shared rich.progress.Progress instance so a
    directory-level ingest can render one combined display (overall file
    bar + per-file page/slide bar) instead of each file popping up its
    own separate progress bar.

    Raises FileNotFoundError or ValueError for unsupported types.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = p.suffix.lower()
    if ext == ".pdf":
        return _ingest_pdf(str(p), progress=progress)
    elif ext in {".pptx", ".ppt"}:
        return _ingest_pptx(str(p), progress=progress)
    elif ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return _ingest_image(str(p))
    elif ext in {".xlsx", ".xls"}:
        # Try complex ingestor first for files that might have hierarchical structure
        try:
            from ingestion.complex_xlsx_ingestor import ingest_complex_xlsx
            chunks = ingest_complex_xlsx(str(p))
            if chunks:
                return chunks
        except Exception as exc:
            logger.warning("Complex XLSX ingestor failed for %s: %s; falling back to simple ingestor.", p, exc)
        # Fallback to simple ingestor
        from ingestion.xlsx_ingestor import ingest_xlsx
        return ingest_xlsx(str(p))
    elif ext == ".csv":
        from ingestion.xlsx_ingestor import ingest_csv
        return ingest_csv(str(p))
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def ingest_directory(directory: str, extensions: Optional[List[str]] = None) -> List[IngestedChunk]:
    """
    Recursively ingest all supported files in a directory.

    Parameters
    ----------
    directory  : Path to the folder to scan.
    extensions : Limit to specific extensions, e.g. ['.pdf', '.pptx'].
                 Defaults to all supported types.
    """
    supported = {".pdf", ".pptx", ".ppt", ".png", ".jpg", ".jpeg", ".webp", ".bmp",
                 ".xlsx", ".xls", ".csv"}
    if extensions:
        supported = {e.lower() for e in extensions} & supported

    targets = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            if Path(fpath).suffix.lower() in supported:
                targets.append(fpath)

    all_chunks: List[IngestedChunk] = []
    progress = _make_progress()

    if progress is not None:
        with progress:
            files_task = progress.add_task(
                "[green]Overall progress", total=len(targets), extra="starting…"
            )
            for fpath in targets:
                progress.update(files_task, extra=f"{os.path.basename(fpath)}")
                logger.info("Ingesting: %s", fpath)
                try:
                    chunks = ingest_file(fpath, progress=progress)
                    all_chunks.extend(chunks)
                    logger.info("  → %d chunks", len(chunks))
                except Exception as exc:
                    logger.error("  ✗ Failed to ingest %s: %s", fpath, exc)
                progress.update(files_task, advance=1)
    else:
        # rich not installed — fall back to plain logging, no bar.
        for fpath in targets:
            logger.info("Ingesting: %s", fpath)
            try:
                chunks = ingest_file(fpath)
                all_chunks.extend(chunks)
                logger.info("  → %d chunks", len(chunks))
            except Exception as exc:
                logger.error("  ✗ Failed to ingest %s: %s", fpath, exc)

    return all_chunks
