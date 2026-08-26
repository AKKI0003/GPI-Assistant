"""
ingestion/image_extractor.py

Extracts embedded images (charts, figures, diagrams, logos, photos) from
a PDF, saving each to disk and recording page number, bounding box,
dimensions, and (best-effort) caption.

Primary backend: PyMuPDF (`fitz`), which can pull the actual embedded
image bytes plus an accurate bounding box per page. Fallback backend
(when PyMuPDF is not installed): pdfplumber, which can report image
position/size but -- unlike PyMuPDF -- cannot cleanly hand back the
original embedded raster bytes for every PDF image-encoding variant, so
the fallback records bounding-box/dimension metadata only and appends an
explicit extraction warning rather than silently pretending a full
extraction happened.
"""

from __future__ import annotations

import itertools
import logging
import os
from typing import List, Optional, Tuple

from .image import Image

logger = logging.getLogger(__name__)

_IMAGE_ID_COUNTER = itertools.count(1)

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _FITZ_AVAILABLE = False
    logger.info("PyMuPDF (fitz) not installed; image extraction will use the reduced pdfplumber path.")

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _PDFPLUMBER_AVAILABLE = False


class ImageExtractor:
    """Extracts images from a PDF into `output_dir`, returning a list of
    `Image` model objects with full provenance."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def extract(self, file_path: str) -> Tuple[List[Image], List[str]]:
        """Returns (images, warnings). Warnings are returned rather than
        raised/logged-only so the caller (`PDFIngestor`) can attach them
        to the Document's own `extraction_warnings` list for downstream
        visibility."""
        if _FITZ_AVAILABLE:
            return self._extract_with_fitz(file_path)
        if _PDFPLUMBER_AVAILABLE:
            return self._extract_with_pdfplumber_bbox_only(file_path)
        logger.error("Cannot extract images: neither PyMuPDF nor pdfplumber is installed.")
        return [], ["Image extraction unavailable: no supported PDF library installed."]

    # -- PyMuPDF path (full extraction) ------------------------------------

    def _extract_with_fitz(self, file_path: str) -> Tuple[List[Image], List[str]]:
        images: List[Image] = []
        warnings: List[str] = []
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        doc = fitz.open(file_path)
        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                page_number = page_index + 1
                try:
                    image_list = page.get_images(full=True)
                except Exception as exc:
                    warnings.append(f"Failed to list images on page {page_number}: {exc}")
                    continue

                for image_info in image_list:
                    xref = image_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                    except Exception as exc:
                        warnings.append(
                            f"Failed to extract image xref {xref} on page {page_number}: {exc}"
                        )
                        continue

                    image_bytes = base_image.get("image")
                    if not image_bytes:
                        continue
                    ext = base_image.get("ext", "png")
                    file_name = f"{base_name}_p{page_number}_{next(_IMAGE_ID_COUNTER):04d}.{ext}"
                    file_path_out = os.path.join(self.output_dir, file_name)
                    try:
                        with open(file_path_out, "wb") as handle:
                            handle.write(image_bytes)
                    except OSError as exc:
                        warnings.append(f"Failed to save image on page {page_number}: {exc}")
                        continue

                    bbox = self._safe_bbox(page, xref)
                    caption = self._guess_caption(page, bbox)

                    images.append(
                        Image(
                            image_id=f"image-{next(_IMAGE_ID_COUNTER):06d}",
                            page_number=page_number,
                            file_path=file_path_out,
                            width=base_image.get("width", 0),
                            height=base_image.get("height", 0),
                            image_format=ext,
                            bounding_box=bbox,
                            caption=caption,
                            extraction_method="pymupdf",
                        )
                    )
        finally:
            doc.close()

        return images, warnings

    @staticmethod
    def _safe_bbox(page, xref: int) -> Optional[Tuple[float, float, float, float]]:
        try:
            rects = page.get_image_rects(xref)
            if rects:
                rect = rects[0]
                return (rect.x0, rect.y0, rect.x1, rect.y1)
        except Exception:
            pass
        return None

    @staticmethod
    def _guess_caption(page, bbox: Optional[Tuple[float, float, float, float]]) -> Optional[str]:
        """Best-effort caption detection: look for a short line of text
        immediately below the image's bounding box (the conventional
        position for a figure caption in most corporate report layouts).
        """
        if bbox is None:
            return None
        try:
            _, _, x1, y1 = bbox
            caption_zone = fitz.Rect(bbox[0], y1, x1, y1 + 40)
            text = page.get_textbox(caption_zone).strip()
            if text and len(text) <= 200:
                return text
        except Exception:
            return None
        return None

    # -- pdfplumber fallback (bounding box / metadata only) ----------------

    def _extract_with_pdfplumber_bbox_only(self, file_path: str) -> Tuple[List[Image], List[str]]:
        images: List[Image] = []
        warnings = [
            "PyMuPDF not installed: image extraction is running in reduced mode "
            "(bounding-box and dimension metadata only, no image files saved)."
        ]
        with pdfplumber.open(file_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                for raw_image in page.images:
                    width = int(raw_image.get("width", 0))
                    height = int(raw_image.get("height", 0))
                    bbox = (
                        raw_image.get("x0", 0.0),
                        raw_image.get("top", 0.0),
                        raw_image.get("x1", 0.0),
                        raw_image.get("bottom", 0.0),
                    )
                    images.append(
                        Image(
                            image_id=f"image-{next(_IMAGE_ID_COUNTER):06d}",
                            page_number=page_index,
                            file_path="",  # not extractable in this reduced mode
                            width=width,
                            height=height,
                            image_format="unknown",
                            bounding_box=bbox,
                            caption=None,
                            extraction_method="pdfplumber_bbox_only",
                        )
                    )
        return images, warnings
