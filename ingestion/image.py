"""
ingestion/image.py

Structured model for images, charts, figures, diagrams, and logos
extracted from a source document. Kept as a distinct model (rather than
folded into Table or Section) because images need their own file-path
and dimension bookkeeping, and because a future vision-model pass (to
caption/describe charts the way `ingest/ingest.py` already does for
slide images) will want a single, stable place to attach that output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

BoundingBox = Tuple[float, float, float, float]  # (x0, top, x1, bottom)


@dataclass
class Image:
    """A single extracted image/figure/chart/logo."""

    image_id: str
    page_number: int
    file_path: str
    width: int = 0
    height: int = 0
    image_format: str = "png"
    bounding_box: Optional[BoundingBox] = None
    caption: Optional[str] = None
    extraction_method: str = "unknown"  # "pymupdf" | "pdfplumber_bbox_only"

    @property
    def aspect_ratio(self) -> float:
        if not self.height:
            return 0.0
        return self.width / self.height

    def is_likely_logo(self) -> bool:
        """Small, roughly-square images near the top of the first few
        pages are almost always a masthead/logo rather than a
        chart/figure worth indexing for retrieval. This is a coarse
        heuristic used only to de-prioritize (never to silently drop)
        such images."""
        if self.width == 0 or self.height == 0:
            return False
        near_square = 0.6 <= self.aspect_ratio <= 1.6
        small = self.width < 400 and self.height < 400
        return near_square and small

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "page_number": self.page_number,
            "file_path": self.file_path,
            "width": self.width,
            "height": self.height,
            "image_format": self.image_format,
            "bounding_box": self.bounding_box,
            "caption": self.caption,
            "extraction_method": self.extraction_method,
        }
