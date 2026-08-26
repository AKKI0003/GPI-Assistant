"""
ingestion/

Enterprise document-intelligence ingestion package.

This package is a SECOND, independent knowledge source that sits
alongside (never inside/instead of) the existing JSON-based
`KnowledgeIndex`. The intended final architecture:

    User Question
        |
    Intent Detection
        |
    Company Knowledge? --> JSON Retriever  ---\
        |                                       >-- Merge Context --> LLM
    Annual Report / Financial / PDF Knowledge? --> PDF Retriever  ---/

Nothing in `core/`, `retrieval/knowledge_index.py`, or any other
existing module is modified by this package. A future retrieval-merge
step is expected to import `PDFIngestor` (or, for a corpus of many
files, iterate `ingestion.utils.iter_files_lazily`), call
`document.all_chunks()` on each resulting `Document`, and feed those
chunks into whatever merge/ranking step sits between "JSON Retriever"
and "Merge Context" in the diagram above.

Adding a new format later (Word, PowerPoint, Excel, HTML, Markdown, a
web page, SharePoint) requires only a new `BaseIngestor` subclass
decorated with `@register_ingestor` -- nothing here needs to change.
"""

from .base_ingestor import BaseIngestor, get_ingestor_for, register_ingestor, registered_extensions
from .document import Document, DocumentStatistics, PageInfo
from .image import Image
from .metadata import DocumentMetadata
from .pdf_ingestor import PDFIngestor
from .section import Chunk, Section
from .table import Table

__all__ = [
    "BaseIngestor",
    "register_ingestor",
    "get_ingestor_for",
    "registered_extensions",
    "Document",
    "DocumentStatistics",
    "PageInfo",
    "DocumentMetadata",
    "Section",
    "Chunk",
    "Table",
    "Image",
    "PDFIngestor",
]

__version__ = "1.0.0"
