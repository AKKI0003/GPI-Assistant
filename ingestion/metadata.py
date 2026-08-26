"""
ingestion/metadata.py

Document-level metadata model for the Document Intelligence ingestion
pipeline, plus the heuristics used to auto-detect fields (company, year,
document type) that are not present verbatim in a PDF's embedded
metadata dictionary.

This module intentionally has ZERO dependency on any specific parsing
library (fitz / pdfplumber / etc). Callers extract the raw values with
whatever library they use, then hand them to `MetadataExtractor` here to
normalize into a `DocumentMetadata` instance. This keeps the data model
stable even if the extraction backend changes later.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ingestion.ner import extract_organizations

logger = logging.getLogger(__name__)

# Known companies this assistant's knowledge base cares about. Extend this
# list as new issuers/competitors are ingested -- it is only ever used as a
# best-effort heuristic, never as a hard requirement.
KNOWN_COMPANIES: List[str] = [
    "Godfrey Phillips India",
    "Godfrey Phillips",
    "GPI",
    "ITC Limited",
    "ITC",
    "VST Industries",
    "VST",
    "Philip Morris International",
    "Philip Morris",
    "British American Tobacco",
    "BAT",
    "Golden Tobacco",
    "NTC Industries",
]

_DOCUMENT_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "annual_report": ["annual report", "integrated report", "report and accounts"],
    "financial_statement": [
        "balance sheet", "profit and loss", "cash flow statement",
        "statement of financial position", "auditor's report",
    ],
    "esg_report": ["esg report", "sustainability report", "environmental social governance"],
    "presentation": ["investor presentation", "earnings call", "quarterly presentation"],
    "press_release": ["press release", "media release"],
    "policy": ["policy document", "code of conduct", "whistleblower policy"],
    "regulatory_filing": ["form 20-f", "form 6-k", "sec filing", "annual return"],
}

_YEAR_PATTERN = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")


@dataclass
class DocumentMetadata:
    """Normalized, best-effort metadata for one ingested document."""

    source_file: str = ""
    title: str = ""
    company: Optional[str] = None
    year: Optional[int] = None
    language: str = "unknown"
    page_count: int = 0
    document_type: str = "unknown"

    author: Optional[str] = None
    pdf_producer: Optional[str] = None
    pdf_creator: Optional[str] = None

    file_size_bytes: int = 0
    checksum_sha256: str = ""

    created_date: Optional[str] = None      # from filesystem
    modified_date: Optional[str] = None     # from filesystem
    pdf_created_date: Optional[str] = None  # from embedded PDF metadata
    pdf_modified_date: Optional[str] = None # from embedded PDF metadata

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "title": self.title,
            "company": self.company,
            "year": self.year,
            "language": self.language,
            "page_count": self.page_count,
            "document_type": self.document_type,
            "author": self.author,
            "pdf_producer": self.pdf_producer,
            "pdf_creator": self.pdf_creator,
            "file_size_bytes": self.file_size_bytes,
            "checksum_sha256": self.checksum_sha256,
            "created_date": self.created_date,
            "modified_date": self.modified_date,
            "pdf_created_date": self.pdf_created_date,
            "pdf_modified_date": self.pdf_modified_date,
            "extra": self.extra,
        }


def compute_checksum(file_path: str, chunk_size: int = 1 << 20) -> str:
    """SHA-256 checksum of a file, read in chunks so large PDFs (annual
    reports can run to hundreds of MB with embedded images) never need to
    be loaded fully into memory just to be hashed."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            sha256.update(block)
    return sha256.hexdigest()


def guess_company(text_sample: str, filename: str) -> Optional[str]:
    """
    Best-effort company detection from the filename and a text sample
    (typically the first 1-2 pages, which is where a title page / cover
    page company name almost always appears).

    Two bugs this fixes:

    1. NAIVE SUBSTRING MATCHING. The old version did `candidate.lower()
       in haystack` with no word-boundary check. Short names like "ITC"
       (3 letters) are substrings of completely ordinary English words —
       "switch", "pitch", "kitchen", "district" — any of which showing
       up in a report's running text was enough to mistag it. Matching
       is now done on whole words only.

    2. FIRST-MATCH-IN-A-FIXED-PRIORITY-LIST, SCANNING FILENAME+TEXT AS
       ONE BLOB. The old version combined the filename and text sample
       into a single string and returned whichever KNOWN_COMPANIES entry
       happened to appear earliest in the (arbitrary) priority list —
       NOT whichever company the document actually is. This is exactly
       why a file literally named "VST_Annual_Report_2023.pdf" was
       getting tagged as "Godfrey Phillips India": a totally normal
       industry-overview sentence early in VST's OWN annual report
       ("...the market is led by ITC, followed by Godfrey Phillips India
       and VST Industries...") mentions GPI, and "Godfrey Phillips
       India"/"GPI" sit earlier in KNOWN_COMPANIES than "VST" — so GPI
       won regardless of the filename or of which company the report was
       actually about. Every competitor's annual report is close to
       guaranteed to mention its peers in exactly this kind of
       industry-overview sentence, so this wasn't a rare edge case — it
       was a near-certain mistag for VST/ITC specifically, since GPI's
       own name sits first in the priority list.

       Fixed by: (a) trusting the filename first, since a file literally
       named "VST_Annual_Report_2023.pdf" is a far more reliable signal
       than one incidental mention buried in the text, and only falling
       back to the text sample if the filename doesn't resolve it
       unambiguously; (b) when falling back to text, picking whichever
       company is named MOST FREQUENTLY rather than merely first/at-all
       — a report ABOUT VST says "VST" dozens of times and mentions ITC
       or GPI only in passing, so frequency correctly identifies the
       actual subject instead of whichever name happens to sit earliest
       in an arbitrary list.
    """
    fname_norm = filename.lower().replace("_", " ").replace("-", " ")
    fname_hits = [
        candidate for candidate in KNOWN_COMPANIES
        if re.search(r"\b" + re.escape(candidate.lower()) + r"\b", fname_norm)
    ]
    if fname_hits:
        # Prefer the longest/most specific matching name (e.g. "VST
        # Industries" over the bare "VST") in case several entries match.
        return max(fname_hits, key=len)

    # NER-assisted counting (see ingestion/ner.py): restrict the
    # frequency count to text spans spaCy actually recognizes as an
    # organization name, rather than any raw substring match anywhere on
    # the page -- a table caption, a footnote, or unrelated boilerplate
    # that happens to contain the word no longer counts the same as a
    # genuine mention. Falls straight through to the plain whole-text
    # scan below (unchanged) if spaCy isn't available or found nothing
    # matching a known company.
    orgs = extract_organizations(text_sample)
    if orgs:
        org_text_lower = " ".join(orgs).lower()
        ner_counts: Dict[str, int] = {}
        for candidate in KNOWN_COMPANIES:
            pattern = r"\b" + re.escape(candidate.lower()) + r"\b"
            hits = len(re.findall(pattern, org_text_lower))
            if hits:
                ner_counts[candidate] = hits
        if ner_counts:
            best = max(ner_counts.values())
            tied = [c for c, n in ner_counts.items() if n == best]
            return max(tied, key=len)

    text_lower = (text_sample or "").lower()
    counts: Dict[str, int] = {}
    for candidate in KNOWN_COMPANIES:
        pattern = r"\b" + re.escape(candidate.lower()) + r"\b"
        hits = len(re.findall(pattern, text_lower))
        if hits:
            counts[candidate] = hits
    if not counts:
        return None
    # Longer/more specific names ("Godfrey Phillips India") naturally
    # also satisfy shorter ones' patterns in surrounding context less
    # often than the reverse, but ties are still possible (e.g. "ITC"
    # counted separately from "ITC Limited") — break ties by preferring
    # the longer, more specific name.
    best_count = max(counts.values())
    tied = [c for c, n in counts.items() if n == best_count]
    return max(tied, key=len)


def guess_year(text_sample: str, filename: str) -> Optional[int]:
    """Best-effort fiscal/report year detection. Prefers a year found in
    the filename (usually the most reliable signal for a report named
    e.g. `VST_Annual_Report_2023.pdf`), falling back to the most
    frequently occurring plausible year in the sampled text."""
    filename_match = _YEAR_PATTERN.search(filename)
    if filename_match:
        return int(filename_match.group(0))

    matches = _YEAR_PATTERN.findall(text_sample)
    if not matches:
        return None
    counts: Dict[str, int] = {}
    for match in matches:
        counts[match] = counts.get(match, 0) + 1
    most_common = max(counts.items(), key=lambda item: item[1])[0]
    return int(most_common)


def guess_document_type(text_sample: str, filename: str) -> str:
    """Best-effort document-type classification via keyword scoring. This
    deliberately stays a flat keyword scan (not a model call) -- it runs
    once per ingested file across potentially thousands of documents and
    must stay cheap and deterministic."""
    haystack = f"{filename}\n{text_sample}".lower()
    scores: Dict[str, int] = {doc_type: 0 for doc_type in _DOCUMENT_TYPE_KEYWORDS}
    for doc_type, keywords in _DOCUMENT_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in haystack:
                scores[doc_type] += 1
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    return best_type if best_score > 0 else "unknown"


def file_system_dates(file_path: str) -> Dict[str, str]:
    stat = os.stat(file_path)
    return {
        "created_date": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }
