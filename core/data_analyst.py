"""
core/data_analyst.py

A LIVE pandas analysis engine for spreadsheet data (.xlsx/.xls/.csv dropped
into documents/raw/), completely separate from the PDF retrieval/RAG
pipeline in retrieval/pdf_retriever.py and retrieval/knowledge_index.py.

Why this exists
----------------
The RAG approach used for PDFs (retrieve pre-written chunks, including
the aggregate summaries ingestion/xlsx_ingestor.py precomputes) can only
ever answer a question shaped the same way as something already
precomputed. "Total sales by state" works because that exact grouping
was computed at ingestion time. "Total sales in Delhi for Product X in
June, compared to May" does NOT work that way -- it's a multi-condition,
dynamic query no amount of precomputed summaries can anticipate in
advance.

This module answers those questions for real: it loads the actual
spreadsheet into a pandas DataFrame, asks the LLM to translate the
question into a small, STRICT JSON query spec (filters / group-by /
aggregate / sort / chart), executes that spec directly against the
DataFrame with real pandas operations, and formats the actual computed
result. No arbitrary code from the LLM is ever executed -- the JSON spec
only describes filter/group/aggregate operations, which this module
carries out itself using validated column names and a small fixed set of
operators. This keeps it as safe as the rest of the pipeline while still
being genuinely dynamic.

Isolation from PDF retrieval (important)
-----------------------------------------
`is_analysis_query()` is the ONLY entry point core/engine.py needs to
check before doing anything else. It is deliberately cheap and returns
False immediately (before touching pandas, the LLM, or anything else) in
the two cases that matter for not disturbing PDF ingestion/retrieval:

  1. No spreadsheet has been ingested at all (the common case while
     someone is mid-way through ingesting PDFs) -- the registry scan
     finds nothing, so this always returns False and every question
     flows through the existing PDF/RAG path completely unchanged.
  2. The question doesn't reference anything spreadsheet-shaped (no
     analysis verb, and no column name / distinct value from any
     ingested sheet appears in it) -- again False, so a financial-
     statement question about a PDF is never intercepted here.

Only when both an analysis intent AND a spreadsheet-specific reference
are present does engine.py route here instead of the normal pipeline --
and even then, this module touches only the DataFrames it loaded itself;
it never calls into pdf_retriever/knowledge_index or modifies anything
they use.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.config import DOCUMENTS_DIR, CHARTS_OUTPUT_DIR
from core.xlsx_reshape import sheet_needs_reshaping, reshape_pivoted_sheet

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _PANDAS_AVAILABLE = False

_RAW_DIR = os.path.join(DOCUMENTS_DIR, "raw")

# How many distinct values of a low-cardinality text column to remember
# for (a) intent detection ("Delhi"/"Okhla" should trigger analysis
# routing) and (b) showing the LLM real examples when it writes the
# filter spec, so it fills in "Delhi" rather than inventing "New Delhi"
# or "delhi branch".
#
# Raised from 200 -> 3000: a column genuinely can have several hundred
# distinct values and still be a meaningful category to filter/group by
# -- e.g. a town/zone/branch column across many Indian cities easily
# exceeds 200 distinct entries in an 87,000+ row sales file, while still
# being nowhere near "every value is basically unique" (an actual
# unique-ID column in a file that size would have on the order of tens
# of thousands of distinct values, still safely excluded at this cap).
# At the old 200 cap, the ENTIRE column was silently dropped from both
# uses above the moment it had 201+ distinct values -- meaning a
# specific real value like "Okhla" was invisible to is_analysis_query's
# detection, and the question fell through to ordinary sampled RAG
# retrieval instead of the full-data analysis path, which is exactly
# the "only sees a handful of rows for that town/zone" symptom this
# fixes.
_MAX_SAMPLE_VALUES = 3000


def _coerce_numeric(series: "pd.Series") -> "pd.Series":
    """
    pd.to_numeric() alone silently turns any comma-thousands-separated
    or currency-symbol-prefixed cell ("4,000", "₹4,000", "$-500",
    accounting-style "(500)" for negative) into NaN instead of a real
    number, because none of those are valid numeric literals as-is.
    On an Excel export where a sales/value column is stored as TEXT
    with formatting like that (extremely common for Indian sales data:
4,000.50", "(120.00)" for a negative), the result is that almost
    every row silently drops out of outlier/aggregate calculations,
    leaving only the handful of rows that happened to already be
clean" numbers (e.g. a small percentage/ratio column) -- which is
    exactly how a real column of 4-figure and negative sales values
    produced outlier results that were all some unrelated small number
    like "7.68": that value was probably real, but from surviving rows
    in the WRONG scale, not the actual sales figures which were all
    silently discarded as unparseable.

    Only ever WIDENS what parses successfully compared to plain
    pd.to_numeric -- never reinterprets an already-clean numeric column
    differently, so this is safe as a drop-in replacement everywhere
    numeric coercion happens in this module.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series

    cleaned = series.astype(str).str.strip()
    # Accounting-style negatives: "(1,234.50)" -> "-1,234.50"
    is_paren_negative = cleaned.str.match(r"^\(.*\)$")
    cleaned = cleaned.where(
        ~is_paren_negative, "-" + cleaned.str.slice(1, -1)
    )
    # Strip anything that isn't a digit, minus sign, or decimal point
    # (currency symbols, thousands commas, stray whitespace, unit
    # suffixes like "Rs." or "Cr" someone typed into a cell).
    cleaned = cleaned.str.replace(r"[^\d\.\-]", "", regex=True)
    # A lone "-" or "" (came from a blank/dash-only cell, e.g. "-" used
    # as a placeholder for "no value") must not become "-0" or error.
    cleaned = cleaned.replace({"": None, "-": None, ".": None})
    return pd.to_numeric(cleaned, errors="coerce")


# ─── Registry: discover + lazily load every ingested spreadsheet ──────────────

@dataclass
class _SheetInfo:
    file_path: str
    sheet_name: str
    df: "pd.DataFrame"
    columns: List[str] = field(default_factory=list)
    categorical_values: Dict[str, List[str]] = field(default_factory=dict)


_registry: Optional[List[_SheetInfo]] = None
_registry_mtime_signature: Optional[Tuple] = None


def _spreadsheet_files() -> List[str]:
    if not os.path.isdir(_RAW_DIR):
        return []
    files = []
    for ext in ("*.xlsx", "*.xls", "*.csv"):
        files.extend(glob.glob(os.path.join(_RAW_DIR, ext)))
    return sorted(files)


def _mtime_signature(files: List[str]) -> Tuple:
    """Cheap fingerprint so the registry reloads automatically if a
    spreadsheet is added/changed, without re-reading every file on every
    single question."""
    return tuple((f, os.path.getmtime(f)) for f in files)


def _load_registry(force: bool = False) -> List[_SheetInfo]:
    global _registry, _registry_mtime_signature

    files = _spreadsheet_files()
    signature = _mtime_signature(files)

    if not force and _registry is not None and signature == _registry_mtime_signature:
        return _registry

    sheets: List[_SheetInfo] = []
    if not files or not _PANDAS_AVAILABLE:
        _registry = sheets
        _registry_mtime_signature = signature
        return sheets

    for path in files:
        try:
            if path.lower().endswith(".csv"):
                df = _read_csv_robust(path)
                if df is None:
                    continue
                all_sheets = {os.path.splitext(os.path.basename(path))[0]: df}
            else:
                all_sheets = pd.read_excel(path, sheet_name=None)
        except Exception:
            logger.exception("data_analyst: failed to load %s", path)
            continue
        for sheet_name, df in all_sheets.items():
            if df.empty:
                continue

            # Some real-world spreadsheets are cross-tab / pivoted MIS
            # reports (a merged multi-row header -- e.g. a state name
            # spanning several "Plan"/"MTD"/"%Ach" columns beneath it)
            # rather than one row per record. A normal header=0 load
            # can't represent that at all -- it comes back full of
            # "Unnamed: N" columns with the real sub-headers sitting in
            # what looks like the first data row, so nothing can be
            # filtered/grouped/aggregated on it correctly. Detection is
            # deliberately conservative (see xlsx_reshape.py) and this
            # ONLY activates for a sheet that actually shows that
            # shape -- an already-flat sheet (the common case) takes
            # the exact same path it always has, completely unchanged.
            if not path.lower().endswith(".csv") and sheet_needs_reshaping(df):
                try:
                    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
                    reshaped = reshape_pivoted_sheet(raw, sheet_name=sheet_name)
                    # BUG (root cause of almost every "wrong column name"
                    # symptom reported): reshaped was computed here but
                    # never actually assigned back to `df` on success --
                    # only the FAILURE branch below touched `df` at all
                    # (and even then, wrongly set it to None). That meant
                    # the reshape result was silently thrown away on
                    # EVERY successful run: `df` stayed as the original
                    # naive flat/multi-header load the entire time, while
                    # the LLM's prompt (built around the reshaped
                    # State/Metric/Is_Rollup-style schema) described a
                    # dataframe that didn't actually exist. That mismatch
                    # is why the model kept "hallucinating" column names
                    # that seemed to come from nowhere -- they were real,
                    # they just belonged to data that was never actually
                    # loaded.
                    if reshaped is not None and not reshaped.empty:
                        df = reshaped
                except Exception:
                    logger.exception(
                        "data_analyst: pivoted-sheet reshape failed for %s / %s -- falling back to the plain flat load", path, sheet_name,
                    )
                    # Keep the original flat `df` as a fallback -- it's
                    # usually still readable even if not ideally shaped,
                    # which beats crashing on `None.columns` below.

            df.columns = [str(c).strip() for c in df.columns]
            categorical_values: Dict[str, List[str]] = {}
            for col in df.columns:
                dtype_str = str(df[col].dtype)
                if df[col].dtype == object or dtype_str in ("string", "str"):
                    distinct = df[col].dropna().astype(str).str.strip().unique()
                    if 1 < len(distinct) <= _MAX_SAMPLE_VALUES:
                        categorical_values[col] = sorted(distinct.tolist())[:_MAX_SAMPLE_VALUES]
            sheets.append(_SheetInfo(
                file_path=path, sheet_name=sheet_name, df=df,
                columns=list(df.columns), categorical_values=categorical_values,
            ))

    _registry = sheets
    _registry_mtime_signature = signature
    return sheets


def _read_csv_robust(path: str) -> Optional["pd.DataFrame"]:
    """
    pd.read_csv(path) alone assumes comma-delimited UTF-8 with no BOM --
    fine for a CSV written by pandas itself, but real-world exports
    routinely aren't:
      - Excel-on-Windows "Save As CSV" writes a UTF-8 BOM (\\ufeff),
        which otherwise ends up glued onto the first column's NAME
        (e.g. "\\ufeffBranch" instead of "Branch"), silently breaking
        every filter/group-by that references that column by name.
      - Regional/European locale exports often use ";" as the field
        separator (since "," is the decimal separator there), which
        pandas' default comma parser turns into one single garbage
        column instead of erroring -- an easy silent failure to miss.
      - Non-UTF-8 encodings (Windows-1252/"ANSI", Latin-1) are common
        from older export tools and raise UnicodeDecodeError under
        plain UTF-8 decoding.

    Tries the common combinations in order and returns the first one
    that parses into more than one column (the tell that delimiter
    detection actually worked), or None if nothing usable was found.
    """
    encodings = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
        except UnicodeDecodeError:
            continue
        except Exception:
            # Delimiter sniffing itself can raise on a genuinely weird
            # file -- fall back to a plain comma-separated attempt
            # before giving up on this encoding entirely.
            try:
                df = pd.read_csv(path, encoding=encoding)
            except Exception:
                continue
        if df.shape[1] > 1:
            return df
    # Nothing produced more than one column -- last resort: whatever
    # the first encoding gives with a plain comma parse, so a
    # genuinely single-column CSV still loads rather than being
    # dropped entirely.
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        logger.exception("data_analyst: could not parse CSV %s with any encoding/delimiter", path)
        return None


# ─── Intent detection ──────────────────────────────────────────────────────────

_ANALYSIS_VERB_PATTERN = re.compile(
    r"\b("
    r"analy[sz]\w*|"
    r"total\w*|sum\b|average|avg|mean\b|median|count\w*|how many|"
    r"top\s*\d*|bottom\s*\d*|highest|lowest|maximum|minimum|max\b|min\b|"
    r"major|primary|leading|biggest|largest|most\b|main\b|key\b|"
    r"best\b|worst\b|perform\w*|rank\w*|"
    r"compar\w*|versus|\bvs\b|"
    r"trend\w*|growth|breakdown|distribut\w*|division|"
    r"filter\w*|only\s+(show|include)|where|"
    r"group\s*by|per\s+\w+|by\s+\w+|"
    r"chart|graph|plot|visuali[sz]e|"
    # These trunk words previously had a trailing \b applied directly
    # after the literal, which requires the very next character to be a
    # non-word character -- that's true for the bare word itself, but
    # NOT for its own ordinary inflections ("outlier" -> "outlierS",
    # "deviat" -> "deviatION"/"deviatING", both have a word character
    # immediately following the trunk, so \b silently fails to match
    # there). "detect OUTLIERS for X" -- almost certainly the single
    # most common way someone actually phrases an outlier question --
    # was therefore NEVER recognized as an analysis query at all, and
    # got answered by the generic LLM/RAG path instead of the real
    # outlier-detection engine below (which is what produced the
    # small-scale, wrong-branch, made-up-looking numbers reported: that
    # answer was assembled by an LLM guessing from coarse pre-aggregated
    # summary text, not computed from the real 87k-row data at all).
    r"outlier\w*|anomal\w*|unusual|abnormal|irregular|spike\w*|deviat\w*|"
    r"stand(s)?\s+out|too\s+(high|low)"
    r")\b",
    re.IGNORECASE,
)


_STRONG_ANALYSIS_VERB_PATTERN = re.compile(
    r"\b(outlier\w*|anomal\w*|unusual|abnormal|irregular|spike\w*|deviat\w*|"
    r"stand(s)?\s+out|too\s+(high|low)|top\s*\d+|bottom\s*\d+)\b",
    re.IGNORECASE,
)


def is_analysis_query(question: str) -> bool:
    """Cheap, fast-failing check. Returns False immediately (no pandas
    load, no LLM call) unless a spreadsheet is actually registered AND
    the question plausibly refers to it -- see the module docstring for
    why both conditions matter for leaving PDF retrieval untouched."""
    if not _PANDAS_AVAILABLE:
        return False
    sheets = _load_registry()
    if not sheets:
        return False

    q_lower = question.lower()
    has_analysis_verb = bool(_ANALYSIS_VERB_PATTERN.search(q_lower))
    if not has_analysis_verb:
        return False

    # "find outliers/anomalies/..." has no sensible meaning other than
    # "run real statistical analysis on the data" -- there's no realistic
    # PDF-financial-report question that would ALSO happen to contain one
    # of these words, unlike "total"/"compare"/"by X" which legitimately
    # show up in ordinary prose questions about a PDF too. Requiring an
    # exact column/value text match before trusting an outlier request
    # just reopens the door to it silently falling through to the
    # generic LLM/RAG path -- which, having no real pandas access, can
    # only answer from whatever pre-chunked row-batch text (e.g. "rows
    # 201-300") happened to get retrieved, i.e. a few hundred of 87,000+
    # rows presented as if it were the whole file. Since a spreadsheet
    # IS registered (checked above) and the verb is this unambiguous,
    # route straight to the real engine and let spec-generation (which
    # sees the sheet's actual column list) resolve which column/value
    # the question means, rather than silently guessing via RAG.
    if _STRONG_ANALYSIS_VERB_PATTERN.search(q_lower):
        return True

    from rapidfuzz import fuzz

    q_words_all = set(re.findall(r"[a-z0-9]+", q_lower))

    # A single generic business word (the kind of word that shows up in
    # almost EVERY question about a company, spreadsheet-related or not)
    # is not enough on its own to prove a question is really about a
    # specific ingested spreadsheet. This was the exact cause of "tell me
    # the strategies/policies/sales techniques ... and compare them"
    # (a plain PDF/strategy question) getting routed here: "compare"
    # satisfied _ANALYSIS_VERB_PATTERN, then the single word "sales" --
    # present in the question only because of the unrelated phrase "sales
    # techniques" -- happened to also be (part of) a column name or file
    # name in an ingested spreadsheet, which was treated as proof the
    # question was about that spreadsheet. Excluding common generic terms
    # from the loose single-word fallback keeps real matches (a real
    # spreadsheet question almost always also names something SPECIFIC --
    # a branch, a product, a person, a date -- not just "sales" alone)
    # while refusing to let one generic word hijack an unrelated question.
    _GENERIC_TERMS = {
        "sales", "sale", "revenue", "total", "amount", "value", "values",
        "data", "name", "names", "date", "dates", "year", "years",
        "company", "companies", "report", "reports", "state", "states",
        "branch", "branches", "region", "regions", "product", "products",
        "category", "categories", "type", "types", "status", "code",
        "codes", "number", "numbers", "record", "records", "file", "files",
    }
    # A bare 4-digit calendar/fiscal year ("2023", "2024", ...) is just as
    # generic, for this purpose, as the word "year" itself -- almost
    # every financial question names a specific year, so a spreadsheet
    # having a Year/FY column whose values are literally "2023"/"2024"
    # would otherwise let THAT alone satisfy the match for nearly any
    # finance-flavoured question, spreadsheet-relevant or not. This is
    # what let "...that GPI followed in 2023 and 2024 and compare them"
    # keep matching even after every word-based fix above: the year
    # digits themselves, not any word, were the categorical-value match.
    _YEAR_TOKEN_RE = re.compile(r"^(19|20)\d{2}$")

    def _is_bare_generic(text_lower: str) -> bool:
        return text_lower in _GENERIC_TERMS or bool(_YEAR_TOKEN_RE.match(text_lower))

    def word_overlaps(text_lower: str, min_len: int = 4) -> bool:
        """Exact-substring match first (cheapest, catches most cases);
        falls back to per-word overlap (>=min_len chars, so short noise
        words like "of"/"the" can't false-positive) so a column/value
        matches regardless of singular/plural or word order -- "BRANDS"
        matching a question that says "brand" (singular), or "State"
        matching "states". Both checks are still exact-text, not
        typo-tolerant -- see fuzzy_overlaps() below for that.

        A bare generic term (see _GENERIC_TERMS) is excluded from BOTH
        the substring check and the per-word fallback -- it can still
        contribute to a match as part of a more specific multi-word
        phrase (e.g. "Sales Amount" as a whole still counts, since that
        phrase is specific even though "sales" alone isn't), just not by
        itself.

        BUG (reproduced live): "tell me the strategies, policies, sales
        techniques ... that GPI followed in 2023 and 2024 and compare
        them" -- a plain PDF/strategy question with no spreadsheet
        relevance at all -- still routed to analysis and returned "No
        rows matched that." "sales" was already excluded by
        _GENERIC_TERMS above, but the substring check itself had NO
        minimum length: a company/categorical value as short as "GPI"
        (3 characters, sitting in a "Company" column in an ingested
        spreadsheet) matched via `text_lower in q_lower` the instant the
        question mentioned "GPI" -- which is nearly every question,
        analysis or not, since GPI is the company this whole assistant
        is about. The per-word fallback below already required
        min_len=4 for exactly this reason; the substring check was
        never given the same floor. Applying it here means a short
        value like "GPI"/"ITC"/"VST" can no longer single-handedly
        prove a question is about a specific spreadsheet -- it still
        contributes normally as part of a longer, more distinctive
        phrase (e.g. "GPI Delhi Branch" as a whole, if that's a real
        categorical value, remains long enough to match on its own).
        """
        if _is_bare_generic(text_lower):
            return False
        if len(text_lower) >= min_len and text_lower in q_lower:
            return True
        words = set(
            w for w in re.findall(r"[a-z0-9]+", text_lower)
            if len(w) >= min_len and not _is_bare_generic(w)
        )
        return bool(words & q_words_all)

    def fuzzy_overlaps(text_lower: str, min_len: int = 5, threshold: int = 82) -> bool:
        """
        Typo-tolerant fallback: does any individual word in the question
        closely resemble any individual word in this column/value (e.g.
        "eurrope" vs "europe")? Only applied to longer words (>=5 chars)
        and only as a LAST resort after exact/word-overlap matching
        already failed, since fuzzy matching on short words risks
        false-positiving on unrelated short words that just happen to
        look similar. Generic terms are excluded here too, for the same
        reason as in word_overlaps() above.
        """
        if _is_bare_generic(text_lower):
            return False
        text_words = [
            w for w in re.findall(r"[a-z0-9]+", text_lower)
            if len(w) >= min_len and not _is_bare_generic(w)
        ]
        if not text_words:
            return False
        q_words_long = [w for w in q_words_all if len(w) >= min_len and not _is_bare_generic(w)]
        for tw in text_words:
            for qw in q_words_long:
                if fuzz.ratio(tw, qw) >= threshold:
                    return True
        return False

    for sheet in sheets:
        for col in sheet.columns:
            col_lower = col.lower()
            if word_overlaps(col_lower) or fuzzy_overlaps(col_lower):
                return True
        for values in sheet.categorical_values.values():
            for v in values:
                v_lower = v.lower()
                # A composite value ("NCT of Delhi", "Delhi - HQ Branch")
                # requiring its ENTIRE string to appear in the question
                # misses the common case of the question naming just the
                # meaningful part ("delhi branch") -- word_overlaps()
                # already handles matching on any individual real word.
                if word_overlaps(v_lower) or fuzzy_overlaps(v_lower):
                    return True
        # Also match on the sheet/file name itself ("primary sales data").
        #
        # BUG (reproduced live, second time): this used to be a raw
        # substring check with NO length guard and NO generic-term
        # filter at all -- it never went through word_overlaps() above,
        # so every fix applied there (the length floor, the _GENERIC_TERMS
        # stoplist) had zero effect on it. A query containing the single
        # word "csr" ("...which is not related to CSR that GPI
        # followed...") matched instantly the moment ANY ingested
        # spreadsheet's sheet/file name happened to contain "csr"
        # anywhere in it (e.g. a CSR-spend tracking sheet) -- exactly
        # the kind of single-generic-word hijack the word_overlaps() fix
        # was meant to prevent everywhere, just reached through a
        # different, unguarded code path. Routing both checks through
        # the same word_overlaps() function they should have used from
        # the start applies the identical length/genericness rules here.
        sheet_name_lower = sheet.sheet_name.lower()
        if word_overlaps(sheet_name_lower) or fuzzy_overlaps(sheet_name_lower):
            return True
        base = os.path.splitext(os.path.basename(sheet.file_path))[0].lower()
        base_spaced = base.replace("_", " ")
        if base_spaced and (word_overlaps(base_spaced) or fuzzy_overlaps(base_spaced)):
            return True
        # Match month names to jan2024/feb2024 etc. sheet naming convention
        month_map = {
            "january": "jan", "february": "feb", "march": "mar", "april": "apr",
            "may": "may", "june": "jun", "july": "jul", "august": "aug",
            "september": "sep", "october": "oct", "november": "nov", "december": "dec",
            "jan": "jan", "feb": "feb", "mar": "mar", "apr": "apr",
            "jun": "jun", "jul": "jul", "aug": "aug", "sep": "sep",
            "oct": "oct", "nov": "nov", "dec": "dec",
        }
        for month_name, month_abbr in month_map.items():
            if month_name in q_lower and month_abbr in sheet.sheet_name.lower():
                return True
            if month_name in q_lower and month_abbr in base:
                return True

    return False


# ─── LLM-driven query spec ─────────────────────────────────────────────────────

_SPEC_SYSTEM_PROMPT = """\
You translate a natural-language data question into a STRICT JSON query \
spec against a spreadsheet. You do not answer the question yourself and \
you do not write pandas/Python code -- you only output the JSON spec \
below, which a separate program will execute exactly.

Schema (all fields optional except where noted):
{
"sheet": "<exact sheet name from the AVAILABLE SHEETS list -- REQUIRED unless using "compare">",
"filters": [
    {"column": "<exact column name>", "op": "==|!=|>|>=|<|<=|contains|in|date==|date!=|date>|date>=|date<|date<=|date_between", "value": <value or [values] for 'in' or [start, end] for 'date_between'>}
  ],
"group_by": ["<column name>", ...],
"aggregate": {"column": "<numeric column name>", "func": "sum|mean|count|min|max|median"},
"outliers": {"column": "<numeric column name>", "method": "iqr|zscore", "group_by": ["<column name>", ...]},
"exclude_outliers": {"column": "<numeric column name>", "method": "iqr|zscore", "group_by": ["<column name>", ...]},
"statistics": {"column": "<numeric column name>", "funcs": ["mean|median|mode|std|var|min|max|sum|count|skew|kurtosis", ...], "group_by": ["<column name>", ...]},
"best_per_group": {"group_by": ["<column name>"], "rank_by": "<different column name>", "aggregate": {"column": "<numeric column>", "func": "sum"}, "limit_per_group": <integer>},
"top_per_group": <integer>,
"compare": {
"sheets": ["<exact sheet name 1>", "<exact sheet name 2>", ...],
"column": "<numeric column name to compare -- must exist, or have an equivalent, in every listed sheet>",
"func": "sum|mean|count|min|max|median",
"group_by": ["<column name>", ...],
"filters": [{"column": "...", "op": "...", "value": ...}]
  },
"compute": {
    "operations": [{"name": "<var name>", "spec": { ...full sub-spec, same schema as this one... }}, ...],
    "formula": "<python expression using the operation names as variables>",
    "label": "<human-readable result name>"
  },
"include_rollups": true,
"sort_descending": true,
"limit": <integer, e.g. 10 for "top 10">,
"chart": {"wanted": true|false, "type": "auto|bar|line|pie|area|scatter", "limit": <integer, optional>}
}

GENERAL PRINCIPLE: every field above is a REUSABLE, COMPOSABLE building block, not a menu you pick \
one item from. A real question is very often several of these AT ONCE (e.g. filter + group_by + \
outliers + aggregate, all together, all scoped to the same filtered rows) -- read the WHOLE question \
before writing the spec, identify every distinct thing being asked for, and set every field that \
applies. Never drop part of a compound question just because another part was easier to express.

Rules:
- Use ONLY column names and values that literally appear in the AVAILABLE SHEETS section below for \
THIS question's sheet. Never invent a column or guess a value's spelling -- copy it \
character-for-character from the real column list / sample values shown there. Every worked example \
below uses placeholder names like <numeric column> purely to illustrate the PATTERN -- always \
substitute the real column/sheet names for the file actually being queried, never the placeholder \
text itself, and never a column/value name borrowed from a different file or a previous conversation.
- Multiple filters are combined with AND.
- If the question asks for a total/sum/count/average of something, set "aggregate" accordingly. If \
it asks to group by more than one thing ("by state and product"), list all of them in "group_by".
- OUTLIERS: if the question asks to find outliers, anomalies, unusual/abnormal values, or what \
"stands out", set "outliers" -- the numeric column being checked goes in "column". If the question \
implies outliers should be judged WITHIN each division/category rather than across the whole sheet at \
once ("which towns have unusually high sales for their region", "find outliers by state"), put that \
column in outliers.group_by so each group is checked against its own normal range. Leave \
outliers.group_by empty when the question is about the sheet as a single whole. Default method to \
"iqr" unless the question specifically asks for a z-score/standard-deviation based check.
- AGGREGATE-THEN-OUTLIER: "outliers BY <category>" (e.g. "which regions have unusually high total \
sales") means aggregate the numeric column by that category FIRST, then find outliers in the \
resulting totals. Set "outliers" (column = the numeric column, group_by = [] empty) AND "aggregate" \
(same column, func "sum") AND "group_by" = [the category]. Do NOT put the category inside \
outliers.group_by for this pattern -- that would check outliers WITHIN each category instead of \
ACROSS categories, a different question.
- EXCLUDE-OUTLIERS: different from "outliers" above. When a ranking question ("best/highest/top X") \
adds a qualifier like "excluding outliers", "not an outlier", "ignoring spikes" -- it wants outlier \
rows REMOVED from the ranking, not reported. Set "exclude_outliers" (same shape as "outliers") \
instead of "outliers", and still set "group_by"/"aggregate"/"sort_descending"/"limit" to produce the \
actual ranking. Never set both "outliers" and "exclude_outliers" together -- they mean opposite things.
- "Best/worst/highest/lowest/top N performing" (for ONE flat ranking, not "per group" -- see below) \
ALWAYS means: set "sort_descending" (true for best/highest/top, false for worst/lowest/bottom) and \
"limit" (use the number given, e.g. "top 5" -> 5; otherwise 1 for a single "best X", or 10-15 for an \
open-ended "which X perform well"). A group_by/aggregate with no sort_descending/limit is NOT an \
answer to "best performing" -- it just lists every group in whatever order pandas produces.
- BEST/TOP PER GROUP: a question asking for the best/top X for EACH/PER Y (e.g. "best product for \
each region", "top 3 SKUs per category") wants ONE ranking PER outer group, not one flat global \
ranking. Use "best_per_group": group_by = [the outer thing you get one result group per, e.g. \
"region"], rank_by = [the different column being ranked inside each group, e.g. the product column], \
aggregate = the numeric column/func, limit_per_group = how many winners per group (1 for "best", 3 \
for "top 3"). group_by and rank_by must be two DIFFERENT columns. This can be combined with ordinary \
"filters" exactly as normal -- best_per_group only changes how the result is shaped/ranked, not which \
rows are eligible in the first place.
- STATISTICS: "statistical analysis", "descriptive statistics", "mean median mode", "variance/std \
dev" -> set "statistics" with the numeric column and the functions asked for (or a sensible full set: \
mean, median, std, var, min, max, sum, count if unspecified). Only add statistics.group_by if the \
question explicitly says "by <category>"/"per <category>" -- otherwise compute global statistics \
across the whole filtered set.
- COMPARE: only for comparing the SAME figure across TWO OR MORE DIFFERENT SHEETS/FILES (e.g. "compare \
totals between this month's file and last month's file"). List every sheet in compare.sheets in the \
order the question mentions them (this order drives % change direction), and put the shared numeric \
column/group-by in compare.column/compare.group_by. Do not use "compare" for grouping within a single \
sheet, and do not combine "compare" with "outliers"/"aggregate" in the same spec.
- COMPUTE: for arithmetic BETWEEN two or more separately-filtered figures (percentage difference, \
growth rate, sum of two specific slices, a ratio) -- run each figure as its own named sub-spec under \
compute.operations, then combine them with compute.formula (a python expression using each \
operation's "name" as a variable). Do not use compute for a single ordinary aggregate.
- ROLLUP/SUBTOTAL ROWS: if a sheet has an "Is_Rollup" (or similarly-named) column, rows flagged true \
(subtotal/grand-total rows) are EXCLUDED AUTOMATICALLY by the execution code for every query against \
that sheet -- you do not need an explicit Is_Rollup filter for normal per-entity rankings. Only set \
"include_rollups": true (a top-level field, not inside "filters") when the question genuinely wants \
those rows included (e.g. it's specifically asking about a named total/subtotal itself).
- HIERARCHICAL/MULTI-LEVEL IDENTIFIER COLUMNS: some reshaped reports have one identifier column that \
mixes several different LEVELS of thing together (e.g. one column holding both broad-category rows \
and fine-item rows, with no separate column telling them apart). For such a column, the reshaper adds \
"<column>_Level" (an integer, 0 = rollup, increasing = more granular) and "<column>_Level_Name" (that \
level's own plain-text name, read directly from the sheet's real data -- check AVAILABLE SHEETS for \
the actual names, which vary per report). When a question asks for one specific level ("by <level \
name>", "at the <finest-level> level", "for each <level name>"), filter "<column>_Level_Name" to that \
level -- do NOT filter or group by the raw identifier column alone, or rows from different levels get \
summed together incorrectly. A question saying "product" or "item" with no exact level name usually \
means the FINEST level (the one with the highest _Level number / most distinct values).
- A "<metric-type>" column some reports have (report-specific sub-measures like plan/actual/percent- \
achieved stored as VALUES in one column rather than as separate columns) needs a filter on that \
column, not a separate column reference -- match the value asked about to the closest real sample \
value shown for that column in AVAILABLE SHEETS. If the question doesn't specify which one, do not \
silently default to any particular value -- either use every value present (group_by that column \
instead of filtering it) or pick the one the question's wording most plausibly refers to; never guess \
a value from a different file's vocabulary.
- Only use date filters/date_between when the sheet is NOT already scoped to one period by its own \
name (e.g. a sheet already named for a specific month doesn't need an extra date filter for that same \
month) -- prefer picking the already-scoped sheet over adding a redundant date filter.
- Output ONLY the JSON object -- no markdown fences, no explanation.

Worked examples (READ CAREFULLY: <angle-bracket> text is a placeholder for whatever the REAL column/\
sheet names are for the file actually being queried -- never copy the placeholder text itself, and \
never reuse a column/value name from one of these examples for an unrelated file):

Question: "total <numeric column> for <value> in <category column>"
JSON SPEC:
{
"sheet": "<sheet name>",
"filters": [{"column": "<category column>", "op": "==", "value": "<value>"}],
"aggregate": {"column": "<numeric column>", "func": "sum"},
"chart": {"wanted": false}
}

Question: "best performing <entity> by <numeric column>"
JSON SPEC:
{
"sheet": "<sheet name>",
"group_by": ["<entity column>"],
"aggregate": {"column": "<numeric column>", "func": "sum"},
"sort_descending": true,
"limit": 5,
"chart": {"wanted": false}
}
(Note: "Is_Rollup" exclusion happens automatically if that column exists -- no explicit filter needed.)

Question: "best <entity> for each <category>"
JSON SPEC:
{
"sheet": "<sheet name>",
"best_per_group": {
    "group_by": ["<category column>"],
    "rank_by": "<entity column>",
    "aggregate": {"column": "<numeric column>", "func": "sum"},
    "limit_per_group": 1
  },
"chart": {"wanted": false}
}
(Note: "for each <category>" means ONE winner PER category -- group_by+sort+limit alone would only \
ever give one flat ranking across ALL categories combined, a different and wrong shape of answer here.)

Question: "top 3 <entity> for each <category>"
JSON SPEC:
{
"sheet": "<sheet name>",
"group_by": ["<category column>", "<entity column>"],
"aggregate": {"column": "<numeric column>", "func": "sum"},
"sort_descending": true,
"top_per_group": 3,
"chart": {"wanted": false}
}

Question: "detect outliers in <numeric column> by <category column>" (outliers ACROSS categories, using each category's total)
JSON SPEC:
{
"sheet": "<sheet name>",
"group_by": ["<category column>"],
"aggregate": {"column": "<numeric column>", "func": "sum"},
"outliers": {"column": "<numeric column>", "method": "iqr", "group_by": []},
"chart": {"wanted": false}
}

Question: "which <entity> has the best <numeric column>, but make sure it's not an outlier"
JSON SPEC:
{
"sheet": "<sheet name>",
"group_by": ["<entity column>"],
"aggregate": {"column": "<numeric column>", "func": "sum"},
"exclude_outliers": {"column": "<numeric column>", "method": "iqr", "group_by": []},
"sort_descending": true,
"limit": 1,
"chart": {"wanted": false}
}
(Note: exclude_outliers' column matching aggregate's column with an empty group_by means each \
<entity>'s own TOTAL is what's checked for being an outlier, computed AFTER aggregating -- an \
extreme one-off shouldn't "win" a best-performing ranking just for being unusual.)

Question: "for <filter value>, find outliers in <numeric column> grouped by <category>, and also \
give me the count of rows per <category>" (COMBINED -- outliers AND a separate aggregate in one ask)
JSON SPEC:
{
"sheet": "<sheet name>",
"filters": [{"column": "<filter column>", "op": "contains", "value": "<filter value>"}],
"outliers": {"column": "<numeric column>", "method": "iqr", "group_by": ["<category column>"]},
"group_by": ["<category column>"],
"aggregate": {"column": "<category column>", "func": "count"},
"chart": {"wanted": false}
}
(Note: set BOTH "outliers" AND "aggregate"/"group_by" -- they're independent parts of the SAME \
question about the same filtered rows, not alternatives. Never invent a field like \
"outliers.count_by" to smuggle the second part inside the first -- give each part its own real field.)

Question: "statistical analysis of <numeric column>" (global, no grouping)
JSON SPEC:
{
"sheet": "<sheet name>",
"statistics": {"column": "<numeric column>", "funcs": ["mean", "median", "std", "var", "min", "max", "sum", "count"]},
"chart": {"wanted": false}
}

Question: "statistical analysis of <numeric column> by <category>"
JSON SPEC:
{
"sheet": "<sheet name>",
"statistics": {"column": "<numeric column>", "funcs": ["mean", "median", "std", "var", "min", "max", "sum", "count"], "group_by": ["<category column>"]},
"chart": {"wanted": false}
}

Question: "what's the percentage difference between <figure A> and <figure B>"
JSON SPEC:
{
"sheet": "<sheet name>",
"compute": {
    "operations": [
      {"name": "a", "spec": {"filters": [{"column": "<column>", "op": "==", "value": "<value for A>"}], "aggregate": {"column": "<numeric column>", "func": "sum"}}},
      {"name": "b", "spec": {"filters": [{"column": "<column>", "op": "==", "value": "<value for B>"}], "aggregate": {"column": "<numeric column>", "func": "sum"}}}
    ],
    "formula": "(b - a) / a * 100",
    "label": "% change B vs A"
  },
"chart": {"wanted": false}
}

Question: "compare total <numeric column> between <sheet 1> and <sheet 2>, by <category>"
JSON SPEC:
{
"compare": {
    "sheets": ["<sheet 1>", "<sheet 2>"],
    "column": "<numeric column>",
    "func": "sum",
    "group_by": ["<category column>"]
  },
"chart": {"wanted": false}
}

Question: "show a chart of total <numeric column> by <category>"
JSON SPEC:
{
"sheet": "<sheet name>",
"group_by": ["<category column>"],
"aggregate": {"column": "<numeric column>", "func": "sum"},
"chart": {"wanted": true, "type": "auto"}
}
(Note: to chart outliers instead, include both "outliers" and "chart" -- the chart is then built from \
the outlier rows/aggregated-outlier rows rather than a plain aggregate.)

Question: "at the <finest level name, e.g. product/SKU/item> level, what's the best <entity> for \
<metric value>" (hierarchical identifier column -- see the rule above)
JSON SPEC:
{
"sheet": "<sheet name>",
"filters": [
    {"column": "<metric-type column>", "op": "==", "value": "<matching real value from AVAILABLE SHEETS>"},
    {"column": "<identifier column>_Level_Name", "op": "==", "value": "<the real finest-level name from AVAILABLE SHEETS>"}
  ],
"group_by": ["<identifier column>"],
"aggregate": {"column": "<numeric column>", "func": "sum"},
"sort_descending": true,
"limit": 5,
"chart": {"wanted": false}
}
"""

def _describe_sheet_for_prompt(sheet: _SheetInfo) -> str:
    lines = [f"Sheet: \"{sheet.sheet_name}\" (file: {os.path.basename(sheet.file_path)})",
             f"Columns: {', '.join(sheet.columns)}"]
    for col, values in sheet.categorical_values.items():
        sample = values[:25]
        lines.append(f"  - \"{col}\" sample values: {sample}")
    return "\n".join(lines)


def _build_spec_prompt(question: str, sheets: List[_SheetInfo],
                        previous_spec: Optional[Dict[str, Any]] = None) -> str:
    sheets_desc = "\n\n".join(_describe_sheet_for_prompt(s) for s in sheets)
    followup_block = ""
    if previous_spec:
        # A real follow-up ("tell me the state for the above", "now
        # show that by segment") needs the PREVIOUS spec as a base --
        # its own text alone is almost always too underspecified to
        # build a sensible spec from scratch (no sheet, no filters, no
        # metric). Giving the model the actual prior JSON spec (not
        # just the prior question's prose) lets it inherit exactly
        # what carries over -- same sheet, same filters, same metric --
        # and change only what this new question explicitly asks to
        # change, rather than guessing at the whole thing fresh.
        followup_block = (
            "\n=== THIS IS A FOLLOW-UP TO THE PREVIOUS QUESTION ===\n"
            "The previous question's spec was:\n"
            f"{json.dumps(previous_spec, indent=2)}\n"
            "Start from this spec as your base. Keep whatever the new "
            "question doesn't mention (same \"sheet\", same \"filters\", "
            "same metric/column) and change ONLY what it explicitly asks "
            "to change. E.g. if the previous question filtered to a "
            "specific product/branch and this one just asks for a "
            "different breakdown of the same thing, keep those filters. "
            "If this question asks for something that isn't a further "
            "refinement at all (a genuinely new, unrelated question), "
            "ignore the previous spec and build a fresh one instead.\n"
            "=== END PREVIOUS SPEC ===\n"
        )
    return (
        f"{_SPEC_SYSTEM_PROMPT}\n\n"
        f"=== AVAILABLE SHEETS ===\n{sheets_desc}\n=== END AVAILABLE SHEETS ===\n"
        f"{followup_block}\n"
        f"QUESTION: {question}\n\nJSON SPEC:"
    )

def _extract_json_object(text: str) -> Optional[str]:
    """
    Find the first balanced {...} object in text via depth counting
    (string- and escape-aware), rather than a greedy \\{.*\\} regex.

    The old greedy regex spans from the very FIRST '{' to the very LAST
    '}' anywhere in the whole response. The system prompt tells the
    model to output ONLY the JSON object, but a smaller local model
    asked for a genuinely complex compound spec (outliers + group_by +
    a separate count aggregate, per this exact combined-query case) can
    still occasionally add one extra sentence or an aside containing its
    own brace before/after the real object -- and the greedy regex then
    captures everything from the first '{' to that unrelated trailing
    '}', producing text that isn't valid JSON at all and making the
    whole analysis silently error out and fall back to the vastly
    inferior plain-RAG path (an LLM doing arithmetic in prose over a
    handful of sampled rows) instead of ever reaching pandas execution.
    Proper brace-depth matching finds exactly the real object regardless
    of what surrounds it.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_spec(raw: str) -> Optional[Dict[str, Any]]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    candidate = _extract_json_object(text)
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Local models produce near-miss JSON often enough that a cheap,
    # local salvage pass is worth trying before spending a second LLM
    # round-trip (the retry further up the call chain) -- every failure
    # here means the question falls back to plain chunk-based RAG
    # instead of real pandas execution, so it's worth being generous.
    # Fixes only well-understood, common local-model mistakes: trailing
    # commas before a closing brace/bracket, and curly/smart quotes
    # instead of straight ones.
    salvaged = re.sub(r",\s*([}\]])", r"\1", candidate)
    salvaged = salvaged.replace("\u201c", '"').replace("\u201d", '"')
    salvaged = salvaged.replace("\u2018", "'").replace("\u2019", "'")
    try:
        return json.loads(salvaged)
    except json.JSONDecodeError:
        return None


# ─── Safe execution of the spec against real pandas data ───────────────────────

_OPS = {
    "==": lambda s, v: s.astype(str).str.strip().str.lower() == str(v).strip().lower(),
    "!=": lambda s, v: s.astype(str).str.strip().str.lower() != str(v).strip().lower(),
    "contains": lambda s, v: s.astype(str).str.strip().str.lower().str.contains(str(v).strip().lower(), na=False),
    "in": lambda s, v: s.astype(str).str.strip().str.lower().isin([str(x).strip().lower() for x in v]) if isinstance(v, list) else False,
    ">":  lambda s, v: _coerce_numeric(s) > float(v),
    ">=": lambda s, v: _coerce_numeric(s) >= float(v),
    "<":  lambda s, v: _coerce_numeric(s) < float(v),
    "<=": lambda s, v: _coerce_numeric(s) <= float(v),
    # Date comparison operators - parse both sides as dates
    "date==": lambda s, v: pd.to_datetime(s, errors="coerce") == pd.Timestamp(v),
    "date!=": lambda s, v: pd.to_datetime(s, errors="coerce") != pd.Timestamp(v),
    "date>":  lambda s, v: pd.to_datetime(s, errors="coerce") > pd.Timestamp(v),
    "date>=": lambda s, v: pd.to_datetime(s, errors="coerce") >= pd.Timestamp(v),
    "date<":  lambda s, v: pd.to_datetime(s, errors="coerce") < pd.Timestamp(v),
    "date<=": lambda s, v: pd.to_datetime(s, errors="coerce") <= pd.Timestamp(v),
    # Date range: value is [start, end]
    "date_between": lambda s, v: (
        (pd.to_datetime(s, errors="coerce") >= pd.Timestamp(v[0]))
        & (pd.to_datetime(s, errors="coerce") <= pd.Timestamp(v[1]))
    ),
}


def _resolve_column(name: str, columns: List[str]) -> Optional[str]:
    """Match the LLM's column name against the real ones, tolerating
    case/whitespace differences the model might introduce."""
    if name in columns:
        return name
    norm = name.strip().lower()
    for c in columns:
        if c.strip().lower() == norm:
            return c
    return None


def _resolve_column_fuzzy(name: str, columns: List[str]) -> Optional[str]:
    """Same as _resolve_column, but falls back to fuzzy matching when
    comparing ACROSS different files -- two spreadsheets tracking the
    same thing frequently don't use byte-identical column names (e.g.
    "Sales Amount" vs "SalesAmount" vs "sales_amount"). An exact match
    is always preferred; fuzzy matching only kicks in when nothing
    matched exactly, and only accepts a reasonably confident match.

    Honest limitation: this is character-level similarity, not a
    semantic understanding of the words -- it reliably handles spacing/
    case/punctuation/underscore differences in essentially the SAME
    words ("Sales Amount" / "SalesAmount" / "sales_amount"), but it
    cannot know that "Sales Amount" and "Revenue" mean the same thing;
    truly different wording for the same concept across two files will
    need the column named explicitly in the question, or the sheet's
    own column renamed to match, for a comparison to work.
    """
    exact = _resolve_column(name, columns)
    if exact:
        return exact
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return None

    def _normalize(s: str) -> str:
        return re.sub(r"[\s_\-]+", "", s.strip().lower())

    norm = _normalize(name)
    best_col, best_score = None, 0
    for c in columns:
        c_norm = _normalize(c)
        score = max(fuzz.ratio(norm, c_norm), fuzz.token_sort_ratio(name.lower(), c.lower()))
        if score > best_score:
            best_col, best_score = c, score
    return best_col if best_score >= 80 else None


def _resolve_sheet_name(name: Optional[str], sheets: List["_SheetInfo"]) -> Optional["_SheetInfo"]:
    """
    Match the LLM's "sheet" value against the real registered sheets,
    tolerating the kind of near-miss a model easily introduces:
    different casing/whitespace, or reproducing an EARLIER conversation
    turn's sheet name slightly wrong from memory rather than copying it
    character-for-character from the current AVAILABLE SHEETS list.

    This matters more than it might look: with the exact-string-only
    match this replaces, a single character of drift in the sheet name
    made `sheet_name` fail to match ANY registered sheet, which then
    required the LLM's every single filter/group_by/aggregate column to
    ALSO exactly, collectively match one sheet's full column set before
    a fallback sheet could even be picked. With more than one sheet
    registered, one small slip anywhere was enough to return
    "ambiguous_sheet" and silently fall all the way out of the analysis
    route entirely -- back out to the generic LLM/RAG path, which has no
    idea the question was ever about a spreadsheet at all.
    """
    if not sheets:
        return None
    if not name:
        return sheets[0] if len(sheets) == 1 else None

    exact = next((s for s in sheets if s.sheet_name == name), None)
    if exact:
        return exact
    norm = name.strip().lower()
    ci_match = next((s for s in sheets if s.sheet_name.strip().lower() == norm), None)
    if ci_match:
        return ci_match

    try:
        from rapidfuzz import fuzz
    except ImportError:
        return None

    def _normalize(s: str) -> str:
        return re.sub(r"[\s_\-]+", "", s.strip().lower())

    name_norm = _normalize(name)
    best_sheet, best_score = None, 0
    for s in sheets:
        score = max(
            fuzz.ratio(name_norm, _normalize(s.sheet_name)),
            fuzz.token_sort_ratio(name.lower(), s.sheet_name.lower()),
        )
        if score > best_score:
            best_sheet, best_score = s, score
    return best_sheet if best_score >= 75 else None


@dataclass
class AnalysisResult:
    text: str
    chart_path: Optional[str] = None
    error: Optional[str] = None


# Last successfully-executed spec, kept so a genuine follow-up ("tell me
# the state for the above", "now break that down by segment") can be
# built from real prior context instead of from scratch. This is
# intentionally simple (one slot, last-writer-wins, no per-session
# keying) -- this app is single-user/single-process, so there is only
# ever one "previous analysis answer" to follow up on at a time; a
# multi-user deployment would need to key this per conversation/session
# instead of at module level.
_last_analysis_spec: Optional[Dict[str, Any]] = None
_last_analysis_question: Optional[str] = None


def get_last_analysis_spec() -> Optional[Dict[str, Any]]:
    """Used by core/engine.py to decide whether a question that doesn't
    match is_analysis_query() on its own (a bare follow-up has no
    analysis verb or column name of its own to match against) should
    still be routed to answer_analysis_query() with this as context."""
    return _last_analysis_spec


def _remember_analysis_spec(question: str, spec: Dict[str, Any]) -> None:
    global _last_analysis_spec, _last_analysis_question
    _last_analysis_spec = spec
    _last_analysis_question = question


class _OutlierDataQualityError(Exception):
    """Raised by _detect_outliers when too few values in the target
    column parsed as numbers to trust the result -- caught in
    _execute_spec and surfaced as a normal, readable error instead of a
    silently wrong table."""


@dataclass
class SpecExecution:
    """Result of running a spec against a sheet. Both fields can be
    populated at once -- a question like "find outliers for Delhi by
    region, and also give me the count per region" asks two independent
    things about the SAME filtered data in one breath, and previously
    the spec format could only ever answer one of them (whichever the
    LLM happened to pick), silently dropping the other. Each populated
    field gets its own section in the final answer."""
    outliers_df: Optional["pd.DataFrame"] = None
    aggregate_df: Optional["pd.DataFrame"] = None
    statistics_df: Optional["pd.DataFrame"] = None
    # Result of a "best_per_group" spec -- e.g. "best state for each
    # segment": one (or a few, if limit_per_group > 1) row(s) PER
    # distinct value of the outer grouping column, showing that group's
    # own top entity by the aggregated metric. Kept as its own field
    # rather than overloading aggregate_df because the shape is
    # different (a rank WITHIN each group, not one flat global ranking)
    # and both can legitimately be requested together.
    best_per_group_df: Optional["pd.DataFrame"] = None
    # Result of a "compute" spec -- mathematical operations between sub-query results
    compute_df: Optional["pd.DataFrame"] = None
    error: Optional[str] = None
    filter_warnings: List[str] = field(default_factory=list)
    # The filtered DataFrame (before aggregation) - useful for showing
    # source rows / breakdowns
    filtered_df: Optional["pd.DataFrame"] = None


def _detect_outliers(df: "pd.DataFrame", column: str, group_by: List[str],
                      method: str) -> "pd.DataFrame":
    """
    Local, fully offline statistical outlier detection -- no external
    API/model call, nothing leaves the process. Two standard, well-
    understood methods:

    - "iqr" (default): flags any value outside
      [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. Robust to skewed sales-style data
      (doesn't assume a normal distribution), which is why it's the
      default over z-score.
    - "zscore": flags any value more than 3 standard deviations from
      the mean.

    If group_by is given, the bounds are computed SEPARATELY within
    each group (e.g. per division, per state) rather than once across
    the whole column -- a value can be perfectly normal for one
    division and a genuine outlier in another, and computing one
    global range would hide that entirely.

    Returns the original rows that are flagged, with two extra columns
    appended: "_outlier_bound_low"/"_outlier_bound_high" (or
_group_mean"/"_group_std" for zscore) so the reader can see WHY a
    row was flagged, not just that it was.
    """
    working = df.copy()
    total_rows = len(working)
    working["_value"] = _coerce_numeric(working[column])
    working = working[working["_value"].notna()]

    # If the "numeric" column mostly didn't parse, the outliers computed
    # from whatever DID survive are not a trustworthy answer -- they're
    # just whatever few rows happened to already be clean numbers, which
    # can be a completely different scale/meaning than the real column
    # (see _coerce_numeric's docstring for exactly this failure mode).
    # Surface that as an explicit error instead of silently returning a
    # plausible-looking but wrong table.
    if total_rows and len(working) / total_rows < 0.5:
        raise _OutlierDataQualityError(
            f"Only {len(working)} of {total_rows} rows in \"{column}\" could be "
            f"read as numbers -- the rest didn't parse (unexpected text, "
            f"merged/blank cells, or a formatting pandas can't read as-is). "
            f"Outliers computed from that small a surviving slice would be "
            f"misleading, so this was stopped rather than shown."
        )

    if working.empty:
        return working

    def _flag(group: "pd.DataFrame") -> "pd.DataFrame":
        vals = group["_value"]
        if method == "zscore":
            mean = vals.mean()
            std = vals.std()
            if not std or pd.isna(std):
                return group.iloc[0:0]
            z = (vals - mean) / std
            flagged = group[z.abs() > 3].copy()
            flagged["_deviation"] = z[z.abs() > 3].abs()
            flagged["_group_mean"] = round(mean, 4)
            flagged["_group_std"] = round(std, 4)
            return flagged
        else:
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            if not iqr or pd.isna(iqr):
                return group.iloc[0:0]
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (vals < low) | (vals > high)
            flagged = group[mask].copy()
            center = (low + high) / 2 or 1
            flagged["_deviation"] = (vals[mask] - center).abs()
            flagged["_outlier_bound_low"] = round(low, 4)
            flagged["_outlier_bound_high"] = round(high, 4)
            return flagged

    valid_group_by = [g for g in group_by if g in working.columns]
    if valid_group_by:
        parts = [
            _flag(g) for _, g in working.groupby(valid_group_by, dropna=False)
        ]
        result = pd.concat(parts) if parts else working.iloc[0:0]
    else:
        result = _flag(working)

    if result.empty:
        return result
    result = result.sort_values(by="_deviation", ascending=False)
    return result.drop(columns=["_value"], errors="ignore")


def _format_outliers_markdown(result: "pd.DataFrame", column: str, group_by: List[str]) -> str:
    if result.empty:
        return f"No outliers found in \"{column}\" by that measure."
    # Show complete row (all original columns) plus outlier reason columns
    # Exclude internal columns like _value, _deviation
    internal_cols = {"_value", "_deviation"}
    reason_cols = [c for c in result.columns if c.startswith("_") and c not in internal_cols]
    # Show all non-internal columns (original data) + reason columns
    show_cols = [c for c in result.columns if c not in internal_cols and not c.startswith("_")]
    cols = show_cols + reason_cols
    result = result[cols]
    # NOTE: deliberately no .head(N) here -- every flagged row is shown.
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in result.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:,.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    # Add statistical interpretation note
    method = "IQR"  # default, could be passed in but kept simple
    if "_outlier_bound_low" in result.columns and "_outlier_bound_high" in result.columns:
        low = result["_outlier_bound_low"].iloc[0]
        high = result["_outlier_bound_high"].iloc[0]
        lines.append(f"\n*Statistical note: Values outside [{low:,.2f}, {high:,.2f}] are flagged as outliers using the IQR method (Q1-1.5×IQR, Q3+1.5×IQR). These represent unusual data points that may indicate data entry errors, exceptional events, or genuine anomalies — they should be investigated, not assumed to represent 'good' or 'bad' performance.*")
    elif "_group_mean" in result.columns and "_group_std" in result.columns:
        mean = result["_group_mean"].iloc[0]
        std = result["_group_std"].iloc[0]
        lines.append(f"\n*Statistical note: Values more than 3 standard deviations from the group mean ({mean:,.2f} ± {3*std:,.2f}) are flagged as outliers using the z-score method. These represent unusual data points that may indicate data entry errors, exceptional events, or genuine anomalies — they should be investigated, not assumed to represent 'good' or 'bad' performance.*")

    return "\n".join(lines)


def _maybe_save_outliers_csv(result: "pd.DataFrame", column: str, sheet: "_SheetInfo",
                              csv_threshold: int = 100) -> Optional[str]:
    """
    CSV saving for outliers is disabled - outliers are displayed inline only.
    """
    return None


def _execute_outliers(outlier_spec: Dict[str, Any], filtered: "pd.DataFrame",
                       sheet: "_SheetInfo", limit: Optional[int]) -> Tuple[Optional["pd.DataFrame"], Optional[str]]:
    col = _resolve_column_fuzzy(str(outlier_spec.get("column", "")), sheet.columns)
    if col is None:
        return None, f"Unknown outlier column: {outlier_spec}"
    method = outlier_spec.get("method", "iqr")
    if method not in ("iqr", "zscore"):
        method = "iqr"
    og_raw = outlier_spec.get("group_by") or []
    og = [c for c in (_resolve_column_fuzzy(g, sheet.columns) for g in og_raw) if c]
    try:
        result = _detect_outliers(filtered, col, og, method)
    except _OutlierDataQualityError as exc:
        return None, str(exc)
    if isinstance(limit, int) and limit > 0:
        result = result.head(limit)
    return result, None


def _execute_statistics(spec: Dict[str, Any], filtered: "pd.DataFrame",
                       sheet: "_SheetInfo") -> Tuple["pd.DataFrame", Optional[str]]:
    """Compute descriptive statistics on a numeric column, optionally grouped."""
    stats_spec = spec.get("statistics") or {}
    col_name = _resolve_column_fuzzy(str(stats_spec.get("column", "")), sheet.columns)
    if col_name is None:
        return pd.DataFrame(), f"Unknown statistics column: {stats_spec}"

    funcs = stats_spec.get("funcs") or ["mean", "median", "std", "var", "min", "max", "sum", "count"]
    group_by_raw = stats_spec.get("group_by") or []
    group_by = [c for c in (_resolve_column_fuzzy(g, sheet.columns) for g in group_by_raw) if c]

    # Coerce to numeric
    numeric = _coerce_numeric(filtered[col_name])
    working = filtered.copy()
    working["_stat_value"] = numeric

    # Formula references for output (ASCII-safe for terminal compatibility)
    FORMULAS = {
        "mean": "mean = sum(x) / n",
        "median": "Middle value (50th percentile)",
        "mode": "Most frequent value",
        "std": "std = sqrt(sum((x - mean)^2) / (n-1))",
        "var": "var = sum((x - mean)^2) / (n-1)",
        "min": "Minimum value",
        "max": "Maximum value",
        "sum": "sum(x)",
        "count": "n (non-null values)",
        "skew": "sum((x - mean)/std)^3 / n",
        "kurtosis": "sum((x - mean)/std)^4 / n - 3",
    }

    if group_by:
        # Per-group statistics
        results = []
        for name, group in working.groupby(group_by, dropna=False):
            vals = group["_stat_value"].dropna()
            if vals.empty:
                row = {g: name[i] if isinstance(name, tuple) else name for i, g in enumerate(group_by)}
                for f in funcs:
                    row[f] = None
                results.append(row)
                continue
            row = {g: name[i] if isinstance(name, tuple) else name for i, g in enumerate(group_by)}
            row["n"] = len(vals)
            if "mean" in funcs:
                row["mean"] = vals.mean()
            if "median" in funcs:
                row["median"] = vals.median()
            if "mode" in funcs:
                mode_vals = vals.mode()
                row["mode"] = mode_vals.iloc[0] if not mode_vals.empty else None
            if "std" in funcs:
                row["std"] = vals.std(ddof=1)
            if "var" in funcs:
                row["var"] = vals.var(ddof=1)
            if "min" in funcs:
                row["min"] = vals.min()
            if "max" in funcs:
                row["max"] = vals.max()
            if "sum" in funcs:
                row["sum"] = vals.sum()
            if "count" in funcs:
                row["count"] = len(vals)
            if "skew" in funcs:
                row["skew"] = vals.skew()
            if "kurtosis" in funcs:
                row["kurtosis"] = vals.kurtosis()
            results.append(row)
        result = pd.DataFrame(results)
    else:
        # Global statistics
        vals = working["_stat_value"].dropna()
        if vals.empty:
            result = pd.DataFrame([{f: None for f in funcs}])
        else:
            row = {"n": len(vals)}
            if "mean" in funcs:
                row["mean"] = vals.mean()
            if "median" in funcs:
                row["median"] = vals.median()
            if "mode" in funcs:
                mode_vals = vals.mode()
                row["mode"] = mode_vals.iloc[0] if not mode_vals.empty else None
            if "std" in funcs:
                row["std"] = vals.std(ddof=1)
            if "var" in funcs:
                row["var"] = vals.var(ddof=1)
            if "min" in funcs:
                row["min"] = vals.min()
            if "max" in funcs:
                row["max"] = vals.max()
            if "sum" in funcs:
                row["sum"] = vals.sum()
            if "count" in funcs:
                row["count"] = len(vals)
            if "skew" in funcs:
                row["skew"] = vals.skew()
            if "kurtosis" in funcs:
                row["kurtosis"] = vals.kurtosis()
            result = pd.DataFrame([row])

    # Add formula references as metadata (not in the table)
    result.attrs["formulas"] = {f: FORMULAS.get(f, "") for f in funcs if f in FORMULAS}
    result.attrs["column"] = col_name
    result.attrs["group_by"] = group_by

    limit = spec.get("limit")
    if isinstance(limit, int) and limit > 0:
        result = result.head(limit)
    return result, None


def _execute_aggregate(spec: Dict[str, Any], filtered: "pd.DataFrame",
                        sheet: "_SheetInfo", apply_sort_limit: bool = True) -> Tuple["pd.DataFrame", Optional[str]]:
    group_by_raw = spec.get("group_by") or []
    group_by = [c for c in (_resolve_column_fuzzy(g, sheet.columns) for g in group_by_raw) if c]

    agg_spec = spec.get("aggregate") or None
    agg_col = None
    agg_func = None
    if agg_spec:
        agg_col = _resolve_column_fuzzy(str(agg_spec.get("column", "")), sheet.columns)
        agg_func = agg_spec.get("func", "sum")
        if agg_func not in ("sum", "mean", "count", "min", "max", "median"):
            agg_func = "sum"

    if group_by and agg_col:
        working = filtered.copy()
        if agg_func == "count":
            # BUG (reproduced): "count" doesn't need the column to be
            # numeric at all -- it just counts non-null rows per group.
            # This used to run every aggregate column through
            # _coerce_numeric() unconditionally BEFORE aggregating, so a
            # question like "count how many times each franchise occurs"
            # (aggregate column = a text column, e.g. "Franchise Name")
            # turned every value into NaN (a name isn't a number), and
            # pandas' .count() explicitly excludes NaN -- so every single
            # group came back as a count of exactly 0, regardless of how
            # many real rows actually matched. Counting the column as
            # given, with no numeric coercion, fixes this.
            working["_agg_value"] = filtered[agg_col]
            result = working.groupby(group_by, dropna=False)["_agg_value"].count().reset_index()
        else:
            numeric = _coerce_numeric(filtered[agg_col])
            working["_agg_value"] = numeric
            result = working.groupby(group_by, dropna=False)["_agg_value"].agg(agg_func).reset_index()
        result = result.rename(columns={"_agg_value": f"{agg_func}({agg_col})"})
    elif agg_col:
        if agg_func == "count":
            value = filtered[agg_col].count()
        else:
            numeric = _coerce_numeric(filtered[agg_col])
            value = getattr(numeric, agg_func)()
        result = pd.DataFrame([{f"{agg_func}({agg_col})": value}])
    elif group_by:
        result = filtered.groupby(group_by, dropna=False).size().reset_index(name="count")
    else:
        result = filtered

    value_col = result.columns[-1] if (group_by or agg_col) and len(result.columns) else None
    if not apply_sort_limit:
        return result, None

    if spec.get("sort_descending") and value_col is not None and value_col in result.columns:
        try:
            result = result.sort_values(by=value_col, ascending=False)
        except Exception:
            pass

    # TOP PER GROUP: If group_by has 2+ columns and top_per_group is set,
    # return only the top N rows per outer group (first group_by column)
    top_per_group = spec.get("top_per_group")
    if (isinstance(top_per_group, int) and top_per_group > 0
            and len(group_by) >= 2
            and value_col is not None
            and value_col in result.columns):
        outer_group = group_by[0]
        result = result.groupby(outer_group, group_keys=False).head(top_per_group)

    limit = spec.get("limit")
    if isinstance(limit, int) and limit > 0:
        result = result.head(limit)
    elif not group_by and not agg_col:
        # A raw (non-aggregated) row dump could be huge (this is exactly
        # the 87k-row case) -- cap it hard even if the spec didn't ask
        # for a limit, so this never tries to format thousands of rows
        # into one answer.
        result = result.head(200)

    return result, None


def _execute_best_per_group(spec_block: Dict[str, Any], filtered: "pd.DataFrame",
                             sheet: "_SheetInfo") -> Tuple[Optional["pd.DataFrame"], Optional[str]]:
    """
    "best_per_group" answers "best/top X FOR EACH Y" -- a fundamentally
    different shape of question from a flat "top N" ranking. E.g. "best
    state for each segment": for EVERY distinct segment, rank states by
    total sales and keep the top one -- NOT one global top-N list mixing
    all segments together (which is what group_by+sort+limit alone
    produces, and is what "best state for each segment" kept getting
    answered with before this existed: one flat table with no per-
    segment structure at all).

    spec_block shape:
      {
        "group_by": ["<column the answer should have one row-group per>"],
        "rank_by": "<column being ranked WITHIN each group, e.g. State>",
        "aggregate": {"column": "<numeric column>", "func": "sum"},
        "limit_per_group": 1   (optional, defaults to 1)
      }
    """
    group_cols_raw = spec_block.get("group_by") or []
    group_cols = [c for c in (_resolve_column_fuzzy(g, sheet.columns) for g in group_cols_raw) if c]
    rank_by = _resolve_column_fuzzy(str(spec_block.get("rank_by", "")), sheet.columns)
    agg_spec = spec_block.get("aggregate") or {}
    agg_col = _resolve_column_fuzzy(str(agg_spec.get("column", "")), sheet.columns)
    agg_func = agg_spec.get("func", "sum")
    if agg_func not in ("sum", "mean", "count", "min", "max", "median"):
        agg_func = "sum"
    limit_per_group = spec_block.get("limit_per_group") or 1
    if not isinstance(limit_per_group, int) or limit_per_group < 1:
        limit_per_group = 1

    if not group_cols:
        return None, "best_per_group needs at least one \"group_by\" column"
    if not rank_by:
        return None, f"Unknown best_per_group \"rank_by\" column: {spec_block.get('rank_by')}"
    if not agg_col:
        return None, f"Unknown best_per_group aggregate column: {agg_spec.get('column')}"

    # Defensive: dedupe group_cols against itself, then drop rank_by out
    # of group_cols if the two collided (e.g. a spec that set
    # group_by=["State"] AND rank_by="State" -- whether from the model
    # picking the same column for both, or two differently-worded
    # column names both fuzzy-resolving to the same real column).
    # Without this, `groupby(group_cols + [rank_by])` builds a column
    # list with the same name twice, and reset_index() then tries to
    # insert a column that's already there -- "cannot insert State,
    # already exists" -- a confusing pandas-internals error to surface
    # to whoever asked the question, for what's really just a
    # self-referential spec ("rank State within State" isn't a
    # meaningful per-group question in the first place). Removing the
    # collision and re-checking group_cols catches this as the same
    # clear "needs at least one group_by column" message above, rather
    # than crashing.
    group_cols = list(dict.fromkeys(group_cols))
    if rank_by in group_cols:
        group_cols = [c for c in group_cols if c != rank_by]
    if not group_cols:
        return None, (
            f"best_per_group's \"group_by\" and \"rank_by\" both resolved to "
            f"the same column ({rank_by!r}) -- ranking a column within "
            f"itself isn't a meaningful \"best X for each Y\" question. "
            f"Use a different column for \"group_by\" (the Y) than for "
            f"\"rank_by\" (the X being ranked)."
        )

    # If the group_by column has a unique value for every row, each
    # "group" is effectively a single row -- no per-group ranking is
    # possible.  Catch this before pandas internals leak a confusing
    # error, and tell the user to pick a column with multiple categories.
    # Use `filtered` here since `working` hasn't been copied yet;
    # the column set is the same.
    first_group_col = group_cols[0]
    nunique = filtered[first_group_col].nunique()
    nrows = len(filtered)
    if nunique <= 1:
        return None, (
            f"best_per_group's \"group_by\" column {first_group_col!r} "
            f"has only one distinct value ({nunique}) -- cannot rank "
            f"within a group that contains a single value.  Use a "
            f"column with multiple categories for \"group_by\"."
        )
    if nunique >= nrows:
        return None, (
            f"best_per_group's \"group_by\" column {first_group_col!r} "
            f"has {nunique} distinct values against {nrows} rows -- each "
            f"row would be its own group, which isn't a meaningful "
            f"\"best X for each Y\" question.  Use a column with fewer "
            f"categories for \"group_by\"."
        )

    working = filtered.copy()
    working["_value"] = _coerce_numeric(working[agg_col])
    working = working.dropna(subset=["_value"])
    if working.empty:
        return working, None

    # Two-level aggregate: first collapse to one row per (group..., rank_by)
    # combination -- e.g. one row per (Segment, State) -- THEN take the
    # top `limit_per_group` rank_by values within each group. Doing the
    # inner aggregate first (rather than ranking raw rows) is what makes
    # this correct when a group/rank_by combination spans many raw rows
    # (e.g. many SKUs for one state within one segment): the ranking has
    # to be by each state's OWN TOTAL, not by any single row.
    inner = (
        working.groupby(group_cols + [rank_by], dropna=False)["_value"]
        .agg(agg_func)
        .reset_index()
        .rename(columns={"_value": f"{agg_func}({agg_col})"})
    )
    value_col = f"{agg_func}({agg_col})"

    # NOT groupby(...).apply(lambda g: g.nlargest(...)) -- confirmed via
    # direct testing that pandas 3.0's groupby-apply silently DROPS the
    # grouping column from the result when the applied function returns
    # a row subset of its own group (a real behavior difference from
    # older pandas, not a hypothetical concern -- caught by testing this
    # against a real multi-segment file, where the returned frame was
    # missing the very column ("BRANDS"/segment) the whole point of
    # "best X for EACH segment" was to keep). Sorting first, then
    # groupby().head(), is the standard "top N per group" idiom that
    # reliably keeps every column across pandas versions.
    result = (
        inner.sort_values(value_col, ascending=False)
        .groupby(group_cols, dropna=False, sort=False)
        .head(limit_per_group)
        .reset_index(drop=True)
    )
    return result, None


def _execute_spec(spec: Dict[str, Any], sheet: "_SheetInfo") -> SpecExecution:
    """Runs the validated spec against sheet.df. Every column/operator
    is checked against the real schema before use -- this never
    executes arbitrary code, only a fixed set of pandas filter/group/
    aggregate calls the spec is allowed to name.

    A spec can carry BOTH "outliers" and "aggregate"/"group_by" at once
    (see SpecExecution's docstring) -- both get computed against the
    same filtered rows and returned together, instead of the old
    behavior of running only whichever one an early-return happened to
    hit first and silently ignoring the other.

    Special case: when both "outliers" and "aggregate"/"group_by" are
    present AND the outlier column matches the aggregate column, this
    indicates the user wants outliers ACROSS groups (e.g. "which towns
    have unusually high total sales") not WITHIN groups. In this case
    we first aggregate, then run outlier detection on the aggregated values.

    A separate "exclude_outliers" key (same shape as "outliers": column/
    method/group_by) means the OPPOSITE of "outliers" -- rather than
    reporting outlier rows, they're dropped from the pool before any
    aggregate/sort/limit runs, for a "best performing X, but not an
    outlier" style question.
    """
    df = sheet.df
    mask = pd.Series(True, index=df.index)
    filter_warnings = []

    for f in spec.get("filters", []) or []:
        col = _resolve_column_fuzzy(str(f.get("column", "")), sheet.columns)
        op = f.get("op")
        if col is None or op not in _OPS:
            return SpecExecution(error=f"Unknown column or operator: {f}", filter_warnings=filter_warnings)
        try:
            new_mask = _OPS[op](df[col], f.get("value"))
            # Check if this filter eliminated all rows or matched nothing
            if new_mask.sum() == 0 and mask.sum() > 0:
                filter_warnings.append(
                    f"Filter {f.get('column')} {op} '{f.get('value')}' matched 0 rows"
                )
            # For "in" operator, check which specific values didn't match
            elif op == "in" and isinstance(f.get("value"), list):
                values = f.get("value")
                col_vals = df[col].astype(str).str.strip().str.lower().unique()
                for val in values:
                    if str(val).strip().lower() not in col_vals:
                        filter_warnings.append(
                            f"Value '{val}' not found in column '{f.get('column')}'"
                        )
            mask &= new_mask
        except Exception as exc:
            return SpecExecution(error=f"Could not apply filter {f}: {exc}", filter_warnings=filter_warnings)

    filtered = df[mask]

    # Default-exclude rollup/subtotal rows (Is_Rollup, added by
    # core/xlsx_reshape.py for any sheet with merged cross-tab headers --
    # see that module for how "ALL EUROPE"/"TOTAL"/"BRANCH 3" style rows
    # get flagged) UNLESS the spec explicitly opted back in with
    # "include_rollups": true, or a filter already directly targets the
    # Is_Rollup column itself (respect that filter's own value rather
    # than double-applying).
    #
    # This used to be something the spec-generation PROMPT asked the
    # model to remember to do on every single query ("always add a
    # filter excluding Is_Rollup=false unless the question wants
    # totals") -- and it kept forgetting, which is exactly what put
    # "ALL EUROPE"/"TOTAL"/"BRANCH 1..8" into "best performing"/"top N"
    # results answer after answer. A default that just always applies,
    # with an explicit escape hatch for the rare case a rollup/total
    # figure genuinely is what's being asked for, doesn't depend on a
    # small model remembering anything.
    if "Is_Rollup" in df.columns:
        explicit_rollup_filter = any(
            _resolve_column_fuzzy(str(f.get("column", "")), sheet.columns) == "Is_Rollup"
            for f in (spec.get("filters", []) or [])
        )
        if not explicit_rollup_filter and not spec.get("include_rollups"):
            filtered = filtered[~filtered["Is_Rollup"].astype(bool)]

    if filtered.empty:
        want_outliers = bool(spec.get("outliers"))
        want_agg = bool(spec.get("aggregate")) or bool(spec.get("group_by")) or not want_outliers
        return SpecExecution(
            outliers_df=filtered if want_outliers else None,
            aggregate_df=filtered if want_agg else None,
            filter_warnings=filter_warnings,
            filtered_df=filtered,
        )

    outlier_spec = spec.get("outliers") or None
    exclude_outlier_spec = spec.get("exclude_outliers") or None
    agg_wanted = bool(spec.get("aggregate")) or bool(spec.get("group_by"))
    stats_wanted = bool(spec.get("statistics"))

    outliers_df = None
    aggregate_df = None
    statistics_df = None

    # "best performing town ... but make sure it's not an outlier" is a
    # DIFFERENT request from "show me the outliers" (the "outliers" key,
    # handled below) -- it wants outlier rows/groups REMOVED from the
    # ranking pool, not reported alongside it. Previously there was no
    # way to express this at all: the spec had "outliers" (show them)
    # and "aggregate"/"group_by" (rank them), computed completely
    # independently, so the two sections never talked to each other --
    # asking for "the best X, excluding outliers" got an outliers
    # section AND a separate, unfiltered, unsorted full ranking that
    # still included whatever outliers existed.
    #
    # Two different things can be meant by "not an outlier" here, and
    # they need different handling:
    #   1. "best performing TOWN ... not an outlier" -- the thing being
    #      judged for outlier-ness is each town's own AGGREGATE total,
    #      not any single raw row. This mirrors the existing "aggregate
    #      then outlier" pattern below, but for EXCLUSION: aggregate
    #      first, then drop whichever group(s) are themselves an
    #      outlier among the aggregated totals, THEN sort/limit for
    #      "best performing" -- sort/limit has to happen AFTER removal,
    #      or a genuine outlier sitting at the top could still win.
    #   2. Everything else (excluding outliers.group_by set, or the
    #      exclude_outliers column not matching the aggregate column,
    #      or no aggregate/group_by at all) -- outlier ROWS are dropped
    #      from the raw filtered data before anything else runs.
    do_aggregate_then_exclude = False
    if exclude_outlier_spec and agg_wanted:
        excl_col_raw = exclude_outlier_spec.get("column")
        agg_spec_check = spec.get("aggregate") or {}
        agg_col_check = agg_spec_check.get("column")
        group_by_check = spec.get("group_by") or []
        excl_group_by_check = exclude_outlier_spec.get("group_by") or []
        if (excl_col_raw and agg_col_check
                and _resolve_column_fuzzy(excl_col_raw, sheet.columns) == _resolve_column_fuzzy(agg_col_check, sheet.columns)
                and group_by_check and not excl_group_by_check):
            do_aggregate_then_exclude = True

    if exclude_outlier_spec and not do_aggregate_then_exclude:
        excl_col = _resolve_column_fuzzy(str(exclude_outlier_spec.get("column", "")), sheet.columns)
        if excl_col is None:
            return SpecExecution(error=f"Unknown exclude_outliers column: {exclude_outlier_spec}", filter_warnings=filter_warnings)
        excl_method = exclude_outlier_spec.get("method", "iqr")
        if excl_method not in ("iqr", "zscore"):
            excl_method = "iqr"
        excl_group_raw = exclude_outlier_spec.get("group_by") or []
        excl_group = [c for c in (_resolve_column_fuzzy(g, sheet.columns) for g in excl_group_raw) if c]
        try:
            flagged = _detect_outliers(filtered, excl_col, excl_group, excl_method)
        except _OutlierDataQualityError as exc:
            return SpecExecution(error=str(exc), filter_warnings=filter_warnings)
        if not flagged.empty:
            filtered = filtered.drop(index=flagged.index)

    # Check if we need "aggregate then outlier" pattern:
    # Both outliers and aggregate/group_by requested, same column, no outliers.group_by
    # This means: aggregate by group_by, then find outliers in the aggregated values
    do_aggregate_then_outlier = False
    if outlier_spec and agg_wanted:
        outlier_col = outlier_spec.get("column")
        agg_spec = spec.get("aggregate") or {}
        agg_col = agg_spec.get("column")
        group_by = spec.get("group_by") or []
        outlier_group_by = outlier_spec.get("group_by") or []

        # Pattern: aggregate by group_by, then outliers on the aggregate column
        # (no outliers.group_by means across groups, not within)
        if (outlier_col and agg_col and _resolve_column_fuzzy(outlier_col, sheet.columns) == _resolve_column_fuzzy(agg_col, sheet.columns)
                and group_by and not outlier_group_by):
            do_aggregate_then_outlier = True

    if do_aggregate_then_outlier:
        # Step 1: Compute aggregate (group_by + aggregate)
        aggregate_df, error = _execute_aggregate(spec, filtered, sheet)
        if error:
            return SpecExecution(error=error, filter_warnings=filter_warnings)

        # Step 2: Run outlier detection on the aggregated result
        # The aggregated column name will be like "sum(sales_qty)"
        agg_col_name = aggregate_df.columns[-1] if len(aggregate_df.columns) > 0 else None
        if agg_col_name and not aggregate_df.empty:
            # Create a temporary sheet-like object with the aggregated data
            class _TempSheet:
                def __init__(self, df, columns):
                    self.df = df
                    self.columns = columns
                    self.sheet_name = sheet.sheet_name
                    self.file_path = sheet.file_path

            temp_sheet = _TempSheet(aggregate_df, list(aggregate_df.columns))
            # Outlier spec for the aggregated data: no group_by (across all groups), same method
            agg_outlier_spec = {
                "column": agg_col_name,
                "method": outlier_spec.get("method", "iqr"),
                "group_by": [],  # Across all aggregated groups
            }
            outliers_df, error = _execute_outliers(agg_outlier_spec, aggregate_df, temp_sheet, spec.get("limit"))
            if error:
                return SpecExecution(error=error, filter_warnings=filter_warnings)
    else:
        # Original behavior: independent outliers and aggregate on filtered data
        if outlier_spec:
            outliers_df, error = _execute_outliers(outlier_spec, filtered, sheet, spec.get("limit"))
            if error:
                return SpecExecution(error=error, filter_warnings=filter_warnings)

        if do_aggregate_then_exclude:
            # Aggregate WITHOUT sort/limit first, so "best performing"
            # can't be decided until after the outlier group(s) are
            # actually removed -- applying limit before removal could
            # cut off the real best-performing (non-outlier) row while
            # keeping an outlier that happened to sort higher.
            aggregate_df, error = _execute_aggregate(spec, filtered, sheet, apply_sort_limit=False)
            if error:
                return SpecExecution(error=error, filter_warnings=filter_warnings)
            if aggregate_df is not None and not aggregate_df.empty:
                agg_col_name = aggregate_df.columns[-1]
                try:
                    flagged = _detect_outliers(aggregate_df, agg_col_name, [], exclude_outlier_spec.get("method", "iqr"))
                except _OutlierDataQualityError as exc:
                    return SpecExecution(error=str(exc), filter_warnings=filter_warnings)
                if not flagged.empty:
                    aggregate_df = aggregate_df.drop(index=flagged.index)
                # Now apply the sort/limit that was deferred above.
                if spec.get("sort_descending") and agg_col_name in aggregate_df.columns:
                    try:
                        aggregate_df = aggregate_df.sort_values(by=agg_col_name, ascending=False)
                    except Exception:
                        pass
                limit = spec.get("limit")
                if isinstance(limit, int) and limit > 0:
                    aggregate_df = aggregate_df.head(limit)
        elif agg_wanted:
            aggregate_df, error = _execute_aggregate(spec, filtered, sheet)
            if error:
                return SpecExecution(error=error, filter_warnings=filter_warnings)

    if stats_wanted:
        statistics_df, error = _execute_statistics(spec, filtered, sheet)
        if error:
            return SpecExecution(error=error, filter_warnings=filter_warnings)

    best_per_group_df = None
    if spec.get("best_per_group"):
        best_per_group_df, error = _execute_best_per_group(spec["best_per_group"], filtered, sheet)
        if error:
            return SpecExecution(error=error, filter_warnings=filter_warnings)

    compute_df = None
    if spec.get("compute"):
        compute_df, error = _execute_compute(spec["compute"], sheet)
        if error:
            return SpecExecution(error=error, filter_warnings=filter_warnings)

    return SpecExecution(
        outliers_df=outliers_df, aggregate_df=aggregate_df,
        statistics_df=statistics_df, best_per_group_df=best_per_group_df,
        compute_df=compute_df,
        filter_warnings=filter_warnings,
        filtered_df=filtered,
    )


def _execute_compute(compute_spec: Dict[str, Any], sheet: "_SheetInfo") -> Tuple[Optional["pd.DataFrame"], Optional[str]]:
    """Execute mathematical operations between sub-query results."""
    operations = compute_spec.get("operations") or []
    formula = compute_spec.get("formula")
    label = compute_spec.get("label", "computed")

    if not operations or not formula:
        return None, "compute spec needs 'operations' list and 'formula'"

    # Execute each sub-spec
    op_results = {}
    for op in operations:
        name = op.get("name")
        sub_spec = op.get("spec")
        if not name or not sub_spec:
            return None, f"each operation needs 'name' and 'spec': {op}"
        # Recursively execute the sub-spec (but without compute to avoid infinite recursion)
        sub_spec_copy = dict(sub_spec)
        sub_spec_copy.pop("compute", None)
        execution = _execute_spec(sub_spec_copy, sheet)
        if execution.error:
            return None, f"sub-spec '{name}' failed: {execution.error}"
        # Get the result value - prefer aggregate, then outliers, then statistics
        result_df = None
        for df in (execution.aggregate_df, execution.outliers_df, execution.statistics_df):
            if df is not None and not df.empty:
                result_df = df
                break
        if result_df is None:
            return None, f"sub-spec '{name}' returned no data"
        # Extract the numeric value (last column, first row)
        value_col = result_df.columns[-1]
        val = result_df[value_col].iloc[0]
        try:
            op_results[name] = float(val)
        except (ValueError, TypeError):
            return None, f"sub-spec '{name}' produced non-numeric result: {val}"

    # Evaluate the formula
    try:
        # Safe eval - only allow basic math operations
        allowed_names = {k: v for k, v in op_results.items()}
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
        result_val = eval(formula, {"__builtins__": {}}, allowed_names)
    except Exception as exc:
        return None, f"formula evaluation failed: {exc}"

    # Build result DataFrame
    result_data = {name: [val] for name, val in op_results.items()}
    result_data[label] = [result_val]
    result_df = pd.DataFrame(result_data)
    return result_df, None


def _should_show_breakdown(question: str) -> bool:
    """
    Check if user explicitly asked for source rows/breakdown/citation.

    BUG (reproduced against a real transcript): "cite the sources as
    well" and "cite me the sources for it" matched NONE of the old
    keywords ("show source(s)", "source rows", ... -- note the absence
    of the word "cite" anywhere in that list), so a same-turn citation
    request silently did nothing. "source"/"cite"/"citation" as bare
    words fix the same-turn case; the FOLLOW-UP case ("give me the
    sources for it") is handled separately in answer_analysis_query
    via _is_bare_citation_followup, since a bare follow-up like that
    has no analytical content of its own for a fresh LLM spec call to
    work with at all.
    """
    q = question.lower()
    return any(kw in q for kw in [
        "show breakdown", "show source", "show sources", "verify", "show rows",
        "source rows", "breakdown", "detail rows", "raw rows", "underlying data",
        "cite", "citation", "sources", "source",
    ])


def _should_show_insights(question: str) -> bool:
    """Check if user explicitly asked for insights/analysis/explanation."""
    q = question.lower()
    return any(kw in q for kw in [
        "insight", "analyze", "analysis", "explain", "why", "interpret",
        "what does this mean", "derive insight", "business insight"
    ])
    if result.empty:
        return "No rows matched that."
    cols = list(result.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in result.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:,.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_result_markdown(result: "pd.DataFrame", spec: Dict[str, Any]) -> str:
    if result.empty:
        return "No rows matched that."
    cols = list(result.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in result.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:,.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_filter_warnings(warnings: List[str]) -> str:
    if not warnings:
        return ""
    lines = ["**Filter Warnings:**"]
    for w in warnings:
        lines.append(f"- {w}")
    return "\n".join(lines)


def _format_breakdown(filtered: "pd.DataFrame", spec: Dict[str, Any], sheet: "_SheetInfo", max_rows: int = 50) -> str:
    """Show the raw rows that contributed to the aggregate result for verification."""
    group_by = spec.get("group_by") or []
    agg_spec = spec.get("aggregate") or {}
    agg_col = agg_spec.get("column")
    agg_func = agg_spec.get("func", "sum")
    
    if not group_by or not agg_col:
        return ""
    
    # Resolve column names
    resolved_group_by = [c for c in (_resolve_column_fuzzy(g, sheet.columns) for g in group_by) if c]
    resolved_agg_col = _resolve_column_fuzzy(str(agg_col), sheet.columns)
    
    if not resolved_group_by or not resolved_agg_col:
        return ""
    
    # Show sample of raw rows per group
    lines = ["**Source Rows (Breakdown):**"]
    for group_vals, group_df in filtered.groupby(resolved_group_by, dropna=False):
        if isinstance(group_vals, tuple):
            group_key = ", ".join(str(v) for v in group_vals)
        else:
            group_key = str(group_vals)
        
        # Get the aggregated value for this group
        numeric = _coerce_numeric(group_df[resolved_agg_col])
        if agg_func == "count":
            agg_val = numeric.count()
        elif agg_func == "sum":
            agg_val = numeric.sum()
        elif agg_func == "mean":
            agg_val = numeric.mean()
        elif agg_func == "median":
            agg_val = numeric.median()
        elif agg_func == "min":
            agg_val = numeric.min()
        elif agg_func == "max":
            agg_val = numeric.max()
        else:
            agg_val = numeric.sum()
        
        lines.append(f"\n--- {group_key} ({agg_func}({resolved_agg_col}) = {agg_val:,.2f}) ---")
        
        # Show up to max_rows per group
        show_cols = resolved_group_by + [resolved_agg_col]
        # Add a few more context columns if available
        for col in filtered.columns:
            if col not in show_cols and len(show_cols) < 8:
                show_cols.append(col)
        
        sample = group_df[show_cols].head(10)
        if not sample.empty:
            cols = list(sample.columns)
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("|" + "|".join(["---"] * len(cols)) + "|")
            for _, row in sample.iterrows():
                cells = []
                for v in row:
                    if isinstance(v, float):
                        cells.append(f"{v:,.2f}")
                    else:
                        cells.append(str(v))
                lines.append("| " + " | ".join(cells) + " |")
        
        if len(group_df) > 10:
            lines.append(f"*... and {len(group_df) - 10} more rows*")
    
    return "\n".join(lines)


def _maybe_render_chart(result: "pd.DataFrame", spec: Dict[str, Any], question: str) -> Optional[str]:
    chart_req = spec.get("chart") or {}
    if not chart_req.get("wanted") or result.empty or len(result.columns) < 2:
        return None
    try:
        from core.chart_generator import ChartSpec
        from ui.chart_renderer import render_chart
    except Exception:
        logger.exception("data_analyst: chart modules unavailable")
        return None

    label_col = result.columns[0]
    value_cols = [c for c in result.columns[1:] if pd.api.types.is_numeric_dtype(result[c])]
    if not value_cols:
        return None

    # Intelligent chart type selection
    requested_type = chart_req.get("type", "auto")
    n_rows = len(result)
    n_value_cols = len(value_cols)
    
    # Determine best chart type if auto
    if requested_type == "auto":
        # Time series detection: check if label column looks like dates/times
        is_time_series = False
        if n_rows > 1:
            sample_vals = result[label_col].dropna().astype(str).head(10).tolist()
            date_patterns = ["-", "/", ":", "q1", "q2", "q3", "q4", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
            is_time_series = any(any(p in str(v).lower() for p in date_patterns) for v in sample_vals)
        
        # Categorical with many categories -> bar
        # Time series -> line
        # Few categories with single value -> pie
        # Multi-value columns -> grouped bar or line
        if is_time_series and n_rows > 5:
            chart_type = "line"
        elif n_value_cols > 1:
            chart_type = "bar"  # grouped bar for multi-series
        elif n_rows <= 8:
            chart_type = "pie"
        else:
            chart_type = "bar"
    else:
        chart_type = requested_type
        # Validate chart type
        valid_types = ("bar", "line", "pie", "area", "scatter")
        if chart_type not in valid_types:
            chart_type = "bar"

    # Use ALL data for charts (not just top 15), unless explicitly limited
    limit = chart_req.get("limit")
    if isinstance(limit, int) and limit > 0:
        chart_data = result.head(limit)
    else:
        chart_data = result  # Use all rows

    # Build series
    series = []
    for vc in value_cols:
        series.append({
"name": vc,
"values": [float(v) if pd.notna(v) else 0.0 for v in chart_data[vc]]
        })

    categories = [str(v) for v in chart_data[label_col]]

    chart_spec = ChartSpec(
        chart_type=chart_type,
        title=question[:120],
        x_label=label_col,
        y_label=", ".join(value_cols) if len(value_cols) > 1 else value_cols[0],
        categories=categories,
        series=series,
    )
    try:
        return render_chart(chart_spec, out_dir=CHARTS_OUTPUT_DIR)
    except Exception:
        logger.exception("data_analyst: chart render failed")
        return None


# ─── Public entry point ─────────────────────────────────────────────────────────

def _run_single_sheet_aggregate(sheet: _SheetInfo, column: str, func: str,
                                 group_by: List[str], filters: List[Dict[str, Any]]) -> Tuple["pd.DataFrame", Optional[str]]:
    """Shared helper: apply filters + group-by + aggregate to one
    sheet -- used both by the normal single-sheet path (via
    _execute_spec) and by _execute_compare for each sheet being
    compared, so the two paths can never silently compute a value
    differently."""
    inner_spec = {
"filters": filters or [],
"group_by": group_by or [],
"aggregate": {"column": column, "func": func},
    }
    execution = _execute_spec(inner_spec, sheet)
    if execution.error:
        return pd.DataFrame(), execution.error
    return execution.aggregate_df if execution.aggregate_df is not None else pd.DataFrame(), None


def _execute_compare(spec: Dict[str, Any], sheets: List[_SheetInfo]) -> Tuple["pd.DataFrame", Optional[str]]:
    """
    Computes the SAME aggregate independently on each of several
    sheets/files, then joins the results side by side on the shared
    group-by dimension(s) (e.g. Division, State, Town, Product) so they
    can be read/compared directly -- e.g. "compare total sales by state
    between file A and file B" produces one row per state with a
    column for each file's total, plus the difference and % change.

    Column names don't have to match byte-for-byte between files --
    _resolve_column_fuzzy tolerates minor naming differences (e.g.
Sales Value" vs "Sales Amount") since two independently-maintained
    spreadsheets tracking the same thing frequently don't use identical
    headers.

    Purely local pandas joins/arithmetic -- no data ever leaves this
    process, same as the rest of this module.
    """
    compare_spec = spec.get("compare") or {}
    col_name = str(compare_spec.get("column", ""))
    func = compare_spec.get("func", "sum")
    if func not in ("sum", "mean", "count", "min", "max", "median"):
        func = "sum"
    group_by_raw = compare_spec.get("group_by") or []
    filters = compare_spec.get("filters") or []

    if not sheets:
        return pd.DataFrame(), "no_sheets_to_compare"

    per_sheet_results = []
    for sheet in sheets:
        col = _resolve_column_fuzzy(col_name, sheet.columns)
        if col is None:
            return pd.DataFrame(), (
                f"Could not find a column matching \"{col_name}\" in "
                f"sheet \"{sheet.sheet_name}\" ({os.path.basename(sheet.file_path)})"
            )
        gb = [c for c in (_resolve_column_fuzzy(g, sheet.columns) for g in group_by_raw) if c]
        result, error = _run_single_sheet_aggregate(sheet, col, func, gb, filters)
        if error:
            return pd.DataFrame(), error
        per_sheet_results.append((sheet, gb, result))

    label = lambda sheet: os.path.splitext(os.path.basename(sheet.file_path))[0]

    if not group_by_raw:
        # No dimension to join on -- just line up one total per sheet.
        row = {}
        for sheet, _gb, result in per_sheet_results:
            value = result.iloc[0, 0] if not result.empty and len(result.columns) else None
            row[f"{func}({col_name}) -- {label(sheet)}"] = value
        merged = pd.DataFrame([row])
    else:
        merged = None
        for sheet, gb, result in per_sheet_results:
            value_col = result.columns[-1] if len(result.columns) else None
            renamed = result.rename(columns={value_col: f"{label(sheet)}"}) if value_col else result
            # Every sheet's own group-by columns get aligned to the
            # FIRST sheet's group-by column names so rows actually join
            # up instead of producing all-NaN duplicate columns when
            # two files spell the same dimension slightly differently.
            if merged is None:
                canonical_gb = group_by_raw[:len(gb)]
                renamed = renamed.rename(columns=dict(zip(gb, canonical_gb)))
                merged = renamed
            else:
                canonical_gb = group_by_raw[:len(gb)]
                renamed = renamed.rename(columns=dict(zip(gb, canonical_gb)))
                merged = merged.merge(renamed, on=canonical_gb, how="outer")

    value_cols = [c for c in merged.columns if c not in group_by_raw]
    if len(value_cols) == 2:
        a, b = value_cols
        merged["difference"] = merged[b] - merged[a]
        pct = ((merged[b] - merged[a]) / merged[a].replace(0, pd.NA)) * 100
        merged["% change"] = pct.replace([float("inf"), float("-inf")], pd.NA)

    if spec.get("sort_descending") and "difference" in merged.columns:
        merged = merged.sort_values(by="difference", key=lambda s: s.abs(), ascending=False)

    limit = spec.get("limit")
    if isinstance(limit, int) and limit > 0:
        merged = merged.head(limit)

    return merged, None


def _format_statistics_markdown(result: "pd.DataFrame", spec: Dict[str, Any]) -> str:
    if result.empty:
        return "No statistics computed."
    cols = list(result.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in result.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:,.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    # Add formula references
    formulas = result.attrs.get("formulas", {})
    if formulas:
        lines.append("\n**Formulas:**")
        for func, formula in formulas.items():
            lines.append(f"- {func}: `{formula}`")

    return "\n".join(lines)


_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract_numbers(text: str) -> List[float]:
    """Every numeric literal appearing in text, comma-separators and
    all, parsed to float. Used to check what a piece of LLM-written
    prose actually claims numerically."""
    numbers = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(0).replace(",", "")
        if raw in ("", "-", "."):
            continue
        try:
            numbers.append(float(raw))
        except ValueError:
            continue
    return numbers


def _validate_insights_grounded(insights: str, stats_text: str) -> Optional[str]:
    """
    Strip any line that cites a number not actually present (within a
    small rounding tolerance) in the real computed statistics it's
    supposedly describing.

    This is a load-bearing SAFETY CHECK, not just a prompt instruction.
    A small local model asked to derive "business implications" from a
    table of numbers will very readily invent a plausible-sounding
    extra percentage, comparison, or figure that was never actually
    computed -- simply telling it not to in the prompt doesn't reliably
    stop this on a 4-8B model. Checking what numbers the generated text
    actually contains against what the real statistics actually contain
    does. A line with no numbers in it at all (e.g. "the distribution
    is right-skewed") is left alone -- this only catches invented
    NUMBERS, not general qualitative commentary, which is a much
    smaller and harder-to-verify surface.
    """
    if not insights:
        return insights
    stats_numbers = _extract_numbers(stats_text)
    if not stats_numbers:
        return insights  # nothing to check against -- let it through as-is

    def grounded(n: float) -> bool:
        for real in stats_numbers:
            if abs(n - real) <= 0.5:
                return True  # small absolute tolerance for prose rounding
            if real != 0 and abs(n - real) / abs(real) <= 0.02:
                return True  # 2% relative tolerance for the same reason
        return False

    kept = []
    dropped_any = False
    for line in insights.split("\n"):
        nums = _extract_numbers(line)
        if not nums or all(grounded(n) for n in nums):
            kept.append(line)
        else:
            dropped_any = True
    result = "\n".join(kept).strip()
    if dropped_any:
        logger.warning(
            "Dropped one or more insight line(s) that cited a number "
            "not present in the actual computed statistics."
        )
    return result or None


def _get_llm_insights(question: str, stats_df: "pd.DataFrame", spec: Dict[str, Any]) -> Optional[str]:
    """Send statistics to LLM for insight derivation."""
    from core.config import CHART_MODEL, CHART_TIMEOUT, CHART_MAX_TOKENS
    from llm.ollama_client import send_prompt

    # Format statistics for LLM
    stats_text = _format_statistics_markdown(stats_df, spec)

    prompt = f"""Analyze the following statistical results computed from real data.

Question: {question}

Statistical Results:
{stats_text}

Provide 2-4 short, specific observations about what these EXACT numbers show --
distribution shape, notable spread, or how groups compare TO EACH OTHER.

STRICT RULES:
- Only ever reference a number that appears VERBATIM (or trivially rounded)
  in the Statistical Results above. Never introduce a new number, percentage,
  ratio, or comparison that isn't already shown there.
- Do NOT speculate about business causes, external context, industry
  benchmarks, or anything not directly computable from the table above --
  describe what the numbers show, not why they might have happened.
- If there's genuinely nothing notable beyond the numbers already shown,
  say so briefly rather than inventing something to fill space.

Bullet points only."""

    # Use a shorter timeout for insights (60s) to avoid long waits
    insights_timeout = min(60, CHART_TIMEOUT)
    try:
        insights = send_prompt(prompt, temperature=0.0, max_tokens=CHART_MAX_TOKENS,
                               timeout=insights_timeout, model=CHART_MODEL)
        insights = insights.strip() if insights else None
        return _validate_insights_grounded(insights, stats_text)
    except Exception:
        return None


_GENERIC_FINEST_LEVEL_WORDS = ("product", "products", "item", "items", "sku", "skus")


def _ensure_hierarchy_level_filter(spec: Dict[str, Any], sheet: "_SheetInfo", question: str) -> Dict[str, Any]:
    """
    Safety net for hierarchical identifier columns (see
    core/xlsx_reshape.py's detect_hierarchy_levels): if the question
    names a specific level ("top 5 SKU", "by segment", "at the
    franchise level") but the LLM-generated spec forgot to actually
    filter that column's "_Level_Name" companion column, inject the
    filter here rather than silently returning a ranking that mixes
    every level together.

    This exists because the corresponding prompt RULE -- filter
    "<column>_Level_Name" to the requested level -- is easy for a
    smaller local model to skip when the question has no OTHER filter
    to anchor it to (e.g. "top 5 sku" with no metric/value mentioned at
    all), even though the rule itself is stated clearly. A prompt
    instruction is a request; this is a guarantee, checked in code
    against the sheet's own REAL level names (read from
    sheet.categorical_values / sheet.df -- never a hardcoded word
    list), so it works identically on any file with this kind of
    column, not just the one this was first noticed on.

    Only ever ADDS a filter that was implied but missing -- never
    removes or overrides a level filter the spec already has, and does
    nothing at all to a sheet with no "_Level_Name" columns.
    """
    if not isinstance(spec, dict):
        return spec
    level_name_cols = [c for c in sheet.columns if c.endswith("_Level_Name")]
    if not level_name_cols:
        return spec

    existing_filter_cols = {
        str(f.get("column", "")).lower()
        for f in (spec.get("filters") or [])
        if isinstance(f, dict)
    }
    # Also check best_per_group/group_by/aggregate in case the level
    # filter was folded in some other way already.
    q_lower = question.lower()
    q_words = set(re.findall(r"[a-z0-9]+", q_lower))

    try:
        from rapidfuzz import fuzz
    except ImportError:
        fuzz = None

    new_filters = list(spec.get("filters") or [])
    changed = False

    for level_col in level_name_cols:
        if level_col.lower() in existing_filter_cols:
            continue  # already filtered explicitly -- respect the LLM's choice

        real_values = sheet.categorical_values.get(level_col) or []
        if not real_values:
            continue

        matched_value = None

        # 1) Direct word/fuzzy match: does the question mention one of
        # this column's REAL level names (e.g. "segment", "sub
        # segment", "franchise", or whatever this file's own data uses)?
        for real_value in real_values:
            rv_lower = real_value.lower()
            rv_words = set(re.findall(r"[a-z0-9]+", rv_lower))
            if rv_lower in q_lower or (rv_words and rv_words <= q_words):
                matched_value = real_value
                break
            if fuzz is not None:
                for qw in q_words:
                    if len(qw) >= 4 and fuzz.ratio(qw, rv_lower.replace(" ", "")) >= 85:
                        matched_value = real_value
                        break
            if matched_value:
                break

        # 2) Generic synonym: "product"/"item"/"SKU" with no exact level
        # name match -> the FINEST level (highest numeric _Level for
        # this column), read from the sheet's own data, not assumed.
        if matched_value is None and q_words & set(_GENERIC_FINEST_LEVEL_WORDS):
            level_col_num = level_col[: -len("_Name")]  # "<col>_Level_Name" -> "<col>_Level"
            if level_col_num in sheet.df.columns and level_col in sheet.df.columns:
                pairs = sheet.df[[level_col_num, level_col]].dropna().drop_duplicates()
                if not pairs.empty:
                    finest = pairs.sort_values(level_col_num, ascending=False).iloc[0]
                    matched_value = finest[level_col]

        if matched_value:
            new_filters.append({"column": level_col, "op": "==", "value": matched_value})
            changed = True

    if changed:
        spec = dict(spec)
        spec["filters"] = new_filters
    return spec


_BARE_CITATION_FOLLOWUP_RE = re.compile(
    r"^\s*(can you |could you |please )?"
    r"(give me|cite|show me|list|provide)?\s*"
    r"(me\s+)?(the\s+)?(source|sources|citation|citations)\s*"
    r"(for it|for that|for the above|of it|of that)?\s*\.?\s*$",
    re.IGNORECASE,
)


def _is_bare_citation_followup(question: str) -> bool:
    """
    True for a content-free citation/source request with no new
    analytical ask of its own -- "give me the sources for it", "cite
    the sources", "cite me the sources for it". These have essentially
    nothing for a fresh LLM spec call to work with beyond copying the
    previous turn's spec verbatim -- a real risk of drift for a smaller
    local model (a dropped filter, a changed limit, a subtly different
    sheet) even with the previous spec injected as context.

    BUG (reproduced against a real transcript): this exact phrasing
    used to fall through to plain LLM/RAG entirely and produce a
    HALLUCINATED citation ("derived from publicly available industry
    reports, government press releases...") for an answer that was
    actually computed from a local Excel file with no such sources
    involved at all. Detecting this pattern and reusing the stored
    previous_spec UNCHANGED instead removes the regeneration risk
    entirely -- the citation is attached to the SAME real result the
    user is asking about, not a freshly (and separately) reconstructed
    approximation of it, and it can never fall through to a
    fabricated-sounding non-answer.
    """
    return bool(_BARE_CITATION_FOLLOWUP_RE.match(question.strip()))


def answer_analysis_query(question: str, previous_spec: Optional[Dict[str, Any]] = None) -> AnalysisResult:
    """
    Full pipeline: load registry -> ask LLM for a spec -> execute it
    safely -> format the real computed result. Called after
    is_analysis_query() has confirmed this is worth doing, OR when
    core/engine.py has determined this is a follow-up to a previous
    successful analysis answer (see get_last_analysis_spec()) -- a bare
    follow-up like "tell me the state for the above" has no analysis
    verb or column name of its own for is_analysis_query() to match on,
    so engine.py routes those here directly using the stored context.

    `previous_spec`, when given, is threaded into the spec-generation
    prompt (see _build_spec_prompt) so the LLM can inherit the prior
    question's sheet/filters/metric instead of trying to reconstruct a
    full spec from a few words with no data context of their own.

    NOTE: worked examples for the spec format live in
    _SPEC_SYSTEM_PROMPT (the string actually sent to the LLM) and in
    _build_spec_prompt's previous_spec handling -- NOT here. An earlier
    version of this docstring contained several "worked examples" for
    follow-up questions that were never actually part of any prompt --
    plain documentation text inside a Python docstring, invisible to
    the model at request time. That's why a follow-up like "tell me the
    state for the above" used to fail outright: the fix that looked
    like it existed never actually ran. See get_last_analysis_spec()/
    _build_spec_prompt() for the real, functioning version.
    """
    from core.config import CHART_MODEL, CHART_TIMEOUT, CHART_MAX_TOKENS
    from llm.ollama_client import send_prompt

    sheets = _load_registry()
    if not sheets:
        return AnalysisResult(text="", error="no_spreadsheet")

    if previous_spec is not None and _is_bare_citation_followup(question):
        # See _is_bare_citation_followup's docstring: this question has
        # no analytical content of its own, so skip the LLM spec call
        # entirely and reuse the prior turn's spec exactly -- the
        # citation/breakdown then attaches to the SAME real result
        # already computed, guaranteed, rather than a freshly (and
        # separately) reconstructed approximation of it that a smaller
        # local model could easily get slightly wrong.
        spec = previous_spec
    else:
        prompt = _build_spec_prompt(question, sheets, previous_spec=previous_spec)
        try:
            raw = send_prompt(prompt, temperature=0.0, max_tokens=CHART_MAX_TOKENS,
                               timeout=CHART_TIMEOUT, model=CHART_MODEL)
        except Exception as exc:
            logger.exception("data_analyst: LLM spec generation failed")
            return AnalysisResult(text="", error=f"llm_error: {exc}")

        spec = _parse_spec(raw)
        for attempt in range(2):
            if spec:
                break
            # A bounded corrective retry (up to 2 attempts total): a smaller
            # local model asked for a genuinely compound spec (outliers +
            # multiple group-bys + a separate aggregate) occasionally
            # produces near-miss JSON -- trailing prose, an unescaped quote,
            # a dangling comma the salvage pass above couldn't fix. Retrying
            # blind with the exact same prompt tends to reproduce the same
            # mistake, so this shows the model its own broken output and
            # asks specifically for valid JSON, which local models follow
            # far more reliably than a generic "try again." This is the
            # difference between falling back to the vastly worse plain-LLM-
            # arithmetic-over-a-sample path and actually answering via real
            # pandas execution over the complete data.
            retry_prompt = (
                f"{prompt}\n\n"
                f"--- YOUR PREVIOUS OUTPUT (INVALID) ---\n{raw.strip()[:1500]}\n"
                f"--- END PREVIOUS OUTPUT ---\n\n"
                f"That was not valid JSON matching the schema above. Output ONLY a "
                f"single valid JSON object for the schema -- no markdown fences, no "
                f"explanation, no trailing text of any kind.\n\nJSON SPEC:"
            )
            try:
                raw = send_prompt(retry_prompt, temperature=0.0, max_tokens=CHART_MAX_TOKENS,
                                   timeout=CHART_TIMEOUT, model=CHART_MODEL)
                spec = _parse_spec(raw)
            except Exception:
                logger.exception("data_analyst: LLM spec retry failed")
        if not spec:
            return AnalysisResult(text="", error="could_not_understand_question")
    _remember_analysis_spec(question, spec)

    has_compare = bool(spec.get("compare"))
    has_statistics = bool(spec.get("statistics"))

    if has_compare:
        wanted_names = spec["compare"].get("sheets") or []
        compare_sheets = [s for s in sheets if s.sheet_name in wanted_names]
        if len(compare_sheets) < 2:
            compare_sheets = [
                s for s in sheets
                if s.sheet_name in wanted_names
                or os.path.splitext(os.path.basename(s.file_path))[0] in wanted_names
            ]
        if len(compare_sheets) < 2:
            return AnalysisResult(text="", error="ambiguous_or_missing_compare_sheets")
        
        # If statistics also requested, do a statistics compare across all sheets
        if has_statistics:
            stat_spec = spec.get("statistics") or {}
            stat_funcs = stat_spec.get("funcs") or ["mean", "median", "std", "var", "min", "max", "sum", "count"]
            stat_group_by = stat_spec.get("group_by") or []
            stat_column = stat_spec.get("column")
            
            # Compute statistics for each sheet
            compare_data = []
            for s in compare_sheets:
                stat_sub_spec = {"statistics": {"column": stat_column, "funcs": stat_funcs, "group_by": stat_group_by}}
                execution = _execute_spec(stat_sub_spec, s)
                if execution.error:
                    return AnalysisResult(text="", error=execution.error)
                if execution.statistics_df is not None:
                    df = execution.statistics_df.copy()
                    df["_sheet"] = s.sheet_name
                    compare_data.append(df)
            
            if compare_data:
                merged = pd.concat(compare_data, ignore_index=True)
                # Pivot to have sheets as columns for comparison
                if stat_group_by:
                    # For grouped stats, pivot on the group_by column
                    pivot_cols = stat_group_by + stat_funcs
                    # Melt and pivot
                    id_vars = stat_group_by + ["_sheet"]
                    value_vars = stat_funcs
                    melted = merged.melt(id_vars=id_vars, value_vars=value_vars, var_name="statistic", value_name="value")
                    pivoted = melted.pivot_table(index=stat_group_by + ["statistic"], columns="_sheet", values="value").reset_index()
                    compare_text = _format_result_markdown(pivoted, spec)
                else:
                    # Global stats - pivot on statistic
                    melted = merged.melt(id_vars=["_sheet"], value_vars=stat_funcs, var_name="statistic", value_name="value")
                    pivoted = melted.pivot_table(index="statistic", columns="_sheet", values="value").reset_index()
                    compare_text = _format_result_markdown(pivoted, spec)
                
                sections = [f"**Statistics Comparison**\n\n{compare_text}"]
                
                # Also show individual statistics for first sheet with insights
                stat_sheet = compare_sheets[0]
                stat_sub_spec = {"statistics": {"column": stat_column, "funcs": stat_funcs, "group_by": stat_group_by}}
                execution = _execute_spec(stat_sub_spec, stat_sheet)
                if execution.error:
                    return AnalysisResult(text="", error=execution.error)
                
                if execution.statistics_df is not None:
                    stats_text = _format_statistics_markdown(execution.statistics_df, stat_sub_spec)
                    sections.append(f"**Statistics (detail: {stat_sheet.sheet_name})**\n\n{stats_text}")
                    insights = _get_llm_insights(question, execution.statistics_df, stat_sub_spec)
                    if insights:
                        sections.append(f"**Insights**\n\n{insights}")
                
                text = "\n\n".join(sections)
                # Generate chart for statistics comparison if requested
                chart_path = _maybe_render_chart(pivoted, spec, question) if spec.get("chart", {}).get("wanted") else None
                return AnalysisResult(text=text, chart_path=chart_path)
            else:
                return AnalysisResult(text="", error="no_statistics_computed")
        else:
            # Original compare behavior
            compare_result, error = _execute_compare(spec, compare_sheets)
            if error:
                return AnalysisResult(text="", error=error)
            compare_text = _format_result_markdown(compare_result, spec)
            # Generate chart for compare if requested
            chart_path = _maybe_render_chart(compare_result, spec, question) if spec.get("chart", {}).get("wanted") else None
            return AnalysisResult(text=compare_text, chart_path=chart_path)

    sheet_name = spec.get("sheet")
    sheet = _resolve_sheet_name(sheet_name, sheets)
    if sheet is None:
        # Fall back to whichever sheet's real (fuzzy-resolved) columns
        # cover the most of what this spec actually references --
        # scored by best match count rather than requiring a PERFECT
        # exact-name subset match, since a single near-miss column name
        # (e.g. the model writing "Brand_Level_Name" for the real
        # "BRANDS_Level_Name") used to be enough on its own to make an
        # otherwise-correct spec unresolvable.
        candidate_cols = {f.get("column") for f in spec.get("filters", []) or [] if f.get("column")}
        candidate_cols |= set(c for c in (spec.get("group_by") or []) if c)
        if spec.get("aggregate", {}).get("column"):
            candidate_cols.add(spec["aggregate"]["column"])
        for key in ("outliers", "exclude_outliers", "statistics"):
            sub = spec.get(key) or {}
            if sub.get("column"):
                candidate_cols.add(sub["column"])
            candidate_cols |= set(c for c in (sub.get("group_by") or []) if c)

        if len(sheets) == 1:
            sheet = sheets[0]
        elif candidate_cols:
            best_sheet, best_matches = None, 0
            for s in sheets:
                matches = sum(1 for c in candidate_cols if _resolve_column_fuzzy(c, s.columns))
                if matches > best_matches:
                    best_sheet, best_matches = s, matches
            # Require at least half the referenced columns to resolve
            # against a sheet before trusting it as "the" sheet -- this
            # still refuses to silently guess between two genuinely
            # unrelated sheets, it just no longer demands perfection.
            if best_sheet and best_matches >= max(1, len(candidate_cols) // 2):
                sheet = best_sheet
    if sheet is None:
        return AnalysisResult(text="", error="ambiguous_sheet")

    spec = _ensure_hierarchy_level_filter(spec, sheet, question)

    execution = _execute_spec(spec, sheet)
    if execution.error:
        return AnalysisResult(text="", error=execution.error)

    sections: List[str] = []
    chart_path = None

    # Add filter warnings if any
    if execution.filter_warnings:
        sections.append(_format_filter_warnings(execution.filter_warnings))

    if execution.outliers_df is not None:
        col = (spec.get("outliers") or {}).get("column", "")
        group_by = (spec.get("outliers") or {}).get("group_by") or []
        outliers_text = _format_outliers_markdown(execution.outliers_df, col, group_by)
        csv_path = _maybe_save_outliers_csv(execution.outliers_df, col, sheet)
        if csv_path:
            outliers_text += (
                f"\n\n*All {len(execution.outliers_df):,} outliers are shown above, and also "
                f"saved in full to `{csv_path}`.*"
            )
        if execution.aggregate_df is not None:
            sections.append(f"**Outliers**\n\n{outliers_text}")
        else:
            sections.append(outliers_text)

    if execution.aggregate_df is not None:
        agg_text = _format_result_markdown(execution.aggregate_df, spec)
        if execution.outliers_df is not None:
            sections.append(f"**Aggregate**\n\n{agg_text}")
        else:
            sections.append(agg_text)
        # Add breakdown for verification ONLY when explicitly asked
        if _should_show_breakdown(question):
            breakdown_text = _format_breakdown(execution.filtered_df, spec, sheet)
            if breakdown_text:
                sections.append(breakdown_text)
        # Chart aggregate data unless outliers are also requested and chart is wanted
        if not (execution.outliers_df is not None and spec.get("chart", {}).get("wanted")):
            chart_path = _maybe_render_chart(execution.aggregate_df, spec, question)

    # If outliers are requested and chart is wanted, chart the outliers data
    if execution.outliers_df is not None and spec.get("chart", {}).get("wanted"):
        chart_path = _maybe_render_chart(execution.outliers_df, spec, question)

    # Get insights for aggregate/outlier results ONLY when explicitly asked
    if _should_show_insights(question) and (execution.aggregate_df is not None or execution.outliers_df is not None):
        insights_df = None
        if execution.aggregate_df is not None:
            insights_df = execution.aggregate_df
        if execution.outliers_df is not None:
            if insights_df is not None:
                insights_df = pd.concat([insights_df, execution.outliers_df], ignore_index=True)
            else:
                insights_df = execution.outliers_df
        
        if insights_df is not None:
            insights = _get_llm_insights(question, insights_df, spec)
            if insights:
                sections.append(f"**Insights**\n\n{insights}")

    # Handle compute results
    if execution.compute_df is not None:
        compute_text = _format_result_markdown(execution.compute_df, spec)
        sections.append(f"**Computed Result**\n\n{compute_text}")
        if chart_path is None:
            chart_path = _maybe_render_chart(execution.compute_df, spec, question)

    if execution.statistics_df is not None:
        stats_text = _format_statistics_markdown(execution.statistics_df, spec)
        if execution.outliers_df is not None or execution.aggregate_df is not None:
            sections.append(f"**Statistics**\n\n{stats_text}")
        else:
            sections.append(stats_text)
            # Generate chart for statistics-only queries
            if chart_path is None:
                chart_path = _maybe_render_chart(execution.statistics_df, spec, question)
        # If only statistics requested (no outliers/aggregate), get LLM insights
        if execution.outliers_df is None and execution.aggregate_df is None:
            insights = _get_llm_insights(question, execution.statistics_df, spec)
            if insights:
                sections.append(f"**Insights**\n\n{insights}")

    if execution.best_per_group_df is not None:
        bpg_text = _format_result_markdown(execution.best_per_group_df, spec)
        other_sections_present = any(
            x is not None for x in
            (execution.outliers_df, execution.aggregate_df, execution.statistics_df)
        )
        if other_sections_present:
            sections.append(f"**Best per group**\n\n{bpg_text}")
        else:
            sections.append(bpg_text)
            if chart_path is None:
                chart_path = _maybe_render_chart(execution.best_per_group_df, spec, question)

    text = "\n\n".join(sections) if sections else "No rows matched that."
    return AnalysisResult(text=text, chart_path=chart_path)