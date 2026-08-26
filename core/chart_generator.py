"""
core/chart_generator.py

Chart generation for explicit "make me a chart/graph" requests.

Design
------
A local text model cannot draw a chart, and asking it to hand-produce an
ASCII/SVG chart from memory is exactly the kind of open-ended task that
invites invented numbers. This module instead treats chart generation as
two separate, much more reliable steps:

  1. EXTRACTION (LLM): pull real numbers out of the already-retrieved
     knowledge context into a small, strict JSON schema. This is a
     narrow, mechanical task with one correct answer per number, which is
     why even a 7-8B local model can do it reasonably reliably when the
     alternative (freeform chart description) would not be.

  2. RENDERING (matplotlib, no model involved): deterministically turn
     that JSON into an actual PNG. See ui/chart_renderer.py.

If extraction can't find enough real data, it says so explicitly
(``insufficient_data``) rather than the caller ever falling back to
inventing placeholder numbers.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.config import CHART_MODEL, OLLAMA_MODEL, CHART_TIMEOUT, CHART_MAX_TOKENS
from llm.ollama_client import send_prompt
from llm.prompt_builder import build_chart_extraction_prompt

logger = logging.getLogger(__name__)


# ─── Intent detection ──────────────────────────────────────────────────────

# Deliberately explicit and fairly narrow — chart generation is an extra
# LLM call plus rendering, so it should only fire when the user actually
# asked for a visual, not on every question that happens to compare two
# numbers in passing.
_CHART_TRIGGER_PATTERN = re.compile(
    r"\b("
    r"chart|graph|plot|"
    r"visuali[sz]e|visuali[sz]ation|"
    r"pie\s*chart|bar\s*chart|line\s*chart|line\s*graph|"
    r"trend\s*line|"
    r"draw\s+(a|me)\s+(a\s+)?(chart|graph|plot)|"
    r"show\s+(me\s+)?(this|that|it)?\s*(as\s+a\s+)?(chart|graph|plot)"
    r")\b",
    re.IGNORECASE,
)


def wants_chart(question: str) -> bool:
    """True if the user explicitly asked for a chart/graph/plot/visualization."""
    return bool(_CHART_TRIGGER_PATTERN.search(question))


# ─── Chart spec ────────────────────────────────────────────────────────────

@dataclass
class ChartSpec:
    chart_type: str = "bar"          # "bar" | "line" | "pie"
    title:      str = ""
    x_label:    str = ""
    y_label:    str = ""
    categories: List[str] = field(default_factory=list)
    series:     List[Dict[str, Any]] = field(default_factory=list)
    insufficient_data: bool = False
    reason:     str = ""


_VALID_TYPES = {"bar", "line", "pie"}


def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    """
    Small local models frequently wrap JSON in markdown fences or add a
    stray sentence before/after it despite instructions not to. Strip
    fences first, then fall back to extracting the first {...} block
    rather than failing outright on an otherwise-valid response.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _validate_spec(data: Dict[str, Any]) -> ChartSpec:
    """
    Turn a raw parsed dict into a ChartSpec, catching the ways a small
    model's JSON can be technically-parseable but still unusable: wrong
    chart_type, mismatched series/category lengths, empty series, etc.
    Anything that fails validation is treated as insufficient data rather
    than rendered as a broken or misleading chart.
    """
    if data.get("insufficient_data"):
        return ChartSpec(
            insufficient_data=True,
            reason=str(data.get("reason") or "Not enough data was found in the knowledge base."),
        )

    chart_type = str(data.get("chart_type", "bar")).lower().strip()
    if chart_type not in _VALID_TYPES:
        chart_type = "bar"

    categories = data.get("categories") or []
    series = data.get("series") or []

    if not isinstance(categories, list) or not categories:
        return ChartSpec(insufficient_data=True,
                          reason="No category labels (years/companies/etc.) were found.")
    if not isinstance(series, list) or not series:
        return ChartSpec(insufficient_data=True,
                          reason="No numeric data series were found.")

    clean_series = []
    for s in series:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name", "")).strip() or "Series"
        values = s.get("values")
        if not isinstance(values, list):
            continue
        # Pad/truncate rather than reject outright — a model that dropped
        # one trailing null is still usable data for the other points.
        if len(values) < len(categories):
            values = values + [None] * (len(categories) - len(values))
        elif len(values) > len(categories):
            values = values[: len(categories)]
        numeric_values = []
        for v in values:
            if v is None:
                numeric_values.append(None)
                continue
            try:
                numeric_values.append(float(v))
            except (TypeError, ValueError):
                numeric_values.append(None)
        if any(v is not None for v in numeric_values):
            clean_series.append({"name": name, "values": numeric_values})

    if not clean_series:
        return ChartSpec(insufficient_data=True,
                          reason="The values found couldn't be read as numbers.")

    # A pie chart only makes sense as a single share-of-whole breakdown —
    # if the model returned multiple series for a "pie" request, that's a
    # comparison, not a breakdown; treat it as a bar chart instead of
    # rendering a misleading pie.
    if chart_type == "pie" and len(clean_series) > 1:
        chart_type = "bar"

    return ChartSpec(
        chart_type=chart_type,
        title=str(data.get("title") or "").strip() or "Chart",
        x_label=str(data.get("x_label") or "").strip(),
        y_label=str(data.get("y_label") or "").strip(),
        categories=[str(c) for c in categories],
        series=clean_series,
    )


def build_chart_spec_from_markdown_table(question: str, markdown_text: str) -> ChartSpec:
    """
    Deterministically build a ChartSpec straight out of a Markdown table
    that was ALREADY shown to the user in a previous turn (e.g. "top 5
    brands by sale value" followed by "can you draw a chart based on
    it"). No LLM call involved -- the table is already clean, real data,
    so re-extracting it through an LLM would just add latency and a new
    chance to mangle a number that's already correct on screen.

    Picks the first column as the category labels and the LAST column
    that parses as numeric across every row as the value series (that's
    reliably the "value"/"amount" column in this app's tables — rank/id
    columns are earlier, if present, and text labels never parse as
    numeric).
    """
    lines = [l for l in markdown_text.strip().splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return ChartSpec(insufficient_data=True,
                          reason="No table was found in the previous answer to chart.")

    def _split(row: str) -> List[str]:
        row = row.strip().strip("|")
        return [c.strip() for c in row.split("|")]

    header = _split(lines[0])
    data_lines = lines[1:]
    # Skip a Markdown separator row ("|---|---|") if present.
    if data_lines and re.match(r"^\|?[\s:|-]+\|?$", data_lines[0].strip()):
        data_lines = data_lines[1:]

    rows = [_split(r) for r in data_lines if len(_split(r)) == len(header)]
    if not header or not rows:
        return ChartSpec(insufficient_data=True,
                          reason="The previous table couldn't be parsed for charting.")

    def _to_number(s: str) -> Optional[float]:
        cleaned = re.sub(r"[,\s%₹$]", "", s)
        cleaned = re.sub(r"^\((.*)\)$", r"-\1", cleaned)  # (123) -> -123
        try:
            return float(cleaned)
        except ValueError:
            return None

    # Find the last column where every row parses as numeric.
    value_col_idx = None
    for idx in range(len(header) - 1, -1, -1):
        values = [_to_number(r[idx]) for r in rows]
        if all(v is not None for v in values):
            value_col_idx = idx
            break

    if value_col_idx is None or value_col_idx == 0:
        return ChartSpec(insufficient_data=True,
                          reason="No numeric column was found in the previous table.")

    # Prefer a text label column (not a bare rank/index) for categories.
    label_col_idx = 0
    for idx, col_name in enumerate(header):
        if idx == value_col_idx:
            continue
        sample = [r[idx] for r in rows]
        if not all(_to_number(v) is not None for v in sample):
            label_col_idx = idx
            break

    categories = [r[label_col_idx] for r in rows]
    values = [_to_number(r[value_col_idx]) for r in rows]

    return ChartSpec(
        chart_type="bar",
        title=question[:80] or header[value_col_idx],
        x_label=header[label_col_idx],
        y_label=header[value_col_idx],
        categories=categories,
        series=[{"name": header[value_col_idx], "values": values}],
    )


def try_build_chart_spec_from_facts(question: str, results: List[Dict[str, Any]]) -> Optional[ChartSpec]:
    """
    Deterministic chart-spec construction from ingestion-time-extracted
    facts (ingestion/financial_facts.py) -- same principle as
    core.direct_answer.try_financial_fact_lookup: a plain "chart of
    <metric> for <years>" request doesn't need an LLM to re-read raw
    table text and hope it finds the right numbers, when those numbers
    were already deterministically extracted and tagged to their own
    year at ingestion time.

    Returns None (falls through to the LLM-based `extract_chart_spec`)
    if the metric/year(s) can't be confidently identified from the
    question, or none of the requested years have a fact on record --
    same conservative principle as the rest of the direct-answer paths.
    """
    from core.direct_answer import _detect_requested_metric, collect_financial_facts
    from retrieval.knowledge_index import detect_years_in_question, detect_companies_in_question

    metric = _detect_requested_metric(question)
    if metric is None:
        return None
    years = sorted(int(y) for y in detect_years_in_question(question) if y.isdigit())
    if len(years) < 1:
        return None
    companies = detect_companies_in_question(question)
    company = companies[0] if companies else None

    # collect_financial_facts returns facts grouped BY COMPANY
    # ({company_or_"default": {year: value}}) and expects a LIST of
    # companies. Pass the list and then pull out this company's own dict
    # -- previously a single company STRING was passed as the list arg,
    # so the function iterated its characters ('G','P','I') as if each
    # were a company, produced a dict keyed by those letters, and the
    # `year in facts_by_year` check below never matched -- silently
    # forcing every deterministic chart back onto the LLM path. Using
    # the same ranked facts here keeps charts as accurate as the text
    # answers.
    all_facts, _sources = collect_financial_facts(metric, years, companies, results)
    facts_by_year = all_facts.get(company or "default", {})
    if not facts_by_year:
        return None

    found_years = [y for y in years if y in facts_by_year]
    if not found_years:
        return None

    chart_type = "pie" if "pie" in question.lower() else ("line" if "line" in question.lower() else "bar")
    title = f"{metric}" + (f" — {company}" if company else "")
    return ChartSpec(
        chart_type=chart_type,
        title=title,
        x_label="Year",
        y_label=metric,
        categories=[str(y) for y in found_years],
        series=[{"name": metric, "values": [facts_by_year[y] for y in found_years]}],
        insufficient_data=False,
    )


def extract_chart_spec(question: str, context: str) -> ChartSpec:
    """
    Run the extraction LLM call and return a validated ChartSpec.

    Tries CHART_MODEL first (see core.config — defaults to
    qwen2.5:7b-instruct for its stronger strict-JSON adherence). If that
    model isn't pulled on this machine, falls back to OLLAMA_MODEL
    automatically rather than failing the whole chart request over a
    missing model pull.
    """
    prompt = build_chart_extraction_prompt(question, context)

    raw = None
    try:
        raw = send_prompt(
            prompt, temperature=0.0, max_tokens=CHART_MAX_TOKENS,
            timeout=CHART_TIMEOUT, model=CHART_MODEL,
        )
    except RuntimeError as exc:
        if CHART_MODEL != OLLAMA_MODEL:
            logger.warning(
                "Chart model '%s' unavailable (%s) — falling back to '%s'. "
                "Run `ollama pull %s` to use the preferred chart-extraction model.",
                CHART_MODEL, exc, OLLAMA_MODEL, CHART_MODEL,
            )
            try:
                raw = send_prompt(
                    prompt, temperature=0.0, max_tokens=CHART_MAX_TOKENS,
                    timeout=CHART_TIMEOUT, model=OLLAMA_MODEL,
                )
            except RuntimeError as exc2:
                return ChartSpec(insufficient_data=True,
                                  reason=f"Could not reach the local model: {exc2}")
        else:
            return ChartSpec(insufficient_data=True,
                              reason=f"Could not reach the local model: {exc}")

    if not raw:
        return ChartSpec(insufficient_data=True, reason="The model returned an empty response.")

    data = _parse_json_response(raw)
    if data is None:
        return ChartSpec(
            insufficient_data=True,
            reason="The model's response wasn't valid JSON, so no chart could be built.",
        )

    return _validate_spec(data)