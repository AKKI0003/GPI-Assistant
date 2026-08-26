"""
llm/ollama_client.py

Thin, robust wrapper around the Ollama /api/generate endpoint.

Provides:
- send_prompt()    — blocking call, returns full response string
- stream_prompt()  — generator that yields token chunks
"""

import json
import os
import re
from typing import Generator

import requests

from core.config import (
    OLLAMA_ENDPOINT, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_THINK,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_CTX_WINDOW,
    LLM_TOP_K, LLM_TOP_P, LLM_REPEAT_PENALTY, LLM_NUM_THREAD,
)

# Some "thinking"-capable models (qwen3 among them) don't reliably honor
# a client-side "think": false request on every Ollama version/build --
# when that happens, the model's raw internal reasoning either arrives
# wrapped in <think>...</think> tags inside the normal `response` field
# (rather than Ollama's newer separate `thinking` field), or -- on a
# small/weak model especially -- leaks out as unwrapped, narrated
# stream-of-consciousness prose ("Let me check...", "Let me look for...",
# "I think I need to be clear...") that spirals without ever committing
# to a final answer. Relying solely on the API-level flag to prevent
# this is fragile across Ollama versions and model builds, so this is
# stripped/detected here too as a second, independent line of defense
# regardless of whether the flag worked.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_NARRATION_PHRASES = (
    "let me check", "let me look", "let me try", "let me double-check",
    "let me double check", "let me see", "let me verify", "let me search",
    "i think i need to", "i need to be clear", "let me be more precise",
)

# A second, distinct failure mode from the narration loop above: at
# greedy decoding (LLM_TEMPERATURE=0, LLM_TOP_K=1 in core/config.py --
# deliberately deterministic, so there's no sampling randomness to knock
# the model off a bad path) a model that has been given a context with
# nothing usable in it for the specific thing being asked (e.g. a year's
# PDF chunk that failed deterministic fact-extraction and so contributed
# no grounding text at all) doesn't trail off or say "I don't know" once
# -- it locks onto that one short sentence and reproduces it verbatim,
# again and again, until num_predict/LLM_MAX_TOKENS is reached. A mild
# repeat_penalty (1.05) is not always enough to break this on its own.
# Unlike the narration case, there is no fixed phrase list to match on
# here -- the repeated sentence is different every time -- so this
# detects the PATTERN (the same sentence appearing 3+ times in a row)
# instead of specific wording.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _has_exact_sentence_repeat_loop(text: str, min_repeats: int = 3) -> bool:
    """True if the tail of ``text`` is the same sentence repeated
    ``min_repeats``+ times back to back (whitespace-normalized,
    case-insensitive). Cheap on purpose -- called on every streamed
    chunk, so it only ever looks at the last handful of sentences."""
    if not text:
        return False
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text[-2000:]) if s.strip()]
    if len(sentences) < min_repeats:
        return False
    tail = sentences[-min_repeats:]
    normalized = {re.sub(r"\s+", " ", s).lower() for s in tail}
    return len(normalized) == 1 and len(tail[0]) > 8


def _strip_leaked_thinking(text: str) -> str:
    """Remove any <think>...</think> block regardless of whether the
    API-level think flag suppressed it, then detect an unwrapped
    narration loop (the same "let me check X... let me check Y..."
    pattern repeated several times with no final commitment) and cut
    the response down to its LAST coherent block, on the theory that a
    model looping through several self-correction attempts usually
    lands closest to a real answer in its final attempt before hitting
    its token limit mid-loop."""
    if not text:
        return text

    cleaned = _THINK_TAG_RE.sub("", text).strip()

    if _has_exact_sentence_repeat_loop(cleaned):
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]
        # Keep everything up to and including the FIRST occurrence of the
        # repeated sentence, then drop the rest -- that first occurrence
        # is the model's actual answer; every copy after it is the loop.
        seen_norm = set()
        kept = []
        for s in sentences:
            norm = re.sub(r"\s+", " ", s).lower()
            kept.append(s)
            if norm in seen_norm:
                break
            seen_norm.add(norm)
        return " ".join(kept)

    lower = cleaned.lower()
    hits = sum(lower.count(phrase) for phrase in _NARRATION_PHRASES)
    if hits < 3:
        return cleaned

    # Split on blank-line paragraph boundaries and drop any paragraph
    # that's just narration ("Let me check if there's any other
    # document...") rather than actual content -- keep everything else,
    # including the real answer paragraphs that were mixed in between
    # narration attempts.
    paragraphs = re.split(r"\n\s*\n", cleaned)
    kept = [
        p for p in paragraphs
        if not any(phrase in p.lower() for phrase in _NARRATION_PHRASES)
    ]
    result = "\n\n".join(p.strip() for p in kept if p.strip())
    return result if result else cleaned


def _default_options() -> dict:
    return {
        "temperature":    LLM_TEMPERATURE,
        "num_predict":    LLM_MAX_TOKENS,
        "num_ctx":        LLM_CTX_WINDOW,
        "top_k":          LLM_TOP_K,
        "top_p":          LLM_TOP_P,
        "repeat_penalty": LLM_REPEAT_PENALTY,
        "num_thread":     LLM_NUM_THREAD,
    }


def send_prompt(
    prompt: str,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int    = LLM_MAX_TOKENS,
    timeout: int       = OLLAMA_TIMEOUT,
    model: str         = OLLAMA_MODEL,
) -> str:
    """
    Send a prompt to Ollama and return the complete response string.

    ``timeout``: override the default read timeout. Callers generating a
    large response from a large context (tables/comparisons) should pass
    a longer value — see core.config.OLLAMA_TIMEOUT_TABLE — since prefill
    and generation time both scale with size, and a premature client-side
    timeout can abort a request that was genuinely still working.

    ``model``: override which Ollama model handles this call. Used by
    core.chart_generator to route strict-JSON extraction to a model
    better suited to that (see core.config.CHART_MODEL) without changing
    the model used for normal conversational answers.

    Raises RuntimeError on HTTP or connection failure.
    """
    options = _default_options()
    options["temperature"] = temperature
    options["num_predict"] = max_tokens

    payload = {
        "model":      model,
        "prompt":     prompt,
        "stream":     False,
        "think":      OLLAMA_THINK,
        "keep_alive": "30m",
        "options":    options,
    }

    try:
        resp = requests.post(
            f"{OLLAMA_ENDPOINT}/api/generate",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return _strip_leaked_thinking(resp.json().get("response", "").strip())

    except requests.exceptions.ReadTimeout:
        raise RuntimeError(
            f"Ollama did not respond within {timeout}s. For large "
            "table/comparison requests this usually means the model "
            "needs more time on this hardware — try increasing "
            "OLLAMA_TIMEOUT_TABLE, or ask a narrower question."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. "
            "Make sure Ollama is running: `ollama serve`"
        )
    except requests.exceptions.HTTPError as exc:
        body = exc.response.text if exc.response else ""
        raise RuntimeError(f"Ollama HTTP {exc.response.status_code}: {body}")
    except Exception as exc:
        raise RuntimeError(f"Ollama error: {exc}")


def stream_prompt(
    prompt: str,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int    = LLM_MAX_TOKENS,
    timeout: int       = OLLAMA_TIMEOUT,
) -> Generator[str, None, None]:
    """
    Stream token chunks from Ollama.

    ``timeout``: override the default read timeout — see send_prompt()
    above. This matters even more for streaming: Ollama does not emit
    the first chunk until it finishes prefilling the input context, so a
    large table/comparison context can go several minutes with zero
    bytes on the wire before the first token streams, which a short
    timeout will mistake for a hang.

    Yields individual string chunks as they arrive.
    Raises RuntimeError on connection failure.
    """
    options = _default_options()
    options["temperature"] = temperature
    options["num_predict"] = max_tokens

    payload = {
        "model":      OLLAMA_MODEL,
        "prompt":     prompt,
        "stream":     True,
        "think":      OLLAMA_THINK,
        "keep_alive": "30m",
        "options":    options,
    }

    try:
        resp = requests.post(
            f"{OLLAMA_ENDPOINT}/api/generate",
            json=payload,
            stream=True,
            timeout=timeout,
        )
        resp.raise_for_status()

        accumulated = ""
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line.decode("utf-8"))
                chunk = data.get("response", "")
                if chunk:
                    accumulated += chunk
                    yield chunk
                if data.get("done"):
                    break
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            # A model stuck narrating its own search process instead of
            # answering ("Let me check... Let me look for... Let me try
            # to be more precise...") will otherwise keep generating,
            # unproductively, all the way to num_predict/LLM_MAX_TOKENS
            # -- several more minutes of real wall-clock time on a
            # CPU-bound model for output that gets thrown away anyway.
            # Once the pattern is unmistakable (the same handful of
            # narration phrases repeated 3+ times with no sign of a
            # final answer), close the connection outright instead of
            # waiting for it to hit the ceiling on its own.
            lower_tail = accumulated[-800:].lower()
            hits = sum(lower_tail.count(phrase) for phrase in _NARRATION_PHRASES)
            if hits >= 3:
                resp.close()
                yield (
                    "\n\n[Stopped early: the model was repeating its own "
                    "search process instead of answering. Try asking a "
                    "narrower question, or re-run this one -- generation "
                    "was cut off here rather than left to run for several "
                    "more minutes producing more of the same.]"
                )
                return

            # Same idea as the narration check above, but for the OTHER
            # loop shape: the exact same short sentence repeated verbatim
            # (e.g. "I don't have GPI's PAT for 2023." over and over).
            # stream_prompt() previously had no guard for this at all --
            # send_prompt()'s _strip_leaked_thinking() only ever ran on
            # the BLOCKING path, so a streaming caller (main.py's terminal
            # app, and the web app) saw the raw, unfiltered loop the whole
            # way to num_predict. Checked on the accumulated text so it
            # fires as soon as the pattern is unmistakable, not only after
            # generation finishes.
            if _has_exact_sentence_repeat_loop(accumulated):
                resp.close()
                yield (
                    "\n\n[Stopped early: the model got stuck repeating the "
                    "same sentence, usually a sign the retrieved context "
                    "had nothing usable for part of this question. Try "
                    "asking about one year/company at a time, or re-run "
                    "this question.]"
                )
                return

    except requests.exceptions.ReadTimeout:
        raise RuntimeError(
            f"Ollama did not respond within {timeout}s. For large "
            "table/comparison requests this usually means the model "
            "needs more time on this hardware — try increasing "
            "OLLAMA_TIMEOUT_TABLE, or ask a narrower question."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. "
            "Make sure Ollama is running: `ollama serve`"
        )
    except requests.exceptions.HTTPError as exc:
        body = exc.response.text if exc.response else ""
        raise RuntimeError(f"Ollama HTTP {exc.response.status_code}: {body}")
