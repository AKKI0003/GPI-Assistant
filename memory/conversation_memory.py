"""
memory/conversation_memory.py

Conversation memory tracks:
- Chat history (question / answer pairs)
- Last retrieval results (for follow-up reuse)
- Last resolved intent (for follow-up category inheritance)
- Primary entity in focus (for anaphora resolution)
"""

from collections import deque
from typing import Any, Dict, List, Optional

from core.config import MEMORY_MAX_HISTORY, MEMORY_CONTEXT_TURNS


class ConversationMemory:
    """
    Rolling conversation window plus retrieval-state tracking.
    """

    def __init__(self, max_history: int = MEMORY_MAX_HISTORY) -> None:
        self._history: deque = deque(maxlen=max_history)
        self._last_results:        List[Dict[str, Any]] = []
        self._last_intent:         Dict[str, Any] = {}
        self._last_answer:         str = ""
        self._last_question:       str = ""
        self._primary_entity:      Optional[str] = None
        self._last_category:       Optional[str] = None

    # ── write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        question: str,
        answer: str,
        results: Optional[List[Dict[str, Any]]] = None,
        intent:  Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one Q&A turn."""
        self._history.append({"question": question, "answer": answer})
        self._last_question = question
        self._last_answer   = answer

        if intent:
            self._last_intent = dict(intent)

        if results is not None:
            self._last_results = list(results)
            self._update_entities(results)

    def _update_entities(self, results: List[Dict[str, Any]]) -> None:
        primary: Optional[str] = None
        category: Optional[str] = None
        for r in results:
            category = r.get("category")
            name = r.get("record", {}).get("name")
            if name and primary is None:
                primary = name
        self._primary_entity = primary
        self._last_category  = category

    # ── read ──────────────────────────────────────────────────────────────────

    def has_last_results(self) -> bool:
        return len(self._last_results) > 0 or bool(self._last_answer)

    def get_last_results(self) -> List[Dict[str, Any]]:
        return self._last_results

    def get_last_intent(self) -> Dict[str, Any]:
        return self._last_intent

    def get_last_answer(self) -> str:
        return self._last_answer

    def get_last_question(self) -> str:
        return self._last_question

    def get_primary_entity(self) -> Optional[str]:
        return self._primary_entity

    def get_last_category(self) -> Optional[str]:
        return self._last_category

    def get_context_string(self, turns: int = MEMORY_CONTEXT_TURNS) -> str:
        """
        Return the last *turns* Q&A pairs as a plain-text string
        suitable for injection into the LLM prompt.
        """
        recent = list(self._history)[-turns:]
        lines: List[str] = []
        for item in recent:
            lines.append(f"User: {item['question']}")
            # Trim long answers so the prompt doesn't explode
            answer_preview = str(item["answer"])[:400]
            lines.append(f"Assistant: {answer_preview}")
        return "\n".join(lines)

    # ── admin ─────────────────────────────────────────────────────────────────

    def clear(self) -> None:
        self._history.clear()
        self._last_results   = []
        self._last_intent    = {}
        self._last_answer    = ""
        self._last_question  = ""
        self._primary_entity = None
        self._last_category  = None


# Module-level singleton
memory = ConversationMemory()
