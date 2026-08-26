"""
core/intent_resolver.py

Intent Resolver V3

Responsibilities
----------------
1. Detect the semantic category of a question (company / department / …)
2. Detect the desired action (overview / detail / table / compare)
3. Detect follow-up signals (references to previous topic)
4. Detect correction signals
5. Load category keywords from the external intent_dictionary.json so that
   extending intent coverage requires no code changes.

The resolver is stateless; it receives the memory object externally.
"""

import json
import re
from typing import Any, Dict, Optional

from core.config import INTENT_DICT_FILE
from memory.conversation_memory import ConversationMemory


# ─── Follow-up signals ────────────────────────────────────────────────────────

_FOLLOWUP_EXACT = {
    "more", "tell me more", "elaborate", "expand", "continue",
    "go on", "keep going", "details", "detail",
}

_FOLLOWUP_STARTS = (
    "tell me more", "elaborate on", "expand on",
    "more about", "more on", "continue from",
)

_ANAPHORA_PRONOUNS = {"it", "its", "they", "them", "their", "those", "that", "these", "the above", "the below", "above", "below", "this"}

_CORRECTION_WORDS = {"i meant", "instead", "rather", "no i meant", "not that"}

# Phrases that explicitly request reformatting of the PREVIOUS answer.
# These always count as follow-ups regardless of length, since they
# contain a back-reference pronoun ("it") combined with a format verb.
_REFORMAT_PATTERNS = (
    "make it a table", "make it into a table", "put it in a table",
    "put it into a table", "show it as a table", "show it in a table",
    "convert it to a table", "convert it into a table",
    "can you make it a table", "can you make it into a table",
    "turn it into a table", "format it as a table",
    "table format", "as a table", "in table form",
)

# ─── Action keywords ─────────────────────────────────────────────────────────

_ACTION_TABLE = {"table", "tabular", "grid", "matrix", "top", "bottom"}
_ACTION_COMPARE = {"compare", "comparison", "versus", "vs"}
_ACTION_DETAIL = {"detail", "details", "deep", "deeply", "elaborate", "analysis", "thorough", "explain"}


# ─── Intent Resolver ──────────────────────────────────────────────────────────

class IntentResolver:

    def __init__(self) -> None:
        self._category_map: Dict[str, list] = self._load_category_map()

    # ── setup ─────────────────────────────────────────────────────────────────

    def _load_category_map(self) -> Dict[str, list]:
        """
        Load category → keyword list from intent_dictionary.json.
        Falls back to a hardcoded minimal map if the file is unavailable.
        """
        try:
            with open(INTENT_DICT_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data.get("categories", {})
        except Exception:
            return {
                "company":    ["company", "organization", "business", "gpi", "profile"],
                "department": ["department", "team", "division"],
                "project":    ["project", "initiative"],
                "process":    ["process", "workflow"],
                "kpi":        ["kpi", "metric", "target"],
                "problem":    ["problem", "issue", "risk", "challenge"],
            }

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"['']s\b", "", text)
        return text

    @staticmethod
    def _is_table_request(q: str) -> bool:
        return any(w in q for w in _ACTION_TABLE)

    # Categories that exist in intent_dictionary.json as topical keyword
    # groups but do NOT correspond to a real top-level data category in
    # KnowledgeIndex (which only ever indexes: company, department, project,
    # process, kpi, problem). These topics live INSIDE the company record as
    # sections. Without this mapping, detect_category() would return e.g.
    # "product" or "competitor", which never matches any indexed item's
    # .category, silently breaking the category-preference logic in
    # engine._retrieve() and causing it to fall back to unfiltered/noisy
    # results.
    _NON_DATA_CATEGORIES = {"product", "competitor"}

    # ── category detection ────────────────────────────────────────────────────

    def detect_category(self, question: str) -> Optional[str]:

        q = self._normalize(question)

        scores = {}

        for category, keywords in self._category_map.items():

            score = 0

            for kw in keywords:

                pattern = r"\b" + re.escape(kw) + r"\b"

                if re.search(pattern, q):

                    score += len(kw.split()) * 5

            if category in q:
                score += 20

            scores[category] = score

        # "operations" is deliberately listed under BOTH "department" (the
        # Operations department as an org unit) and "process" (an
        # "operation"/business-process sense) in intent_dictionary.json,
        # since the same word genuinely means either thing depending on
        # phrasing. A bare score tie was previously resolved only by
        # incidental dict key order, which is fragile. Use explicit
        # phrasing cues instead: "operations department/team" or a
        # department-style question ("head/manager of operations") means
        # the org unit; "operations process/workflow" or "business
        # operations" means the process sense.
        if "operations" in q or "operation" in q:
            if re.search(r"\boperations?\s+(department|team|division|unit)\b", q) or \
               re.search(r"\b(head|manager|owner|lead)\s+of\s+operations?\b", q):
                scores["department"] = scores.get("department", 0) + 15
            elif re.search(r"\boperations?\s+(process|workflow|procedure)\b", q) or \
                 re.search(r"\bbusiness\s+operations?\b", q):
                scores["process"] = scores.get("process", 0) + 15

        if not scores:
            return None

        best = max(scores, key=scores.get)

        if scores[best] <= 0:
            return None

        if best in self._NON_DATA_CATEGORIES:
            return "company"

        return best

    # ── action detection ──────────────────────────────────────────────────────

    @staticmethod
    def detect_action(question: str) -> str:

        q = question.lower()

        if any(w in q for w in _ACTION_TABLE):
            return "table"

        if any(w in q for w in _ACTION_COMPARE):
            return "compare"

        if any(
            w in q
            for w in (
                "timeline",
                "chronological",
                "chronology",
                "over the years",
                "year wise",
            )
        ):
            return "timeline"

        if any(
            w in q
            for w in (
                "list",
                "show all",
                "all",
                "every",
            )
        ):
            return "list"

        if any(w in q for w in _ACTION_DETAIL):
            return "detail"

        return "overview"

    # ── follow-up detection ───────────────────────────────────────────────────

    @staticmethod
    def is_followup(question: str, memory: ConversationMemory) -> bool:
        """Return True if the question refers to the previous topic."""
        if not memory.has_last_results():
            return False

        q = question.lower().strip().rstrip("?")
        words = set(q.split())

        if q in _FOLLOWUP_EXACT:
            return True
        if any(q.startswith(p) for p in _FOLLOWUP_STARTS):
            return True
        if any(pattern in q for pattern in _REFORMAT_PATTERNS):
            return True
        # BUG (reproduced against a live transcript): "tell me the state
        # for the above" is 6 words -- a real, short, topic-free
        # follow-up purely by virtue of its "the above" backreference --
        # but the old <= 4 cutoff meant it fell straight through to the
        # longer-sentence branch below, which ADDITIONALLY requires a
        # reformat verb ("table"/"chart"/etc.) to be present. This
        # question has no such verb, so is_followup() returned False
        # entirely, get_last_analysis_spec() was never consulted, and
        # the question fell through to the generic LLM/RAG path instead
        # of the analysis follow-up logic -- producing a vague "could
        # you clarify" non-answer after wasting 95 seconds, when the
        # previous turn's spec had everything needed to answer it
        # immediately. 8 covers this and similar short analytical
        # follow-ups ("what's the town for that one", "which segment
        # for the above") while still requiring a reformat verb for
        # anything longer, so a long unrelated sentence that happens to
        # contain "that"/"it" incidentally still isn't over-matched.
        if len(words) <= 8 and words & _ANAPHORA_PRONOUNS:
            return True
        # Longer sentences that still contain an explicit back-reference
        # pronoun AND a reformatting/elaboration verb (table, list, summarize,
        # expand, elaborate) are also follow-ups, even if longer than 4 words.
        if words & _ANAPHORA_PRONOUNS:
            # "chart"/"graph"/"plot"/"visualize" belong here too — "can you
            # draw a chart based on it" has an explicit back-reference
            # ("it") plus a clear reformat/visualize verb, exactly like
            # "put it in a table" does. Without these, that question fell
            # through to is_followup()==False, which meant the chart
            # request re-resolved category/section_hint from scratch on a
            # nearly content-free sentence ("can you draw a chart based on
            # it") — with no topic words of its own to match, retrieval
            # picked up unrelated context instead of the previous turn's
            # topic, so "chart based on it" could come back with a
            # completely different table (e.g. material topics) instead of
            # a chart of what "it" actually referred to.
            reformat_verbs = {
                "table", "list", "summarize", "summarise",
                "expand", "elaborate", "format", "convert", "turn",
                "chart", "graph", "plot", "visualize", "visualise",
                "draw", "diagram",
                # Analysis-specific follow-up verbs (source, breakdown, verify, etc.)
                "source", "breakdown", "verify", "show", "rows", "detail",
                "explain", "insight", "analyze", "analyse",
            }
            if words & reformat_verbs:
                return True
        return False

    # ── correction detection ──────────────────────────────────────────────────

    @staticmethod
    def is_correction(question: str) -> bool:
        q = question.lower()
        return any(w in q for w in _CORRECTION_WORDS)

    # ── section hint detection ────────────────────────────────────────────────

    @staticmethod
    def detect_section_hint(question: str) -> Optional[str]:
        """
        Return a sub-section hint for company record splitting.
        Maps to actual section names in the KnowledgeIndex.
        """
        q = question.lower()

        # "Owner" is asked about far more often, and far more ambiguously,
        # than any other single keyword here — "who is the owner", "owner
        # of the company", "who owns GPI" all mean the same specific thing
        # (ownership/control), never a job title. Resolve it directly with
        # a real word-boundary check rather than letting it compete against
        # generic substring matches (like a leftover "who is the" phrase)
        # in the scoring loop below, which is what caused owner questions
        # to get mis-routed to the leadership/CEO section before.
        if re.search(r"\bowner(s)?\b", q) or re.search(r"\bwho owns\b", q):
            return "ownership"

        mapping = {
            # NOTE: this dict previously defined "leadership"/"history"/
            # "financials"/"products"/"operations"/"overview" TWICE — a
            # plain Python dict literal silently keeps only the LAST
            # definition of a duplicate key, so the earlier keyword lists
            # were dropped entirely. In particular, the first "products"
            # definition (with "cigarette", "tobacco", "confectionery",
            # "nicotine", "smoke") got shadowed by a second, narrower
            # "products" definition that only had "products", "portfolio",
            # "offerings", "brands", "range", "categories" — so a question
            # that just said "cigarettes" (no "brand"/"portfolio" word)
            # never resolved a "products" section_hint at all, starving
            # retrieval of its section-hint boost and making the strict,
            # context-only LLM prompt correctly (but confusingly) say it
            # couldn't find the answer, even though the data was there.
            # Merged into one clean definition per section below (union of
            # both keyword lists), and "company" removed from the overview
            # keyword list — company vs. department is decided by
            # detect_category(), not by this method, and "company" showing
            # up here nudged genuinely department-scoped questions (e.g.
            # "the company's marketing overview") toward the company
            # overview section.
            # NOTE: "who is the" used to be a leadership keyword, but it's
            # a generic phrase that also appears in "who is the OWNER of
            # the company" — since it's 3 words it scored higher (9) than
            # "owner" alone (3), so ownership questions were silently
            # routed to the leadership section (CEO/chairperson) instead.
            # Removed here; "who runs"/"leads" are specific enough to keep,
            # and the explicit "owner" whole-word check below (which runs
            # BEFORE this generic scoring loop) now takes priority whenever
            # the question is actually asking about ownership.
            "leadership": ["ceo", "cfo", "coo", "leader", "leadership", "management", "executive",
                           "director", "board", "chairperson", "chairwoman", "who runs", "who leads"],
            "history": ["history", "historical", "founded", "founding", "when was", "origin",
                        "established", "started", "incorporation", "legacy", "milestones",
                        "timeline", "journey", "background", "evolution", "story"],
            "financials": ["revenue", "profit", "financial", "financials", "turnover", "earnings",
                           "how much", "finance", "income", "pnl", "p&l", "p & l",
                           "profit and loss", "profit & loss", "balance sheet", "cash flow",
                           "ebitda", "net profit", "gross profit", "income statement"],
            # NOTE: this used to also list "itc" and "vst" as keywords — meaning the
            # mere presence of a competitor's NAME in the question (e.g. "VST's PnL
            # statement", "revenue and net profit of VST") pushed the section_hint to
            # "competitor_finances" purely because the company was named, regardless
            # of what was actually being asked about. That's a category error: WHICH
            # company a question is about is already handled separately by
            # company_filter/_detect_company — this mapping is only supposed to
            # decide the TOPIC (financials vs. leadership vs. products, etc.), and a
            # competitor's own revenue/profit is exactly as much a "financials"
            # question as GPI's own. Conflating the two caused a genuine "what is
            # VST's revenue" question to be scored as a competitor_finances/company-
            # gossip question instead of a financials question, which then favored
            # whatever vague competitor commentary existed in the curated JSON over
            # VST's own real, PDF-extracted figures. This section_hint should now
            # only fire for questions that are actually ABOUT competitive dynamics
            # (market share, rivalry) rather than any question that merely names a
            # competitor.
            "competitor_finances": ["competitor", "competitors", "rival", "rivals", "competition",
                                    "market player", "market share", "rivalry"],
            "cigarette_segmentation": ["segment", "segmentation", "product lines", "cigarette segment"],
            "pmi_strategic_segments": ["pmi segment", "pmi product", "smokefree", "iqos", "heated tobacco"],
            "products": ["product", "products", "brand", "brands", "cigarette", "tobacco",
                         "confectionery", "nicotine", "smoke", "portfolio", "offerings",
                         "range", "categories"],
            "location": ["headquarters", "hq", "where is", "office", "location", "address"],
            "mission_vision": ["mission", "vision", "purpose", "values"],
            "ownership": ["owner", "ownership", "shareholding", "promoter", "stake"],
            "operations": ["manufacturing", "factory", "plant", "production", "operations",
                           "facilities", "supply chain"],
            "indian_market_overview": ["indian market", "india market", "market size", "market overview"],
            "pmi_global_strategy": ["pmi strategy", "philip morris strategy", "global strategy"],
            "overview": ["company profile", "summary", "introduction",
                         "tell me about the company", "about the company", "about gpi",
                         "company overview", "overview"],
        }

        best = None
        best_score = 0

        for section, keywords in mapping.items():

            score = 0

            for kw in keywords:

                if kw in q:

                    score += len(kw.split()) * 3

            if score > best_score:

                best = section

                best_score = score

        return best

    # ── main resolve ──────────────────────────────────────────────────────────

    def resolve(
        self,
        question: str,
        memory: ConversationMemory,
    ) -> Dict[str, Any]:
        """
        Resolve a question into a structured intent dict.

        Returns
        -------
        {
            original_question : str,
            category          : str | None,
            section_hint      : str | None,
            action            : "overview" | "detail" | "table" | "compare",
            is_followup       : bool,
            is_correction     : bool,
            table_requested   : bool,
        }
        """
        followup    = self.is_followup(question, memory)
        words = question.lower().split()

        if len(words) == 1:

            word = words[0].rstrip("s")

            direct_map = {
                "project": "project",
                "process": "process",
                "problem": "problem",
                "kpi": "kpi",
                "department": "department",
            }

            if word in direct_map:

                return {
                    "original_question": question,
                    "category": direct_map[word],
                    "section_hint": None,
                    "action": "overview",
                    "is_followup": False,
                    "is_correction": False,
                    "table_requested": False,
                }
        correction  = self.is_correction(question)
        action      = self.detect_action(question)
        table_req   = self._is_table_request(question.lower())

        if followup:
            # Follow-up phrases ("tell me more about it", "convert it into a
            # table") are usually content-free — running detect_category /
            # detect_section_hint on them is noisy and can return a wrong,
            # unrelated topic (e.g. "department"/"overview" purely because
            # the word "about" happens to match a keyword). For genuine
            # follow-ups we default to the previous turn's resolved intent,
            # UNLESS the follow-up itself names an explicit new topic
            # (e.g. "tell me more about the financials"), in which case the
            # explicit topic wins.
            prev = memory.get_last_intent() or {}
            fresh_category     = self.detect_category(question)
            fresh_section_hint = self.detect_section_hint(question)

            category     = fresh_category if fresh_category else prev.get("category")
            section_hint = fresh_section_hint if fresh_section_hint else prev.get("section_hint")
        else:
            category     = self.detect_category(question)
            section_hint = self.detect_section_hint(question)

        # For corrections, force category resolution
        if correction and not category:
            category = memory.get_last_intent().get("category", "company")

        return {
            "original_question": question,
            "category":          category,
            "section_hint":      section_hint,
            "action":            action,
            "is_followup":       followup,
            "is_correction":     correction,
            "table_requested":   table_req,
        }


# Module-level singleton
resolver = IntentResolver()