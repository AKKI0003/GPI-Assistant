"""
core/query_normalizer.py

Previously corrected "branch"/"branches" -> "brand"/"brands" on the
assumption that this knowledge base had no branch/location concept at
all. That assumption is no longer true: ingested Excel data now has
real zone/branch/town columns (e.g. "Delhi branch", "Okhla zone"), so
that correction was silently rewriting every such question into asking
about the wrong thing entirely (a real "outliers for Delhi branch"
question was being sent to retrieval/analysis as "outliers for Delhi
brand") -- this is likely a major contributor to the "only sees a
handful of rows" symptom reported for zone/town/branch-based queries,
since the corrected wording pointed the analysis engine at the wrong
column/value in the first place, before the aggregation-completeness
question even entered the picture.

Kept as a no-op passthrough (rather than deleted outright) so any other
module still importing normalize_question doesn't break; there are
currently no active corrections.
"""

_CORRECTIONS = []


def normalize_question(question: str) -> str:
    """
    Return *question* unchanged. See module docstring: the one
    correction this used to make (branch -> brand) is no longer valid
    now that real branch/zone/town data exists in ingested Excel files.
    """
    return question
