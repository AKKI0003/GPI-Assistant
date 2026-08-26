"""
diagnose.py — run this from the project root (same place you run main.py):

    python diagnose.py

It checks, against your REAL ingested data, exactly why the two problem
questions are misbehaving, and prints what it finds. Paste the full
output back — no guessing needed after this.
"""
import sys

print("=" * 70)
print("1) Is the ANALYSIS-route bug still triggering, and on what?")
print("=" * 70)
try:
    from core.data_analyst import is_analysis_query, _load_registry
    q1 = "tell me the strategies , policies, sales techniques etc. which is not related to csr thhat GPI followed in 2023 and 2024 and compare them"
    result = is_analysis_query(q1)
    print(f"is_analysis_query() on query 1 -> {result}")
    if result:
        print("Still True. Checking which sheet/column/value matched...")
        sheets = _load_registry()
        q_lower = q1.lower()
        import re
        q_words_all = set(re.findall(r"[a-z0-9]+", q_lower))
        GENERIC = {
            "sales", "sale", "revenue", "total", "amount", "value", "values",
            "data", "name", "names", "date", "dates", "year", "years",
            "company", "companies", "report", "reports", "state", "states",
            "branch", "branches", "region", "regions", "product", "products",
            "category", "categories", "type", "types", "status", "code",
            "codes", "number", "numbers", "record", "records", "file", "files",
        }
        year_re = re.compile(r"^(19|20)\d{2}$")
        def is_generic(w):
            return w in GENERIC or bool(year_re.match(w))
        def overlaps(text_lower, min_len=4):
            if is_generic(text_lower):
                return False
            if len(text_lower) >= min_len and text_lower in q_lower:
                return True
            words = set(w for w in re.findall(r"[a-z0-9]+", text_lower) if len(w) >= min_len and not is_generic(w))
            return bool(words & q_words_all)
        for sheet in sheets:
            print(f"  sheet: {sheet.sheet_name}  file: {sheet.file_path}")
            print(f"    sheet_name match: {overlaps(sheet.sheet_name.lower())}")
            import os
            base = os.path.splitext(os.path.basename(sheet.file_path))[0].lower().replace("_", " ")
            print(f"    base filename match: {overlaps(base)}")
            for col in sheet.columns:
                if overlaps(col.lower()):
                    print(f"    COLUMN MATCH: '{col}'")
            for colname, values in sheet.categorical_values.items():
                for v in values:
                    if overlaps(str(v).lower()):
                        print(f"    VALUE MATCH: column '{colname}' value '{v}'")
except Exception as e:
    print(f"ERROR running check 1: {e!r}")

print()
print("=" * 70)
print("2) Does financial_facts extraction find ANY GPI PAT figures at all?")
print("=" * 70)
try:
    from retrieval.pdf_retriever import pdf_retriever
    pdf_retriever.ensure_index_built()
    total_chunks = len(pdf_retriever._chunks)
    print(f"Total cached PDF chunks: {total_chunks}")
    pat_facts = []
    for cached in pdf_retriever._chunks:
        cd = cached.chunk_dict
        for f in (cd.get("financial_facts") or []):
            if f.get("metric") == "PAT":
                pat_facts.append((cd.get("source"), cd.get("page"), f.get("year"), f.get("value"), f.get("raw_label")))
    print(f"Chunks with a PAT fact extracted: {len(pat_facts)}")
    for row in pat_facts[:30]:
        print(f"  source={row[0]}  page={row[1]}  year={row[2]}  value={row[3]}  label={row[4]!r}")
    if not pat_facts:
        print("NO PAT facts were extracted from ANY ingested PDF chunk.")
        print("This means the problem is at INGESTION time (extract_financial_facts")
        print("in ingestion/financial_facts.py never matches this document's real")
        print("table formatting), not at retrieval time. Run part 3 below to see why.")
except Exception as e:
    print(f"ERROR running check 2: {e!r}")

print()
print("=" * 70)
print("3) Sample raw text of chunks that LOOK like they should have PAT")
print("=" * 70)
try:
    from retrieval.pdf_retriever import pdf_retriever
    shown = 0
    for cached in pdf_retriever._chunks:
        cd = cached.chunk_dict
        text = cd.get("text") or ""
        if "profit" in text.lower() and shown < 3:
            shown += 1
            print(f"--- chunk from {cd.get('source')} page {cd.get('page')} (section={cd.get('section')}) ---")
            print(text[:1200])
            print("--- financial_facts extracted from THIS chunk: ---")
            print(cd.get("financial_facts") or "(none)")
            print()
    if shown == 0:
        print("No chunk containing the word 'profit' was found at all -- PDF ingestion")
        print("may not have run, or the PDF text extraction itself is failing.")
except Exception as e:
    print(f"ERROR running check 3: {e!r}")

print()
print("=" * 70)
print("4) Is spreadsheet content still leaking into the JSON/document index?")
print("=" * 70)
try:
    from retrieval.knowledge_index import index as knowledge_index
    polluted = []
    for item in knowledge_index.items:
        rec = item.record
        sf = str(rec.get("source_file") or "")
        if sf.lower().endswith((".xlsx", ".xls", ".xlsm", ".csv")):
            polluted.append(sf)
    print(f"Entries in knowledge_index still sourced from a spreadsheet file: {len(polluted)}")
    for p in polluted[:10]:
        print(f"  {p}")
    if polluted:
        print("These should be ZERO after the fix. If this is non-zero, either the")
        print("fixed retrieval/knowledge_index.py wasn't actually loaded (check for a")
        print("stale __pycache__/knowledge_index.cpython-*.pyc), or these chunks'")
        print("source_file field doesn't literally end in one of those extensions --")
        print("paste a couple of the printed paths above and that'll show the real")
        print("naming convention to match against instead.")
except Exception as e:
    print(f"ERROR running check 4: {e!r}")

print()
print("Done. Paste this whole output back.")