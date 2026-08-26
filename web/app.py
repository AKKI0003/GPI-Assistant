"""
web/app.py

GPI Knowledge Assistant — Web UI (Streamlit).
Dark, minimal aesthetic. Messages rendered as HTML for full layout control.

Run with:
    streamlit run web/app.py
"""

import os
import re
import sys
import html

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from core.config import ASSISTANT_NAME, COMPANY_FULL_NAME, COMPANY_SHORT_NAME
from retrieval.knowledge_index import index as knowledge_index
from web.session_manager import SessionRegistry

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROUTE_BADGE = {
    "FAST":       ("●", "#4ade80"),
    "DIRECT":     ("●", "#a78bfa"),
    "LLM":        ("●", "#fbbf24"),
    "CHART":      ("●", "#38bdf8"),
    "NO_RESULTS": ("●", "#f87171"),
    "ERROR":      ("●", "#f87171"),
}

QUICK_QUESTIONS = [
    "Who leads GPI?",
    "Department structure",
    "Key financial KPIs",
    "Recent projects",
]

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=ASSISTANT_NAME,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Tokens ───────────────────────────────────────────────────────────── */
:root {
    --bg-page:        #0d0f14;
    --bg-sidebar:     #11141b;
    --bg-surface:     #171a22;
    --bg-surface-2:   #1b1f29;
    --bg-hover:       #1e222c;
    --bg-active:      #20242f;
    --bg-user-bubble: #2a5adf;
    --border:         rgba(255,255,255,0.06);
    --border-soft:    rgba(255,255,255,0.045);
    --text-primary:   #eef1f6;
    --text-secondary: #b8bfcc;
    --text-muted:     #6b7280;
    --accent:         #4f7cff;
    --accent-soft:    rgba(79,124,255,0.12);
}

/* ── Reset & base ─────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="st-"], .stMarkdown, p, span, div, button, textarea, input {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Full app background (covers every wrapper Streamlit renders) ──────── */
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"],
[data-testid="stMainBlockContainer"],
section.main,
section.main > div {
    background: var(--bg-page) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Main column ─────────────────────────────────────────────────────── */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 7rem !important;
    max-width: 720px !important;
    margin: 0 auto !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
    width: 244px !important;
    min-width: 244px !important;
}
section[data-testid="stSidebar"] > div { width: 244px !important; }
section[data-testid="stSidebar"] .block-container {
    padding: 1.1rem 0.85rem !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] hr {
    margin: 0.85rem 0 !important;
    border-color: var(--border) !important;
}

/* Sidebar brand */
.sb-brand {
    display: flex; align-items: center; gap: 9px;
    padding: 0.1rem 0.2rem 1rem;
}
.sb-brand-mark {
    width: 26px; height: 26px; border-radius: 7px;
    background: linear-gradient(135deg, var(--accent), #7aa2ff);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.62rem; font-weight: 700; color: #fff;
    flex-shrink: 0;
}
.sb-brand-name { font-size: 0.82rem; font-weight: 600; color: var(--text-primary); }

/* Sidebar buttons — flat, minimal, no boxes */
section[data-testid="stSidebar"] button {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-size: 0.79rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    padding: 0.4rem 0.55rem !important;
    height: auto !important;
    min-height: unset !important;
    box-shadow: none !important;
    width: 100% !important;
    transition: background 0.12s ease, color 0.12s ease !important;
}
section[data-testid="stSidebar"] button:hover {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
}
/* Primary "New chat" button */
section[data-testid="stSidebar"] button[kind="primary"] {
    background: var(--accent) !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    border-radius: 9px !important;
    justify-content: center !important;
    text-align: center !important;
    margin-bottom: 0.6rem !important;
}
section[data-testid="stSidebar"] button[kind="primary"]:hover {
    background: #3f6bef !important;
}
section[data-testid="stSidebar"] div[data-testid="column"] { padding: 0 1px !important; }
section[data-testid="stSidebar"] .stTextInput input {
    border-radius: 7px !important;
    font-size: 0.76rem !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    padding: 0.32rem 0.5rem !important;
    height: 1.9rem !important;
}

/* Sidebar labels & meta */
.sidebar-label {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 0 0.25rem;
    margin-bottom: 0.4rem;
    display: block;
}
.sidebar-meta {
    font-size: 0.7rem;
    color: var(--text-muted);
    line-height: 1.85;
    padding: 0.1rem 0.25rem;
}
.sidebar-meta b { color: #9aa4b2; font-weight: 500; }
.sidebar-meta .dot { color: var(--accent); margin: 0 3px; }

/* ── Page header ─────────────────────────────────────────────────────── */
.gpi-header {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 1.5rem 0 1.15rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.6rem;
}
.gpi-logo {
    width: 34px; height: 34px;
    border-radius: 9px;
    background: linear-gradient(135deg, var(--accent), #7aa2ff);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.66rem; font-weight: 700; color: #ffffff;
    letter-spacing: 0.3px; flex-shrink: 0;
}
.gpi-name { font-size: 0.94rem; font-weight: 600; color: var(--text-primary); margin: 0 0 1px; }
.gpi-sub  { font-size: 0.71rem; color: var(--text-muted); margin: 0; }

/* ── Message rows ────────────────────────────────────────────────────── */
.msg-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 1.3rem;
}
.msg-row.user { flex-direction: row-reverse; }

/* Avatars — soft circles, no hard border */
.avatar {
    width: 28px; height: 28px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.6rem; font-weight: 700; line-height: 1;
    margin-top: 3px;
}
.avatar.user      { background: var(--bg-surface-2); color: var(--text-secondary); }
.avatar.assistant { background: linear-gradient(135deg, var(--accent), #7aa2ff); color: #ffffff; }

/* Bubbles — soft, rounded, minimal border weight */
.bubble {
    max-width: 82%;
    font-size: 0.87rem;
    line-height: 1.7;
}
.bubble.user {
    background: var(--bg-user-bubble);
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 0.6rem 1rem;
}
.bubble.assistant {
    color: var(--text-secondary);
    padding-top: 0.15rem;
}

/* Markdown inside assistant bubble */
.bubble.assistant p  { margin: 0 0 0.55rem; color: var(--text-secondary); }
.bubble.assistant p:last-child { margin-bottom: 0; }
.bubble.assistant ul,
.bubble.assistant ol { padding-left: 1.2rem; margin: 0 0 0.5rem; }
.bubble.assistant li { margin-bottom: 0.2rem; color: var(--text-secondary); }
.bubble.assistant h1,
.bubble.assistant h2,
.bubble.assistant h3 {
    font-size: 0.9rem; font-weight: 600;
    color: var(--text-primary); margin: 0.7rem 0 0.25rem;
}
.bubble.assistant strong { color: var(--text-primary); }
.bubble.assistant code {
    background: var(--bg-surface); border-radius: 5px;
    padding: 0.12em 0.36em; font-size: 0.8rem;
    color: #8fb8ff;
    font-family: 'SFMono-Regular', Consolas, monospace;
}
.bubble.assistant pre {
    background: var(--bg-surface); border-radius: 10px;
    padding: 0.8rem 1rem; overflow-x: auto; margin: 0.5rem 0;
}
.bubble.assistant pre code { background: none; padding: 0; color: var(--text-secondary); }

/* ── Chat input — flush with page background, soft floating pill ──────── */
div[data-testid="stChatInput"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35) !important;
    padding: 0.15rem !important;
}
div[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft), 0 4px 20px rgba(0,0,0,0.35) !important;
}
div[data-testid="stChatInput"] textarea {
    font-size: 0.87rem !important;
    color: var(--text-primary) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.65rem 0.85rem !important;
    caret-color: var(--accent) !important;
}
div[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }
div[data-testid="stChatInput"] button {
    background: var(--accent) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    margin: 4px !important;
    border: none !important;
}
div[data-testid="stChatInput"] button:hover { background: #3f6bef !important; }

/* ── Streaming placeholder ───────────────────────────────────────────── */
.stream-placeholder {
    font-size: 0.87rem;
    line-height: 1.7;
    color: var(--text-secondary);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Diagnostics — flat panel, no hard box ──────────────────────────── */
div[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 12px !important;
    margin-top: 0.4rem;
}
div[data-testid="stExpander"] summary {
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
    padding: 0.35rem 0.65rem !important;
}
div[data-testid="stExpander"] summary:hover { color: var(--text-secondary) !important; }
div[data-testid="stMetricValue"] { font-size: 0.86rem !important; font-weight: 500 !important; color: var(--text-primary) !important; }
div[data-testid="stMetricLabel"] { font-size: 0.62rem !important; color: var(--text-muted) !important; }

/* ── Dataframes ──────────────────────────────────────────────────────── */
.stDataFrame { border: 1px solid var(--border-soft); border-radius: 10px; overflow: hidden; }

/* ── Empty state ─────────────────────────────────────────────────────── */
.empty-state { text-align: center; padding: 3.6rem 1rem 1.4rem; }
.empty-state h2 { font-size: 1.06rem; font-weight: 600; color: var(--text-primary); margin: 0 0 0.4rem; }
.empty-state p  { font-size: 0.8rem; color: var(--text-muted); max-width: 320px; margin: 0 auto 1.5rem; line-height: 1.6; }

/* Quick-question pills — real buttons, styled flat & round */
.st-key-quick_pills button {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    color: var(--text-secondary) !important;
    font-size: 0.76rem !important;
    font-weight: 400 !important;
    padding: 0.45rem 1rem !important;
    height: auto !important;
    min-height: unset !important;
    box-shadow: none !important;
    transition: border-color 0.12s ease, color 0.12s ease, background 0.12s ease !important;
}
.st-key-quick_pills button:hover {
    border-color: var(--accent) !important;
    color: var(--text-primary) !important;
    background: var(--accent-soft) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Session state ─────────────────────────────────────────────────────────────

if "registry" not in st.session_state:
    st.session_state.registry = SessionRegistry()
if "rename_target" not in st.session_state:
    st.session_state.rename_target = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

registry: SessionRegistry = st.session_state.registry


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _markdown_table_to_df(text: str):
    """
    Parse a Markdown pipe table into a DataFrame for clean st.dataframe
    rendering. Tolerant of the ways a small local model's table can be
    slightly imperfect -- a missing/misplaced separator row, or a data
    row with a different cell count than the header -- rather than
    bailing out to raw, unrendered pipe-text (which is what previously
    made financial-statement answers show up as an ugly wall of "|"
    characters) or silently dropping rows that don't line up exactly
    (which is what made those tables look "incomplete").
    """
    lines = text.strip().splitlines()
    pipe_lines = [i for i, l in enumerate(lines) if l.strip().startswith("|")]
    if len(pipe_lines) < 2:
        return None
    start = pipe_lines[0]
    block = []
    for i in range(start, len(lines)):
        if lines[i].strip().startswith("|"):
            block.append(lines[i])
        else:
            break
    if len(block) < 2:
        return None

    def _split(row):
        row = row.strip().strip("|")
        return [c.strip() for c in row.split("|")]

    header = _split(block[0])
    data_lines = block[1:]
    # Skip a genuine separator row ("|---|---|") if the model included
    # one; if it didn't, treat block[1] straight away as the first data
    # row instead of discarding the whole table.
    if data_lines and re.match(r"^\|?[\s:|-]+\|?$", data_lines[0].strip()):
        data_lines = data_lines[1:]

    def _fit(cells):
        """Pad or truncate a row to the header's column count rather
        than dropping it outright over a one-cell mismatch."""
        if len(cells) < len(header):
            return cells + [""] * (len(header) - len(cells))
        return cells[: len(header)]

    rows = [_fit(_split(r)) for r in data_lines if _split(r)]
    if not rows:
        return None
    try:
        return pd.DataFrame(rows, columns=header)
    except Exception:
        return None


def render_user_msg(content: str):
    safe = html.escape(content)
    st.markdown(f"""
    <div class="msg-row user">
        <div class="avatar user">You</div>
        <div class="bubble user">{safe}</div>
    </div>""", unsafe_allow_html=True)


def render_assistant_msg(content: str, chart_path: str = None):
    df = _markdown_table_to_df(content)
    if df is not None:
        st.markdown("""<div class="msg-row assistant">
            <div class="avatar assistant">GPI</div>
            <div class="bubble assistant" style="flex:1">""", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        prose = re.sub(r"(\|.+\|[\r\n]*)+", "", content).strip()
        if prose:
            st.markdown(f'<div class="bubble assistant">{_md_to_html(prose)}</div>',
                        unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-row assistant">
            <div class="avatar assistant">GPI</div>
            <div class="bubble assistant">{_md_to_html(content)}</div>
        </div>""", unsafe_allow_html=True)

    if chart_path and os.path.exists(chart_path):
        st.image(chart_path, use_container_width=True)


def _md_to_html(text: str) -> str:
    """Minimal markdown → HTML for dark-themed assistant bubbles."""
    lines = text.split("\n")
    out = []
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"__(.+?)__",     r"<strong>\1</strong>", s)
        s = re.sub(r"\*(.+?)\*",     r"<em>\1</em>", s)
        s = re.sub(r"`(.+?)`",       r"<code>\1</code>", s)
        return s

    for line in lines:
        stripped = line.rstrip()

        if stripped.startswith("### "):
            close_lists(); out.append(f"<h3>{inline(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            close_lists(); out.append(f"<h2>{inline(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            close_lists(); out.append(f"<h1>{inline(stripped[2:])}</h1>")
        elif re.match(r"^[-*+] ", stripped):
            if not in_ul:
                close_lists(); out.append("<ul>"); in_ul = True
            out.append(f"<li>{inline(stripped[2:])}</li>")
        elif re.match(r"^\d+\. ", stripped):
            if not in_ol:
                close_lists(); out.append("<ol>"); in_ol = True
            out.append(f"<li>{inline(re.sub(r'^\\d+\\.\\s', '', stripped))}</li>")
        elif stripped == "":
            close_lists(); out.append("<br>")
        else:
            if in_ul or in_ol:
                out.append(f"{inline(stripped)} ")
            else:
                out.append(f"<p>{inline(stripped)}</p>")

    close_lists()
    result = re.sub(r"(<br>\s*){2,}", "<br>", "\n".join(out))
    return result


def render_diagnostics(result) -> None:
    badge, color = ROUTE_BADGE.get(result.route, ("●", "#888"))
    with st.expander(f"{result.route}  ·  {result.total_time:.2f}s", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieval", f"{result.retrieval_time:.3f}s")
        c2.metric("LLM",       f"{result.llm_time:.2f}s" if result.llm_time else "—")
        c3.metric("Prompt",    f"{result.prompt_size:,} chars" if result.prompt_size else "—")
        if result.error:
            st.error(result.error)
        if result.results:
            st.caption("Sources")
            rows = [
                {"Category": r.get("category", "—").title(),
                 "Section":  r.get("section",  "—"),
                 "Name":     r.get("record", {}).get("name", "—")}
                for r in result.results[:8]
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div class="sb-brand">
        <div class="sb-brand-mark">GPI</div>
        <div class="sb-brand-name">{html.escape(ASSISTANT_NAME)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sidebar-label">Chats</span>', unsafe_allow_html=True)

    if st.button("+ New chat", use_container_width=True, type="primary"):
        registry.create_chat()
        st.session_state.rename_target = None
        st.rerun()

    current = registry.current_chat()

    for name in registry.list_chats():
        is_active = name == current

        if st.session_state.rename_target == name:
            new_name = st.text_input("Rename", value=name,
                                     key=f"rename_input_{name}",
                                     label_visibility="collapsed")
            bc = st.columns(2)
            if bc[0].button("Save", key=f"save_{name}", use_container_width=True):
                if new_name and new_name != name:
                    registry.rename_chat(name, new_name)
                st.session_state.rename_target = None
                st.rerun()
            if bc[1].button("Cancel", key=f"cancel_{name}", use_container_width=True):
                st.session_state.rename_target = None
                st.rerun()
            continue

        cols = st.columns([6, 1, 1], gap="small")
        label = ("▸ " if is_active else "  ") + name
        if cols[0].button(label, key=f"sel_{name}", use_container_width=True):
            registry.switch_chat(name)
            st.rerun()
        if cols[1].button("✎", key=f"ren_{name}", help="Rename"):
            st.session_state.rename_target = name
            st.rerun()
        if cols[2].button("✕", key=f"del_{name}", help="Delete"):
            registry.delete_chat(name)
            st.rerun()

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        registry.clear_chat(current)
        st.session_state.pending_question = None
        st.rerun()

    st.divider()
    st.markdown('<span class="sidebar-label">System</span>', unsafe_allow_html=True)
    st.markdown(
        f"""<div class="sidebar-meta">
            <b>{len(knowledge_index.items)}</b> sections<span class="dot">·</span>Semantic + lexical<br>
            Llama 3.1:8b<span class="dot">·</span>RAG
        </div>""",
        unsafe_allow_html=True,
    )


# ─── Header ────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="gpi-header">
    <div class="gpi-logo">GPI</div>
    <div>
        <p class="gpi-name">{html.escape(ASSISTANT_NAME)}</p>
        <p class="gpi-sub">{html.escape(COMPANY_FULL_NAME)}</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Conversation history ──────────────────────────────────────────────────────

current_chat = registry.current_chat()
messages     = registry.messages(current_chat)

if not messages:
    st.markdown("""
    <div class="empty-state">
        <h2>What would you like to know?</h2>
        <p>Ask anything about leadership, departments, financials, products, or company history.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(key="quick_pills"):
        pill_cols = st.columns(len(QUICK_QUESTIONS))
        for col, q in zip(pill_cols, QUICK_QUESTIONS):
            if col.button(q, key=f"pill_{q}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()
else:
    for msg in messages:
        if msg["role"] == "user":
            render_user_msg(msg["content"])
        else:
            render_assistant_msg(msg["content"], chart_path=msg.get("chart_path"))


# ─── Chat input & streaming ────────────────────────────────────────────────────

typed_question = st.chat_input(f"Ask about {COMPANY_SHORT_NAME}…")

question = None
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
elif typed_question:
    question = typed_question

if question:
    render_user_msg(question)

    st.markdown("""
    <div class="msg-row assistant">
        <div class="avatar assistant">GPI</div>
        <div class="bubble assistant" id="stream-bubble">""", unsafe_allow_html=True)

    placeholder = st.empty()
    acc = {"text": ""}

    def _on_chunk(chunk: str) -> None:
        acc["text"] += chunk
        placeholder.markdown(
            f'<div class="stream-placeholder">{html.escape(acc["text"])}▌</div>',
            unsafe_allow_html=True,
        )

    try:
        result = registry.ask(current_chat, question, stream_callback=_on_chunk)
    except Exception as exc:
        placeholder.empty()
        st.markdown("</div></div>", unsafe_allow_html=True)
        st.error(f"Something went wrong: {exc}")
    else:
        placeholder.empty()
        st.markdown("</div></div>", unsafe_allow_html=True)
        render_assistant_msg(result.answer, chart_path=result.chart_path)
        render_diagnostics(result)

    st.rerun()
