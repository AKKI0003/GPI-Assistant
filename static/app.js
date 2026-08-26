// static/app.js
// Vanilla JS on purpose: no build step, no CDN dependency (npm/CDN
// resources could easily be blocked on a locked-down company network),
// runs from a single static file served by api/server.py itself.

// ── Auth guard ────────────────────────────────────────────────────────────
// Every page under /app/ requires a logged-in session -- redirect to the
// login page immediately if there's no token, before any chat state
// loads. The token lives in localStorage (survives a refresh/reopen);
// api() below attaches it to every request, and a 401 response (expired/
// revoked session) clears it and redirects back to login.
const AUTH_TOKEN_KEY = "gpi_auth_token";
const AUTH_USER_KEY  = "gpi_auth_user";

function getToken() { return localStorage.getItem(AUTH_TOKEN_KEY); }
function getStoredUser() {
  try { return JSON.parse(localStorage.getItem(AUTH_USER_KEY) || "null"); }
  catch { return null; }
}
function clearAuth() {
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_USER_KEY);
}
function requireAuth() {
  if (!getToken()) {
    window.location.href = "login.html";
    throw new Error("not authenticated");
  }
}
requireAuth();

const chatListEl   = document.getElementById("chatList");
const messagesEl   = document.getElementById("messages");
const emptyStateEl = document.getElementById("emptyState");
const chatTitleEl  = document.getElementById("chatTitle");
const statusLineEl = document.getElementById("statusLine");
const form         = document.getElementById("composerForm");
const input        = document.getElementById("questionInput");
const sendBtn      = document.getElementById("sendBtn");
const userLineEl   = document.getElementById("userLine");
const logoutBtn    = document.getElementById("logoutBtn");

let currentChat = null;
let chats = [];

// ── API helpers ─────────────────────────────────────────────────────────
// Relative paths: the frontend is served by the SAME FastAPI process
// it's talking to, so this is always same-origin -- no CORS, no base
// URL to configure, and it keeps working unchanged no matter what
// domain/tunnel you're reaching it through.

async function api(path, options = {}) {
  const token = getToken();
  const res = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(token ? { "Authorization": `Bearer ${token}` } : {}),
    },
    ...options,
  });
  if (res.status === 401) {
    clearAuth();
    window.location.href = "login.html";
    throw new Error("session expired");
  }
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.status === 204 ? null : res.json();
}

if (logoutBtn) {
  logoutBtn.onclick = async () => {
    try { await api("/auth/logout", { method: "POST" }); } catch {}
    clearAuth();
    window.location.href = "login.html";
  };
}
if (userLineEl) {
  const u = getStoredUser();
  userLineEl.textContent = u ? `${u.email} · ${u.department}` : "";
  // Fill the small circular avatar next to the name with the first
  // letter of the email's local-part, so the "signed in as" row reads
  // as an actual account card instead of a bare line of text.
  const avatarEl = document.getElementById("userAvatar");
  if (avatarEl && u && u.email) {
    avatarEl.textContent = u.email.trim().charAt(0).toUpperCase();
  }
}

// ── Hero greeting (Claude-style "Good afternoon, X") ────────────────────
// There's no dedicated display-name field in the auth system -- only
// email + department (see getStoredUser() above) -- so the name here is
// a best-effort read of the email's local-part: split on the first
// non-letter character (a dot, underscore, digit run, etc.) and title-
// case whatever comes before it. Falls back to a plain "Hello" with no
// name if that local-part is empty/unparseable for any reason, rather
// than showing something obviously wrong.
function greetingTimeOfDay() {
  const h = new Date().getHours();
  if (h < 5) return "Good night";
  if (h < 12) return "Good morning";
  if (h < 17) return "Good afternoon";
  if (h < 21) return "Good evening";
  return "Good night";
}
function bestEffortFirstName(email) {
  if (!email) return "";
  const local = email.split("@")[0] || "";
  const match = local.match(/^[a-zA-Z]+/);
  if (!match || match[0].length < 2) return "";
  const name = match[0];
  return name.charAt(0).toUpperCase() + name.slice(1).toLowerCase();
}
const heroGreetingEl = document.getElementById("heroGreeting");
if (heroGreetingEl) {
  const u = getStoredUser();
  const name = u ? bestEffortFirstName(u.email) : "";
  heroGreetingEl.textContent = name ? `${greetingTimeOfDay()}, ${name}` : greetingTimeOfDay();
}

// ── Minimal markdown rendering (tables, bold, inline code) ─────────────
// No markdown library -- answers here are mostly plain prose plus the
// occasional pipe-delimited financial table, so a small hand-rolled
// renderer covers what's actually needed without pulling in a CDN
// dependency that might not be reachable at all on a locked-down network.

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderInline(text) {
  let out = escapeHtml(text);
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  return out;
}

function isTableSeparatorRow(line) {
  return /^\|?[\s:|-]+\|?$/.test(line.trim()) && line.includes("-");
}

function renderMarkdown(text) {
  const lines = text.split("\n");
  let html = "";
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Pipe-delimited table: a row of |cells| followed by a |---|---| rule.
    if (line.trim().startsWith("|") && i + 1 < lines.length && isTableSeparatorRow(lines[i + 1])) {
      const headerCells = line.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim());
      let tableHtml = "<table><thead><tr>" +
        headerCells.map(c => `<th>${renderInline(c)}</th>`).join("") +
        "</tr></thead><tbody>";
      i += 2;
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        const cells = lines[i].trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim());
        tableHtml += "<tr>" + cells.map(c => `<td>${renderInline(c)}</td>`).join("") + "</tr>";
        i++;
      }
      tableHtml += "</tbody></table>";
      html += tableHtml;
      continue;
    }

    html += renderInline(line) + "\n";
    i++;
  }
  return html;
}

// ── Rendering ────────────────────────────────────────────────────────────

function renderChatList() {
  chatListEl.innerHTML = "";
  const items = [];
  for (const c of chats) {
    const item = document.createElement("div");
    item.className = "chat-item" + (c.name === currentChat ? " active" : "");
    item.innerHTML = `<span class="chat-item__name">${escapeHtml(c.name)}</span>` +
      `<button class="chat-item__del" title="Delete">×</button>`;
    item.querySelector(".chat-item__name").onclick = () => selectChat(c.name);
    item.querySelector(".chat-item__del").onclick = (e) => { e.stopPropagation(); deleteChat(c.name); };
    chatListEl.appendChild(item);
    items.push(item);
  }
  if (typeof animateStagger === "function") animateStagger(items, 30);
}

function hideHero() {
  if (!emptyStateEl || emptyStateEl.classList.contains("hero--leaving")) return;
  emptyStateEl.classList.add("hero--leaving");
  setTimeout(() => { emptyStateEl.style.display = "none"; }, 400);
}

function showHero() {
  if (!emptyStateEl) return;
  emptyStateEl.classList.remove("hero--leaving");
  emptyStateEl.style.display = "flex";
}

function addMessageEl(role, bodyHtml, chartUrl, metaText) {
  hideHero();
  const wrap = document.createElement("div");
  wrap.className = "msg msg--" + role;
  wrap.innerHTML = `
    <div class="msg__role">${role === "user" ? "You" : "Assistant"}</div>
    <div class="msg__body">${bodyHtml}</div>
    ${chartUrl ? `<img class="msg__chart" src="${chartUrl}">` : ""}
    ${metaText ? `<div class="msg__meta">${metaText}</div>` : ""}
  `;
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  if (typeof animateIn === "function") animateIn(wrap);
  return wrap;
}

async function loadMessages(name) {
  messagesEl.innerHTML = "";
  const msgs = await api(`/chats/${encodeURIComponent(name)}/messages`);
  if (msgs.length === 0) {
    messagesEl.appendChild(emptyStateEl);
    showHero();
  }
  for (const m of msgs) {
    addMessageEl(m.role, renderMarkdown(m.content), m.chart_path || null);
  }
}

// ── Chat management ──────────────────────────────────────────────────────

async function refreshChats() {
  chats = await api("/chats");
  const current = chats.find(c => c.is_current);
  currentChat = current ? current.name : (chats[0] && chats[0].name);
  chatTitleEl.textContent = currentChat || "—";
  renderChatList();
}

async function selectChat(name) {
  await api(`/chats/${encodeURIComponent(name)}/switch`, { method: "POST" });
  currentChat = name;
  chatTitleEl.textContent = name;
  renderChatList();
  await loadMessages(name);
}

async function newChat() {
  const chat = await api("/chats", { method: "POST", body: JSON.stringify({}) });
  await refreshChats();
  await selectChat(chat.name);
}

async function deleteChat(name) {
  if (!confirm(`Delete chat "${name}"?`)) return;
  await api(`/chats/${encodeURIComponent(name)}`, { method: "DELETE" });
  await refreshChats();
  if (currentChat) await loadMessages(currentChat);
}

document.getElementById("newChatBtn").onclick = newChat;

document.getElementById("renameBtn").onclick = async () => {
  if (!currentChat) return;
  const newName = prompt("Rename chat to:", currentChat);
  if (!newName || newName === currentChat) return;
  await api(`/chats/${encodeURIComponent(currentChat)}/rename`, {
    method: "POST", body: JSON.stringify({ new_name: newName }),
  });
  await refreshChats();
  await selectChat(newName);
};

document.getElementById("clearBtn").onclick = async () => {
  if (!currentChat || !confirm("Clear all messages in this chat?")) return;
  await api(`/chats/${encodeURIComponent(currentChat)}/clear`, { method: "POST" });
  await loadMessages(currentChat);
};

// ── Sending a question (SSE streaming) ───────────────────────────────────
// Uses fetch() + a manual ReadableStream reader rather than EventSource,
// since EventSource can't send a POST body. fetch() has NO client-side
// timeout by default in browsers, so a slow local model taking minutes
// to finish generating doesn't get cut off by the frontend itself --
// only OLLAMA_TIMEOUT (server-side) can end it early.

async function sendMessage(question) {
  addMessageEl("user", renderMarkdown(question));
  const assistantEl = addMessageEl("assistant", `<span class="thinking"><span></span><span></span><span></span></span>`);
  const bodyEl = assistantEl.querySelector(".msg__body");

  let accumulated = "";
  let gotFirstChunk = false;

  try {
    const res = await fetch("/ask/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${getToken()}`,
      },
      body: JSON.stringify({ question, chat_name: currentChat }),
    });
    if (res.status === 401) {
      clearAuth();
      window.location.href = "login.html";
      return;
    }
    if (!res.ok || !res.body) throw new Error(`Server error: ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const frames = buffer.split("\n\n");
      buffer = frames.pop(); // last (possibly incomplete) frame stays in buffer

      for (const frame of frames) {
        const eventMatch = frame.match(/^event:\s*(.+)$/m);
        const dataMatch = frame.match(/^data:\s*(.+)$/m);
        if (!dataMatch) continue;
        const eventType = eventMatch ? eventMatch[1].trim() : "message";
        let data;
        try { data = JSON.parse(dataMatch[1]); } catch { continue; }

        if (eventType === "chunk" && typeof data.chunk === "string") {
          if (!gotFirstChunk) { bodyEl.innerHTML = ""; gotFirstChunk = true; }
          accumulated += data.chunk;
          bodyEl.innerHTML = renderMarkdown(accumulated);
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (eventType === "chart" && data.chart_path) {
          const img = document.createElement("img");
          img.className = "msg__chart";
          img.src = data.chart_path;
          assistantEl.appendChild(img);
        } else if (eventType === "done") {
          if (!gotFirstChunk) bodyEl.innerHTML = renderMarkdown(data.answer || accumulated);
          if (data.restricted) {
            assistantEl.classList.add("msg--restricted");
          }
          if (data.chart_path && !assistantEl.querySelector(".msg__chart")) {
            const img = document.createElement("img");
            img.className = "msg__chart";
            img.src = data.chart_path;
            assistantEl.appendChild(img);
          }
          if (data.error) {
            const err = document.createElement("div");
            err.className = "msg__meta";
            err.style.color = "#d9765a";
            err.textContent = "Error: " + data.error;
            assistantEl.appendChild(err);
          }
          const meta = document.createElement("div");
          meta.className = "msg__meta";
          const total = (data.total_time || 0).toFixed(1);
          meta.textContent = `${data.route || ""} · ${total}s`;
          assistantEl.appendChild(meta);
        }
      }
    }
  } catch (err) {
    bodyEl.innerHTML = renderMarkdown("Something went wrong: " + err.message);
  }

  await refreshChats(); // chat may have been auto-named from the first question
}

form.onsubmit = (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  input.style.height = "auto";
  sendBtn.disabled = true;
  sendMessage(question).finally(() => { sendBtn.disabled = false; input.focus(); });
};

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
});

// ── Boot ──────────────────────────────────────────────────────────────────

async function boot() {
  try {
    const health = await api("/health");
    statusLineEl.textContent = `${health.indexed_sections} sections indexed`;
  } catch {
    statusLineEl.textContent = "server unreachable";
  }
  await refreshChats();
  if (currentChat) await loadMessages(currentChat);
}

boot();