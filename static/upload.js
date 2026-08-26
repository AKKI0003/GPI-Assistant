// static/upload.js
// Same auth pattern as app.js -- separate file since this page doesn't
// need chat state, but shares the exact same token/redirect logic.

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
if (!getToken()) { window.location.href = "login.html"; }

async function api(path, options = {}) {
  const token = getToken();
  const res = await fetch(path, {
    headers: { ...(token ? { "Authorization": `Bearer ${token}` } : {}) },
    ...options,
  });
  if (res.status === 401) {
    clearAuth();
    window.location.href = "login.html";
    throw new Error("session expired");
  }
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail || res.statusText);
  }
  return res.status === 204 ? null : res.json();
}

const userLineEl = document.getElementById("userLine");
const u = getStoredUser();
if (userLineEl) userLineEl.textContent = u ? `${u.email} · ${u.department}` : "";
document.getElementById("logoutBtn").onclick = async () => {
  try { await fetch("/auth/logout", { method: "POST", headers: { "Authorization": `Bearer ${getToken()}` } }); } catch {}
  clearAuth();
  window.location.href = "login.html";
};

// ── Access-mode toggle (everyone / departments / people) ────────────────
let accessMode = "everyone";
const modeButtons = document.querySelectorAll(".access-mode button");
const deptPicker = document.getElementById("deptPicker");
const userPicker = document.getElementById("userPicker");

modeButtons.forEach(btn => {
  btn.onclick = () => {
    accessMode = btn.dataset.mode;
    modeButtons.forEach(b => b.classList.toggle("active", b === btn));
    deptPicker.style.display = accessMode === "departments" ? "block" : "none";
    userPicker.style.display = accessMode === "users" ? "block" : "none";
  };
});

// ── Department list (for the datalist + validation) ──────────────────────
let knownDepartments = [];
fetch("/auth/departments").then(r => r.json()).then(depts => {
  knownDepartments = depts;
  document.getElementById("deptList").innerHTML =
    depts.map(d => `<option value="${d}">`).join("");
});

// ── Chip inputs (departments / users) ─────────────────────────────────────
function setupChipInput(containerId, inputId, validate) {
  const container = document.getElementById(containerId);
  const input = document.getElementById(inputId);
  const chips = [];

  function render() {
    container.querySelectorAll(".chip").forEach(el => el.remove());
    for (const value of chips) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.innerHTML = `${value} <button type="button">×</button>`;
      chip.querySelector("button").onclick = () => {
        chips.splice(chips.indexOf(value), 1);
        render();
      };
      container.insertBefore(chip, input);
    }
  }

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      const value = input.value.trim().replace(/,$/, "");
      if (!value) return;
      if (validate && !validate(value)) {
        input.style.borderColor = "#c96a4e";
        setTimeout(() => { input.style.borderColor = ""; }, 900);
        return;
      }
      if (!chips.includes(value)) chips.push(value);
      input.value = "";
      render();
    } else if (e.key === "Backspace" && !input.value && chips.length) {
      chips.pop();
      render();
    }
  });

  return { get: () => chips };
}

const deptChips = setupChipInput("deptChips", "deptInput", (v) => knownDepartments.includes(v));
const userChips = setupChipInput("userChips", "userInput", (v) => /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(v));

// ── File picker (click or drag-drop) ──────────────────────────────────────
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const dropzoneLabel = document.getElementById("dropzoneLabel");
let selectedFile = null;

dropzone.onclick = () => fileInput.click();
fileInput.onchange = () => {
  if (fileInput.files.length) setSelectedFile(fileInput.files[0]);
};
["dragenter", "dragover"].forEach(evt =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach(evt =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) setSelectedFile(e.dataTransfer.files[0]);
});

function setSelectedFile(file) {
  selectedFile = file;
  dropzoneLabel.innerHTML = `<strong>${file.name}</strong> selected (${(file.size / 1024).toFixed(0)} KB)`;
}

// ── Upload ──────────────────────────────────────────────────────────────
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");

uploadBtn.onclick = async () => {
  if (!selectedFile) {
    uploadStatus.textContent = "Choose a file first.";
    uploadStatus.className = "upload-status error";
    return;
  }
  uploadBtn.disabled = true;
  uploadStatus.textContent = "Uploading…";
  uploadStatus.className = "upload-status";

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("allowed_departments", accessMode === "departments" ? deptChips.get().join(",") : "");
  formData.append("allowed_users", accessMode === "users" ? userChips.get().join(",") : "");

  try {
    await api("/files/upload", { method: "POST", body: formData });
    uploadStatus.textContent = "Uploaded. PDFs are indexed in the background and become searchable in a moment; spreadsheets are available immediately.";
    uploadStatus.className = "upload-status ok";
    selectedFile = null;
    fileInput.value = "";
    dropzoneLabel.innerHTML = "Drag a file here, or <strong>click to browse</strong>";
    loadFileList();
  } catch (err) {
    uploadStatus.textContent = err.message;
    uploadStatus.className = "upload-status error";
  } finally {
    uploadBtn.disabled = false;
  }
};

// ── Existing files list ────────────────────────────────────────────────────
async function loadFileList() {
  const listEl = document.getElementById("fileList");
  try {
    const files = await api("/files");
    if (!files.length) {
      listEl.textContent = "No files uploaded yet.";
      return;
    }
    listEl.innerHTML = "";
    const rows = [];
    for (const f of files) {
      const access = (!f.allowed_departments.length && !f.allowed_users.length)
        ? "Everyone"
        : [...f.allowed_departments, ...f.allowed_users].join(", ");
      const row = document.createElement("div");
      row.className = "file-row";
      row.innerHTML = `
        <span class="file-row__name">${f.source_file}</span>
        <span class="file-row__access">${access} · uploaded by ${f.uploaded_by}</span>
      `;
      listEl.appendChild(row);
      rows.push(row);
    }
    if (typeof animateStagger === "function") animateStagger(rows, 35);
  } catch (err) {
    listEl.textContent = "Could not load files: " + err.message;
  }
}

loadFileList();
