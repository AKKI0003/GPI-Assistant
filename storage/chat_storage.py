"""
storage/chat_storage.py

Persistent chat session storage backed by a single JSON file.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from core.config import CHAT_DIR, CHAT_FILE, CURRENT_CHAT_FILE

os.makedirs(CHAT_DIR, exist_ok=True)


# ─── File I/O ─────────────────────────────────────────────────────────────────

def load_chats() -> Dict[str, List]:
    if not os.path.exists(CHAT_FILE):
        return {}
    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_chats(sessions: Dict[str, List]) -> None:
    with open(CHAT_FILE, "w", encoding="utf-8") as fh:
        json.dump(sessions, fh, indent=2, ensure_ascii=False)


def load_current_chat() -> Optional[str]:
    if not os.path.exists(CURRENT_CHAT_FILE):
        return None
    try:
        with open(CURRENT_CHAT_FILE, "r", encoding="utf-8") as fh:
            return fh.read().strip() or None
    except Exception:
        return None


def save_current_chat(name: str) -> None:
    with open(CURRENT_CHAT_FILE, "w", encoding="utf-8") as fh:
        fh.write(name)


# ─── Chat Manager ─────────────────────────────────────────────────────────────

class ChatManager:
    """
    High-level chat session manager.
    Used by terminal UI, Streamlit, and future interfaces.
    """

    def __init__(self) -> None:
        self.chats = load_chats()
        if not self.chats:
            self.chats = {"New Chat": []}
            save_chats(self.chats)

        saved = load_current_chat()
        self.current_chat: str = (
            saved if saved and saved in self.chats
            else list(self.chats.keys())[0]
        )

    # ── Navigation ────────────────────────────────────────────────────────────

    def switch_chat(self, name: str) -> bool:
        if name in self.chats:
            self.current_chat = name
            save_current_chat(name)
            return True
        return False

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create_chat(self, name: Optional[str] = None) -> str:
        if not name:
            ts = datetime.now().strftime("%b %d %H:%M")
            name = f"Chat {ts}"
        # Ensure uniqueness
        base, i = name, 2
        while name in self.chats:
            name = f"{base} ({i})"
            i += 1
        self.chats[name] = []
        save_chats(self.chats)
        self.switch_chat(name)
        return name

    def delete_chat(self, name: str) -> bool:
        if len(self.chats) <= 1 or name not in self.chats:
            return False
        del self.chats[name]
        save_chats(self.chats)
        self.current_chat = list(self.chats.keys())[0]
        save_current_chat(self.current_chat)
        return True

    def rename_chat(self, old: str, new: str) -> bool:
        if old not in self.chats or not new or new in self.chats:
            return False
        self.chats[new] = self.chats.pop(old)
        if self.current_chat == old:
            self.current_chat = new
        save_chats(self.chats)
        save_current_chat(self.current_chat)
        return True

    # ── Messages ──────────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str, chart_path: Optional[str] = None) -> None:
        # Every message now carries its own timestamp (ISO-8601, local
        # time), not just the chat session as a whole. Previously only
        # the chat's *creation* time was ever recorded (baked into the
        # auto-generated "Chat <date> <time>" session name) -- there was
        # no way to tell when the last query in a session was actually
        # asked. `timestamp` is stored per-message so `log`/`chats` can
        # show exactly when each question (and each answer) happened.
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        if chart_path:
            msg["chart_path"] = chart_path
        self.chats.setdefault(self.current_chat, []).append(msg)
        save_chats(self.chats)

    def get_last_message_time(self, chat_name: Optional[str] = None) -> Optional[str]:
        """ISO timestamp of the most recent message in a chat (defaults
        to the current chat), or None if the chat has no messages yet."""
        name = chat_name or self.current_chat
        messages = self.chats.get(name, [])
        for msg in reversed(messages):
            ts = msg.get("timestamp")
            if ts:
                return ts
        return None

    def get_messages(self) -> List[Dict]:
        return self.chats.get(self.current_chat, [])

    def clear_messages(self) -> None:
        self.chats[self.current_chat] = []
        save_chats(self.chats)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_chat_names(self) -> List[str]:
        return list(self.chats.keys())

    def get_current(self) -> str:
        return self.current_chat

    def chat_exists(self, name: str) -> bool:
        return name in self.chats
