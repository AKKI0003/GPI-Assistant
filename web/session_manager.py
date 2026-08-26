"""
web/session_manager.py

Shared per-chat session orchestration for the web UI and the API server.

This is a thin layer ON TOP of the existing, unmodified core modules
(core.engine, memory.conversation_memory, storage.chat_storage). It does
not change any of their behavior — it exists purely so two different
front ends (a Streamlit UI and a FastAPI server) can share one consistent
way of mapping a *named, persisted* chat (as already managed by
ChatManager) to its own isolated ConversationMemory + AssistantEngine
pair.

Why this is needed
-------------------
ChatManager (storage/chat_storage.py) persists chat *messages* to disk,
but it has no concept of retrieval/follow-up state — that lives in
ConversationMemory, which is purely in-process. The original terminal app
(main.py) only ever has one active chat at a time, so it gets away with a
single global ConversationMemory. A web UI or API, by contrast, may have
several chats "open" (visible in a sidebar / addressable by name)
simultaneously, so each one needs its own ConversationMemory + engine —
otherwise asking a follow-up in Chat B could resolve against Chat A's
last retrieval results.

This registry is intentionally independent of core/: deleting this whole
module would not affect main.py or any core/retrieval/llm/memory file.
"""

from typing import Any, Dict, List, Optional

from core.engine import AssistantEngine, EngineResult
from memory.conversation_memory import ConversationMemory
from storage.chat_storage import ChatManager


class SessionRegistry:
    """
    Owns one ChatManager (persisted chat list + messages) plus a per-chat
    map of ConversationMemory/AssistantEngine pairs (in-memory only, like
    the original app's session memory).
    """

    def __init__(self) -> None:
        self.chat_manager = ChatManager()
        self._memories: Dict[str, ConversationMemory] = {}
        self._engines: Dict[str, AssistantEngine] = {}

    # ── engine / memory access ──────────────────────────────────────────────

    def engine_for(self, chat_name: str) -> AssistantEngine:
        """Return (creating if necessary) the engine for *chat_name*."""
        if chat_name not in self._engines:
            memory = ConversationMemory()
            self._memories[chat_name] = memory
            self._engines[chat_name] = AssistantEngine(memory=memory, stream=False)
        return self._engines[chat_name]

    def memory_for(self, chat_name: str) -> ConversationMemory:
        self.engine_for(chat_name)  # ensures both are created together
        return self._memories[chat_name]

    def reset_chat_memory(self, chat_name: str) -> None:
        """Wipe follow-up/retrieval memory for a chat without deleting it."""
        memory = ConversationMemory()
        self._memories[chat_name] = memory
        self._engines[chat_name] = AssistantEngine(memory=memory, stream=False)

    def _forget(self, chat_name: str) -> None:
        self._memories.pop(chat_name, None)
        self._engines.pop(chat_name, None)

    # ── chat CRUD (delegates to ChatManager, keeps engines/memories in sync) ──

    def list_chats(self) -> List[str]:
        return self.chat_manager.get_chat_names()

    def current_chat(self) -> str:
        return self.chat_manager.get_current()

    def create_chat(self, name: Optional[str] = None) -> str:
        return self.chat_manager.create_chat(name)

    def switch_chat(self, name: str) -> bool:
        return self.chat_manager.switch_chat(name)

    def delete_chat(self, name: str) -> bool:
        ok = self.chat_manager.delete_chat(name)
        if ok:
            self._forget(name)
        return ok

    def rename_chat(self, old: str, new: str) -> bool:
        ok = self.chat_manager.rename_chat(old, new)
        if ok:
            if old in self._engines:
                self._engines[new] = self._engines.pop(old)
            if old in self._memories:
                self._memories[new] = self._memories.pop(old)
        return ok

    def messages(self, chat_name: Optional[str] = None) -> List[Dict]:
        name = chat_name or self.current_chat()
        if name != self.current_chat():
            self.chat_manager.switch_chat(name)
        return self.chat_manager.get_messages()

    def clear_chat(self, chat_name: Optional[str] = None) -> None:
        name = chat_name or self.current_chat()
        if name != self.current_chat():
            self.chat_manager.switch_chat(name)
        self.chat_manager.clear_messages()
        self.reset_chat_memory(name)

    # ── the actual ask pipeline ─────────────────────────────────────────────

    def ask(
        self,
        chat_name: str,
        question: str,
        stream_callback=None,
        user: Optional[Dict[str, Any]] = None,
    ) -> EngineResult:
        """
        Run the full engine pipeline for *question* against *chat_name*'s
        isolated session, then persist both turns to the chat's message
        history via ChatManager. Returns the EngineResult unchanged.

        `user` (see auth/store.py's User.public_dict()) is passed straight
        through to AssistantEngine.ask() for department-based file access
        control -- see auth/permissions.py. None (the default) means
        "no restriction", matching every caller that existed before this
        feature (terminal CLI, tests).
        """
        engine = self.engine_for(chat_name)

        if chat_name != self.current_chat():
            self.chat_manager.switch_chat(chat_name)

        if stream_callback is not None:
            engine.stream = True
            engine.set_stream_callback(stream_callback)
        else:
            engine.stream = False

        result = engine.ask(question, user=user)

        self.chat_manager.add_message("user", question)
        self.chat_manager.add_message("assistant", result.answer, chart_path=result.chart_path)

        return result