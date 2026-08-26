"""
auth/store.py

Minimal, dependency-free user auth: registration, login, and bearer-token
sessions, all backed by a plain JSON file (data/users.json) -- consistent
with how the rest of this app stores its data (departments.json,
chat_sessions.json, etc.), no database server to stand up.

Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib `hashlib`, no extra
dependency) with a random per-user salt and a deliberately high iteration
count. Plaintext passwords are never stored or logged.

Sessions are opaque bearer tokens (`secrets.token_urlsafe`), stored
server-side with an expiry, sent back to the client once at login/register
and attached as `Authorization: Bearer <token>` on every subsequent
request (see api/server.py's `get_current_user` dependency).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from core.config import DATA_DIR, SIGNUP_DEPARTMENTS

USERS_FILE = os.path.join(DATA_DIR, "users.json")
SESSION_TTL_SECONDS = 60 * 60 * 24 * 14  # 14 days

_LOCK = threading.Lock()

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PBKDF2_ITERATIONS = 260_000


@dataclass
class User:
    email: str
    password_hash: str
    salt: str
    designation: str
    department: str
    created_at: float = field(default_factory=time.time)
    is_admin: bool = False

    def public_dict(self) -> Dict[str, Any]:
        """Everything EXCEPT password_hash/salt -- safe to send to the client."""
        return {
            "email": self.email,
            "designation": self.designation,
            "department": self.department,
            "is_admin": self.is_admin,
        }


class AuthError(Exception):
    """Raised for any registration/login failure with a user-facing message."""


def _load_users() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_users(users: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    tmp = USERS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
    os.replace(tmp, USERS_FILE)  # atomic on both POSIX and Windows


def _hash_password(password: str, salt: Optional[str] = None) -> tuple:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ITERATIONS
    ).hex()
    return digest, salt


def register(
    email: str, password: str, designation: str, department: str
) -> User:
    email = email.strip().lower()
    designation = designation.strip()

    if not _EMAIL_RE.match(email):
        raise AuthError("That doesn't look like a valid email address.")
    if len(password) < 8:
        raise AuthError("Password must be at least 8 characters.")
    if not designation:
        raise AuthError("Please enter your designation/post.")
    if department not in SIGNUP_DEPARTMENTS:
        raise AuthError(
            f"Unknown department. Choose one of: {', '.join(SIGNUP_DEPARTMENTS)}"
        )

    with _LOCK:
        users = _load_users()
        if email in users:
            raise AuthError("An account with that email already exists.")

        password_hash, salt = _hash_password(password)
        user = User(
            email=email,
            password_hash=password_hash,
            salt=salt,
            designation=designation,
            department=department,
            # First-ever registered user becomes admin (can manage file
            # permissions/department list) -- everyone after is a normal
            # user. Promote/demote by hand-editing data/users.json's
            # "is_admin" field if you need a different arrangement.
            is_admin=(len(users) == 0),
        )
        users[email] = asdict(user)
        _save_users(users)
        return user


def login(email: str, password: str) -> User:
    email = email.strip().lower()
    with _LOCK:
        users = _load_users()
    raw = users.get(email)
    if raw is None:
        raise AuthError("No account with that email, or the password is wrong.")

    check_hash, _ = _hash_password(password, salt=raw["salt"])
    if not secrets.compare_digest(check_hash, raw["password_hash"]):
        raise AuthError("No account with that email, or the password is wrong.")

    return User(**raw)


def get_user(email: str) -> Optional[User]:
    with _LOCK:
        users = _load_users()
    raw = users.get(email.strip().lower())
    return User(**raw) if raw else None


def list_users() -> List[User]:
    with _LOCK:
        users = _load_users()
    return [User(**u) for u in users.values()]


# ── Sessions ─────────────────────────────────────────────────────────────
# In-memory is fine for a single-process personal-laptop-as-server setup
# (see RUNNING_REMOTELY.md) -- sessions reset if the server restarts,
# which just means signing back in, not a real problem for this scale.

_SESSIONS: Dict[str, Dict[str, Any]] = {}
_SESSIONS_LOCK = threading.Lock()


def create_session(email: str) -> str:
    token = secrets.token_urlsafe(32)
    with _SESSIONS_LOCK:
        _SESSIONS[token] = {"email": email, "expires": time.time() + SESSION_TTL_SECONDS}
    return token


def resolve_session(token: str) -> Optional[User]:
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(token)
        if session is None:
            return None
        if session["expires"] < time.time():
            del _SESSIONS[token]
            return None
    return get_user(session["email"])


def revoke_session(token: str) -> None:
    with _SESSIONS_LOCK:
        _SESSIONS.pop(token, None)
