"""
auth/permissions.py

Per-FILE access control: when a user uploads a document or spreadsheet,
they choose which department(s) and/or which specific person(s) are
allowed to see it -- including through the chatbot, not just a file
browser. This module is the single source of truth both the upload
endpoint (writes permissions) and the retrieval/answer path
(core/engine.py -- enforces them) go through, so there's exactly one
place this logic can go wrong rather than two that could drift apart.

Storage: data/file_permissions.json, one entry per uploaded source file,
keyed by the same source_file name the rest of the app already uses to
tag document-sourced knowledge_index/pdf_retriever chunks. This is
deliberately the SAME identifier already threaded through the whole
retrieval pipeline (record["source_file"]) -- enforcement plugs in as a
filter on that existing field, it doesn't require changing how
documents are chunked, tagged, or indexed at all.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from core.config import DATA_DIR

PERMISSIONS_FILE = os.path.join(DATA_DIR, "file_permissions.json")

_LOCK = threading.Lock()


@dataclass
class FilePermission:
    source_file: str
    uploaded_by: str
    # Empty list on EITHER of these means "no restriction on this axis".
    # A file is visible to a user if their department is in
    # allowed_departments OR their email is in allowed_users OR both
    # lists are empty (nobody restricted it, so it's fully open) OR the
    # uploader themself is asking OR the requester is an admin.
    allowed_departments: List[str] = field(default_factory=list)
    allowed_users: List[str] = field(default_factory=list)


def _load() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(PERMISSIONS_FILE):
        return {}
    try:
        with open(PERMISSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save(data: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(PERMISSIONS_FILE), exist_ok=True)
    tmp = PERMISSIONS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, PERMISSIONS_FILE)


def set_permission(
    source_file: str,
    uploaded_by: str,
    allowed_departments: Optional[List[str]] = None,
    allowed_users: Optional[List[str]] = None,
) -> FilePermission:
    perm = FilePermission(
        source_file=source_file,
        uploaded_by=uploaded_by,
        allowed_departments=[d.strip() for d in (allowed_departments or [])],
        allowed_users=[u.strip().lower() for u in (allowed_users or [])],
    )
    with _LOCK:
        data = _load()
        data[source_file] = asdict(perm)
        _save(data)
    return perm


def get_permission(source_file: str) -> Optional[FilePermission]:
    with _LOCK:
        data = _load()
    raw = data.get(source_file)
    return FilePermission(**raw) if raw else None


def list_permissions() -> List[FilePermission]:
    with _LOCK:
        data = _load()
    return [FilePermission(**v) for v in data.values()]


def is_allowed(source_file: Optional[str], user: Optional[Dict[str, Any]]) -> bool:
    """
    True if `user` may see content from `source_file`.

    - No source_file at all (curated JSON knowledge, not an uploaded
      file) -> always allowed; file-level ACLs only ever apply to
      actual uploaded documents/spreadsheets.
    - No permission record for this file -> always allowed (a file
      nobody ever restricted is open by default, matching how every
      file behaved before this feature existed -- ACL is opt-in per
      file via the uploader's choice, not opt-out).
    - No `user` at all (unauthenticated caller, e.g. a bare API call
      with no session) -> DENIED for any file that DOES have a
      permission record, since there's no department/identity to check
      against. Public/curated content is still unaffected (see above).
    - Otherwise: allowed if the user uploaded it, is an admin, has an
      empty-restriction file (both lists empty), their department is in
      allowed_departments, or their email is in allowed_users.
    """
    if not source_file:
        return True
    perm = get_permission(source_file)
    if perm is None:
        return True
    if user is None:
        return False
    if user.get("is_admin"):
        return True
    if user.get("email", "").strip().lower() == perm.uploaded_by.strip().lower():
        return True
    if not perm.allowed_departments and not perm.allowed_users:
        return True
    if user.get("department") in perm.allowed_departments:
        return True
    if user.get("email", "").strip().lower() in perm.allowed_users:
        return True
    return False


def filter_allowed(results: List[Dict[str, Any]], user: Optional[Dict[str, Any]]) -> tuple:
    """
    Split retrieval results into (allowed, was_anything_restricted).

    `was_anything_restricted` lets the caller distinguish "nothing
    relevant existed at all" from "something relevant existed but this
    user isn't allowed to see it" -- these need different answers (a
    plain "I don't have that information" vs. an explicit "Restricted
    access" -- see core/engine.py). Silently returning an empty list
    either way would make a real access restriction look exactly like
    the assistant simply not knowing something, which defeats the point
    of telling a user their access is limited rather than leaving them
    to assume the data doesn't exist.
    """
    allowed = []
    restricted_hit = False
    for r in results:
        source_file = (r.get("record") or {}).get("source_file")
        if is_allowed(source_file, user):
            allowed.append(r)
        else:
            restricted_hit = True
    return allowed, restricted_hit
