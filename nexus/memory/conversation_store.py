"""Local-first conversation memory backed by SQLite.

Zero-config, zero-dependency persistent chat history.  Works offline,
survives dashboard refreshes, and provides full-text search across sessions.
Falls back gracefully — the system works even if this is never called.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ConversationStore:
    """SQLite-backed conversation memory.

    Usage::

        store = ConversationStore()
        store.save_message("session_1", "user", "build a REST API")
        store.save_message("session_1", "assistant", "Here's your Express server...",
                           metadata={"agent": "coding", "reflect_score": 0.05})
        history = store.get_history("session_1")
        results = store.search("REST API")
    """

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            from nexus.config import config
            db_path = config.data_dir / "conversations.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a message. Returns the message ID."""
        msg_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO messages (id, session_id, role, content, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (msg_id, session_id, role, content, json.dumps(metadata or {}), now),
        )
        self._sync_fts_row(msg_id)
        # Update session last_active
        self._conn.execute(
            """INSERT INTO sessions (id, created_at, last_active, message_count)
               VALUES (?, ?, ?, 1)
               ON CONFLICT(id) DO UPDATE SET
                   last_active = excluded.last_active,
                   message_count = message_count + 1""",
            (session_id, now, now),
        )
        self._conn.commit()
        return msg_id

    def get_history(
        self,
        session_id: str,
        limit: int = 50,
        before: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get messages for a session, most recent last."""
        if before:
            rows = self._conn.execute(
                """SELECT id, role, content, metadata, created_at
                   FROM messages
                   WHERE session_id = ? AND created_at < ?
                   ORDER BY created_at ASC
                   LIMIT ?""",
                (session_id, before, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT id, role, content, metadata, created_at
                   FROM messages
                   WHERE session_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (session_id, limit),
            ).fetchall()
            rows = list(reversed(rows))  # Return in chronological order

        return [self._row_to_dict(row) for row in rows]

    def get_context(self, session_id: str, limit: int = 10) -> list[dict[str, str]]:
        """Return last N messages formatted for LLM context injection.

        Returns a list of ``{role, content}`` dicts ready to be appended
        to an LLM prompt.
        """
        history = self.get_history(session_id, limit=limit)
        return [{"role": m["role"], "content": m["content"]} for m in history]

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search across all sessions."""
        rows = []
        try:
            rows = self._conn.execute(
                """SELECT m.id, m.session_id, m.role, m.content, m.metadata, m.created_at
                   FROM messages_fts f
                   JOIN messages m ON m.id = f.id
                   WHERE messages_fts MATCH ?
                   ORDER BY bm25(messages_fts), m.created_at DESC
                   LIMIT ?""",
                (self._fts_query(query), limit),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = self._conn.execute(
                """SELECT id, session_id, role, content, metadata, created_at
                   FROM messages
                   WHERE content LIKE ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (f"%{query}%", limit),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent sessions with metadata."""
        rows = self._conn.execute(
            """SELECT id, created_at, last_active, message_count
               FROM sessions
               ORDER BY last_active DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_session(self, session_id: str) -> int:
        """Delete a session and all its messages. Returns deleted message count."""
        message_ids = [
            row[0]
            for row in self._conn.execute(
                "SELECT id FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        ]
        cursor = self._conn.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id,)
        )
        if message_ids:
            self._conn.executemany("DELETE FROM messages_fts WHERE id = ?", [(msg_id,) for msg_id in message_ids])
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()
        return cursor.rowcount

    def message_count(self, session_id: str | None = None) -> int:
        """Total message count, optionally filtered by session."""
        if session_id:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        return row[0] if row else 0

    def close(self):
        """Close the database connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, created_at);
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
                USING fts5(id UNINDEXED, session_id UNINDEXED, role, content, created_at UNINDEXED);

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_active TEXT NOT NULL,
                message_count INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_active
                ON sessions(last_active DESC);
        """)
        self._conn.execute(
            """INSERT INTO messages_fts (id, session_id, role, content, created_at)
               SELECT id, session_id, role, content, created_at FROM messages
               WHERE id NOT IN (SELECT id FROM messages_fts)"""
        )
        self._conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        if "metadata" in d:
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = {}
        return d

    def _sync_fts_row(self, msg_id: str) -> None:
        self._conn.execute("DELETE FROM messages_fts WHERE id = ?", (msg_id,))
        self._conn.execute(
            """INSERT INTO messages_fts (id, session_id, role, content, created_at)
               SELECT id, session_id, role, content, created_at
               FROM messages
               WHERE id = ?""",
            (msg_id,),
        )

    def _fts_query(self, query: str) -> str:
        terms = [term.strip() for term in query.split() if term.strip()]
        if not terms:
            return query
        cleaned_terms = [term.replace('"', "") for term in terms]
        return " OR ".join(f'"{term}"' for term in cleaned_terms)
