"""
Supabase memory layer for NEXUS.
Stores conversation history, agent outputs, and user preferences.
Tables: nexus_memory, nexus_sessions, nexus_preferences
"""

import json
from datetime import datetime
from rich.console import Console

console = Console()


class SupabaseMemory:
    """
    Persistent memory layer using Supabase.
    Used by MemoryAgent and JarvisInterface for session persistence.
    """

    def __init__(self):
        from nexus.config import config
        self.config = config
        self._client = None
        self._unavailable_reason = None

    def _is_configured(self) -> bool:
        """Return True when Supabase credentials are present."""
        return bool(self.config.supabase_url and self.config.supabase_key)

    def _get_client(self):
        """Lazy-load Supabase client."""
        if not self._is_configured():
            self._unavailable_reason = (
                "Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY to enable persistent memory."
            )
            return None
        if not self._client:
            try:
                from supabase import create_client
            except ImportError:
                self._unavailable_reason = (
                    "Supabase support is not installed. Install the optional `supabase` package to enable memory."
                )
                return None
            try:
                self._client = create_client(self.config.supabase_url, self.config.supabase_key)
            except Exception:
                self._unavailable_reason = (
                    "Supabase configuration looks invalid. Recheck SUPABASE_URL and SUPABASE_KEY."
                )
                return None
            self._unavailable_reason = None
        return self._client

    async def save_message(self, session_id: str, role: str, content: str, agent: str = None):
        """Save a message to memory."""
        try:
            client = self._get_client()
            if client is None:
                return
            client.table("nexus_memory").insert({
                "session_id": session_id,
                "role": role,
                "content": content,
                "agent": agent,
                "created_at": datetime.utcnow().isoformat(),
            }).execute()
        except Exception:
            console.print("[yellow]Memory save skipped because the Supabase backend is unavailable.[/yellow]")

    async def get_history(self, session_id: str, limit: int = 20) -> list:
        """Get conversation history for a session."""
        try:
            client = self._get_client()
            if client is None:
                return []
            result = client.table("nexus_memory")\
                .select("role, content, agent, created_at")\
                .eq("session_id", session_id)\
                .order("created_at", desc=False)\
                .limit(limit)\
                .execute()
            return result.data or []
        except Exception:
            console.print("[yellow]Memory history is unavailable because the Supabase backend is not ready.[/yellow]")
            return []

    async def clear_session(self, session_id: str):
        """Clear all messages in a session."""
        try:
            client = self._get_client()
            if client is None:
                return
            client.table("nexus_memory").delete().eq("session_id", session_id).execute()
        except Exception:
            console.print("[yellow]Memory clear skipped because the Supabase backend is unavailable.[/yellow]")

    def status_message(self) -> str:
        """Return a human-readable memory backend status message."""
        self._get_client()
        return self._unavailable_reason or "Supabase memory is configured."
