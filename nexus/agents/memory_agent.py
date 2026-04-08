"""
MemoryAgent — persistent memory using Supabase.
Stores and retrieves conversation history, facts, preferences.
Schema: nexus_memory table (id, session_id, role, content, created_at, tags)
"""

from nexus.agents.base_agent import BaseAgent
from nexus.memory.supabase_memory import SupabaseMemory
from rich.console import Console
from rich.table import Table
import json
from datetime import datetime

console = Console()


class MemoryAgent(BaseAgent):
    """
    Persistent memory agent backed by Supabase.
    Handles: store memory, recall memory, search memory, forget.
    """

    name = "memory"
    capabilities = ("memory_read", "memory_write", "reasoning")
    system_prompt = "You are a memory assistant. Help store and retrieve important information accurately."

    def __init__(self):
        super().__init__()
        self._client = None

    def _get_client(self):
        """Lazy-load Supabase client."""
        if not self.config.supabase_url or not self.config.supabase_key:
            return None
        if not self._client:
            try:
                from supabase import create_client
            except ImportError:
                return None
            self._client = create_client(self.config.supabase_url, self.config.supabase_key)
        return self._client

    def _status_message(self) -> str:
        return SupabaseMemory().status_message()

    async def run(self, task: str) -> str:
        """Route memory task."""
        task_lower = task.lower()
        if any(w in task_lower for w in ["remember", "save", "store", "note"]):
            return await self.store(task)
        elif any(w in task_lower for w in ["recall", "what did", "history", "previous", "last"]):
            return await self.recall(task)
        elif any(w in task_lower for w in ["forget", "delete", "remove"]):
            return await self.forget(task)
        else:
            return await self.recall(task)

    async def store(self, content: str, tags: list = None, session_id: str = "default") -> str:
        """Store a memory in Supabase."""
        try:
            client = self._get_client()
            if client is None:
                return self._status_message()
            client.table("nexus_memory").insert({
                "session_id": session_id,
                "role": "user",
                "content": content,
                "tags": json.dumps(tags or []),
                "created_at": datetime.utcnow().isoformat(),
            }).execute()
            return f"Memory stored: {content[:100]}"
        except Exception:
            return "Memory store failed. Check your Supabase project and table schema, then try again."

    async def recall(self, query: str, limit: int = 10) -> str:
        """Recall memories related to a query."""
        try:
            client = self._get_client()
            if client is None:
                return self._status_message()
            result = client.table("nexus_memory")\
                .select("content, created_at, tags")\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()

            if not result.data:
                return "No memories found."

            memories = "\n".join([
                f"[{m['created_at'][:10]}] {m['content']}"
                for m in result.data
            ])

            prompt = f"Based on these stored memories, answer: {query}\n\nMemories:\n{memories}"
            return await self._call_local(prompt)
        except Exception:
            return "Memory recall failed. Check your Supabase project and table schema, then try again."

    async def forget(self, content: str) -> str:
        """Delete memories matching content."""
        try:
            client = self._get_client()
            if client is None:
                return self._status_message()
            client.table("nexus_memory")\
                .delete()\
                .ilike("content", f"%{content}%")\
                .execute()
            return f"Forgot memories related to: {content}"
        except Exception:
            return "Forget failed. Check your Supabase project and table schema, then try again."
