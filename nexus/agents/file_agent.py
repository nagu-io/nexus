"""
FileAgent — safe local file system operations.
Handles: read, write, list, search, summarize files.
Restricted to user's home directory and current working directory.
"""

from pathlib import Path

from nexus.agents.base_agent import BaseAgent
from rich.console import Console

console = Console()


class FileAgent(BaseAgent):
    """
    File system agent with safety restrictions.
    Can read, write, list, and summarize files.
    """

    name = "file"
    system_prompt = "You are a file management assistant. Help read, write, and organize files."

    def _safe_dirs(self) -> list[Path]:
        """Return the directories the file agent is allowed to access."""
        return [Path.home().resolve(), Path.cwd().resolve()]

    async def run(self, task: str) -> str:
        """Route file task."""
        task_lower = task.lower()
        if "read" in task_lower or "open" in task_lower or "show" in task_lower:
            return await self._read_task(task)
        elif "write" in task_lower or "create" in task_lower or "save" in task_lower:
            return await self._write_task(task)
        elif "list" in task_lower or "directory" in task_lower or "folder" in task_lower:
            return await self._list_task(task)
        else:
            return await self._call_local(task)

    def _is_safe_path(self, path: Path) -> bool:
        """Check if path is within safe directories."""
        resolved_path = path.expanduser().resolve()
        return any(
            resolved_path == safe_dir or resolved_path.is_relative_to(safe_dir)
            for safe_dir in self._safe_dirs()
        )

    async def _read_task(self, task: str) -> str:
        """Read a file and optionally summarize it."""
        # Extract file path from task using local model
        prompt = f"Extract only the file path from this request (return just the path, nothing else): {task}"
        path_str = await self._call_local(prompt)
        path_str = path_str.strip().strip('"').strip("'")

        try:
            path = Path(path_str)
            if not self._is_safe_path(path):
                return f"Access denied: {path} is outside safe directories."
            if not path.exists():
                return f"File not found: {path}"
            content = path.read_text(encoding="utf-8", errors="replace")
            if len(content) > 4000:
                summary_prompt = f"Summarize this file content:\n\n{content[:4000]}"
                return await self._call_local(summary_prompt)
            return content
        except Exception as e:
            return f"Read failed: {e}"

    async def _write_task(self, task: str) -> str:
        """Write content to a file."""
        prompt = f"Extract the file path and content to write from this request. Return JSON with 'path' and 'content' keys only: {task}"
        response = await self._call_local(prompt)
        try:
            import json
            data = json.loads(response)
            path = Path(data["path"])
            if not self._is_safe_path(path):
                return f"Access denied: {path}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data["content"], encoding="utf-8")
            return f"Written to {path} ({len(data['content'])} chars)"
        except Exception as e:
            return f"Write failed: {e}"

    async def _list_task(self, task: str) -> str:
        """List directory contents."""
        path = Path.cwd()
        try:
            files = list(path.iterdir())
            listing = "\n".join([
                f"{'[DIR]' if f.is_dir() else '[FILE]'} {f.name}"
                for f in sorted(files)
            ])
            return f"Contents of {path}:\n{listing}"
        except Exception as e:
            return f"List failed: {e}"
