"""Safe file tool for NEXUS agent runtime actions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FileTool:
    """Restricted file read/write/edit tool with action logging."""

    VALID_ACTIONS = {"read_file", "write_file", "edit_file", "read", "write", "edit"}

    def __init__(
        self,
        *,
        allowed_roots: list[Path] | None = None,
        log_path: Path | None = None,
    ):
        from nexus.config import config

        self.allowed_roots = [
            Path(root).expanduser().resolve()
            for root in (allowed_roots or [Path.cwd()])
        ]
        runtime_log_dir = config.data_dir / "runtime_logs"
        runtime_log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path) if log_path else runtime_log_dir / "file_tool_actions.jsonl"

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute a normalized file-tool request and return a structured result."""
        tool_name = (request.get("tool") or request.get("tool_name") or "file_tool").strip()
        action = self._normalize_action(request.get("action"))
        arguments = dict(request.get("arguments") or request.get("args") or {})

        result: dict[str, Any]
        try:
            if action == "read_file":
                result = self._read_file(arguments)
            elif action == "write_file":
                result = self._write_file(arguments)
            elif action == "edit_file":
                result = self._edit_file(arguments)
            else:
                raise ValueError(f"Unsupported file tool action: {action}")
        except Exception as error:
            result = {
                "ok": False,
                "tool": tool_name,
                "action": action,
                "summary": f"Tool error: {error}",
                "error": str(error),
            }

        result.setdefault("tool", tool_name)
        result.setdefault("action", action)
        self._log_action(result)
        return result

    def _read_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(arguments.get("path"))
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.is_dir():
            raise IsADirectoryError(f"Expected a file but received a directory: {path}")

        content = path.read_text(encoding="utf-8", errors="replace")
        return {
            "ok": True,
            "path": str(path),
            "summary": f"Read {len(content)} chars from {path}",
            "content": content,
            "chars": len(content),
        }

    def _write_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(arguments.get("path"))
        content = arguments.get("content")
        if content is None:
            raise ValueError("write_file requires 'content'")
        content = str(content)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "ok": True,
            "path": str(path),
            "summary": f"Wrote {len(content)} chars to {path}",
            "content": content,
            "chars": len(content),
        }

    def _edit_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(arguments.get("path"))
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.is_dir():
            raise IsADirectoryError(f"Expected a file but received a directory: {path}")

        old_text = arguments.get("old_text")
        new_text = arguments.get("new_text")
        replace_all = bool(arguments.get("replace_all", False))
        if old_text is None or new_text is None:
            raise ValueError("edit_file requires 'old_text' and 'new_text'")

        original = path.read_text(encoding="utf-8", errors="replace")
        if str(old_text) not in original:
            raise ValueError(f"Could not find the requested text to replace in {path}")

        if replace_all:
            updated = original.replace(str(old_text), str(new_text))
            replacements = original.count(str(old_text))
        else:
            updated = original.replace(str(old_text), str(new_text), 1)
            replacements = 1

        path.write_text(updated, encoding="utf-8")
        return {
            "ok": True,
            "path": str(path),
            "summary": f"Edited {path} with {replacements} replacement(s)",
            "content": updated,
            "replacements": replacements,
        }

    def _resolve_path(self, raw_path: Any) -> Path:
        if not raw_path:
            raise ValueError("A file path is required")

        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = self.allowed_roots[0] / path
        resolved = path.resolve()
        if not any(resolved == root or resolved.is_relative_to(root) for root in self.allowed_roots):
            raise PermissionError(f"Access denied: {resolved} is outside allowed roots")
        return resolved

    def _normalize_action(self, action: Any) -> str:
        normalized = str(action or "").strip().lower()
        if normalized not in self.VALID_ACTIONS:
            raise ValueError(f"Unsupported file tool action: {action}")
        return {
            "read": "read_file",
            "write": "write_file",
            "edit": "edit_file",
        }.get(normalized, normalized)

    def _log_action(self, result: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": result.get("tool", "file_tool"),
            "action": result.get("action"),
            "ok": bool(result.get("ok", False)),
            "path": result.get("path"),
            "summary": result.get("summary"),
            "allowed_roots": [str(root) for root in self.allowed_roots],
        }
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
