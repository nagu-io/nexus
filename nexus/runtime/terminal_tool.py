"""Safe terminal execution for autonomous coding workflows."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any


class TerminalTool:
    """Run bounded workspace commands with path and command restrictions."""

    VALID_ACTIONS = {"run_command", "run"}
    ALLOWED_COMMANDS = {
        "python",
        "py",
        "node",
        "npm",
        "npm.cmd",
        "pnpm",
        "pnpm.cmd",
        "npx",
        "npx.cmd",
        "pytest",
        "uv",
    }

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
        self.log_path = Path(log_path) if log_path else runtime_log_dir / "terminal_tool_actions.jsonl"

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute a terminal request and return stdout/stderr/exit metadata."""
        tool_name = (request.get("tool") or request.get("tool_name") or "terminal_tool").strip()
        action = self._normalize_action(request.get("action"))
        arguments = dict(request.get("arguments") or request.get("args") or {})

        try:
            if action != "run_command":
                raise ValueError(f"Unsupported terminal tool action: {action}")
            result = self._run_command(arguments)
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

    def _run_command(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = self._normalize_command(arguments.get("command"))
        executable = self._resolve_executable(command[0])
        cwd = self._resolve_cwd(arguments.get("cwd"))
        timeout_seconds = max(1, min(int(arguments.get("timeout_seconds", 60) or 60), 300))
        env = os.environ.copy()
        for key, value in dict(arguments.get("env") or {}).items():
            env[str(key)] = str(value)

        completed = subprocess.run(
            [executable, *command[1:]],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            shell=False,
        )
        stdout = completed.stdout[-4000:]
        stderr = completed.stderr[-4000:]
        ok = completed.returncode == 0
        summary = (
            f"Command succeeded ({completed.returncode}): {' '.join(command)}"
            if ok
            else f"Command failed ({completed.returncode}): {' '.join(command)}"
        )
        return {
            "ok": ok,
            "command": command,
            "cwd": str(cwd),
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "summary": summary,
        }

    def _resolve_cwd(self, raw_cwd: Any) -> Path:
        cwd = Path(str(raw_cwd or self.allowed_roots[0])).expanduser()
        if not cwd.is_absolute():
            cwd = self.allowed_roots[0] / cwd
        resolved = cwd.resolve()
        if not any(resolved == root or resolved.is_relative_to(root) for root in self.allowed_roots):
            raise PermissionError(f"Access denied: {resolved} is outside allowed roots")
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def _normalize_action(self, action: Any) -> str:
        normalized = str(action or "").strip().lower()
        if normalized not in self.VALID_ACTIONS:
            raise ValueError(f"Unsupported terminal tool action: {action}")
        return "run_command" if normalized == "run" else normalized

    def _normalize_command(self, raw_command: Any) -> list[str]:
        if isinstance(raw_command, (list, tuple)):
            command = [str(part) for part in raw_command if str(part)]
        elif isinstance(raw_command, str):
            if any(token in raw_command for token in ("&&", "||", ";", "|", ">", "<")):
                raise ValueError("Shell chaining and redirection are not allowed")
            command = shlex.split(raw_command, posix=os.name != "nt")
        else:
            raise ValueError("A command list or string is required")

        if not command:
            raise ValueError("A command is required")
        command_name = Path(command[0]).name.lower()
        if command_name not in self.ALLOWED_COMMANDS:
            raise PermissionError(f"Command '{command[0]}' is not allowed")
        return command

    def _resolve_executable(self, command_name: str) -> str:
        candidates = [command_name]
        base_name = Path(command_name).name.lower()
        if os.name == "nt":
            if base_name == "npm":
                candidates.insert(0, "npm.cmd")
            if base_name == "pnpm":
                candidates.insert(0, "pnpm.cmd")
            if base_name == "npx":
                candidates.insert(0, "npx.cmd")
        for candidate in candidates:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        raise FileNotFoundError(f"Could not find executable for {command_name}")

    def _log_action(self, result: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": result.get("tool", "terminal_tool"),
            "action": result.get("action"),
            "ok": bool(result.get("ok", False)),
            "cwd": result.get("cwd"),
            "command": result.get("command"),
            "exit_code": result.get("exit_code"),
            "summary": result.get("summary"),
            "allowed_roots": [str(root) for root in self.allowed_roots],
        }
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
