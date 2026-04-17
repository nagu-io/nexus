"""Workspace management for autonomous coding runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


@dataclass
class WorkspaceDirectory:
    """A dedicated project workspace for one workflow execution."""

    IGNORED_DIRS = {
        ".git",
        ".hg",
        ".nexus",
        ".next",
        ".ruff_cache",
        ".svn",
        ".turbo",
        ".venv",
        ".pytest_cache",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "venv",
    }
    IGNORED_SUFFIXES = {
        ".7z",
        ".avi",
        ".bin",
        ".class",
        ".db",
        ".dll",
        ".dylib",
        ".eot",
        ".exe",
        ".gif",
        ".gz",
        ".ico",
        ".jar",
        ".jpeg",
        ".jpg",
        ".lock",
        ".mp3",
        ".mp4",
        ".mov",
        ".o",
        ".obj",
        ".pdf",
        ".png",
        ".pyc",
        ".pyd",
        ".pyo",
        ".so",
        ".sqlite",
        ".svg",
        ".tar",
        ".ttf",
        ".wav",
        ".webm",
        ".webp",
        ".woff",
        ".woff2",
        ".zip",
    }
    TEXT_PREVIEW_NAMES = {
        ".env",
        "dockerfile",
        "justfile",
        "makefile",
        "procfile",
        "requirements.txt",
    }
    TEXT_PREVIEW_SUFFIXES = {
        ".bat",
        ".c",
        ".cfg",
        ".cmd",
        ".cpp",
        ".css",
        ".csv",
        ".env",
        ".go",
        ".h",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".md",
        ".ps1",
        ".py",
        ".rs",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
    MAX_PREVIEW_BYTES = 16_384

    root_dir: Path
    workflow_id: str
    goal_slug: str

    @classmethod
    def for_workflow(
        cls,
        *,
        workflow_id: str,
        goal: str,
        base_dir: Path | None = None,
    ) -> "WorkspaceDirectory":
        from nexus.config import config

        workspace_base = Path(base_dir) if base_dir else config.data_dir / "workspaces"
        workspace_base.mkdir(parents=True, exist_ok=True)
        goal_slug = cls._slugify(goal)
        root_dir = workspace_base / f"{goal_slug}-{workflow_id}"
        root_dir.mkdir(parents=True, exist_ok=True)
        return cls(root_dir=root_dir.resolve(), workflow_id=workflow_id, goal_slug=goal_slug)

    @classmethod
    def for_project(
        cls,
        *,
        project_dir: Path | str,
        workflow_id: str,
        goal: str,
    ) -> "WorkspaceDirectory":
        """Bind the workflow to an existing project directory in project mode."""
        root_dir = Path(project_dir).expanduser().resolve()
        root_dir.mkdir(parents=True, exist_ok=True)
        return cls(root_dir=root_dir, workflow_id=workflow_id, goal_slug=cls._slugify(goal))

    def snapshot(self, *, max_files: int = 40, max_preview_chars: int = 240) -> dict[str, Any]:
        """Return a compact workspace snapshot suitable for shared memory and prompts."""
        files: list[dict[str, Any]] = []
        for path in self.root_dir.rglob("*"):
            if len(files) >= max_files:
                break
            if not path.is_file() or self._should_skip_path(path):
                continue
            relative = path.relative_to(self.root_dir)
            files.append(
                {
                    "path": relative.as_posix(),
                    "size": path.stat().st_size,
                    "preview": self._read_preview(path, max_preview_chars=max_preview_chars),
                }
            )
        return {
            "root_dir": str(self.root_dir),
            "workflow_id": self.workflow_id,
            "goal_slug": self.goal_slug,
            "file_count": len(files),
            "files": files,
        }

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
        return slug[:40] or "workspace"

    def _should_skip_path(self, path: Path) -> bool:
        relative_parts = path.relative_to(self.root_dir).parts
        if any(part in self.IGNORED_DIRS for part in relative_parts):
            return True
        return path.suffix.lower() in self.IGNORED_SUFFIXES

    def _read_preview(self, path: Path, *, max_preview_chars: int) -> str:
        lower_name = path.name.lower()
        lower_suffix = path.suffix.lower()
        if lower_suffix not in self.TEXT_PREVIEW_SUFFIXES and lower_name not in self.TEXT_PREVIEW_NAMES:
            return ""

        try:
            with path.open("rb") as handle:
                payload = handle.read(self.MAX_PREVIEW_BYTES)
        except Exception:
            return ""

        if b"\x00" in payload:
            return ""

        try:
            text = payload.decode("utf-8", errors="replace")
        except Exception:
            return ""
        return text[:max_preview_chars]
