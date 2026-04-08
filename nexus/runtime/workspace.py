"""Workspace management for autonomous coding runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


@dataclass
class WorkspaceDirectory:
    """A dedicated project workspace for one workflow execution."""

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
        for path in sorted(self.root_dir.rglob("*")):
            if len(files) >= max_files:
                break
            if not path.is_file() or ".nexus" in path.parts or "node_modules" in path.parts:
                continue
            relative = path.relative_to(self.root_dir)
            try:
                preview = path.read_text(encoding="utf-8", errors="replace")[:max_preview_chars]
            except Exception:
                preview = ""
            files.append(
                {
                    "path": relative.as_posix(),
                    "size": path.stat().st_size,
                    "preview": preview,
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
