"""Project-mode helpers for persistent local project sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nexus.memory.environment_memory import EnvironmentMemory
from nexus.runtime.code_reader import CodeReader


class ProjectModeManager:
    """Prepare and persist project-aware context across multiple commands."""

    def __init__(
        self,
        *,
        code_reader: CodeReader | None = None,
        environment_memory: EnvironmentMemory | None = None,
    ):
        self.code_reader = code_reader or CodeReader()
        self.environment_memory = environment_memory or EnvironmentMemory()

    def prepare(
        self,
        *,
        project_dir: str | Path,
        goal: str,
        execution_mode: str = "stable",
    ) -> dict[str, Any]:
        """Scan the project and return persisted project-mode context."""
        project_context = self.code_reader.scan(project_dir)
        return self.environment_memory.begin_project_session(
            project_context=project_context,
            goal=goal,
            execution_mode=execution_mode,
        )
