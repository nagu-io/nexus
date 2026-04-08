"""
Lightweight shared memory for compiled NEXUS workflows.

The orchestrator uses this as the only coordination channel between tasks.
"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SharedMemory:
    """Small persistent state store for workflow execution."""

    def __init__(self, workflow_id: str | None = None, root_dir: Path | None = None):
        from nexus.config import config

        self.workflow_id = workflow_id or uuid.uuid4().hex[:12]
        self.root_dir = Path(root_dir) if root_dir else config.data_dir / "runtime_memory"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root_dir / f"{self.workflow_id}.json"

        self.state: dict[str, Any] = {}
        self.task_results: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self._persist()

    def put(self, key: str, value: Any) -> None:
        """Write a key/value pair into shared workflow state."""
        self.state[key] = value
        self._persist()

    def get(self, key: str, default: Any = None) -> Any:
        """Read a value from shared workflow state."""
        return self.state.get(key, default)

    def append_event(self, kind: str, source: str, payload: dict[str, Any] | None = None) -> None:
        """Record a timestamped workflow event."""
        self.events.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                "source": source,
                "payload": payload or {},
            }
        )
        self._persist()

    def publish_task_result(self, task_id: str, payload: dict[str, Any]) -> None:
        """Store the latest execution payload for a task."""
        self.task_results[task_id] = deepcopy(payload)
        self.put(f"task:{task_id}", self.task_results[task_id])

    def dependency_context(self, task_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return the task outputs available to the next step."""
        return {
            task_id: deepcopy(self.task_results[task_id])
            for task_id in task_ids
            if task_id in self.task_results
        }

    def agent_context(self, agent_name: str) -> dict[str, Any]:
        """Expose the latest workflow data to an agent lifecycle hook."""
        return {
            "workflow_id": self.workflow_id,
            "state_keys": sorted(self.state.keys()),
            "task_ids": sorted(self.task_results.keys()),
            "agent": agent_name,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of the shared state."""
        return {
            "workflow_id": self.workflow_id,
            "state": deepcopy(self.state),
            "task_results": deepcopy(self.task_results),
            "events": deepcopy(self.events),
            "state_path": str(self.state_path),
        }

    def _persist(self) -> None:
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(self.snapshot(), handle, indent=2, default=str)
