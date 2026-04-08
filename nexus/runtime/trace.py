"""
Execution tracing and structured decision logging for NEXUS workflows.

Designed to keep runtime behavior observable without adding heavy overhead:
- task lifecycle events are buffered in memory and persisted at workflow end
- structured decisions are buffered and flushed in batches
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TraceEvent:
    """A single execution lifecycle event."""

    timestamp: str
    kind: str
    task_id: str
    agent: str | None = None
    attempt: int | None = None
    input_preview: str | None = None
    output_preview: str | None = None
    reflect_score: float | None = None
    evaluation_score: float | None = None
    critic_scores: dict[str, float] = field(default_factory=dict)
    retry_count: int | None = None
    fallback_triggered: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionEntry:
    """A structured runtime decision log row."""

    timestamp: str
    decision_type: str
    task_id: str
    agent_selected: str | None
    reason: str
    confidence: float | None = None
    retry: bool = False
    fallback_triggered: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExecutionTrace:
    """Collect trace events and structured decisions for a workflow."""

    def __init__(
        self,
        workflow_id: str,
        root_dir: Path | None = None,
        decision_log_path: Path | None = None,
        decision_flush_interval: int = 10,
        verbosity: str = "full",
    ):
        from nexus.config import config

        base_dir = Path(root_dir) if root_dir else config.data_dir / "runtime_traces"
        base_dir.mkdir(parents=True, exist_ok=True)
        self.workflow_id = workflow_id
        self.trace_path = base_dir / f"{workflow_id}.json"

        runtime_log_dir = self.trace_path.parent.parent / "runtime_logs"
        runtime_log_dir.mkdir(parents=True, exist_ok=True)
        self.decision_log_path = Path(decision_log_path) if decision_log_path else runtime_log_dir / "decision_log.jsonl"
        self.events: list[TraceEvent] = []
        self.decisions: list[DecisionEntry] = []
        self._pending_decisions: list[DecisionEntry] = []
        self._decision_flush_interval = max(1, int(decision_flush_interval))
        self.verbosity = verbosity
        self.started_at = _utc_now()
        self.finished_at: str | None = None
        self.status: str | None = None
        self.final_output_preview: str | None = None
        self.metadata: dict[str, Any] = {}

    def record_task_event(
        self,
        *,
        kind: str,
        task_id: str,
        agent: str | None = None,
        attempt: int | None = None,
        input_text: str | None = None,
        output_text: str | None = None,
        reflect_score: float | None = None,
        evaluation_score: float | None = None,
        critic_scores: dict[str, float] | None = None,
        retry_count: int | None = None,
        fallback_triggered: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Capture a task lifecycle event."""
        input_preview = self._preview(input_text)
        output_preview = self._preview(output_text)
        critic_payload = deepcopy(critic_scores or {})
        if self.verbosity == "compact":
            if kind in {"workflow_started", "task_started"}:
                input_preview = None
            if kind == "task_finished" and (metadata or {}).get("status") == "completed":
                output_preview = None
                critic_payload = {}
        self.events.append(
            TraceEvent(
                timestamp=_utc_now(),
                kind=kind,
                task_id=task_id,
                agent=agent,
                attempt=attempt,
                input_preview=input_preview,
                output_preview=output_preview,
                reflect_score=reflect_score,
                evaluation_score=evaluation_score,
                critic_scores=critic_payload,
                retry_count=retry_count,
                fallback_triggered=fallback_triggered,
                metadata=deepcopy(metadata or {}),
            )
        )

    def record_decision(
        self,
        *,
        decision_type: str,
        task_id: str,
        agent_selected: str | None,
        reason: str,
        confidence: float | None = None,
        retry: bool = False,
        fallback_triggered: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a structured runtime decision row."""
        entry = DecisionEntry(
            timestamp=_utc_now(),
            decision_type=decision_type,
            task_id=task_id,
            agent_selected=agent_selected,
            reason=reason,
            confidence=confidence,
            retry=retry,
            fallback_triggered=fallback_triggered,
            metadata=deepcopy(metadata or {}),
        )
        self.decisions.append(entry)
        self._pending_decisions.append(entry)
        if len(self._pending_decisions) >= self._decision_flush_interval:
            self.flush_decisions()

    def finalize(self, *, status: str, final_output: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        """Persist the aggregated workflow trace."""
        self.finished_at = _utc_now()
        self.status = status
        self.final_output_preview = self._preview(final_output)
        self.metadata = deepcopy(metadata or {})
        self.flush_decisions()
        payload = self.snapshot()
        with open(self.trace_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def flush_decisions(self) -> None:
        """Flush buffered decision rows to disk in one append."""
        if not self._pending_decisions:
            return
        lines = [
            json.dumps(entry.to_dict(), ensure_ascii=True)
            for entry in self._pending_decisions
        ]
        with open(self.decision_log_path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        self._pending_decisions.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return the current in-memory trace payload."""
        return {
            "workflow_id": self.workflow_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "trace_path": str(self.trace_path),
            "decision_log_path": str(self.decision_log_path),
            "final_output_preview": self.final_output_preview,
            "metadata": deepcopy(self.metadata),
            "events": [event.to_dict() for event in self.events],
            "decisions": [decision.to_dict() for decision in self.decisions],
        }

    def _preview(self, text: str | None, limit: int = 240) -> str | None:
        if text is None:
            return None
        return text[:limit]


class ExecutionContext:
    """Holds per-run execution strategy that should not live in shared memory."""

    def __init__(self, trace: ExecutionTrace | None = None, mode: str = "stable"):
        self.trace = trace
        self.mode = mode
        self._task_strategies: dict[str, dict[str, Any]] = {}
        self._task_history: dict[str, list[dict[str, Any]]] = {}
        self._task_started_at: dict[str, float] = {}

    def set_task_strategy(self, task_id: str, strategy: dict[str, Any]) -> None:
        self._task_strategies[task_id] = deepcopy(strategy)

    def get_task_strategy(self, task_id: str) -> dict[str, Any]:
        return deepcopy(self._task_strategies.get(task_id, {}))

    def clear_task_strategy(self, task_id: str) -> None:
        self._task_strategies.pop(task_id, None)

    def start_task(self, task_id: str) -> None:
        self._task_started_at.setdefault(task_id, time.perf_counter())

    def elapsed_seconds(self, task_id: str) -> float:
        started_at = self._task_started_at.get(task_id)
        if started_at is None:
            return 0.0
        return max(0.0, time.perf_counter() - started_at)

    def record_attempt(self, task_id: str, payload: dict[str, Any]) -> None:
        self._task_history.setdefault(task_id, []).append(deepcopy(payload))

    def task_history(self, task_id: str) -> list[dict[str, Any]]:
        return deepcopy(self._task_history.get(task_id, []))
