"""Shared explainability and runtime summaries for CLI and dashboard surfaces."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nexus.memory.skill_memory import SkillMemory
from nexus.runtime.decision_cache import DecisionCache


class RuntimeInsights:
    """Summarize runtime traces into explainable, UI-friendly views."""

    def __init__(
        self,
        trace_dir: Path | None = None,
        skill_memory: SkillMemory | None = None,
        decision_cache: DecisionCache | None = None,
    ):
        from nexus.config import config

        self.trace_dir = Path(trace_dir) if trace_dir else config.data_dir / "runtime_traces"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.skill_memory = skill_memory or SkillMemory()
        self.decision_cache = decision_cache or DecisionCache()

    def explain_run(
        self,
        *,
        goal: str,
        intent: dict[str, Any],
        plan: dict[str, Any],
        blueprint: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Build an explainable summary for a just-finished workflow."""
        trace_payload = deepcopy(result.get("trace", {}))
        trace_payload.setdefault("workflow_id", result.get("workflow_id"))
        trace_payload.setdefault("status", result.get("status"))
        trace_payload.setdefault("metadata", {})
        trace_payload["metadata"]["goal"] = goal
        trace_summary = self.summarize_trace_payload(trace_payload)

        plan_steps = []
        for task in blueprint.get("tasks", []):
            plan_steps.append(
                {
                    "task_id": task.get("id"),
                    "task_type": task.get("task_type"),
                    "agent": task.get("agent"),
                    "depends_on": list(task.get("depends_on", [])),
                    "retry_strategy": task.get("retry_strategy"),
                    "fallback": task.get("fallback"),
                    "confidence_threshold": float(task.get("confidence_threshold", 0.0) or 0.0),
                }
            )

        decision_highlights: list[str] = []
        seen = set()
        for decision in trace_payload.get("decisions", []):
            highlight = self._decision_highlight(decision)
            if not highlight or highlight in seen:
                continue
            seen.add(highlight)
            decision_highlights.append(highlight)

        if not decision_highlights:
            decision_highlights.append("executed without retries, fallbacks, or cache overrides")

        strategy_notes = []
        for decision in trace_payload.get("decisions", []):
            if decision.get("decision_type") not in {"retry", "fallback"}:
                continue
            metadata = decision.get("metadata", {}) or {}
            strategy = metadata.get("strategy") or "repeat"
            target_agent = decision.get("agent_selected") or metadata.get("current_agent")
            strategy_notes.append(
                {
                    "task_id": decision.get("task_id"),
                    "strategy": strategy,
                    "agent": target_agent,
                    "reason": decision.get("reason"),
                }
            )

        return {
            "summary": {
                "workflow_id": result.get("workflow_id"),
                "goal": goal,
                "primary_intent": intent.get("primary_intent"),
                "complexity": intent.get("complexity"),
                "execution_mode": trace_summary.get("execution_mode"),
                "project_mode": trace_summary.get("project_mode"),
                "project_root": trace_summary.get("project_root"),
                "status": result.get("status"),
                "confidence": trace_summary.get("final_confidence"),
                "retry_count": trace_summary.get("retry_count"),
                "cache_used": trace_summary.get("cache_used"),
                "parallel_batches": trace_summary.get("parallel_batches"),
            },
            "plan": plan_steps,
            "decisions": decision_highlights[:8],
            "strategies": strategy_notes,
            "result": {
                "final_output": result.get("final_output", ""),
                "final_output_preview": trace_payload.get("final_output_preview"),
                "trace_path": trace_payload.get("trace_path"),
                "decision_log_path": trace_payload.get("decision_log_path"),
            },
        }

    def overview(self, limit: int = 8) -> dict[str, Any]:
        """Return a dashboard-friendly overview of recent runtime activity."""
        runs = self.recent_runs(limit=limit)
        total_runs = len(runs)
        completed_runs = sum(1 for run in runs if run["status"] == "completed")
        avg_retries = (
            sum(float(run.get("retry_count", 0) or 0) for run in runs) / total_runs
            if total_runs
            else 0.0
        )
        confidence_values = [float(run["final_confidence"]) for run in runs if run.get("final_confidence") is not None]
        avg_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0
        )
        cache_hits = sum(1 for run in runs if run.get("cache_used"))
        skill_summary = self.skill_memory.summary(limit=5)
        cache_summary = self.decision_cache.summary(limit=5)
        return {
            "metrics": {
                "total_runs": total_runs,
                "completed_runs": completed_runs,
                "success_rate": completed_runs / total_runs if total_runs else 0.0,
                "avg_retries": avg_retries,
                "avg_confidence": avg_confidence,
                "cache_reuse_rate": cache_hits / total_runs if total_runs else 0.0,
                "reusable_cache_entries": cache_summary["reusable_entries"],
            },
            "runs": runs,
            "patterns": skill_summary["top_patterns"],
            "pattern_metrics": skill_summary["metrics"],
            "decision_cache": cache_summary,
        }

    def recent_runs(self, limit: int = 8) -> list[dict[str, Any]]:
        """Load and summarize recent workflow traces from disk."""
        traces = []
        for trace_path in sorted(
            self.trace_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[: max(1, int(limit))]:
            try:
                with open(trace_path, encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                continue
            summary = self.summarize_trace_payload(payload)
            summary["trace_path"] = str(trace_path)
            traces.append(summary)
        return traces

    def run_detail(self, workflow_id: str) -> dict[str, Any] | None:
        """Return a full trace plus the derived summary for one workflow."""
        trace_path = self.trace_dir / f"{workflow_id}.json"
        if not trace_path.exists():
            return None
        try:
            with open(trace_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return None
        return {
            "summary": self.summarize_trace_payload(payload),
            "trace": payload,
        }

    def summarize_trace_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Turn a raw trace payload into compact operational metrics."""
        events = list(payload.get("events", []))
        decisions = list(payload.get("decisions", []))
        metadata = payload.get("metadata", {}) or {}

        retry_by_task: dict[str, int] = {}
        for event in events:
            if event.get("kind") != "task_finished" or event.get("task_id") == "workflow":
                continue
            retry_by_task[event["task_id"]] = max(
                retry_by_task.get(event["task_id"], 0),
                int(event.get("retry_count", 0) or 0),
            )

        agents = sorted(
            {
                event.get("agent")
                for event in events
                if event.get("agent") and event.get("task_id") != "workflow"
            }
        )
        execution_mode = (
            metadata.get("execution_mode")
            or next(
                (
                    (decision.get("metadata", {}) or {}).get("mode")
                    for decision in decisions
                    if decision.get("decision_type") == "policy_mode"
                ),
                "stable",
            )
        )

        return {
            "workflow_id": payload.get("workflow_id"),
            "goal": metadata.get("goal") or self._goal_from_events(events) or payload.get("workflow_id"),
            "status": payload.get("status", "unknown"),
            "execution_mode": execution_mode,
            "project_mode": bool(metadata.get("project_mode", False)),
            "project_root": metadata.get("project_root"),
            "started_at": payload.get("started_at"),
            "finished_at": payload.get("finished_at"),
            "duration_seconds": self._duration_seconds(payload.get("started_at"), payload.get("finished_at")),
            "task_count": int(metadata.get("task_count", 0) or len(retry_by_task)),
            "retry_count": sum(retry_by_task.values()),
            "final_confidence": self._final_confidence(payload),
            "cache_used": any(decision.get("decision_type") == "decision_cache" for decision in decisions),
            "parallel_batches": sum(
                1
                for decision in decisions
                if decision.get("decision_type") == "parallel_batch"
                and bool((decision.get("metadata", {}) or {}).get("parallel"))
            ),
            "agents": agents,
            "strategies": [
                (decision.get("metadata", {}) or {}).get("strategy")
                for decision in decisions
                if decision.get("decision_type") in {"retry", "fallback"}
                and (decision.get("metadata", {}) or {}).get("strategy")
            ],
            "skipped_critics": sorted(
                {
                    critic
                    for decision in decisions
                    if decision.get("decision_type") == "evaluation"
                    for critic in ((decision.get("metadata", {}) or {}).get("skipped_critics") or [])
                }
            ),
            "final_output_preview": payload.get("final_output_preview"),
        }

    def _decision_highlight(self, decision: dict[str, Any]) -> str | None:
        decision_type = decision.get("decision_type")
        metadata = decision.get("metadata", {}) or {}
        task_id = decision.get("task_id", "task")

        if decision_type == "skill_pattern_selection":
            success_rate = self._percent(decision.get("confidence"))
            return f"reused a remembered workflow pattern ({success_rate} success)"
        if decision_type == "decision_cache":
            confidence = self._percent(decision.get("confidence"))
            return f"used cached strategy for {task_id} ({confidence} expected confidence)"
        if decision_type == "evaluation" and metadata.get("skipped_critics"):
            critics = ", ".join(metadata.get("skipped_critics", []))
            return f"skipped expensive critics for {task_id}: {critics}"
        if decision_type == "parallel_batch" and metadata.get("parallel"):
            batch = ", ".join(metadata.get("selected_batch", []))
            return f"enabled parallel execution for {batch}"
        if decision_type == "retry":
            return f"retried {task_id} using {metadata.get('strategy', 'repeat')}"
        if decision_type == "fallback":
            return f"triggered fallback for {task_id} using {metadata.get('strategy', 'fallback')}"
        if decision_type == "policy" and metadata.get("action") in {"retry_with_strategy_change", "retry_current_strategy"}:
            return f"policy chose {metadata['action'].replace('_', ' ')} for {task_id}"
        return None

    def _goal_from_events(self, events: list[dict[str, Any]]) -> str | None:
        for event in events:
            if event.get("kind") == "workflow_started" and event.get("input_preview"):
                return event["input_preview"]
        return None

    def _final_confidence(self, payload: dict[str, Any]) -> float | None:
        metadata = payload.get("metadata", {}) or {}
        if metadata.get("final_confidence") is not None:
            return float(metadata["final_confidence"])
        decisions = payload.get("decisions", [])
        for decision in reversed(decisions):
            if decision.get("decision_type") == "evaluation" and decision.get("confidence") is not None:
                return float(decision["confidence"])
        for event in reversed(payload.get("events", [])):
            if event.get("evaluation_score") is not None:
                return float(event["evaluation_score"])
        return None

    def _duration_seconds(self, started_at: str | None, finished_at: str | None) -> float | None:
        if not started_at or not finished_at:
            return None
        started = self._parse_datetime(started_at)
        finished = self._parse_datetime(finished_at)
        if not started or not finished:
            return None
        return max(0.0, (finished - started).total_seconds())

    def _parse_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            timestamp = datetime.fromisoformat(value)
        except ValueError:
            return None
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    def _percent(self, value: float | None) -> str:
        if value is None:
            return "--"
        return f"{round(float(value) * 100)}%"
