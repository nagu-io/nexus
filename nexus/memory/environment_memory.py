"""Persistent local environment memory for project-aware runs."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EnvironmentMemory:
    """Store user preferences, project patterns, and common errors across sessions."""

    def __init__(self, storage_path: Path | None = None):
        from nexus.config import config

        self.storage_path = Path(storage_path) if storage_path else config.data_dir / "environment_memory.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def begin_project_session(
        self,
        *,
        project_context: dict[str, Any],
        goal: str,
        execution_mode: str = "stable",
    ) -> dict[str, Any]:
        """Register a project-mode command and return the learned session context."""
        record = self._project_record(project_context)
        record["project_context"] = self._compact_project_context(project_context)
        record["project_signature"] = project_context.get("project_signature")
        record["session_count"] = int(record.get("session_count", 0) or 0) + 1
        record["last_goal"] = goal
        record["last_used_at"] = self._utc_now()
        record["recent_goals"] = self._append_recent(record.get("recent_goals", []), goal, limit=8)
        self._increment_counter(record.setdefault("execution_modes", {}), execution_mode)
        self._learn_user_preferences(project_context, execution_mode=execution_mode)
        self._save()
        return self.project_mode_context(project_context.get("project_root"))

    def project_mode_context(self, project_root: str | Path | None) -> dict[str, Any]:
        """Return the persisted context NEXUS should use for project mode."""
        if not project_root:
            return {"enabled": False}
        project_id = self._project_id(project_root)
        record = self._normalize_project_record(self._data["projects"].get(project_id))
        if not record:
            return {"enabled": False, "project_root": str(Path(project_root).expanduser().resolve())}
        return {
            "enabled": True,
            "project_id": project_id,
            "project_root": record.get("project_root"),
            "project_signature": record.get("project_signature"),
            "project_context": deepcopy(record.get("project_context", {})),
            "session_iteration": int(record.get("session_count", 0) or 0),
            "recent_goals": list(record.get("recent_goals", [])),
            "user_preferences": self.preference_summary(),
            "successful_patterns": self.project_patterns(project_root, limit=5),
            "common_errors": self.project_common_errors(project_root, limit=5),
        }

    def record_workflow(
        self,
        *,
        project_root: str | Path,
        goal: str,
        blueprint: Any,
        executions: list[dict[str, Any]],
        status: str,
        final_confidence: float | None,
        execution_mode: str = "stable",
    ) -> None:
        """Persist project-specific success patterns and common error signatures."""
        project_id = self._project_id(project_root)
        record = self._normalize_project_record(self._data["projects"].get(project_id)) or self._default_project_record(project_root)
        self._data["projects"][project_id] = record
        record["last_used_at"] = self._utc_now()
        record["last_goal"] = goal
        record["recent_goals"] = self._append_recent(record.get("recent_goals", []), goal, limit=8)
        self._increment_counter(record.setdefault("execution_modes", {}), execution_mode)
        record["total_runs"] = int(record.get("total_runs", 0) or 0) + 1
        if status == "completed":
            record["successful_runs"] = int(record.get("successful_runs", 0) or 0) + 1

        metadata = getattr(blueprint, "metadata", None) or (blueprint.get("metadata", {}) if isinstance(blueprint, dict) else {})
        primary_intent = getattr(blueprint, "primary_intent", None) or metadata.get("primary_intent", "unknown")
        plan_signature = metadata.get("plan_signature") or self._pattern_signature(goal, primary_intent)
        retry_count = sum(max(0, int(item.get("attempts", 1)) - 1) for item in executions)
        if status == "completed":
            self._record_success_pattern(
                record=record,
                signature=plan_signature,
                goal=goal,
                primary_intent=primary_intent,
                executions=executions,
                retry_count=retry_count,
                final_confidence=final_confidence,
            )
        self._record_common_errors(record=record, executions=executions)
        self._save()

    def preference_summary(self) -> dict[str, Any]:
        """Return the user's inferred local preferences from past project work."""
        preferences = deepcopy(self._data.get("user_preferences", {}))
        return {
            "preferred_languages": self._top_keys(preferences.get("languages", {})),
            "preferred_frameworks": self._top_keys(preferences.get("frameworks", {})),
            "preferred_execution_mode": self._top_key(preferences.get("execution_modes", {})),
            "last_project_root": preferences.get("last_project_root"),
        }

    def project_patterns(
        self,
        project_root: str | Path | None,
        *,
        limit: int = 5,
        primary_intent: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return ranked successful patterns for a specific project."""
        record = self._project_record_for_lookup(project_root)
        if not record:
            return []
        patterns = []
        for pattern in (record.get("successful_patterns", {}) or {}).values():
            normalized = self._normalize_pattern(pattern)
            if primary_intent and normalized.get("primary_intent") != primary_intent:
                continue
            patterns.append(normalized)
        ranked = sorted(
            patterns,
            key=lambda item: (
                float(item.get("success_rate", 0.0) or 0.0),
                -float(item.get("avg_retries", 0.0) or 0.0),
                float(item.get("avg_confidence", 0.0) or 0.0),
                int(item.get("total_runs", 0) or 0),
            ),
            reverse=True,
        )
        return ranked[: max(1, int(limit))]

    def project_common_errors(self, project_root: str | Path | None, *, limit: int = 5) -> list[dict[str, Any]]:
        """Return the most common observed failures for a project."""
        record = self._project_record_for_lookup(project_root)
        if not record:
            return []
        entries = sorted(
            (record.get("common_errors", {}) or {}).values(),
            key=lambda item: (
                int(item.get("count", 0) or 0),
                item.get("updated_at") or "",
            ),
            reverse=True,
        )
        return [deepcopy(item) for item in entries[: max(1, int(limit))]]

    def reload(self) -> None:
        self._data = self._load()

    def _project_record(self, project_context: dict[str, Any]) -> dict[str, Any]:
        project_id = self._project_id(project_context.get("project_root"))
        record = self._normalize_project_record(self._data["projects"].get(project_id)) or self._default_project_record(project_context.get("project_root"))
        record["project_root"] = project_context.get("project_root")
        self._data["projects"][project_id] = record
        return record

    def _project_record_for_lookup(self, project_root: str | Path | None) -> dict[str, Any] | None:
        if not project_root:
            return None
        project_id = self._project_id(project_root)
        return self._normalize_project_record(self._data["projects"].get(project_id))

    def _record_success_pattern(
        self,
        *,
        record: dict[str, Any],
        signature: str,
        goal: str,
        primary_intent: str,
        executions: list[dict[str, Any]],
        retry_count: int,
        final_confidence: float | None,
    ) -> None:
        patterns = record.setdefault("successful_patterns", {})
        pattern = self._normalize_pattern(patterns.get(signature)) or {
            "signature": signature,
            "primary_intent": primary_intent,
            "total_runs": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "avg_retries": 0.0,
            "avg_confidence": 0.0,
            "best_agent_sequence": [],
            "examples": [],
            "updated_at": None,
        }
        pattern["total_runs"] = int(pattern.get("total_runs", 0) or 0) + 1
        pattern["success_count"] = int(pattern.get("success_count", 0) or 0) + 1
        pattern["success_rate"] = pattern["success_count"] / max(pattern["total_runs"], 1)
        prior_retry_total = float(pattern.get("avg_retries", 0.0) or 0.0) * max(pattern["total_runs"] - 1, 0)
        pattern["avg_retries"] = (prior_retry_total + retry_count) / max(pattern["total_runs"], 1)
        prior_confidence_total = float(pattern.get("avg_confidence", 0.0) or 0.0) * max(pattern["total_runs"] - 1, 0)
        pattern["avg_confidence"] = (
            (prior_confidence_total + float(final_confidence or 0.0)) / max(pattern["total_runs"], 1)
            if final_confidence is not None
            else float(pattern.get("avg_confidence", 0.0) or 0.0)
        )
        pattern["best_agent_sequence"] = [
            execution.get("agent")
            for execution in executions
            if execution.get("agent")
        ]
        pattern["examples"] = self._append_recent(pattern.get("examples", []), goal, limit=5)
        pattern["updated_at"] = self._utc_now()
        patterns[signature] = pattern

    def _record_common_errors(self, *, record: dict[str, Any], executions: list[dict[str, Any]]) -> None:
        common_errors = record.setdefault("common_errors", {})
        for execution in executions:
            if execution.get("status") == "completed":
                continue
            observation = dict(execution.get("observation") or {})
            summary = str(observation.get("summary") or execution.get("error") or "").strip()
            failure_type = str(observation.get("failure_type") or "runtime_error")
            if not summary:
                continue
            signature = self._error_signature(failure_type, summary)
            entry = dict(common_errors.get(signature) or {})
            entry["signature"] = signature
            entry["failure_type"] = failure_type
            entry["summary"] = summary[:240]
            entry["count"] = int(entry.get("count", 0) or 0) + 1
            entry["last_agent"] = execution.get("agent")
            entry["last_task_id"] = execution.get("task_id")
            entry["updated_at"] = self._utc_now()
            common_errors[signature] = entry

    def _learn_user_preferences(self, project_context: dict[str, Any], *, execution_mode: str) -> None:
        preferences = self._data.setdefault(
            "user_preferences",
            {"languages": {}, "frameworks": {}, "execution_modes": {}, "last_project_root": None},
        )
        for language in project_context.get("languages", []):
            self._increment_counter(preferences.setdefault("languages", {}), language)
        for framework in project_context.get("frameworks", []):
            self._increment_counter(preferences.setdefault("frameworks", {}), framework)
        self._increment_counter(preferences.setdefault("execution_modes", {}), execution_mode)
        preferences["last_project_root"] = project_context.get("project_root")

    def _compact_project_context(self, project_context: dict[str, Any]) -> dict[str, Any]:
        return {
            "project_root": project_context.get("project_root"),
            "project_name": project_context.get("project_name"),
            "project_signature": project_context.get("project_signature"),
            "frameworks": list(project_context.get("frameworks", [])),
            "languages": list(project_context.get("languages", [])),
            "package_manager": project_context.get("package_manager"),
            "entrypoints": list(project_context.get("entrypoints", []))[:8],
            "directories": list(project_context.get("directories", []))[:16],
            "files": list(project_context.get("files", []))[:24],
            "scripts": dict(project_context.get("scripts", {})),
            "summary_text": project_context.get("summary_text"),
        }

    def _load(self) -> dict[str, Any]:
        if not self.storage_path.exists():
            return {"user_preferences": {}, "projects": {}}
        try:
            with open(self.storage_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return {"user_preferences": {}, "projects": {}}
        payload.setdefault("user_preferences", {})
        payload.setdefault("projects", {})
        payload["projects"] = {
            project_id: self._normalize_project_record(record)
            for project_id, record in payload["projects"].items()
            if self._normalize_project_record(record)
        }
        return payload

    def _save(self) -> None:
        with open(self.storage_path, "w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2)

    def _normalize_project_record(self, record: dict[str, Any] | None) -> dict[str, Any] | None:
        if not record:
            return None
        normalized = deepcopy(record)
        normalized.setdefault("project_root", None)
        normalized.setdefault("project_signature", None)
        normalized.setdefault("project_context", {})
        normalized.setdefault("session_count", 0)
        normalized.setdefault("total_runs", 0)
        normalized.setdefault("successful_runs", 0)
        normalized.setdefault("recent_goals", [])
        normalized.setdefault("successful_patterns", {})
        normalized.setdefault("common_errors", {})
        normalized.setdefault("execution_modes", {})
        normalized.setdefault("last_goal", None)
        normalized.setdefault("last_used_at", None)
        return normalized

    def _normalize_pattern(self, pattern: dict[str, Any] | None) -> dict[str, Any] | None:
        if not pattern:
            return None
        normalized = deepcopy(pattern)
        normalized.setdefault("signature", None)
        normalized.setdefault("primary_intent", "unknown")
        normalized.setdefault("total_runs", 0)
        normalized.setdefault("success_count", int(normalized.get("total_runs", 0) or 0))
        normalized.setdefault("success_rate", 0.0)
        normalized.setdefault("avg_retries", 0.0)
        normalized.setdefault("avg_confidence", 0.0)
        normalized.setdefault("best_agent_sequence", [])
        normalized.setdefault("examples", [])
        normalized.setdefault("updated_at", None)
        total_runs = int(normalized.get("total_runs", 0) or 0)
        success_count = int(normalized.get("success_count", total_runs) or total_runs)
        normalized["total_runs"] = max(total_runs, success_count)
        normalized["success_count"] = success_count
        normalized["success_rate"] = success_count / max(normalized["total_runs"], 1) if normalized["total_runs"] else 0.0
        return normalized

    def _default_project_record(self, project_root: str | Path | None) -> dict[str, Any]:
        return {
            "project_root": str(Path(project_root).expanduser().resolve()) if project_root else None,
            "project_signature": None,
            "project_context": {},
            "session_count": 0,
            "total_runs": 0,
            "successful_runs": 0,
            "recent_goals": [],
            "successful_patterns": {},
            "common_errors": {},
            "execution_modes": {},
            "last_goal": None,
            "last_used_at": None,
        }

    def _top_keys(self, values: dict[str, int], limit: int = 3) -> list[str]:
        return [
            key
            for key, _ in sorted(values.items(), key=lambda item: (-int(item[1]), item[0]))[:limit]
        ]

    def _top_key(self, values: dict[str, int]) -> str | None:
        top = self._top_keys(values, limit=1)
        return top[0] if top else None

    def _append_recent(self, values: list[str], item: str, *, limit: int) -> list[str]:
        updated = [value for value in values if value != item]
        updated.append(item)
        return updated[-limit:]

    def _increment_counter(self, mapping: dict[str, int], key: str) -> None:
        if not key:
            return
        mapping[key] = int(mapping.get(key, 0) or 0) + 1

    def _project_id(self, project_root: str | Path | None) -> str:
        root = str(Path(project_root or ".").expanduser().resolve())
        return hashlib.sha1(root.encode("utf-8")).hexdigest()

    def _pattern_signature(self, goal: str, primary_intent: str) -> str:
        payload = f"{primary_intent}|{goal.strip().lower()}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _error_signature(self, failure_type: str, summary: str) -> str:
        payload = f"{failure_type}|{summary.strip().lower()[:240]}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
