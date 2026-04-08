"""
Persistent skill-pattern memory for compiled NEXUS workflows.

Stores successful and failed workflow shapes so future plans can reuse
proven task sequences and retry strategies instead of starting from scratch.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SkillMemory:
    """Lightweight file-backed workflow pattern store."""

    def __init__(self, storage_path: Path | None = None):
        from nexus.config import config

        self.storage_path = Path(storage_path) if storage_path else config.data_dir / "skill_memory.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def signature_for_intent(self, intent: Any) -> str:
        """Build a stable signature for plan reuse."""
        profile = self._profile_from_payload(intent)
        agent_segment = ",".join(sorted(profile["required_agents"]))
        return (
            f"{profile['primary_intent']}|{profile['complexity']}|{agent_segment}|"
            f"file:{profile['file_action']}|memory:{profile['memory_action']}"
        )

    def lookup(self, intent: Any) -> dict | None:
        """Return the best remembered pattern for an intent signature."""
        profile = self._profile_from_payload(intent)
        signature = self.signature_for_intent(intent)
        candidates = []
        for raw_pattern in self._data["patterns"].values():
            pattern = self._normalize_pattern(raw_pattern)
            similarity = self._similarity_score(profile, pattern.get("profile", {}), signature, pattern.get("signature", ""))
            if similarity <= 0:
                continue
            rank_score = self._rank_pattern(pattern, similarity)
            candidates.append((rank_score, pattern))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_pattern = deepcopy(candidates[0][1])
        best_pattern["rank_score"] = candidates[0][0]
        return best_pattern

    def record_workflow(self, blueprint: Any, executions: list[dict], status: str) -> None:
        """Persist workflow outcomes for future planning."""
        metadata = getattr(blueprint, "metadata", {}) or {}
        signature = metadata.get("plan_signature") or metadata.get("signature")
        if not signature:
            return

        profile = {
            "primary_intent": getattr(blueprint, "primary_intent", metadata.get("primary_intent", "unknown")),
            "required_agents": sorted(metadata.get("required_agents", [])),
            "complexity": metadata.get("complexity", "unknown"),
            "file_action": metadata.get("file_action", "none"),
            "memory_action": metadata.get("memory_action", "none"),
        }
        retry_count = sum(max(0, int(entry.get("attempts", 1)) - 1) for entry in executions)
        task_count = len(getattr(blueprint, "tasks", []) or executions)

        patterns = self._data["patterns"]
        pattern = patterns.setdefault(
            signature,
            {
                "signature": signature,
                "profile": profile,
                "total_runs": 0,
                "success_count": 0,
                "failure_count": 0,
                "success_rate": 0.0,
                "total_retries": 0,
                "avg_retries": 0.0,
                "avg_task_count": 0.0,
                "best_score": None,
                "best_agent_sequence": [],
                "examples": [],
                "task_overrides": {},
                "updated_at": None,
            },
        )
        pattern = self._normalize_pattern(pattern)
        patterns[signature] = pattern

        pattern["profile"] = profile
        pattern["total_runs"] += 1
        pattern["total_retries"] += retry_count
        pattern["avg_retries"] = pattern["total_retries"] / max(pattern["total_runs"], 1)
        prior_total_task_count = float(pattern.get("avg_task_count", 0.0)) * max(pattern["total_runs"] - 1, 0)
        pattern["avg_task_count"] = (prior_total_task_count + task_count) / max(pattern["total_runs"], 1)

        if status == "completed":
            pattern["success_count"] += 1
        else:
            pattern["failure_count"] += 1
        pattern["success_rate"] = pattern["success_count"] / max(pattern["total_runs"], 1)

        performance_score = self._performance_score(status, retry_count, task_count)
        if status == "completed" and (
            pattern.get("best_score") is None or performance_score >= float(pattern["best_score"])
        ):
            pattern["best_score"] = performance_score
            pattern["best_agent_sequence"] = [entry.get("agent") for entry in executions if entry.get("agent")]
            pattern["task_overrides"] = self._build_task_overrides(blueprint)

        goal = getattr(blueprint, "goal", None) or metadata.get("goal")
        if goal and goal not in pattern["examples"]:
            pattern["examples"] = (pattern["examples"] + [goal])[-5:]
        pattern["last_status"] = status
        pattern["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def top_patterns(self, limit: int = 5) -> list[dict[str, Any]]:
        """Return the best-performing remembered patterns for UI and planning insight."""
        self.reload()
        patterns = [self._normalize_pattern(pattern) for pattern in self._data["patterns"].values()]
        ranked = sorted(
            patterns,
            key=lambda pattern: self._rank_pattern(pattern, similarity=1.0),
            reverse=True,
        )
        top_matches = []
        for pattern in ranked[: max(1, int(limit))]:
            profile = pattern.get("profile", {})
            top_matches.append(
                {
                    "signature": pattern.get("signature"),
                    "primary_intent": profile.get("primary_intent", "unknown"),
                    "required_agents": list(profile.get("required_agents", [])),
                    "complexity": profile.get("complexity", "unknown"),
                    "success_rate": float(pattern.get("success_rate", 0.0) or 0.0),
                    "avg_retries": float(pattern.get("avg_retries", 0.0) or 0.0),
                    "total_runs": int(pattern.get("total_runs", 0) or 0),
                    "best_agent_sequence": list(pattern.get("best_agent_sequence", [])),
                    "examples": list(pattern.get("examples", [])),
                    "updated_at": pattern.get("updated_at"),
                    "rank_score": self._rank_pattern(pattern, similarity=1.0),
                }
            )
        return top_matches

    def summary(self, limit: int = 5) -> dict[str, Any]:
        """Return aggregate metrics plus the top ranked remembered workflows."""
        self.reload()
        patterns = [self._normalize_pattern(pattern) for pattern in self._data["patterns"].values()]
        total_patterns = len(patterns)
        total_runs = sum(int(pattern.get("total_runs", 0) or 0) for pattern in patterns)
        avg_success_rate = (
            sum(float(pattern.get("success_rate", 0.0) or 0.0) for pattern in patterns) / total_patterns
            if total_patterns
            else 0.0
        )
        avg_retries = (
            sum(float(pattern.get("avg_retries", 0.0) or 0.0) for pattern in patterns) / total_patterns
            if total_patterns
            else 0.0
        )
        return {
            "metrics": {
                "total_patterns": total_patterns,
                "total_runs": total_runs,
                "avg_success_rate": avg_success_rate,
                "avg_retries": avg_retries,
            },
            "top_patterns": self.top_patterns(limit=limit),
        }

    def reload(self) -> None:
        """Refresh in-memory state from disk for UI and planner readers."""
        self._data = self._load()

    def _build_task_overrides(self, blueprint: Any) -> dict[str, dict]:
        overrides: dict[str, dict] = {}
        for task in getattr(blueprint, "tasks", []):
            task_type = getattr(task, "task_type", "generic")
            overrides[task_type] = {
                "agent": getattr(task, "agent", None),
                "candidate_agents": list(getattr(task, "candidate_agents", []) or []),
                "required_capabilities": list(getattr(task, "required_capabilities", []) or []),
                "retry_strategy": getattr(task, "retry_strategy", "repeat"),
                "fallback": getattr(task, "fallback", None),
                "fallback_agent": getattr(task, "fallback_agent", None),
                "confidence_threshold": getattr(task, "confidence_threshold", 0.0),
            }
        return overrides

    def _profile_from_payload(self, payload: Any) -> dict[str, Any]:
        primary_intent = getattr(payload, "primary_intent", None) or payload.get("primary_intent", "unknown")
        required_agents = getattr(payload, "required_agents", None) or payload.get("required_agents", [])
        complexity = getattr(payload, "complexity", None) or payload.get("complexity", "unknown")
        metadata = getattr(payload, "metadata", None) or payload.get("metadata", {})
        return {
            "primary_intent": primary_intent,
            "required_agents": sorted(required_agents),
            "complexity": complexity,
            "file_action": metadata.get("file_action") or "none",
            "memory_action": metadata.get("memory_action") or "none",
        }

    def _similarity_score(self, current: dict, candidate: dict, current_signature: str, candidate_signature: str) -> float:
        if current_signature == candidate_signature:
            return 2.0
        if current["primary_intent"] != candidate.get("primary_intent"):
            return 0.0

        current_agents = set(current["required_agents"])
        candidate_agents = set(candidate.get("required_agents", []))
        shared_agents = len(current_agents & candidate_agents)
        union_agents = len(current_agents | candidate_agents) or 1
        agent_similarity = shared_agents / union_agents

        score = 0.5 + agent_similarity
        if current["complexity"] == candidate.get("complexity"):
            score += 0.25
        if current["file_action"] == candidate.get("file_action"):
            score += 0.1
        if current["memory_action"] == candidate.get("memory_action"):
            score += 0.1
        return score

    def _rank_pattern(self, pattern: dict, similarity: float) -> float:
        success_rate = float(pattern.get("success_rate", 0.0) or 0.0)
        avg_retries = float(pattern.get("avg_retries", 0.0) or 0.0)
        total_runs = int(pattern.get("total_runs", 0) or 0)
        best_score = float(pattern.get("best_score", 0.0) or 0.0)
        failure_penalty = (1.0 - success_rate) * 0.25
        retry_penalty = min(avg_retries, 5.0) * 0.12
        experience_bonus = min(total_runs, 5) * 0.03
        best_score_bonus = max(best_score, 0.0) * 0.10
        return similarity + (success_rate * 1.2) - failure_penalty - retry_penalty + experience_bonus + best_score_bonus

    def _performance_score(self, status: str, retry_count: int, task_count: int) -> float:
        base = 1.0 if status == "completed" else 0.0
        return base - (retry_count * 0.1) - (max(task_count - 1, 0) * 0.02)

    def _load(self) -> dict:
        if not self.storage_path.exists():
            return {"patterns": {}}
        try:
            with open(self.storage_path, encoding="utf-8") as handle:
                data = json.load(handle)
            if "patterns" not in data:
                return {"patterns": {}}
            data["patterns"] = {
                signature: self._normalize_pattern(pattern)
                for signature, pattern in data["patterns"].items()
            }
            return data
        except Exception:
            return {"patterns": {}}

    def _save(self) -> None:
        with open(self.storage_path, "w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2)

    def _normalize_pattern(self, pattern: dict) -> dict:
        normalized = deepcopy(pattern)
        success_count = int(normalized.get("success_count", 0) or 0)
        failure_count = int(normalized.get("failure_count", 0) or 0)
        total_runs = int(normalized.get("total_runs", success_count + failure_count) or 0)
        total_runs = max(total_runs, success_count + failure_count, 1 if success_count or failure_count else 0)
        total_retries = int(normalized.get("total_retries", 0) or 0)

        normalized["total_runs"] = total_runs
        normalized["success_count"] = success_count
        normalized["failure_count"] = failure_count
        normalized["success_rate"] = success_count / max(total_runs, 1) if total_runs else 0.0
        normalized["total_retries"] = total_retries
        normalized["avg_retries"] = total_retries / max(total_runs, 1) if total_runs else 0.0
        normalized.setdefault("avg_task_count", 0.0)
        normalized.setdefault("best_score", None)
        normalized.setdefault("best_agent_sequence", normalized.get("agent_sequence", []))
        normalized.setdefault("examples", [])
        normalized.setdefault("task_overrides", {})
        normalized.setdefault("updated_at", None)
        normalized.setdefault("profile", self._profile_from_signature(normalized.get("signature", "")))
        return normalized

    def _profile_from_signature(self, signature: str) -> dict[str, Any]:
        parts = signature.split("|")
        if len(parts) < 5:
            return {
                "primary_intent": "unknown",
                "required_agents": [],
                "complexity": "unknown",
                "file_action": "none",
                "memory_action": "none",
            }
        primary_intent, complexity, agents_segment, file_segment, memory_segment = parts[:5]
        required_agents = [agent for agent in agents_segment.split(",") if agent]
        file_action = file_segment.split(":", 1)[1] if ":" in file_segment else "none"
        memory_action = memory_segment.split(":", 1)[1] if ":" in memory_segment else "none"
        return {
            "primary_intent": primary_intent,
            "required_agents": sorted(required_agents),
            "complexity": complexity,
            "file_action": file_action or "none",
            "memory_action": memory_action or "none",
        }
