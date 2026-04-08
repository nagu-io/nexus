"""Lightweight cache for successful runtime decisions."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class DecisionCache:
    """Persist high-confidence task decisions for reuse on similar runs."""

    DECAY_PER_DAY = 0.985
    STALE_AFTER_DAYS = 21
    INVALIDATE_AFTER_DAYS = 45
    INVALIDATE_AFTER_FAILURES = 2
    SIMILARITY_THRESHOLD = 0.55
    TOKEN_STOP_WORDS = {"a", "an", "and", "the", "to", "for", "of", "in", "on", "with", "this", "that"}

    def __init__(self, storage_path: Path | None = None):
        from nexus.config import config

        self.storage_path = Path(storage_path) if storage_path else config.data_dir / "decision_cache.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def signature_for(
        self,
        *,
        task: Any,
        blueprint: Any,
        agent: str,
        strategy: str,
        mode: str,
    ) -> str:
        """Build a stable cache key for a task execution shape."""
        metadata = getattr(blueprint, "metadata", {}) or {}
        payload = {
            "plan_signature": metadata.get("plan_signature"),
            "primary_intent": getattr(blueprint, "primary_intent", "unknown"),
            "project_signature": metadata.get("project_signature"),
            "task_id": getattr(task, "id", "unknown"),
            "task_type": getattr(task, "task_type", "generic"),
            "instruction": getattr(task, "instruction", ""),
            "agent": agent,
            "strategy": strategy,
            "mode": mode,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()

    def lookup(
        self,
        *,
        task: Any,
        blueprint: Any,
        agent: str,
        strategy: str,
        mode: str,
        min_confidence: float,
    ) -> dict[str, Any] | None:
        """Return a reusable cached decision when confidence is high enough."""
        signature = self.signature_for(
            task=task,
            blueprint=blueprint,
            agent=agent,
            strategy=strategy,
            mode=mode,
        )
        entry = self._normalize_entry(self._data["entries"].get(signature))
        if not entry:
            entry = self._lookup_similar(
                task=task,
                blueprint=blueprint,
                agent=agent,
                strategy=strategy,
                mode=mode,
                min_confidence=min_confidence,
            )
            if not entry:
                return None
            signature = entry["signature"]
        changed = self._refresh_entry_state(entry)
        if changed:
            self._data["entries"][signature] = entry

        if entry.get("invalidated", False):
            if changed:
                self._save()
            return None

        success_rate = float(entry.get("success_rate", 0.0) or 0.0)
        expected_confidence = self._effective_confidence(entry)
        avg_attempts = float(entry.get("avg_attempts", 0.0) or 0.0)
        if success_rate < 0.85 or expected_confidence < min_confidence or avg_attempts > 1.5:
            if changed:
                self._save()
            return None

        entry["cache_hits"] = int(entry.get("cache_hits", 0) or 0) + 1
        entry["last_used_at"] = self._utc_now()
        self._data["entries"][signature] = entry
        self._save()

        cached = deepcopy(entry)
        cached["signature"] = signature
        cached["decayed_confidence"] = expected_confidence
        return cached

    def record(
        self,
        *,
        task: Any,
        blueprint: Any,
        agent: str,
        strategy: str,
        mode: str,
        status: str,
        attempts: int,
        evaluation: dict[str, Any] | None,
        reflection: dict[str, Any] | None = None,
    ) -> None:
        """Persist a task outcome for future reuse."""
        metadata = getattr(blueprint, "metadata", {}) or {}
        signature = self.signature_for(
            task=task,
            blueprint=blueprint,
            agent=agent,
            strategy=strategy,
            mode=mode,
        )
        entries = self._data["entries"]
        entry = entries.setdefault(
            signature,
            {
                "signature": signature,
                "task_type": getattr(task, "task_type", "generic"),
                "project_signature": metadata.get("project_signature"),
                "agent": agent,
                "strategy": strategy,
                "mode": mode,
                "instruction_tokens": self._instruction_tokens(getattr(task, "instruction", "")),
                "total_runs": 0,
                "success_count": 0,
                "failure_count": 0,
                "success_rate": 0.0,
                "avg_attempts": 0.0,
                "expected_confidence": 0.0,
                "critic_scores": {},
                "weights_used": {},
                "last_failure_type": None,
                "consecutive_failures": 0,
                "cache_hits": 0,
                "last_used_at": None,
                "decay_factor": 1.0,
                "invalidated": False,
                "invalidation_reason": None,
                "updated_at": None,
            },
        )
        entry = self._normalize_entry(entry) or entry
        entries[signature] = entry

        entry["total_runs"] += 1
        prior_attempt_total = float(entry.get("avg_attempts", 0.0)) * max(entry["total_runs"] - 1, 0)
        entry["avg_attempts"] = (prior_attempt_total + max(1, int(attempts))) / max(entry["total_runs"], 1)

        if status == "completed":
            entry["success_count"] += 1
            confidence = float((evaluation or {}).get("confidence", 0.0) or 0.0)
            previous_confidence_total = float(entry.get("expected_confidence", 0.0) or 0.0) * max(entry["success_count"] - 1, 0)
            entry["expected_confidence"] = (
                (previous_confidence_total + confidence) / max(entry["success_count"], 1)
                if entry["success_count"]
                else 0.0
            )
            if evaluation:
                entry["critic_scores"] = dict(evaluation.get("critic_scores", {}))
                entry["weights_used"] = dict(evaluation.get("weights_used", {}))
            entry["last_failure_type"] = None
            entry["consecutive_failures"] = 0
            entry["invalidated"] = False
            entry["invalidation_reason"] = None
        else:
            entry["failure_count"] += 1
            entry["last_failure_type"] = (
                (evaluation or {}).get("failure_type")
                or (reflection or {}).get("failure_type")
            )
            entry["consecutive_failures"] = int(entry.get("consecutive_failures", 0) or 0) + 1
            if entry["success_count"] and entry["consecutive_failures"] >= self.INVALIDATE_AFTER_FAILURES:
                entry["invalidated"] = True
                entry["invalidation_reason"] = "repeated_failures"

        entry["success_rate"] = entry["success_count"] / max(entry["total_runs"], 1)
        entry["decay_factor"] = 1.0
        entry["updated_at"] = self._utc_now()
        if status == "completed":
            entry["last_used_at"] = entry["updated_at"]
        self._save()

    def summary(self, limit: int = 5) -> dict[str, Any]:
        """Return lightweight cache-health metrics for UI surfaces."""
        self.reload()
        entries = []
        changed = False
        for signature, raw_entry in self._data["entries"].items():
            entry = self._normalize_entry(raw_entry)
            if not entry:
                continue
            entry_changed = self._refresh_entry_state(entry)
            changed = changed or entry_changed
            reusable = (
                not entry.get("invalidated", False)
                and float(entry.get("success_rate", 0.0) or 0.0) >= 0.85
                and float(entry.get("avg_attempts", 0.0) or 0.0) <= 1.5
                and self._effective_confidence(entry) >= 0.75
            )
            payload = {
                "signature": signature,
                "task_type": entry.get("task_type", "generic"),
                "agent": entry.get("agent"),
                "strategy": entry.get("strategy"),
                "mode": entry.get("mode", "stable"),
                "success_rate": float(entry.get("success_rate", 0.0) or 0.0),
                "expected_confidence": float(entry.get("expected_confidence", 0.0) or 0.0),
                "decayed_confidence": self._effective_confidence(entry),
                "avg_attempts": float(entry.get("avg_attempts", 0.0) or 0.0),
                "total_runs": int(entry.get("total_runs", 0) or 0),
                "cache_hits": int(entry.get("cache_hits", 0) or 0),
                "decay_factor": float(entry.get("decay_factor", 1.0) or 1.0),
                "last_used_at": entry.get("last_used_at"),
                "updated_at": entry.get("updated_at"),
                "invalidated": bool(entry.get("invalidated", False)),
                "invalidation_reason": entry.get("invalidation_reason"),
                "reusable": reusable,
            }
            entries.append(payload)
            self._data["entries"][signature] = entry

        if changed:
            self._save()

        ranked = sorted(
            entries,
            key=lambda item: (
                item["reusable"],
                item["decayed_confidence"],
                item["success_rate"],
                item["total_runs"],
            ),
            reverse=True,
        )
        return {
            "total_entries": len(entries),
            "reusable_entries": sum(1 for entry in entries if entry["reusable"]),
            "invalidated_entries": sum(1 for entry in entries if entry["invalidated"]),
            "top_entries": ranked[: max(1, int(limit))],
        }

    def reload(self) -> None:
        """Refresh in-memory state from disk for multi-process readers."""
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.storage_path.exists():
            return {"entries": {}}
        try:
            with open(self.storage_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if "entries" not in payload:
                return {"entries": {}}
            payload["entries"] = {
                signature: self._normalize_entry(entry)
                for signature, entry in payload["entries"].items()
                if self._normalize_entry(entry)
            }
            return payload
        except Exception:
            return {"entries": {}}

    def _save(self) -> None:
        with open(self.storage_path, "w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2)

    def _normalize_entry(self, entry: dict[str, Any] | None) -> dict[str, Any] | None:
        if not entry:
            return None
        normalized = deepcopy(entry)
        total_runs = int(normalized.get("total_runs", 0) or 0)
        success_count = int(normalized.get("success_count", 0) or 0)
        failure_count = int(normalized.get("failure_count", 0) or 0)
        total_runs = max(total_runs, success_count + failure_count, 1 if success_count or failure_count else 0)
        normalized["total_runs"] = total_runs
        normalized["success_count"] = success_count
        normalized["failure_count"] = failure_count
        normalized["success_rate"] = success_count / max(total_runs, 1) if total_runs else 0.0
        normalized.setdefault("project_signature", None)
        normalized.setdefault("instruction_tokens", [])
        normalized.setdefault("avg_attempts", 0.0)
        normalized.setdefault("expected_confidence", 0.0)
        normalized.setdefault("critic_scores", {})
        normalized.setdefault("weights_used", {})
        normalized.setdefault("last_failure_type", None)
        normalized.setdefault("consecutive_failures", 0)
        normalized.setdefault("cache_hits", 0)
        normalized.setdefault("last_used_at", normalized.get("updated_at"))
        normalized.setdefault("decay_factor", 1.0)
        normalized.setdefault("invalidated", False)
        normalized.setdefault("invalidation_reason", None)
        normalized.setdefault("updated_at", None)
        return normalized

    def _lookup_similar(
        self,
        *,
        task: Any,
        blueprint: Any,
        agent: str,
        strategy: str,
        mode: str,
        min_confidence: float,
    ) -> dict[str, Any] | None:
        metadata = getattr(blueprint, "metadata", {}) or {}
        target_project_signature = metadata.get("project_signature")
        target_task_type = getattr(task, "task_type", "generic")
        target_tokens = self._instruction_tokens(getattr(task, "instruction", ""))
        candidates = []
        for signature, raw_entry in self._data["entries"].items():
            entry = self._normalize_entry(raw_entry)
            if not entry:
                continue
            if entry.get("agent") != agent or entry.get("mode") != mode:
                continue
            if entry.get("task_type") != target_task_type:
                continue
            if target_project_signature and entry.get("project_signature") != target_project_signature:
                continue
            similarity = self._token_similarity(target_tokens, entry.get("instruction_tokens", []))
            if similarity < self.SIMILARITY_THRESHOLD:
                continue
            if self._refresh_entry_state(entry):
                self._data["entries"][signature] = entry
            if entry.get("invalidated", False):
                continue
            success_rate = float(entry.get("success_rate", 0.0) or 0.0)
            expected_confidence = self._effective_confidence(entry)
            avg_attempts = float(entry.get("avg_attempts", 0.0) or 0.0)
            if success_rate < 0.85 or expected_confidence < min_confidence or avg_attempts > 1.5:
                continue
            candidate = deepcopy(entry)
            candidate["signature"] = signature
            candidate["match_type"] = "similar"
            candidates.append((similarity, expected_confidence, success_rate, candidate))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return candidates[0][3]

    def _refresh_entry_state(self, entry: dict[str, Any]) -> bool:
        changed = False
        age = self._age_since(entry.get("last_used_at") or entry.get("updated_at"))
        if age is not None:
            decay_factor = max(0.2, self.DECAY_PER_DAY ** max(age.days, 0))
            if abs(float(entry.get("decay_factor", 1.0) or 1.0) - decay_factor) > 1e-9:
                entry["decay_factor"] = decay_factor
                changed = True
            if age >= timedelta(days=self.INVALIDATE_AFTER_DAYS) and not entry.get("invalidated", False):
                entry["invalidated"] = True
                entry["invalidation_reason"] = "stale_entry"
                changed = True
        return changed

    def _effective_confidence(self, entry: dict[str, Any]) -> float:
        base_confidence = float(entry.get("expected_confidence", 0.0) or 0.0)
        decay_factor = float(entry.get("decay_factor", 1.0) or 1.0)
        return base_confidence * decay_factor

    def _instruction_tokens(self, value: str) -> list[str]:
        cleaned = "".join(character.lower() if character.isalnum() else " " for character in value)
        tokens = [
            token
            for token in cleaned.split()
            if len(token) > 2 and token not in self.TOKEN_STOP_WORDS
        ]
        return sorted(dict.fromkeys(tokens))

    def _token_similarity(self, left: list[str], right: list[str]) -> float:
        if not left or not right:
            return 0.0
        left_set = set(left)
        right_set = set(right)
        return len(left_set & right_set) / max(1, len(left_set | right_set))

    def _age_since(self, value: str | None) -> timedelta | None:
        if not value:
            return None
        try:
            timestamp = datetime.fromisoformat(value)
        except ValueError:
            return None
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
