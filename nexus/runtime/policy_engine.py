"""Policy-guided control layer for NEXUS runtime decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(float(value), 0.0) for value in weights.values()) or 1.0
    return {
        critic: max(float(weight), 0.0) / total
        for critic, weight in weights.items()
    }


@dataclass
class PolicyProfile:
    """Static policy configuration for one task execution."""

    mode: str
    critic_weights: dict[str, float]
    confidence_target: float
    retry_budget: dict[str, Any]
    trust_memory: bool
    allow_cached_decisions: bool = True
    cache_min_confidence: float = 0.8
    trace_verbosity: str = "full"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyDecision:
    """Runtime policy decision after evaluation."""

    mode: str
    action: str
    should_retry: bool
    reason: str
    failure_type: str | None = None
    critic_weights: dict[str, float] = field(default_factory=dict)
    confidence_target: float = 0.0
    retry_budget: dict[str, Any] = field(default_factory=dict)
    trust_memory: bool = True
    force_strategy_change: bool = False
    continue_current_strategy: bool = False
    preferred_strategy: str | None = None
    note: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PolicyEngine:
    """Global control layer for evaluation weighting, budgets, and retry stability."""

    VALID_MODES = {"stable", "explore"}

    def __init__(self, mode: str = "stable"):
        self.mode = mode if mode in self.VALID_MODES else "stable"

    def build_profile(
        self,
        *,
        task: Any,
        blueprint: Any,
        max_attempts: int,
    ) -> PolicyProfile:
        """Create the policy profile that should govern a task."""
        return PolicyProfile(
            mode=self.mode,
            critic_weights=self._critic_weights(task),
            confidence_target=self._confidence_target(task),
            retry_budget={
                "max_attempts": max(1, int(max_attempts)),
                "max_retries": max(0, int(max_attempts) - 1),
                "max_time_seconds": round(
                    max(1.0, float(getattr(task, "timeout_seconds", 20) or 20.0))
                    * max(1, int(max_attempts))
                    * (1.0 if self.mode == "stable" else 1.35),
                    2,
                ),
            },
            trust_memory=self._trust_memory(blueprint),
            allow_cached_decisions=True,
            cache_min_confidence=0.88 if self.mode == "stable" else 0.82,
            trace_verbosity=self.trace_verbosity(),
            metadata={
                "task_type": getattr(task, "task_type", "generic"),
                "task_id": getattr(task, "id", "unknown"),
            },
        )

    def decide(
        self,
        *,
        task: Any,
        blueprint: Any,
        profile: PolicyProfile,
        attempts: int,
        elapsed_seconds: float,
        observation: dict[str, Any],
        evaluation: dict[str, Any] | None,
        history: list[dict[str, Any]],
        current_strategy: str | None,
    ) -> PolicyDecision:
        """Decide whether runtime should stop, retry, or force a strategy change."""
        del blueprint  # reserved for future policy expansion without widening the signature

        failure_type = (
            observation.get("failure_type")
            or (evaluation or {}).get("failure_type")
        )
        current_score = float((evaluation or {}).get("score", observation.get("confidence", 0.0)) or 0.0)
        previous_score = self._last_value(history, "evaluation_score")
        score_delta = current_score - previous_score if previous_score is not None else None
        repeated_failure = self._recent_repeat(history, "failure_type", failure_type, count=2)
        repeated_strategy = self._recent_repeat(history, "strategy", current_strategy, count=2)
        improving = (
            previous_score is not None
            and score_delta is not None
            and score_delta >= (0.03 if self.mode == "explore" else 0.05)
        )

        within_attempt_budget = attempts < int(profile.retry_budget["max_attempts"])
        within_time_budget = elapsed_seconds < float(profile.retry_budget["max_time_seconds"])
        note = None
        preferred_strategy = None
        force_strategy_change = False
        continue_current_strategy = False

        if observation.get("ok", False) and ((evaluation or {}).get("ok", True)):
            action = "complete"
            should_retry = False
            reason = "Policy accepted the current result."
        elif not within_attempt_budget:
            action = "stop"
            should_retry = False
            reason = f"Retry budget exhausted after {attempts} attempts."
        elif not within_time_budget:
            action = "stop"
            should_retry = False
            reason = (
                "Task exceeded the policy time budget "
                f"({elapsed_seconds:.2f}s / {profile.retry_budget['max_time_seconds']:.2f}s)."
            )
            preferred_strategy = "simplify_task"
        elif failure_type == "safety_risk" and self.mode == "stable":
            safety_score = float((evaluation or {}).get("critic_scores", {}).get("safety", 1.0) or 1.0)
            if safety_score < 0.25:
                action = "stop"
                should_retry = False
                reason = "Stable mode stopped after a severe safety risk."
                preferred_strategy = "plan_modification"
            else:
                action = "retry_with_strategy_change"
                should_retry = True
                reason = "Stable mode requested a safer retry path."
                preferred_strategy = "plan_modification"
                force_strategy_change = True
        elif repeated_failure and repeated_strategy and not improving:
            action = "retry_with_strategy_change"
            should_retry = True
            reason = f"Policy detected repeated {failure_type or 'runtime'} failures without improvement."
            preferred_strategy = "switch_agent" if current_strategy != "switch_agent" else "plan_modification"
            force_strategy_change = True
        elif improving and current_strategy not in {None, "", "none", "switch_agent"}:
            action = "retry_current_strategy"
            should_retry = True
            reason = f"Policy kept the current strategy because score improved by {score_delta:.2f}."
            preferred_strategy = current_strategy
            continue_current_strategy = True
        elif not profile.trust_memory and failure_type in {"low_confidence", "logic_error"}:
            action = "retry_with_strategy_change"
            should_retry = True
            reason = "Policy reduced trust in remembered workflow defaults for this retry."
            preferred_strategy = "plan_modification"
            force_strategy_change = True
            note = "Recompute from live shared context instead of leaning on remembered workflow defaults."
        else:
            action = "retry"
            should_retry = True
            reason = f"Policy approved another attempt for {failure_type or 'recovery'} within budget."
            if failure_type in {"timeout", "inefficient"}:
                preferred_strategy = "simplify_task"

        return PolicyDecision(
            mode=profile.mode,
            action=action,
            should_retry=should_retry,
            reason=reason,
            failure_type=failure_type,
            critic_weights=dict(profile.critic_weights),
            confidence_target=profile.confidence_target,
            retry_budget=dict(profile.retry_budget),
            trust_memory=profile.trust_memory,
            force_strategy_change=force_strategy_change,
            continue_current_strategy=continue_current_strategy,
            preferred_strategy=preferred_strategy,
            note=note,
            metadata={
                "attempt": attempts,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "current_strategy": current_strategy,
                "previous_score": previous_score,
                "score_delta": round(score_delta, 3) if score_delta is not None else None,
                "repeated_failure": repeated_failure,
                "repeated_strategy": repeated_strategy,
                "improving": improving,
            },
        )

    def _critic_weights(self, task: Any) -> dict[str, float]:
        task_type = getattr(task, "task_type", "generic")
        if task_type in {"solution", "file_write"}:
            weights = {"correctness": 0.6, "efficiency": 0.3, "safety": 0.1}
        elif task_type in {"canary_protection"}:
            weights = {"correctness": 0.35, "efficiency": 0.15, "safety": 0.5}
        elif task_type.startswith("memory"):
            weights = {"correctness": 0.55, "efficiency": 0.15, "safety": 0.3}
        elif task_type in {"research_context", "file_context"}:
            weights = {"correctness": 0.55, "efficiency": 0.15, "safety": 0.3}
        else:
            weights = {"correctness": 0.55, "efficiency": 0.2, "safety": 0.25}

        if self.mode == "stable":
            weights["safety"] += 0.1
            weights["efficiency"] -= 0.05
        else:
            weights["efficiency"] += 0.1
            weights["safety"] -= 0.05
        return _normalize_weights(weights)

    def _confidence_target(self, task: Any) -> float:
        target = float(getattr(task, "confidence_threshold", 0.0) or 0.0)
        if target <= 0:
            return 0.0
        if self.mode == "stable":
            return min(target + 0.05, 0.95)
        return max(target - 0.05, 0.0)

    def _trust_memory(self, blueprint: Any) -> bool:
        metadata = getattr(blueprint, "metadata", {}) or {}
        if not metadata.get("skill_pattern_available"):
            return True
        if not metadata.get("skill_pattern_reused"):
            return False

        success_rate = float(metadata.get("skill_pattern_success_rate", 0.0) or 0.0)
        avg_retries = float(metadata.get("skill_pattern_avg_retries", 0.0) or 0.0)
        required_success = 0.8 if self.mode == "stable" else 0.65
        allowed_retries = 1.25 if self.mode == "stable" else 2.0
        return success_rate >= required_success and avg_retries <= allowed_retries

    def _last_value(self, history: list[dict[str, Any]], key: str) -> float | None:
        for entry in reversed(history):
            value = entry.get(key)
            if value is not None:
                return float(value)
        return None

    def trace_verbosity(self) -> str:
        """Return the desired trace detail for the current mode."""
        return "compact" if self.mode == "stable" else "full"

    def parallel_plan(
        self,
        *,
        ready_tasks: list[Any],
        blueprint: Any,
    ) -> dict[str, Any]:
        """Decide whether a ready task batch should run in parallel."""
        if len(ready_tasks) < 2:
            return {
                "parallel": False,
                "task_ids": [ready_tasks[0].id] if ready_tasks else [],
                "reason": "Only one task is ready.",
                "max_parallel_tasks": 1,
            }

        max_parallel_tasks = 2 if self.mode == "stable" else 3
        safe_context_types = {"memory_recall", "file_context", "research_context"}
        selected: list[str] = []
        used_agents: set[str] = set()

        for task in ready_tasks:
            primary_agent = getattr(task, "agent", None)
            if not primary_agent and len(getattr(task, "candidate_agents", []) or []) == 1:
                primary_agent = task.candidate_agents[0]
            if not primary_agent or primary_agent in used_agents:
                continue

            task_type = getattr(task, "task_type", "generic")
            if self.mode == "stable":
                if task_type not in safe_context_types:
                    continue
            elif task_type in {"solution", "file_write", "memory_store"}:
                continue

            selected.append(task.id)
            used_agents.add(primary_agent)
            if len(selected) >= max_parallel_tasks:
                break

        if len(selected) < 2:
            return {
                "parallel": False,
                "task_ids": [ready_tasks[0].id],
                "reason": "Policy kept execution sequential because the ready tasks were not a safe parallel batch.",
                "max_parallel_tasks": max_parallel_tasks,
            }

        return {
            "parallel": True,
            "task_ids": selected,
            "reason": f"Policy approved a safe parallel batch of {len(selected)} independent tasks.",
            "max_parallel_tasks": max_parallel_tasks,
        }

    def _recent_repeat(
        self,
        history: list[dict[str, Any]],
        key: str,
        expected: Any,
        *,
        count: int,
    ) -> bool:
        if expected in {None, ""} or len(history) < count:
            return False
        recent = history[-count:]
        return all(entry.get(key) == expected for entry in recent)
