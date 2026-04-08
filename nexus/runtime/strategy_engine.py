"""Adaptive retry strategy selection for NEXUS orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StrategyDecision:
    """Structured retry or fallback decision."""

    should_retry: bool
    strategy: str
    reason: str
    failure_type: str
    note: str | None = None
    next_agent: str | None = None
    candidate_agents: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    fallback_triggered: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StrategyEngine:
    """Choose how the runtime should adapt after a weak attempt."""

    def choose(
        self,
        *,
        task: Any,
        current_agent: str,
        attempts: int,
        max_attempts: int,
        observation: dict[str, Any],
        evaluation: dict[str, Any] | None = None,
        alternative_agents: list[str] | None = None,
        policy: dict[str, Any] | None = None,
        previous_strategy: str | None = None,
    ) -> StrategyDecision:
        failure_type = (
            observation.get("failure_type")
            or (evaluation or {}).get("failure_type")
            or "runtime_error"
        )
        policy = dict(policy or {})
        reason = policy.get("reason") or observation.get("summary") or (evaluation or {}).get("summary") or "retry requested"
        should_retry = bool(policy.get("should_retry", attempts < max_attempts))
        preferred_strategy = policy.get("preferred_strategy")

        alternatives = self._candidate_agents(task, current_agent, alternative_agents)
        if not should_retry:
            return self._with_policy(
                StrategyDecision(
                    should_retry=False,
                    strategy=preferred_strategy or previous_strategy or task.retry_strategy,
                    reason=reason,
                    failure_type=failure_type,
                    note=policy.get("note"),
                ),
                policy,
            )

        if policy.get("continue_current_strategy") and previous_strategy:
            if previous_strategy == "switch_agent" and alternatives:
                decision = self._switch_agent(
                    task=task,
                    current_agent=current_agent,
                    next_agent=alternatives[0],
                    reason=reason,
                    failure_type=failure_type,
                    alternatives=alternatives,
                    should_retry=should_retry,
                )
            elif previous_strategy == "plan_modification":
                decision = self._plan_modification(reason, failure_type, should_retry, note=policy.get("note"))
            elif previous_strategy == "simplify_task":
                decision = self._simplify(reason, failure_type, should_retry, note=policy.get("note"))
            else:
                decision = StrategyDecision(
                    should_retry=should_retry,
                    strategy=previous_strategy,
                    reason=reason,
                    failure_type=failure_type,
                    note=policy.get("note"),
                )
            return self._with_policy(decision, policy)

        if preferred_strategy == "switch_agent" and alternatives:
            decision = self._switch_agent(
                task=task,
                current_agent=current_agent,
                next_agent=alternatives[0],
                reason=reason,
                failure_type=failure_type,
                alternatives=alternatives,
                should_retry=should_retry,
            )
            return self._apply_policy_controls(decision, task, current_agent, alternatives, policy, previous_strategy)
        if preferred_strategy == "plan_modification":
            decision = self._plan_modification(reason, failure_type, should_retry, note=policy.get("note"))
            return self._apply_policy_controls(decision, task, current_agent, alternatives, policy, previous_strategy)
        if preferred_strategy == "simplify_task":
            decision = self._simplify(reason, failure_type, should_retry, note=policy.get("note"))
            return self._apply_policy_controls(decision, task, current_agent, alternatives, policy, previous_strategy)

        if failure_type in {"low_confidence", "logic_error", "framework_mismatch"}:
            if alternatives:
                decision = self._switch_agent(
                    task=task,
                    current_agent=current_agent,
                    next_agent=alternatives[0],
                    reason=reason,
                    failure_type=failure_type,
                    alternatives=alternatives,
                    should_retry=should_retry,
                )
                return self._apply_policy_controls(decision, task, current_agent, alternatives, policy, previous_strategy)
            decision = StrategyDecision(
                should_retry=should_retry,
                strategy="plan_modification",
                reason=reason,
                failure_type=failure_type,
                note=(
                    "Re-plan before answering: break the task into a few explicit checks, "
                    "verify each step against shared memory, and keep the response grounded."
                ),
                metadata={"mode": "replan"},
            )
            return self._apply_policy_controls(decision, task, current_agent, alternatives, policy, previous_strategy)

        if failure_type in {"timeout", "inefficient"}:
            decision = StrategyDecision(
                should_retry=should_retry,
                strategy="simplify_task",
                reason=reason,
                failure_type=failure_type,
                note=(
                    "Reduce scope on the retry: produce the smallest viable answer, "
                    "skip optional detail, and favor a short grounded result."
                ),
                metadata={"mode": "simplify"},
            )
            return self._apply_policy_controls(decision, task, current_agent, alternatives, policy, previous_strategy)

        if failure_type == "safety_risk":
            if task.fallback_agent and task.fallback_agent != current_agent:
                decision = self._switch_agent(
                    task=task,
                    current_agent=current_agent,
                    next_agent=task.fallback_agent,
                    reason=reason,
                    failure_type=failure_type,
                    alternatives=alternatives,
                    should_retry=should_retry,
                )
                return self._apply_policy_controls(decision, task, current_agent, alternatives, policy, previous_strategy)
            decision = StrategyDecision(
                should_retry=should_retry,
                strategy="plan_modification",
                reason=reason,
                failure_type=failure_type,
                note=(
                    "Retry in safety-first mode: avoid destructive or irreversible actions, "
                    "prefer reversible guidance, and surface uncertainty instead of guessing."
                ),
                metadata={"mode": "safety"},
            )
            return self._apply_policy_controls(decision, task, current_agent, alternatives, policy, previous_strategy)

        if task.retry_strategy == "fallback" and task.fallback_agent and task.fallback_agent != current_agent:
            decision = self._switch_agent(
                task=task,
                current_agent=current_agent,
                next_agent=task.fallback_agent,
                reason=reason,
                failure_type=failure_type,
                alternatives=alternatives,
                should_retry=should_retry,
            )
            return self._apply_policy_controls(decision, task, current_agent, alternatives, policy, previous_strategy)
        if task.retry_strategy == "switch_agent" and alternatives:
            decision = self._switch_agent(
                task=task,
                current_agent=current_agent,
                next_agent=alternatives[0],
                reason=reason,
                failure_type=failure_type,
                alternatives=alternatives,
                should_retry=should_retry,
            )
            return self._apply_policy_controls(decision, task, current_agent, alternatives, policy, previous_strategy)

        decision = StrategyDecision(
            should_retry=should_retry,
            strategy=task.retry_strategy,
            reason=reason,
            failure_type=failure_type,
            note=(
                task.fallback
                or "Retry conservatively, stay grounded in shared memory, and surface uncertainty when needed."
            ),
        )
        return self._apply_policy_controls(decision, task, current_agent, alternatives, policy, previous_strategy)

    def _candidate_agents(
        self,
        task: Any,
        current_agent: str,
        alternative_agents: list[str] | None,
    ) -> list[str]:
        ordered = []
        if task.fallback_agent and task.fallback_agent != current_agent:
            ordered.append(task.fallback_agent)
        for candidate in alternative_agents or []:
            if candidate != current_agent and candidate not in ordered:
                ordered.append(candidate)
        return ordered

    def _switch_agent(
        self,
        *,
        task: Any,
        current_agent: str,
        next_agent: str,
        reason: str,
        failure_type: str,
        alternatives: list[str],
        should_retry: bool,
    ) -> StrategyDecision:
        return StrategyDecision(
            should_retry=should_retry,
            strategy="switch_agent",
            reason=reason,
            failure_type=failure_type,
            note=(
                f"The previous attempt from '{current_agent}' was weak. "
                f"Retry with '{next_agent}', stay grounded in shared memory, and explain only what you can support."
            ),
            next_agent=next_agent,
            candidate_agents=alternatives,
            required_capabilities=list(getattr(task, "required_capabilities", []) or []),
            fallback_triggered=bool(next_agent == task.fallback_agent and next_agent != task.agent),
        )

    def _plan_modification(
        self,
        reason: str,
        failure_type: str,
        should_retry: bool,
        *,
        note: str | None = None,
    ) -> StrategyDecision:
        return StrategyDecision(
            should_retry=should_retry,
            strategy="plan_modification",
            reason=reason,
            failure_type=failure_type,
            note=note
            or (
                "Re-plan the task into a few explicit checks, stay grounded in shared memory, "
                "and prefer verified output over speed."
            ),
            metadata={"mode": "replan"},
        )

    def _simplify(
        self,
        reason: str,
        failure_type: str,
        should_retry: bool,
        *,
        note: str | None = None,
    ) -> StrategyDecision:
        return StrategyDecision(
            should_retry=should_retry,
            strategy="simplify_task",
            reason=reason,
            failure_type=failure_type,
            note=note
            or (
                "Reduce scope on the retry, skip optional detail, and aim for the smallest grounded answer."
            ),
            metadata={"mode": "simplify"},
        )

    def _apply_policy_controls(
        self,
        decision: StrategyDecision,
        task: Any,
        current_agent: str,
        alternatives: list[str],
        policy: dict[str, Any],
        previous_strategy: str | None,
    ) -> StrategyDecision:
        if policy.get("force_strategy_change") and previous_strategy and decision.strategy == previous_strategy:
            if previous_strategy != "switch_agent" and alternatives:
                decision = self._switch_agent(
                    task=task,
                    current_agent=current_agent,
                    next_agent=alternatives[0],
                    reason=policy.get("reason", decision.reason),
                    failure_type=decision.failure_type,
                    alternatives=alternatives,
                    should_retry=decision.should_retry,
                )
            elif previous_strategy != "plan_modification":
                decision = self._plan_modification(
                    policy.get("reason", decision.reason),
                    decision.failure_type,
                    decision.should_retry,
                    note=policy.get("note"),
                )
            elif previous_strategy != "simplify_task":
                decision = self._simplify(
                    policy.get("reason", decision.reason),
                    decision.failure_type,
                    decision.should_retry,
                    note=policy.get("note"),
                )

        if not policy.get("trust_memory", True):
            memory_note = "Recompute from live shared context instead of leaning on remembered workflow defaults."
            decision.note = f"{decision.note} {memory_note}".strip() if decision.note else memory_note
        return self._with_policy(decision, policy)

    def _with_policy(self, decision: StrategyDecision, policy: dict[str, Any]) -> StrategyDecision:
        decision.metadata.update(
            {
                "policy_action": policy.get("action"),
                "policy_mode": policy.get("mode"),
                "force_strategy_change": bool(policy.get("force_strategy_change", False)),
                "continue_current_strategy": bool(policy.get("continue_current_strategy", False)),
            }
        )
        return decision
