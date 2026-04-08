"""
Blueprint generator for the NEXUS agent compiler.

Produces a JSON-serializable workflow blueprint from either an intent profile
or a richer execution plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from nexus.intent_parser import IntentProfile


@dataclass
class TaskBlueprint:
    """A single executable step in a workflow."""

    id: str
    task_type: str
    agent: str | None
    instruction: str
    depends_on: list[str] = field(default_factory=list)
    retries: int = 1
    timeout_seconds: int = 20
    optional: bool = False
    output_key: str | None = None
    fallback: str | None = None
    fallback_agent: str | None = None
    retry_strategy: str = "repeat"
    confidence_threshold: float = 0.0
    required_capabilities: list[str] = field(default_factory=list)
    candidate_agents: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorkflowBlueprint:
    """Compiled workflow representation."""

    goal: str
    primary_intent: str
    tasks: list[TaskBlueprint]
    metadata: dict = field(default_factory=dict)
    version: str = "1.1"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tasks"] = [task.to_dict() for task in self.tasks]
        return payload


class BlueprintGenerator:
    """Compile an intent or plan into an executable task blueprint."""

    def generate(self, source: Any) -> WorkflowBlueprint:
        plan = self._coerce_plan(source)
        tasks = [
            TaskBlueprint(
                id=task.id,
                task_type=task.task_type,
                agent=task.agent,
                instruction=task.objective,
                depends_on=list(task.depends_on),
                retries=task.retries,
                timeout_seconds=task.timeout_seconds,
                optional=task.optional,
                output_key=task.output_key,
                fallback=task.fallback,
                fallback_agent=task.fallback_agent,
                retry_strategy=task.retry_strategy,
                confidence_threshold=task.confidence_threshold,
                required_capabilities=list(task.required_capabilities),
                candidate_agents=list(task.candidate_agents),
                metadata=dict(task.metadata),
            )
            for task in plan.tasks
        ]

        metadata = {
            "compiled_at": datetime.now(timezone.utc).isoformat(),
            "complexity": plan.metadata.get("complexity"),
            "constraints": list(plan.constraints),
            "required_agents": list(plan.metadata.get("required_agents", [])),
            "deliverables": list(plan.metadata.get("deliverables", [])),
            "plan_signature": plan.metadata.get("signature"),
            "skill_pattern_available": bool(plan.metadata.get("skill_pattern_available")),
            "skill_pattern_reused": bool(plan.metadata.get("skill_pattern_reused")),
            "skill_pattern_reuse_mode": plan.metadata.get("skill_pattern_reuse_mode"),
            "skill_pattern_success_rate": plan.metadata.get("skill_pattern_success_rate", 0.0),
            "skill_pattern_avg_retries": plan.metadata.get("skill_pattern_avg_retries", 0.0),
            "skill_pattern_rank_score": plan.metadata.get("skill_pattern_rank_score", 0.0),
            "file_action": plan.metadata.get("file_action", "none"),
            "memory_action": plan.metadata.get("memory_action", "none"),
            "project_mode": bool(plan.metadata.get("project_mode", False)),
            "project_root": plan.metadata.get("project_root"),
            "project_signature": plan.metadata.get("project_signature"),
            "project_frameworks": list(plan.metadata.get("project_frameworks", [])),
            "project_languages": list(plan.metadata.get("project_languages", [])),
            "project_recent_goals": list(plan.metadata.get("project_recent_goals", [])),
            "project_common_errors": list(plan.metadata.get("project_common_errors", [])),
            "project_preferences": dict(plan.metadata.get("project_preferences", {})),
            "goal": plan.goal,
        }
        return WorkflowBlueprint(
            goal=plan.goal,
            primary_intent=plan.primary_intent,
            tasks=tasks,
            metadata=metadata,
        )

    def _coerce_plan(self, source: Any):
        from nexus.compiler.planner_engine import ExecutionPlan, PlannerEngine

        if isinstance(source, ExecutionPlan):
            return source
        if isinstance(source, IntentProfile):
            return PlannerEngine().plan(source)
        if hasattr(source, "tasks") and hasattr(source, "goal"):
            return source
        raise TypeError(f"Unsupported blueprint source: {type(source)!r}")
