"""
Structured planning engine for the NEXUS compiler layer.

Turns an intent profile into a richer execution plan with constraints,
dependencies, fallback behavior, and reusable skill-pattern overrides.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from nexus.intent_parser import IntentProfile
from nexus.memory.skill_memory import SkillMemory


@dataclass
class PlannedTask:
    """A structured plan step before blueprint compilation."""

    id: str
    task_type: str
    objective: str
    agent: str | None
    depends_on: list[str] = field(default_factory=list)
    candidate_agents: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    retries: int = 1
    retry_strategy: str = "repeat"
    fallback: str | None = None
    fallback_agent: str | None = None
    confidence_threshold: float = 0.0
    timeout_seconds: int = 20
    optional: bool = False
    output_key: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutionPlan:
    """Structured execution plan produced by the planning engine."""

    goal: str
    primary_intent: str
    constraints: list[str]
    tasks: list[PlannedTask]
    metadata: dict = field(default_factory=dict)
    version: str = "1.0"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tasks"] = [task.to_dict() for task in self.tasks]
        return payload


class PlannerEngine:
    """Compile an intent profile into a reusable execution plan."""

    def __init__(self, skill_memory: SkillMemory | None = None):
        self.skill_memory = skill_memory or SkillMemory()

    def plan(self, intent: IntentProfile, project_context: dict | None = None) -> ExecutionPlan:
        project_context = project_context or intent.metadata.get("project_context")
        signature = self.skill_memory.signature_for_intent(intent)
        remembered_pattern = self.skill_memory.lookup(intent)
        reuse_mode = self._pattern_reuse_mode(remembered_pattern)
        tasks: list[PlannedTask] = []

        memory_action = intent.metadata.get("memory_action")
        file_action = intent.metadata.get("file_action")
        required_agents = list(intent.required_agents)

        if "memory" in required_agents and memory_action == "recall":
            tasks.append(
                self._task(
                    task_type="memory_recall",
                    agent="memory",
                    ordinal=len(tasks) + 1,
                    objective=f"Recall relevant stored context for this goal: {intent.goal}",
                    required_capabilities=["memory_read"],
                    candidate_agents=["memory"],
                    retry_strategy="fallback",
                    fallback="Continue without persistent memory if the backend is unavailable.",
                    timeout_seconds=10,
                    output_key="memory_context",
                )
            )

        if "file" in required_agents and file_action in {"read", "list"}:
            tasks.append(
                self._task(
                    task_type="file_context",
                    agent="file",
                    ordinal=len(tasks) + 1,
                    objective=f"Inspect requested files or directories for this goal: {intent.goal}",
                    depends_on=[task.id for task in tasks[-1:]],
                    required_capabilities=["filesystem"],
                    candidate_agents=["file"],
                    retry_strategy="fallback",
                    fallback="Continue with the available context if the file system action is blocked.",
                    timeout_seconds=10,
                    output_key="file_context",
                )
            )

        should_add_research = self._should_add_research_context(intent, required_agents)
        if should_add_research:
            tasks.append(
                self._task(
                    task_type="research_context",
                    agent="research",
                    ordinal=len(tasks) + 1,
                    objective=f"Collect grounded supporting context for this goal: {intent.goal}",
                    depends_on=[task.id for task in tasks[-1:]],
                    required_capabilities=["reasoning", "summarization"],
                    candidate_agents=["research", "coding"],
                    retries=2 if intent.complexity == "high" else 1,
                    retry_strategy="tighten_prompt",
                    fallback="Limit the answer to verified context already available in memory.",
                    confidence_threshold=0.55,
                    timeout_seconds=20,
                    output_key="research_context",
                    optional="research" not in required_agents,
                )
            )

        if "coding" in required_agents:
            tasks.append(
                self._task(
                    task_type="solution",
                    agent="coding",
                    ordinal=len(tasks) + 1,
                    objective=f"Produce the main solution for this goal: {intent.goal}",
                    depends_on=[task.id for task in tasks],
                    required_capabilities=["reasoning", "code_generation"],
                    candidate_agents=["coding"],
                    retries=2 if intent.complexity == "high" else 1,
                    retry_strategy="tighten_prompt",
                    fallback="Return a conservative implementation plan if full code is not reliable.",
                    confidence_threshold=0.65,
                    timeout_seconds=45 if intent.complexity == "high" else 30,
                    output_key="final_response",
                )
            )
            tasks.append(
                self._task(
                    task_type="test_generation",
                    agent="coding",
                    ordinal=len(tasks) + 1,
                    objective=f"Generate and validate tests for this solution goal: {intent.goal}",
                    depends_on=[tasks[-1].id],
                    required_capabilities=["reasoning", "code_generation", "testing"],
                    candidate_agents=["coding"],
                    retries=1,
                    retry_strategy="plan_modification",
                    fallback="Return a minimal test plan when reliable executable tests cannot be produced.",
                    confidence_threshold=0.6,
                    timeout_seconds=30,
                    output_key="test_generation_result",
                    metadata={"tests_required": True},
                )
            )

        if "file" in required_agents and file_action == "write":
            tasks.append(
                self._task(
                    task_type="file_write",
                    agent="file",
                    ordinal=len(tasks) + 1,
                    objective=f"Write or save the requested artifact for this goal: {intent.goal}",
                    depends_on=[tasks[-1].id] if tasks else [],
                    required_capabilities=["file_write"],
                    candidate_agents=["file"],
                    retry_strategy="fallback",
                    fallback="Return the artifact inline if writing to disk is not possible.",
                    timeout_seconds=15,
                    output_key="file_write_result",
                )
            )

        if "canary" in required_agents:
            tasks.append(
                self._task(
                    task_type="canary_protection",
                    agent="canary",
                    ordinal=len(tasks) + 1,
                    objective=f"Execute the canary protection workflow for this goal: {intent.goal}",
                    depends_on=[task.id for task in tasks[-1:]],
                    required_capabilities=["security"],
                    candidate_agents=["canary"],
                    retry_strategy="fallback",
                    fallback="Provide a local fallback protection plan when live monitoring is unavailable.",
                    timeout_seconds=20,
                    output_key="canary_result",
                )
            )

        if "memory" in required_agents and memory_action == "store":
            tasks.append(
                self._task(
                    task_type="memory_store",
                    agent="memory",
                    ordinal=len(tasks) + 1,
                    objective=f"Store important takeaways from this goal: {intent.goal}",
                    depends_on=[tasks[-1].id] if tasks else [],
                    required_capabilities=["memory_write"],
                    candidate_agents=["memory"],
                    retry_strategy="fallback",
                    fallback="Surface the memory payload to the user if persistence is unavailable.",
                    timeout_seconds=10,
                    output_key="memory_store_result",
                )
            )

        if not tasks:
            tasks.append(
                self._task(
                    task_type="solution",
                    agent="coding",
                    ordinal=1,
                    objective=f"Produce the main solution for this goal: {intent.goal}",
                    required_capabilities=["reasoning", "code_generation"],
                    candidate_agents=["coding"],
                    retries=2 if intent.complexity == "high" else 1,
                    retry_strategy="tighten_prompt",
                    fallback="Return a conservative execution plan if implementation confidence is low.",
                    confidence_threshold=0.65,
                    timeout_seconds=45 if intent.complexity == "high" else 30,
                    output_key="final_response",
                )
            )

        if remembered_pattern and reuse_mode in {"reuse_high_success_pattern", "reuse_cautiously"}:
            self._apply_skill_pattern(
                tasks,
                remembered_pattern,
                include_runtime_policy=reuse_mode == "reuse_high_success_pattern",
            )
        elif remembered_pattern:
            self._harden_plan(tasks, reason=reuse_mode)

        if project_context and project_context.get("enabled"):
            self._apply_project_context(tasks, intent, project_context)

        metadata = {
            "planned_at": datetime.now(timezone.utc).isoformat(),
            "signature": signature,
            "complexity": intent.complexity,
            "constraints": list(intent.constraints),
            "required_agents": required_agents,
            "deliverables": list(intent.deliverables),
            "skill_pattern_available": bool(remembered_pattern),
            "skill_pattern_reused": bool(remembered_pattern and reuse_mode in {"reuse_high_success_pattern", "reuse_cautiously"}),
            "skill_pattern_reuse_mode": reuse_mode,
            "skill_pattern_success_count": (remembered_pattern or {}).get("success_count", 0),
            "skill_pattern_success_rate": (remembered_pattern or {}).get("success_rate", 0.0),
            "skill_pattern_avg_retries": (remembered_pattern or {}).get("avg_retries", 0.0),
            "skill_pattern_rank_score": (remembered_pattern or {}).get("rank_score", 0.0),
            "file_action": file_action or "none",
            "memory_action": memory_action or "none",
            "project_mode": bool(project_context and project_context.get("enabled")),
            "project_root": (project_context or {}).get("project_root"),
            "project_signature": (project_context or {}).get("project_signature"),
            "project_frameworks": list(((project_context or {}).get("project_context") or {}).get("frameworks", [])),
            "project_languages": list(((project_context or {}).get("project_context") or {}).get("languages", [])),
            "project_recent_goals": list((project_context or {}).get("recent_goals", [])),
            "project_common_errors": list((project_context or {}).get("common_errors", [])),
            "project_preferences": dict((project_context or {}).get("user_preferences", {})),
        }
        return ExecutionPlan(
            goal=intent.goal,
            primary_intent=intent.primary_intent,
            constraints=self._merge_constraints(intent.constraints, project_context),
            tasks=tasks,
            metadata=metadata,
        )

    def _should_add_research_context(
        self,
        intent: IntentProfile,
        required_agents: list[str],
    ) -> bool:
        """Avoid slow research hops for implementation-heavy build requests."""
        if "research" in required_agents:
            return True
        if intent.primary_intent != "coding" or intent.complexity != "high":
            return False

        lowered = intent.goal.lower()
        implementation_signals = {
            "build",
            "create",
            "implement",
            "scaffold",
            "full stack",
            "backend",
            "frontend",
            "api route",
            "api routes",
            "login",
            "express",
            "react",
            "form",
        }
        research_signals = {
            "research",
            "analyze",
            "compare",
            "investigate",
            "summarize",
            "explain",
        }
        if any(signal in lowered for signal in research_signals):
            return True
        if any(signal in lowered for signal in implementation_signals):
            return False
        return True

    def _pattern_reuse_mode(self, pattern: dict | None) -> str:
        if not pattern:
            return "no_pattern"

        success_rate = float(pattern.get("success_rate", 0.0) or 0.0)
        avg_retries = float(pattern.get("avg_retries", 0.0) or 0.0)
        total_runs = int(pattern.get("total_runs", 0) or 0)

        if success_rate >= 0.8 and avg_retries <= 1.25:
            return "reuse_high_success_pattern"
        if success_rate >= 0.6 and avg_retries <= 2.0:
            return "reuse_cautiously"
        if total_runs and avg_retries > 2.5:
            return "avoid_high_retry_pattern"
        return "avoid_low_success_pattern"

    def _apply_skill_pattern(
        self,
        tasks: list[PlannedTask],
        pattern: dict,
        *,
        include_runtime_policy: bool,
    ) -> None:
        overrides = pattern.get("task_overrides", {})
        for task in tasks:
            override = overrides.get(task.task_type)
            if not override:
                continue
            if override.get("agent"):
                task.agent = override["agent"]
            if override.get("candidate_agents"):
                task.candidate_agents = list(override["candidate_agents"])
            if override.get("required_capabilities"):
                task.required_capabilities = list(override["required_capabilities"])
            if include_runtime_policy and override.get("retry_strategy"):
                task.retry_strategy = override["retry_strategy"]
            if override.get("fallback"):
                task.fallback = override["fallback"]
            if override.get("fallback_agent"):
                task.fallback_agent = override["fallback_agent"]
            if include_runtime_policy and override.get("confidence_threshold") is not None:
                task.confidence_threshold = float(override["confidence_threshold"])
            task.metadata["planner_signal"] = "reused_skill_pattern"

    def _harden_plan(self, tasks: list[PlannedTask], reason: str) -> None:
        for task in tasks:
            task.metadata["planner_signal"] = reason
            if task.retry_strategy in {"repeat", "tighten_prompt"}:
                task.retry_strategy = "switch_agent" if len(task.candidate_agents) > 1 else "plan_modification"
            if not task.fallback:
                task.fallback = (
                    "Historical runs of similar workflows needed too many retries. "
                    "Prefer the smallest grounded answer over speculative detail."
                )

    def _apply_project_context(
        self,
        tasks: list[PlannedTask],
        intent: IntentProfile,
        project_context: dict,
    ) -> None:
        project_scan = project_context.get("project_context") or {}
        frameworks = list(project_scan.get("frameworks", []))
        languages = list(project_scan.get("languages", []))
        entrypoints = list(project_scan.get("entrypoints", []))
        preferences = dict(project_context.get("user_preferences", {}))
        recent_goals = list(project_context.get("recent_goals", []))
        common_errors = list(project_context.get("common_errors", []))
        preferred_pattern = self._best_project_pattern(project_context, primary_intent=intent.primary_intent)

        for task in tasks:
            task.metadata.update(
                {
                    "project_mode": True,
                    "project_root": project_context.get("project_root"),
                    "project_signature": project_context.get("project_signature"),
                    "project_frameworks": frameworks,
                    "project_languages": languages,
                    "project_entrypoints": entrypoints[:6],
                    "project_preferences": preferences,
                    "recent_project_goals": recent_goals[-5:],
                    "common_project_errors": common_errors[:3],
                }
            )
            if frameworks:
                task.metadata["planner_signal"] = task.metadata.get("planner_signal", "project_context_applied")
            if task.task_type == "solution" and frameworks:
                existing_stack = ", ".join(frameworks[:4])
                task.fallback = (
                    f"{task.fallback} Match the existing project stack: {existing_stack}."
                    if task.fallback
                    else f"Match the existing project stack: {existing_stack}."
                )
                if common_errors:
                    task.retry_strategy = "plan_modification"
                    task.metadata["known_error_types"] = [
                        error.get("failure_type", "runtime_error")
                        for error in common_errors[:3]
                    ]
            if preferred_pattern and task.task_type == "solution":
                task.metadata["project_pattern_signal"] = "reused_project_pattern"
                best_sequence = list(preferred_pattern.get("best_agent_sequence", []))
                if best_sequence:
                    preferred_agent = best_sequence[0]
                    if preferred_agent in task.candidate_agents:
                        task.agent = preferred_agent
                if float(preferred_pattern.get("avg_retries", 0.0) or 0.0) <= 1.0:
                    task.confidence_threshold = max(task.confidence_threshold, 0.7)

    def _best_project_pattern(self, project_context: dict, *, primary_intent: str) -> dict | None:
        patterns = list(project_context.get("successful_patterns", []))
        candidates = [
            pattern
            for pattern in patterns
            if not primary_intent or pattern.get("primary_intent") == primary_intent
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                float(item.get("success_rate", 0.0) or 0.0),
                -float(item.get("avg_retries", 0.0) or 0.0),
                float(item.get("avg_confidence", 0.0) or 0.0),
                int(item.get("total_runs", 0) or 0),
            ),
            reverse=True,
        )
        return candidates[0]

    def _merge_constraints(self, constraints: list[str], project_context: dict | None) -> list[str]:
        merged = list(constraints)
        if project_context and project_context.get("enabled"):
            merged.extend(["project_mode_active", "respect_existing_project_context"])
        return sorted(dict.fromkeys(merged))

    def _task(
        self,
        task_type: str,
        agent: str | None,
        ordinal: int,
        objective: str,
        depends_on: list[str] | None = None,
        candidate_agents: list[str] | None = None,
        required_capabilities: list[str] | None = None,
        retries: int = 1,
        retry_strategy: str = "repeat",
        fallback: str | None = None,
        fallback_agent: str | None = None,
        confidence_threshold: float = 0.0,
        timeout_seconds: int = 20,
        optional: bool = False,
        output_key: str | None = None,
        metadata: dict | None = None,
    ) -> PlannedTask:
        agent_label = agent or task_type
        return PlannedTask(
            id=f"{agent_label}_{ordinal}",
            task_type=task_type,
            objective=objective,
            agent=agent,
            depends_on=depends_on or [],
            candidate_agents=list(candidate_agents or ([agent] if agent else [])),
            required_capabilities=list(required_capabilities or []),
            retries=retries,
            retry_strategy=retry_strategy,
            fallback=fallback,
            fallback_agent=fallback_agent,
            confidence_threshold=confidence_threshold,
            timeout_seconds=timeout_seconds,
            optional=optional,
            output_key=output_key,
            metadata=dict(metadata or {}),
        )
