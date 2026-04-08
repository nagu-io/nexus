"""
Workflow orchestrator for compiled NEXUS goals.

Execution loop:
plan -> act -> observe -> reflect -> retry
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nexus.blueprint_generator import TaskBlueprint, WorkflowBlueprint
from nexus.critics.multi_critic import MultiCriticEvaluator
from nexus.memory.environment_memory import EnvironmentMemory
from nexus.memory.skill_memory import SkillMemory
from nexus.reflect.reflect_score import ReflectScore
from nexus.runtime.code_reader import CodeReader
from nexus.runtime.context_reducer import BaseContextReducer, ContextReductionResult, build_context_reducer
from nexus.runtime.decision_cache import DecisionCache
from nexus.runtime.doc_generator import DocGenerator
from nexus.runtime.executor import CodeExecutor
from nexus.runtime.event_bus import runtime_event_bus
from nexus.runtime.file_tool import FileTool
from nexus.runtime.git_tool import GitTool
from nexus.runtime.policy_engine import PolicyEngine
from nexus.runtime.project_executor import ProjectExecutor
from nexus.runtime.terminal_tool import TerminalTool
from nexus.runtime.strategy_engine import StrategyEngine
from nexus.runtime.trace import ExecutionContext, ExecutionTrace
from nexus.runtime.workspace import WorkspaceDirectory
from nexus.shared_memory import SharedMemory
from nexus.wiring_engine import WiringEngine


def _build_logger() -> logging.Logger:
    from nexus.config import config

    logger = logging.getLogger("nexus.orchestrator")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    log_dir = Path(config.data_dir) / "runtime_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "orchestrator.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


LOGGER = _build_logger()


@dataclass
class TaskExecutionResult:
    """Final status for a single task."""

    task_id: str
    agent: str
    status: str
    attempts: int
    output: str
    observation: dict[str, Any]
    reflection: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent": self.agent,
            "status": self.status,
            "attempts": self.attempts,
            "output": self.output,
            "observation": self.observation,
            "reflection": self.reflection,
            "error": self.error,
        }


class Orchestrator:
    """Compile-time blueprint executor with evaluation-driven retries and shared memory."""

    def __init__(
        self,
        wiring_engine: WiringEngine | None = None,
        shared_memory: SharedMemory | None = None,
        reflect_scorer: ReflectScore | None = None,
        multi_critic: MultiCriticEvaluator | None = None,
        decision_cache: DecisionCache | None = None,
        file_tool: FileTool | None = None,
        terminal_tool: TerminalTool | None = None,
        code_executor: CodeExecutor | None = None,
        project_executor: ProjectExecutor | None = None,
        git_tool: GitTool | None = None,
        policy_engine: PolicyEngine | None = None,
        strategy_engine: StrategyEngine | None = None,
        skill_memory: SkillMemory | None = None,
        environment_memory: EnvironmentMemory | None = None,
        code_reader: CodeReader | None = None,
        context_reducer: BaseContextReducer | None = None,
        doc_generator: DocGenerator | None = None,
        project_context: dict[str, Any] | None = None,
        execution_mode: str = "stable",
        trace_dir: Path | None = None,
        decision_log_path: Path | None = None,
        workspace_base_dir: Path | None = None,
    ):
        self.wiring_engine = wiring_engine or WiringEngine()
        self.shared_memory = shared_memory or SharedMemory()
        self.reflect_scorer = reflect_scorer or ReflectScore()
        self.execution_mode = execution_mode if execution_mode in PolicyEngine.VALID_MODES else "stable"
        plugin_critics = self.wiring_engine.resolve_critics(reflect_scorer=self.reflect_scorer)
        self.multi_critic = multi_critic or MultiCriticEvaluator(reflect_scorer=self.reflect_scorer)
        if plugin_critics:
            self.multi_critic.critics.extend(plugin_critics)
        self.decision_cache = decision_cache or DecisionCache()
        self.file_tool = file_tool
        self.terminal_tool = terminal_tool
        self.code_executor = code_executor
        self.project_executor = project_executor
        self.git_tool = git_tool
        self.policy_engine = policy_engine or PolicyEngine(mode=self.execution_mode)
        self.execution_mode = self.policy_engine.mode
        self.strategy_engine = strategy_engine or StrategyEngine()
        self.skill_memory = skill_memory or SkillMemory()
        self.environment_memory = environment_memory or EnvironmentMemory()
        self.code_reader = code_reader or CodeReader()
        if context_reducer is not None:
            self.context_reducer = context_reducer
        else:
            from nexus.config import config

            self.context_reducer = build_context_reducer(
                enabled=config.context_reduction_enabled,
                backend=config.context_reduction_backend,
                threshold_chars=config.context_reduction_threshold_chars,
                target_chars=config.context_reduction_target_chars,
                model_name=config.context_reduction_model,
            )
        self.doc_generator = doc_generator or DocGenerator()
        self.project_context = dict(project_context or {})
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.decision_log_path = Path(decision_log_path) if decision_log_path else None
        self.workspace_base_dir = Path(workspace_base_dir) if workspace_base_dir else None
        self._active_workspace: WorkspaceDirectory | None = None
        self._active_file_tool: FileTool | None = None
        self._active_terminal_tool: TerminalTool | None = None
        self._active_code_executor: CodeExecutor | None = None
        self._active_project_executor: ProjectExecutor | None = None
        self._active_git_tool: GitTool | None = None
        self._active_workspace_managed: bool = False

    async def run_blueprint(self, blueprint: WorkflowBlueprint) -> dict[str, Any]:
        """Execute every task in the workflow in dependency order."""
        trace = ExecutionTrace(
            workflow_id=self.shared_memory.workflow_id,
            root_dir=self.trace_dir,
            decision_log_path=self.decision_log_path,
            verbosity=self.policy_engine.trace_verbosity(),
        )
        execution_context = ExecutionContext(trace=trace, mode=self.execution_mode)
        project_context = self._resolve_project_context(blueprint)
        if project_context.get("enabled") and project_context.get("project_root"):
            workspace = WorkspaceDirectory.for_project(
                project_dir=project_context["project_root"],
                workflow_id=self.shared_memory.workflow_id,
                goal=blueprint.goal,
            )
        else:
            workspace = WorkspaceDirectory.for_workflow(
                workflow_id=self.shared_memory.workflow_id,
                goal=blueprint.goal,
                base_dir=self.workspace_base_dir,
            )
        self._active_workspace_managed = not bool(project_context.get("enabled") and project_context.get("project_root"))
        self._active_workspace = workspace
        self._active_file_tool = self.file_tool or FileTool(allowed_roots=[workspace.root_dir])
        self._active_terminal_tool = self.terminal_tool or TerminalTool(allowed_roots=[workspace.root_dir])
        self._active_code_executor = self.code_executor or CodeExecutor()
        self._active_project_executor = self.project_executor or ProjectExecutor(executor=self._active_code_executor)
        self._active_git_tool = self.git_tool or GitTool(allowed_roots=[workspace.root_dir])
        if self._active_workspace_managed:
            try:
                self._active_git_tool.init(workspace.root_dir)
            except Exception as error:  # pragma: no cover - defensive path
                LOGGER.warning("workspace git init failed: %s", error)

        self.shared_memory.put("workflow.goal", blueprint.goal)
        self.shared_memory.put("workflow.primary_intent", blueprint.primary_intent)
        self.shared_memory.put("workflow.version", blueprint.version)
        self.shared_memory.put("workflow.execution_mode", self.execution_mode)
        self.shared_memory.put("workspace.root_dir", str(workspace.root_dir))
        self.shared_memory.put("workspace.managed", self._active_workspace_managed)
        self.shared_memory.put("workspace.project_state", workspace.snapshot())
        self.shared_memory.put("workspace.last_terminal_error", None)
        self.shared_memory.put("workspace.last_command", [])
        if project_context.get("enabled"):
            self.shared_memory.put("project.mode", True)
            self.shared_memory.put("project.root_dir", project_context.get("project_root"))
            self.shared_memory.put("project.context", project_context.get("project_context", {}))
            self.shared_memory.put(
                "project.session",
                {
                    "session_iteration": project_context.get("session_iteration"),
                    "recent_goals": list(project_context.get("recent_goals", [])),
                },
            )
            self.shared_memory.put("project.user_preferences", project_context.get("user_preferences", {}))
            self.shared_memory.put("project.common_errors", project_context.get("common_errors", []))
        if blueprint.metadata.get("plan_signature"):
            self.shared_memory.put("workflow.signature", blueprint.metadata["plan_signature"])
        self.shared_memory.append_event(
            "workflow.started",
            "orchestrator",
            {"goal": blueprint.goal, "tasks": [task.id for task in blueprint.tasks]},
        )
        runtime_event_bus.emit(
            {
                "type": "agent_started",
                "workflow_id": self.shared_memory.workflow_id,
                "task_id": "workflow",
                "status": "started",
                "summary": blueprint.goal,
            }
        )
        trace.record_task_event(
            kind="workflow_started",
            task_id="workflow",
            input_text=blueprint.goal,
            metadata={"task_count": len(blueprint.tasks), "execution_mode": self.execution_mode},
        )
        trace.record_decision(
            decision_type="policy_mode",
            task_id="workflow",
            agent_selected=None,
            reason=f"workflow started in {self.execution_mode} mode",
            metadata={"mode": self.execution_mode},
        )
        if project_context.get("enabled"):
            trace.record_decision(
                decision_type="project_mode",
                task_id="workflow",
                agent_selected=None,
                reason="loaded persisted project context before runtime execution",
                metadata={
                    "project_root": project_context.get("project_root"),
                    "project_signature": project_context.get("project_signature"),
                    "frameworks": (project_context.get("project_context") or {}).get("frameworks", []),
                    "session_iteration": project_context.get("session_iteration"),
                },
            )

        if blueprint.metadata.get("skill_pattern_reused"):
            trace.record_decision(
                decision_type="skill_pattern_selection",
                task_id="workflow",
                agent_selected=None,
                reason="selected the best-ranked remembered workflow pattern for this goal shape",
                confidence=blueprint.metadata.get("skill_pattern_success_rate"),
                metadata={
                    "plan_signature": blueprint.metadata.get("plan_signature"),
                    "success_rate": blueprint.metadata.get("skill_pattern_success_rate"),
                    "avg_retries": blueprint.metadata.get("skill_pattern_avg_retries"),
                },
            )

        executions: list[TaskExecutionResult] = []
        workflow_status = "completed"
        execution_by_task: dict[str, TaskExecutionResult] = {}
        pending_tasks = {task.id: task for task in blueprint.tasks}

        while pending_tasks:
            ready_tasks = [
                task
                for task in blueprint.tasks
                if task.id in pending_tasks and all(dependency in execution_by_task for dependency in task.depends_on)
            ]
            if not ready_tasks:
                workflow_status = "failed"
                trace.record_decision(
                    decision_type="scheduler",
                    task_id="workflow",
                    agent_selected=None,
                    reason="No ready tasks remained; workflow may contain unsatisfied dependencies.",
                    metadata={"pending_tasks": sorted(pending_tasks)},
                )
                break

            parallel_plan = self.policy_engine.parallel_plan(ready_tasks=ready_tasks, blueprint=blueprint)
            batch_lookup = {task.id: task for task in ready_tasks}
            batch_task_ids = parallel_plan.get("task_ids") or [ready_tasks[0].id]
            batch = [batch_lookup[task_id] for task_id in batch_task_ids if task_id in batch_lookup] or [ready_tasks[0]]
            if len(ready_tasks) > 1:
                self._record_parallel_decision(
                    ready_tasks=ready_tasks,
                    batch=batch,
                    parallel_plan=parallel_plan,
                    trace=trace,
                )

            if parallel_plan.get("parallel", False):
                batch_results = list(
                    await asyncio.gather(
                        *[self._execute_task(task, blueprint, execution_context) for task in batch]
                    )
                )
            else:
                batch_results = [await self._execute_task(batch[0], blueprint, execution_context)]
                batch = [batch[0]]

            for task, result in zip(batch, batch_results):
                executions.append(result)
                execution_by_task[task.id] = result
                pending_tasks.pop(task.id, None)

            if any(result.status != "completed" and not task.optional for task, result in zip(batch, batch_results)):
                workflow_status = "failed"
                break

        final_output = self._derive_final_output(blueprint, execution_by_task, executions)
        self.shared_memory.put("workflow.final_output", final_output)
        self.shared_memory.append_event(
            "workflow.finished",
            "orchestrator",
            {"status": workflow_status, "final_output": final_output[:240]},
        )
        trace.record_task_event(
            kind="workflow_finished",
            task_id="workflow",
            output_text=final_output,
            metadata={"status": workflow_status, "completed_tasks": len(executions)},
        )

        execution_payload = [execution.to_dict() for execution in executions]
        total_retries = sum(max(0, execution.attempts - 1) for execution in executions)
        final_confidence = self._final_confidence(executions)
        documentation = None
        touched_files = self._collect_touched_files(execution_payload, workspace.root_dir)
        if touched_files or final_output:
            documentation = self.doc_generator.generate(
                workspace_root=workspace.root_dir,
                blueprint=blueprint,
                executions=execution_payload,
                trace_snapshot=trace.snapshot(),
                touched_files=touched_files,
                managed_workspace=self._active_workspace_managed,
            ).to_dict()
            self.shared_memory.put("workflow.documentation", documentation)
        try:
            self.skill_memory.record_workflow(blueprint, execution_payload, workflow_status)
        except Exception as error:  # pragma: no cover - defensive persistence path
            LOGGER.warning("skill-memory record failed: %s", error)
        if project_context.get("enabled") and project_context.get("project_root"):
            try:
                self.environment_memory.record_workflow(
                    project_root=project_context["project_root"],
                    goal=blueprint.goal,
                    blueprint=blueprint,
                    executions=execution_payload,
                    status=workflow_status,
                    final_confidence=final_confidence,
                    execution_mode=self.execution_mode,
                )
            except Exception as error:  # pragma: no cover - defensive persistence path
                LOGGER.warning("environment-memory record failed: %s", error)

        trace.finalize(
            status=workflow_status,
            final_output=final_output,
            metadata={
                "plan_signature": blueprint.metadata.get("plan_signature"),
                "goal": blueprint.goal,
                "task_count": len(blueprint.tasks),
                "execution_mode": self.execution_mode,
                "total_retries": total_retries,
                "final_confidence": final_confidence,
                "workspace_root": str(workspace.root_dir),
                "project_mode": bool(project_context.get("enabled")),
                "project_root": project_context.get("project_root"),
                "project_signature": project_context.get("project_signature"),
                "documentation": documentation or {},
            },
        )
        runtime_event_bus.emit(
            {
                "type": "workflow_complete",
                "workflow_id": self.shared_memory.workflow_id,
                "status": workflow_status,
                "summary": final_output[:240],
                "documentation": documentation or {},
            }
        )
        return {
            "workflow_id": self.shared_memory.workflow_id,
            "status": workflow_status,
            "blueprint": blueprint.to_dict(),
            "executions": execution_payload,
            "final_output": final_output,
            "execution_mode": self.execution_mode,
            "final_confidence": final_confidence,
            "memory": self.shared_memory.snapshot(),
            "trace": trace.snapshot(),
            "documentation": documentation or {},
        }

    async def _execute_task(
        self,
        task: TaskBlueprint,
        blueprint: WorkflowBlueprint,
        execution_context: ExecutionContext,
    ) -> TaskExecutionResult:
        attempts = 0
        last_output = ""
        last_observation: dict[str, Any] = {"ok": False, "summary": "task did not run"}
        last_reflection: dict[str, Any] = {"should_retry": False, "reason": "task did not run"}
        last_error: str | None = None
        selected_agent_name = task.agent or "unassigned"
        attempt_checkpoint_sha: str | None = None

        max_attempts = max(1, task.retries + 1)
        policy_profile = self.policy_engine.build_profile(
            task=task,
            blueprint=blueprint,
            max_attempts=max_attempts,
        )
        execution_context.start_task(task.id)

        while attempts < max_attempts:
            attempts += 1
            attempt_checkpoint_sha = self._checkpoint_before_attempt(task, attempts)
            current_strategy = execution_context.get_task_strategy(task.id).get("strategy") or task.retry_strategy
            wired_task = self.wiring_engine.wire_task(
                task,
                blueprint,
                self.shared_memory,
                execution_context=execution_context,
            )
            selected_agent_name = wired_task.selected_agent_name
            prompt, reduction = self._prepare_task_prompt(
                task=task,
                agent_name=selected_agent_name,
                prompt=wired_task.prompt,
                attempts=attempts,
                execution_context=execution_context,
            )
            self.shared_memory.append_event(
                "task.started",
                task.id,
                {"attempt": attempts, "agent": selected_agent_name},
            )
            execution_context.trace.record_task_event(
                kind="task_started",
                task_id=task.id,
                agent=selected_agent_name,
                attempt=attempts,
                input_text=prompt,
                retry_count=attempts - 1,
                metadata={
                    "task_type": task.task_type,
                    "selection_reason": wired_task.selection_reason,
                    "execution_mode": execution_context.mode,
                    "context_reduced": bool(reduction and reduction.reduced),
                },
            )
            runtime_event_bus.emit(
                {
                    "type": "agent_started",
                    "workflow_id": self.shared_memory.workflow_id,
                    "task_id": task.id,
                    "agent": selected_agent_name,
                    "attempt": attempts,
                    "status": "running",
                }
            )
            LOGGER.info("task=%s attempt=%s agent=%s", task.id, attempts, selected_agent_name)

            evaluation: dict[str, Any] = {}
            policy_decision: dict[str, Any] = {}
            cache_hint: dict[str, Any] | None = None
            try:
                cycle = await asyncio.wait_for(
                    self._run_agent_cycle_with_tools(
                        agent=wired_task.agent,
                        task=task,
                        prompt=prompt,
                        selected_agent_name=selected_agent_name,
                        attempts=attempts,
                        execution_context=execution_context,
                    ),
                    timeout=task.timeout_seconds,
                )
                last_output = self._stringify_output(cycle.get("result", ""))
                last_observation = dict(cycle["observation"])
                last_reflection = dict(cycle["reflection"])
                last_error = None
                runtime_event_bus.emit(
                    {
                        "type": "agent_output",
                        "workflow_id": self.shared_memory.workflow_id,
                        "task_id": task.id,
                        "agent": selected_agent_name,
                        "attempt": attempts,
                        "summary": last_output[:240],
                    }
                )

                if last_observation.get("ok", False):
                    if policy_profile.allow_cached_decisions and attempts == 1:
                        cache_hint = self.decision_cache.lookup(
                            task=task,
                            blueprint=blueprint,
                            agent=selected_agent_name,
                            strategy=current_strategy,
                            mode=self.execution_mode,
                            min_confidence=max(policy_profile.cache_min_confidence, policy_profile.confidence_target),
                        )
                        if cache_hint:
                            self._record_cache_decision(
                                task=task,
                                current_agent=selected_agent_name,
                                cache_entry=cache_hint,
                                execution_context=execution_context,
                            )
                    evaluation = await self._evaluate_output(
                        task=task,
                        output=last_output,
                        observation=last_observation,
                        attempt=attempts,
                        max_attempts=max_attempts,
                        policy_profile=policy_profile,
                        cache_hint=cache_hint,
                    )
                    if evaluation:
                        last_observation["evaluation"] = evaluation
                        last_observation["confidence"] = evaluation["confidence"]
                        last_observation["critic_scores"] = dict(evaluation["critic_scores"])
                        last_observation["evaluation_summary"] = evaluation["summary"]
                        self._record_evaluation_decision(
                            task=task,
                            current_agent=selected_agent_name,
                            attempts=attempts,
                            evaluation=evaluation,
                            execution_context=execution_context,
                        )
                        if not evaluation["ok"]:
                            last_observation["ok"] = False
                            last_observation["summary"] = evaluation["summary"]
                            last_observation["failure_type"] = evaluation["failure_type"]
                else:
                    last_observation.setdefault("failure_type", "runtime_error")
            except asyncio.TimeoutError:
                last_output = ""
                last_error = f"task timed out after {task.timeout_seconds}s"
                last_observation = {"ok": False, "summary": last_error, "failure_type": "timeout"}
                LOGGER.warning("task=%s timed out on attempt %s", task.id, attempts)
            except Exception as error:
                last_output = ""
                last_error = str(error)
                last_observation = {"ok": False, "summary": f"exception: {last_error}", "failure_type": "runtime_error"}
                LOGGER.exception("task=%s failed on attempt %s", task.id, attempts)

            policy_decision = self.policy_engine.decide(
                task=task,
                blueprint=blueprint,
                profile=policy_profile,
                attempts=attempts,
                elapsed_seconds=execution_context.elapsed_seconds(task.id),
                observation=last_observation,
                evaluation=evaluation or None,
                history=execution_context.task_history(task.id),
                current_strategy=current_strategy,
            ).to_dict()
            last_observation["policy"] = policy_decision
            self._record_policy_decision(
                task=task,
                current_agent=selected_agent_name,
                attempts=attempts,
                policy_decision=policy_decision,
                execution_context=execution_context,
            )

            if last_observation.get("ok", False) and (not evaluation or evaluation.get("ok", True)) and policy_decision["action"] == "complete":
                last_reflection.update(
                    {
                        "should_retry": False,
                        "reason": policy_decision["reason"],
                        "strategy": "none",
                        "combined_score": evaluation.get("score"),
                        "policy_action": policy_decision["action"],
                    }
                )
            elif policy_decision["should_retry"]:
                last_reflection = self.strategy_engine.choose(
                    task=task,
                    current_agent=selected_agent_name,
                    attempts=attempts,
                    max_attempts=max_attempts,
                    observation=last_observation,
                    evaluation=evaluation,
                    alternative_agents=self.wiring_engine.alternative_agents(task, exclude=[selected_agent_name]),
                    policy=policy_decision,
                    previous_strategy=current_strategy,
                ).to_dict()
            else:
                last_reflection = {
                    "should_retry": False,
                    "reason": policy_decision["reason"],
                    "strategy": policy_decision.get("preferred_strategy") or current_strategy,
                    "failure_type": last_observation.get("failure_type"),
                    "note": policy_decision.get("note"),
                    "policy_action": policy_decision["action"],
                }

            payload = {
                "agent": selected_agent_name,
                "attempts": attempts,
                "output": last_output,
                "observation": last_observation,
                "reflection": last_reflection,
                "error": last_error,
            }
            self.shared_memory.publish_task_result(task.id, payload)
            self.shared_memory.append_event(
                "task.finished",
                task.id,
                {"attempt": attempts, "agent": selected_agent_name, "observation": last_observation, "error": last_error},
            )

            fallback_triggered = bool(last_reflection.get("fallback_triggered", False))
            execution_context.trace.record_task_event(
                kind="task_finished",
                task_id=task.id,
                agent=selected_agent_name,
                attempt=attempts,
                output_text=last_output,
                reflect_score=evaluation.get("legacy_reflect_score"),
                evaluation_score=evaluation.get("score"),
                critic_scores=evaluation.get("critic_scores"),
                retry_count=attempts - 1,
                fallback_triggered=fallback_triggered,
                metadata={
                    "status": "completed" if last_observation.get("ok", False) else "retrying" if last_reflection.get("should_retry", False) else "failed",
                    "summary": last_observation.get("summary"),
                    "failure_type": last_observation.get("failure_type"),
                    "policy_action": policy_decision.get("action"),
                },
            )
            execution_context.record_attempt(
                task.id,
                {
                    "attempt": attempts,
                    "agent": selected_agent_name,
                    "evaluation_score": evaluation.get("score"),
                    "confidence": last_observation.get("confidence"),
                    "failure_type": last_observation.get("failure_type"),
                    "strategy": last_reflection.get("strategy"),
                    "policy_action": policy_decision.get("action"),
                },
            )

            if last_observation.get("ok", False):
                self._checkpoint_success(task, attempts)
                execution_context.clear_task_strategy(task.id)
                self.decision_cache.record(
                    task=task,
                    blueprint=blueprint,
                    agent=selected_agent_name,
                    strategy=current_strategy,
                    mode=self.execution_mode,
                    status="completed",
                    attempts=attempts,
                    evaluation=evaluation or None,
                    reflection=last_reflection,
                )
                return TaskExecutionResult(
                    task_id=task.id,
                    agent=selected_agent_name,
                    status="completed",
                    attempts=attempts,
                    output=last_output,
                    observation=last_observation,
                    reflection=last_reflection,
                    error=None,
                )

            if last_reflection.get("should_retry", False):
                self._rollback_failed_attempt(task, attempt_checkpoint_sha, attempts)
                execution_context.set_task_strategy(task.id, last_reflection)
                self._record_retry_decision(
                    task=task,
                    current_agent=selected_agent_name,
                    attempts=attempts,
                    reflection=last_reflection,
                    observation=last_observation,
                    execution_context=execution_context,
                )
                continue

            self._rollback_failed_attempt(task, attempt_checkpoint_sha, attempts)
            break

        self.decision_cache.record(
            task=task,
            blueprint=blueprint,
            agent=selected_agent_name,
            strategy=current_strategy,
            mode=self.execution_mode,
            status="failed",
            attempts=attempts,
            evaluation=evaluation or None,
            reflection=last_reflection,
        )
        return TaskExecutionResult(
            task_id=task.id,
            agent=selected_agent_name,
            status="failed",
            attempts=attempts,
            output=last_output,
            observation=last_observation,
            reflection=last_reflection,
            error=last_error or last_observation.get("summary"),
        )

    async def _run_agent_cycle_with_tools(
        self,
        *,
        agent: Any,
        task: TaskBlueprint,
        prompt: str,
        selected_agent_name: str,
        attempts: int,
        execution_context: ExecutionContext,
    ) -> dict[str, Any]:
        """Execute one agent attempt and resolve any emitted tool calls."""
        cycle = await agent.execute_cycle(prompt, memory=self.shared_memory)
        tool_actions: list[dict[str, Any]] = []
        tool_budget = 24

        while cycle.get("tool_request") and tool_budget > 0:
            tool_request = dict(cycle["tool_request"])
            tool_result = await self._execute_tool_request(
                agent=agent,
                task=task,
                prompt=prompt,
                current_agent=selected_agent_name,
                attempts=attempts,
                tool_request=tool_request,
                execution_context=execution_context,
            )
            tool_actions.append(tool_result)
            follow_up = await agent.continue_after_tool(
                prompt,
                tool_result,
                memory=self.shared_memory,
                thought=cycle.get("thought"),
            )
            next_tool = agent.normalize_tool_call(follow_up)
            if next_tool is not None:
                cycle = {
                    "agent": selected_agent_name,
                    "thought": cycle.get("thought"),
                    "result": "",
                    "tool_request": next_tool,
                }
                tool_budget -= 1
                continue

            result_text = self._stringify_output(follow_up)
            observation = await agent.observe(
                prompt,
                result_text,
                memory=self.shared_memory,
                thought=cycle.get("thought"),
            )
            if tool_actions:
                observation["tool_actions"] = [dict(action) for action in tool_actions]
            reflection = await agent.reflect(
                prompt,
                result_text,
                observation,
                memory=self.shared_memory,
                thought=cycle.get("thought"),
            )
            return {
                "agent": selected_agent_name,
                "thought": cycle.get("thought"),
                "result": result_text,
                "observation": observation,
                "reflection": reflection,
            }

        if cycle.get("tool_request"):
            failure = {
                "ok": False,
                "summary": "tool budget exhausted before the agent produced a final response",
                "failure_type": "tool_runtime_error",
                "tool_actions": [dict(action) for action in tool_actions],
            }
            return {
                "agent": selected_agent_name,
                "thought": cycle.get("thought"),
                "result": "Tool error: tool budget exhausted",
                "observation": failure,
                "reflection": {
                    "should_retry": True,
                    "reason": failure["summary"],
                },
            }

        cycle["result"] = self._stringify_output(cycle.get("result", ""))
        if tool_actions:
            cycle["observation"] = dict(cycle.get("observation", {}))
            cycle["observation"]["tool_actions"] = [dict(action) for action in tool_actions]
        return cycle

    async def _execute_tool_request(
        self,
        *,
        agent: Any,
        task: TaskBlueprint,
        prompt: str,
        current_agent: str,
        attempts: int,
        tool_request: dict[str, Any],
        execution_context: ExecutionContext,
    ) -> dict[str, Any]:
        """Execute a structured tool request and record it in traces and memory."""
        tool_name = tool_request.get("tool", "file_tool")
        arguments = dict(tool_request.get("arguments") or {})
        arguments.setdefault("workflow_id", self.shared_memory.workflow_id)
        arguments.setdefault("task_id", task.id)
        if tool_name in {"project_executor", "code_executor"}:
            arguments.setdefault("coding_agent", agent)
        tool_request = {
            **tool_request,
            "arguments": arguments,
        }
        if tool_name == "file_tool":
            result = (self._active_file_tool or self.file_tool).execute(tool_request)
        elif tool_name == "terminal_tool":
            result = (self._active_terminal_tool or self.terminal_tool).execute(tool_request)
        elif tool_name == "code_executor":
            executor = self._active_code_executor or CodeExecutor()
            if hasattr(executor, "execute_async"):
                result = await executor.execute_async(tool_request)
            else:
                result = executor.execute(tool_request)
        elif tool_name == "project_executor":
            executor = self._active_project_executor or ProjectExecutor()
            if hasattr(executor, "execute_async"):
                result = await executor.execute_async(tool_request)
            else:
                result = executor.execute(tool_request)
        elif tool_name == "git_tool":
            result = (self._active_git_tool or GitTool()).execute(tool_request)
        else:
            try:
                plugin_tool = self.wiring_engine.resolve_tool(tool_name)
            except KeyError:
                result = {
                    "ok": False,
                    "tool": tool_name,
                    "action": tool_request.get("action"),
                    "summary": f"Tool error: unsupported tool '{tool_name}'",
                    "error": f"unsupported tool '{tool_name}'",
                }
            else:
                if hasattr(plugin_tool, "execute_async"):
                    result = await plugin_tool.execute_async(tool_request)
                else:
                    result = plugin_tool.execute(tool_request)
        self._update_workspace_state(tool_request, result)

        self.shared_memory.append_event(
            "tool.executed",
            current_agent,
            {
                "task_id": task.id,
                "attempt": attempts,
                "tool_request": tool_request,
                "tool_result": result,
            },
        )
        self.shared_memory.put(
            f"tool:{task.id}:{attempts}:{len(self.shared_memory.events)}",
            {
                "prompt": prompt[:240],
                "request": tool_request,
                "result": result,
            },
        )
        execution_context.trace.record_decision(
            decision_type="tool_execution",
            task_id=task.id,
            agent_selected=current_agent,
            reason=result.get("summary", "tool executed"),
            retry=not result.get("ok", False),
            metadata={
                "attempt": attempts,
                "tool": tool_name,
                "action": tool_request.get("action"),
                "arguments": self._sanitize_tool_arguments(tool_request.get("arguments", {})),
                "path": result.get("path"),
            },
        )
        execution_context.trace.record_task_event(
            kind="tool_executed",
            task_id=task.id,
            agent=current_agent,
            attempt=attempts,
            output_text=result.get("summary"),
            retry_count=attempts - 1,
            metadata={
                "tool": tool_name,
                "action": tool_request.get("action"),
                "ok": result.get("ok", False),
                "path": result.get("path"),
            },
        )
        runtime_event_bus.emit(
            {
                "type": "tool_executed",
                "workflow_id": self.shared_memory.workflow_id,
                "task_id": task.id,
                "agent": current_agent,
                "tool": tool_name,
                "action": tool_request.get("action"),
                "summary": result.get("summary", ""),
                "ok": result.get("ok", False),
            }
        )
        if tool_name == "file_tool" and tool_request.get("action") in {"edit_file", "write_file"}:
            runtime_event_bus.emit(
                {
                    "type": "fix_applied",
                    "workflow_id": self.shared_memory.workflow_id,
                    "task_id": task.id,
                    "agent": current_agent,
                    "file": result.get("path") or arguments.get("path"),
                    "summary": result.get("summary", ""),
                }
            )
        LOGGER.info(
            "task=%s attempt=%s agent=%s tool=%s action=%s ok=%s path=%s",
            task.id,
            attempts,
            current_agent,
            tool_name,
            tool_request.get("action"),
            result.get("ok", False),
            result.get("path"),
        )
        return result

    async def _evaluate_output(
        self,
        *,
        task: TaskBlueprint,
        output: str,
        observation: dict[str, Any],
        attempt: int,
        max_attempts: int,
        policy_profile: Any,
        cache_hint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate output quality using the multi-critic runtime layer."""
        if not output:
            return {}
        return await self.multi_critic.evaluate(
            task=task,
            output=output,
            observation=observation,
            attempt=attempt,
            max_attempts=max_attempts,
            weights=policy_profile.critic_weights,
            confidence_target=policy_profile.confidence_target,
            cache_hint=cache_hint,
        )

    def _record_evaluation_decision(
        self,
        *,
        task: TaskBlueprint,
        current_agent: str,
        attempts: int,
        evaluation: dict[str, Any],
        execution_context: ExecutionContext,
    ) -> None:
        """Emit a structured decision row for multi-critic evaluation."""
        execution_context.trace.record_decision(
            decision_type="evaluation",
            task_id=task.id,
            agent_selected=current_agent,
            reason=evaluation.get("summary", "multi-critic evaluation completed"),
            confidence=evaluation.get("confidence"),
            retry=not evaluation.get("ok", True),
            metadata={
                "attempt": attempts,
                "failure_type": evaluation.get("failure_type"),
                "dominant_critic": evaluation.get("dominant_critic"),
                "critic_scores": evaluation.get("critic_scores", {}),
                "weights_used": evaluation.get("weights_used", {}),
                "evaluated_critics": evaluation.get("evaluated_critics", []),
                "skipped_critics": evaluation.get("skipped_critics", []),
                "lazy_path": evaluation.get("lazy_path"),
                "cached": evaluation.get("cached", False),
                "cache_signature": evaluation.get("cache_signature"),
                "critics": evaluation.get("critics", []),
            },
        )
        runtime_event_bus.emit(
            {
                "type": "critic_scored",
                "workflow_id": self.shared_memory.workflow_id,
                "task_id": task.id,
                "agent": current_agent,
                "attempt": attempts,
                "confidence": evaluation.get("confidence"),
                "critic_scores": dict(evaluation.get("critic_scores", {})),
                "summary": evaluation.get("summary"),
            }
        )

    def _record_cache_decision(
        self,
        *,
        task: TaskBlueprint,
        current_agent: str,
        cache_entry: dict[str, Any],
        execution_context: ExecutionContext,
    ) -> None:
        """Emit a structured decision row when a cached task decision is reused."""
        execution_context.trace.record_decision(
            decision_type="decision_cache",
            task_id=task.id,
            agent_selected=current_agent,
            reason="Reused a cached high-confidence task decision before running expensive critics.",
            confidence=cache_entry.get("decayed_confidence", cache_entry.get("expected_confidence")),
            metadata={
                "signature": cache_entry.get("signature"),
                "strategy": cache_entry.get("strategy"),
                "success_rate": cache_entry.get("success_rate"),
                "avg_attempts": cache_entry.get("avg_attempts"),
                "decay_factor": cache_entry.get("decay_factor"),
            },
        )

    def _record_policy_decision(
        self,
        *,
        task: TaskBlueprint,
        current_agent: str,
        attempts: int,
        policy_decision: dict[str, Any],
        execution_context: ExecutionContext,
    ) -> None:
        """Emit a structured decision row for policy control."""
        execution_context.trace.record_decision(
            decision_type="policy",
            task_id=task.id,
            agent_selected=current_agent,
            reason=policy_decision.get("reason", "policy decision recorded"),
            confidence=policy_decision.get("confidence_target"),
            retry=policy_decision.get("should_retry", False),
            metadata={
                "attempt": attempts,
                "mode": policy_decision.get("mode"),
                "action": policy_decision.get("action"),
                "failure_type": policy_decision.get("failure_type"),
                "critic_weights": policy_decision.get("critic_weights", {}),
                "retry_budget": policy_decision.get("retry_budget", {}),
                "trust_memory": policy_decision.get("trust_memory", True),
                "force_strategy_change": policy_decision.get("force_strategy_change", False),
                "continue_current_strategy": policy_decision.get("continue_current_strategy", False),
                "preferred_strategy": policy_decision.get("preferred_strategy"),
                "note": policy_decision.get("note"),
                "policy_metadata": policy_decision.get("metadata", {}),
            },
        )

    def _record_retry_decision(
        self,
        *,
        task: TaskBlueprint,
        current_agent: str,
        attempts: int,
        reflection: dict[str, Any],
        observation: dict[str, Any],
        execution_context: ExecutionContext,
    ) -> None:
        """Emit a structured decision record for retries or fallbacks."""
        next_agent = reflection.get("next_agent", current_agent)
        fallback_triggered = bool(reflection.get("fallback_triggered", False))
        decision_type = "fallback" if fallback_triggered else "retry"
        execution_context.trace.record_decision(
            decision_type=decision_type,
            task_id=task.id,
            agent_selected=next_agent,
            reason=reflection.get("reason", observation.get("summary", "retry requested")),
            confidence=observation.get("confidence"),
            retry=True,
            fallback_triggered=fallback_triggered,
            metadata={
                "attempt": attempts,
                "strategy": reflection.get("strategy", task.retry_strategy),
                "failure_type": reflection.get("failure_type"),
                "note": reflection.get("note"),
                "current_agent": current_agent,
                "policy_action": reflection.get("metadata", {}).get("policy_action") or reflection.get("policy_action"),
            },
        )

    def _record_parallel_decision(
        self,
        *,
        ready_tasks: list[TaskBlueprint],
        batch: list[TaskBlueprint],
        parallel_plan: dict[str, Any],
        trace: ExecutionTrace,
    ) -> None:
        """Emit a structured decision row for workflow-level parallel scheduling."""
        trace.record_decision(
            decision_type="parallel_batch",
            task_id="workflow",
            agent_selected=None,
            reason=parallel_plan.get("reason", "parallel scheduling decision recorded"),
            metadata={
                "parallel": bool(parallel_plan.get("parallel", False)),
                "ready_tasks": [task.id for task in ready_tasks],
                "selected_batch": [task.id for task in batch],
                "max_parallel_tasks": parallel_plan.get("max_parallel_tasks", 1),
            },
        )

    def _derive_final_output(
        self,
        blueprint: WorkflowBlueprint,
        execution_by_task: dict[str, TaskExecutionResult],
        executions: list[TaskExecutionResult],
    ) -> str:
        """Choose the most meaningful final output after sequential or parallel execution."""
        preferred_task_ids = [
            task.id
            for task in blueprint.tasks
            if task.output_key == "final_response" or task.task_type in {"solution", "file_write", "memory_store"}
        ]
        for task_id in reversed(preferred_task_ids):
            result = execution_by_task.get(task_id)
            if result and result.output:
                return result.output

        for result in reversed(executions):
            if result.output:
                return result.output
        return ""

    def _resolve_project_context(self, blueprint: WorkflowBlueprint) -> dict[str, Any]:
        """Merge any explicit project-mode context with blueprint metadata."""
        context = dict(self.project_context or {})
        metadata = dict(getattr(blueprint, "metadata", {}) or {})
        project_enabled = bool(context.get("enabled") or metadata.get("project_mode"))
        project_root = context.get("project_root") or metadata.get("project_root")
        if project_enabled and project_root and not context.get("project_context"):
            try:
                scan = self.code_reader.scan(project_root)
            except Exception:
                scan = {
                    "project_root": str(Path(project_root).expanduser().resolve()),
                    "project_signature": metadata.get("project_signature"),
                    "frameworks": list(metadata.get("project_frameworks", [])),
                    "languages": list(metadata.get("project_languages", [])),
                    "files": [],
                    "directories": [],
                    "entrypoints": [],
                    "scripts": {},
                    "summary_text": None,
                }
            context.update(
                {
                    "enabled": True,
                    "project_root": scan.get("project_root") or str(Path(project_root).expanduser().resolve()),
                    "project_signature": context.get("project_signature") or scan.get("project_signature") or metadata.get("project_signature"),
                    "project_context": scan,
                    "recent_goals": list(context.get("recent_goals", metadata.get("project_recent_goals", []))),
                    "common_errors": list(context.get("common_errors", metadata.get("project_common_errors", []))),
                    "user_preferences": dict(context.get("user_preferences", metadata.get("project_preferences", {}))),
                }
            )
        return context

    def _final_confidence(self, executions: list[TaskExecutionResult]) -> float | None:
        """Return the last measured confidence across executed tasks."""
        for execution in reversed(executions):
            confidence = execution.observation.get("confidence")
            if confidence is not None:
                return float(confidence)
        return None

    def _stringify_output(self, value: Any) -> str:
        """Normalize agent output into text for downstream evaluation."""
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        try:
            return json.dumps(value, ensure_ascii=True)
        except TypeError:
            return str(value)

    def _sanitize_tool_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Keep tool traces inspectable without logging entire file payloads."""
        sanitized = dict(arguments or {})
        for key in ("content", "old_text", "new_text"):
            if key in sanitized:
                value = str(sanitized[key])
                sanitized[f"{key}_chars"] = len(value)
                sanitized[f"{key}_preview"] = value[:120]
                sanitized.pop(key, None)
        return sanitized

    def _update_workspace_state(self, tool_request: dict[str, Any], tool_result: dict[str, Any]) -> None:
        """Persist the latest workspace-facing state for coding retries and explainability."""
        if self._active_workspace is None:
            return

        if tool_request.get("tool") == "terminal_tool":
            self.shared_memory.put("workspace.last_command", list(tool_result.get("command") or []))
            error_text = ""
            if not tool_result.get("ok", False):
                error_text = "\n".join(
                    part
                    for part in (
                        tool_result.get("summary"),
                        tool_result.get("stderr", ""),
                        tool_result.get("stdout", ""),
                    )
                    if part
                )[:4000]
            self.shared_memory.put("workspace.last_terminal_error", error_text or None)
        if tool_request.get("tool") == "file_tool":
            self.shared_memory.put("workspace.last_file_action", tool_result.get("summary"))

        self.shared_memory.put("workspace.last_tool_result", tool_result)
        self.shared_memory.put("workspace.project_state", self._active_workspace.snapshot())

    def _checkpoint_before_attempt(self, task: TaskBlueprint, attempt: int) -> str | None:
        """Checkpoint managed workspaces before retry attempts only."""
        if (
            not self._active_workspace_managed
            or self._active_workspace is None
            or self._active_git_tool is None
            or attempt <= 1
        ):
            return None
        try:
            return self._active_git_tool.checkpoint(
                self._active_workspace.root_dir,
                message=f"nexus: before retry {task.id} attempt {attempt}",
            )
        except Exception as error:  # pragma: no cover - defensive path
            LOGGER.warning("git checkpoint failed for %s attempt %s: %s", task.id, attempt, error)
            return None

    def _rollback_failed_attempt(self, task: TaskBlueprint, checkpoint_sha: str | None, attempt: int) -> None:
        """Restore the last safe state after a failed retry attempt."""
        if (
            not checkpoint_sha
            or not self._active_workspace_managed
            or self._active_workspace is None
            or self._active_git_tool is None
        ):
            return
        try:
            sha = self._active_git_tool.rollback_to(self._active_workspace.root_dir, checkpoint_sha)
            runtime_event_bus.emit(
                {
                    "type": "fix_applied",
                    "workflow_id": self.shared_memory.workflow_id,
                    "task_id": task.id,
                    "summary": f"Rolled back failed retry attempt {attempt} to {sha[:8]}",
                }
            )
        except Exception as error:  # pragma: no cover - defensive path
            LOGGER.warning("git rollback failed for %s attempt %s: %s", task.id, attempt, error)

    def _checkpoint_success(self, task: TaskBlueprint, attempt: int) -> None:
        """Persist a successful managed-workspace state."""
        if (
            not self._active_workspace_managed
            or self._active_workspace is None
            or self._active_git_tool is None
        ):
            return
        try:
            self._active_git_tool.checkpoint(
                self._active_workspace.root_dir,
                message=f"nexus: success {task.id} attempt {attempt}",
            )
        except Exception as error:  # pragma: no cover - defensive path
            LOGGER.warning("git success checkpoint failed for %s: %s", task.id, error)

    def _collect_touched_files(self, executions: list[dict[str, Any]], workspace_root: Path) -> list[str]:
        """Extract workspace file paths touched by tool actions or project builds."""
        touched: list[str] = []
        for execution in executions:
            observation = execution.get("observation", {}) or {}
            for action in observation.get("tool_actions", []) or []:
                path = action.get("path")
                if path:
                    normalized = self._normalize_workspace_path(path, workspace_root)
                    if normalized:
                        touched.append(normalized)
                for file_path in action.get("files_written", []) or []:
                    normalized = self._normalize_workspace_path(file_path, workspace_root)
                    if normalized:
                        touched.append(normalized)
        deduped: list[str] = []
        seen = set()
        for item in touched:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped

    def _normalize_workspace_path(self, raw_path: Any, workspace_root: Path) -> str | None:
        text = str(raw_path or "").strip()
        if not text:
            return None
        candidate = Path(text)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(workspace_root.resolve()).as_posix()
            except ValueError:
                return None
        return candidate.as_posix()

    def _prepare_task_prompt(
        self,
        *,
        task: TaskBlueprint,
        agent_name: str,
        prompt: str,
        attempts: int,
        execution_context: ExecutionContext,
    ) -> tuple[str, ContextReductionResult | None]:
        """Reduce oversized prompts before they reach an agent."""
        if self.context_reducer is None:
            return prompt, None

        reduction = self.context_reducer.reduce(
            prompt,
            metadata={
                "task_id": task.id,
                "agent": agent_name,
                "attempt": attempts,
            },
        )
        if not reduction.reduced:
            return prompt, None

        payload = reduction.to_dict()
        self.shared_memory.put("context_reduction.last", payload)
        self.shared_memory.put(f"context_reduction:{task.id}:{attempts}", payload)
        self.shared_memory.append_event("context.reduced", task.id, payload)
        execution_context.trace.record_decision(
            decision_type="context_reduction",
            task_id=task.id,
            agent_selected=agent_name,
            reason=(
                f"Reduced prompt from {reduction.original_length} to "
                f"{reduction.reduced_length} chars using {reduction.backend}."
            ),
            retry=attempts > 1,
            metadata=payload,
        )
        return reduction.text, reduction
