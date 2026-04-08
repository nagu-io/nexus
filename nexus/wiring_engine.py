"""
Agent wiring engine for the NEXUS compiler/runtime system.

Maps blueprint tasks onto reusable agent instances and builds the prompt each
agent receives from shared memory instead of direct task-to-task coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from nexus.agents.base_agent import BaseAgent
from nexus.blueprint_generator import TaskBlueprint, WorkflowBlueprint
from nexus.critics.base import BaseCritic
from nexus.plugins.loader import PluginLoader
from nexus.runtime.trace import ExecutionContext
from nexus.shared_memory import SharedMemory


@dataclass
class WiredTask:
    """Executable task after agent resolution and prompt composition."""

    task: TaskBlueprint
    agent: BaseAgent
    prompt: str
    selected_agent_name: str
    selection_reason: str
    selection_details: dict


class WiringEngine:
    """Registry and composition layer for workflow agents."""

    def __init__(self, auto_register: bool = True, plugin_loader: PluginLoader | None = None):
        self._factories: dict[str, Callable[[], BaseAgent]] = {}
        self._instances: dict[str, BaseAgent] = {}
        self._tool_factories: dict[str, Callable[[], object]] = {}
        self._tool_instances: dict[str, object] = {}
        self._critic_factories: dict[str, Callable[..., BaseCritic]] = {}
        self._critic_instances: dict[str, BaseCritic] = {}
        self._plugin_loader = plugin_loader or PluginLoader()
        self._plugin_specs: list[dict] = []
        if auto_register:
            self._register_defaults()
            self._register_plugins()

    def register(self, name: str, factory: Callable[[], BaseAgent]) -> None:
        """Register a reusable agent factory."""
        self._factories[name] = factory

    def register_tool(self, name: str, factory: Callable[[], object]) -> None:
        """Register a reusable tool factory."""
        self._tool_factories[name] = factory

    def register_critic(self, name: str, factory: Callable[..., BaseCritic]) -> None:
        """Register a reusable critic factory."""
        self._critic_factories[name] = factory

    def available_agents(self) -> list[str]:
        """Return the current registry."""
        return sorted(self._factories)

    def available_tools(self) -> list[str]:
        """Return tool registrations, including plugin tools."""
        return sorted(self._tool_factories)

    def available_critics(self) -> list[str]:
        """Return critic registrations, including plugin critics."""
        return sorted(self._critic_factories)

    def discovered_plugins(self) -> list[dict]:
        """Return compact metadata about auto-loaded plugins."""
        return list(self._plugin_specs)

    def resolve(self, name: str) -> BaseAgent:
        """Return a singleton-like agent instance for a workflow."""
        if name not in self._factories:
            raise KeyError(f"Unknown agent: {name}")
        if name not in self._instances:
            self._instances[name] = self._factories[name]()
        return self._instances[name]

    def resolve_tool(self, name: str) -> object:
        """Return a singleton-like tool instance."""
        if name not in self._tool_factories:
            raise KeyError(f"Unknown tool: {name}")
        if name not in self._tool_instances:
            self._tool_instances[name] = self._tool_factories[name]()
        return self._tool_instances[name]

    def resolve_critics(self, *, reflect_scorer=None) -> list[BaseCritic]:
        """Instantiate plugin critics, passing the reflect scorer when accepted."""
        critics: list[BaseCritic] = []
        for name in sorted(self._critic_factories):
            if name not in self._critic_instances:
                factory = self._critic_factories[name]
                try:
                    critic = factory(reflect_scorer=reflect_scorer)
                except TypeError:
                    critic = factory()
                self._critic_instances[name] = critic
            critics.append(self._critic_instances[name])
        return critics

    def wire_task(
        self,
        task: TaskBlueprint,
        blueprint: WorkflowBlueprint,
        memory: SharedMemory,
        execution_context: ExecutionContext | None = None,
    ) -> WiredTask:
        """Attach an agent and memory-derived context to a blueprint task."""
        agent, decision = self.resolve_for_task(task, execution_context=execution_context)
        prompt = self._compose_prompt(task, blueprint, memory, execution_context=execution_context)
        return WiredTask(
            task=task,
            agent=agent,
            prompt=prompt,
            selected_agent_name=agent.name,
            selection_reason=decision["reason"],
            selection_details=decision,
        )

    def resolve_for_task(
        self,
        task: TaskBlueprint,
        execution_context: ExecutionContext | None = None,
    ) -> tuple[BaseAgent, dict]:
        """Select the best available agent for a task using capabilities and overrides."""
        override = execution_context.get_task_strategy(task.id) if execution_context else {}
        requested_capabilities = list(override.get("required_capabilities") or task.required_capabilities or [])
        preferred_agent = override.get("next_agent") or task.agent

        candidate_order = []
        if preferred_agent:
            candidate_order.append(preferred_agent)
        for candidate in override.get("candidate_agents", task.candidate_agents or []):
            if candidate not in candidate_order:
                candidate_order.append(candidate)
        if not candidate_order:
            candidate_order = self.available_agents()

        selected_agent = None
        selection_reason = ""
        matched_capabilities = []

        for candidate in candidate_order:
            agent = self.resolve(candidate)
            if not requested_capabilities or agent.supports_capabilities(requested_capabilities):
                selected_agent = agent
                matched_capabilities = requested_capabilities
                if candidate == preferred_agent:
                    if requested_capabilities:
                        selection_reason = (
                            f"selected '{candidate}' because it satisfies capabilities: "
                            f"{', '.join(requested_capabilities)}"
                        )
                    else:
                        selection_reason = f"selected preferred agent '{candidate}' with no capability constraints"
                else:
                    selection_reason = (
                        f"selected '{candidate}' because preferred agent '{preferred_agent}' "
                        f"did not satisfy capabilities: {', '.join(requested_capabilities)}"
                    )
                break

        if selected_agent is None:
            if preferred_agent:
                selected_agent = self.resolve(preferred_agent)
                selection_reason = (
                    f"defaulted to preferred agent '{preferred_agent}' even though no candidate "
                    f"advertised all required capabilities: {', '.join(requested_capabilities)}"
                )
            else:
                raise KeyError(f"No agent satisfies task capabilities: {requested_capabilities}")

        decision = {
            "selected_agent": selected_agent.name,
            "preferred_agent": preferred_agent,
            "candidate_order": candidate_order,
            "required_capabilities": requested_capabilities,
            "matched_capabilities": matched_capabilities,
            "reason": selection_reason,
            "strategy_override": bool(override),
        }
        if execution_context and execution_context.trace:
            execution_context.trace.record_decision(
                decision_type="agent_selection",
                task_id=task.id,
                agent_selected=selected_agent.name,
                reason=selection_reason,
                retry=bool(override),
                fallback_triggered=bool(override.get("next_agent") and override.get("next_agent") != task.agent),
                metadata=decision,
            )
        return selected_agent, decision

    def alternative_agents(
        self,
        task: TaskBlueprint,
        exclude: list[str] | None = None,
    ) -> list[str]:
        """Return alternate agents that satisfy the task capabilities."""
        excluded = set(exclude or [])
        candidates = task.candidate_agents or self.available_agents()
        alternatives = []
        for candidate in candidates:
            if candidate in excluded:
                continue
            agent = self.resolve(candidate)
            if not task.required_capabilities or agent.supports_capabilities(task.required_capabilities):
                alternatives.append(candidate)
        return alternatives

    def _compose_prompt(
        self,
        task: TaskBlueprint,
        blueprint: WorkflowBlueprint,
        memory: SharedMemory,
        execution_context: ExecutionContext | None = None,
    ) -> str:
        dependency_context = memory.dependency_context(task.depends_on)
        prior_outputs = []
        for dependency_id, payload in dependency_context.items():
            output = payload.get("output", "")
            if output:
                prior_outputs.append(f"{dependency_id}: {output}")

        sections = [
            f"Workflow goal: {blueprint.goal}",
            f"Primary intent: {blueprint.primary_intent}",
            f"Task type: {task.task_type}",
            f"Task: {task.instruction}",
        ]
        if prior_outputs:
            sections.append("Shared memory context:\n" + "\n\n".join(prior_outputs))
        if blueprint.metadata.get("constraints"):
            sections.append("Constraints: " + ", ".join(blueprint.metadata["constraints"]))
        workspace_root = memory.get("workspace.root_dir")
        if workspace_root:
            sections.append(f"Workspace root: {workspace_root}")
        project_state = memory.get("workspace.project_state")
        if project_state:
            files = project_state.get("files", [])
            if files:
                file_lines = [
                    f"- {item['path']} ({item.get('size', 0)} bytes)"
                    for item in files[:12]
                ]
                sections.append("Workspace files:\n" + "\n".join(file_lines))
        last_error = memory.get("workspace.last_terminal_error")
        if last_error:
            sections.append(f"Terminal feedback:\n{last_error}")
        last_command = memory.get("workspace.last_command")
        if last_command:
            sections.append(f"Last command: {' '.join(last_command)}")
        project_context = memory.get("project.context")
        if project_context:
            frameworks = ", ".join(project_context.get("frameworks", [])) or "unknown"
            languages = ", ".join(project_context.get("languages", [])) or "unknown"
            sections.append(f"Project context: frameworks={frameworks}; languages={languages}")
            summary_text = project_context.get("summary_text")
            if summary_text:
                sections.append(f"Project summary: {summary_text}")
            key_files = [
                f"- {item['path']}"
                for item in (project_context.get("files", []) or [])[:10]
            ]
            if key_files:
                sections.append("Project files:\n" + "\n".join(key_files))
        project_session = memory.get("project.session")
        if project_session:
            recent_goals = project_session.get("recent_goals", [])
            if recent_goals:
                sections.append("Recent project goals:\n" + "\n".join(f"- {goal}" for goal in recent_goals[-5:]))
        project_preferences = memory.get("project.user_preferences")
        if project_preferences:
            preferred_frameworks = ", ".join(project_preferences.get("preferred_frameworks", []))
            preferred_languages = ", ".join(project_preferences.get("preferred_languages", []))
            if preferred_frameworks or preferred_languages:
                sections.append(
                    "User preferences: "
                    f"frameworks={preferred_frameworks or 'unknown'}; "
                    f"languages={preferred_languages or 'unknown'}"
                )
        common_errors = memory.get("project.common_errors")
        if common_errors:
            error_lines = [
                f"- {item.get('failure_type', 'runtime_error')}: {item.get('summary', '')}"
                for item in common_errors[:3]
            ]
            sections.append("Common project errors:\n" + "\n".join(error_lines))
        if task.fallback:
            sections.append(f"Fallback strategy: {task.fallback}")
        if task.confidence_threshold > 0:
            sections.append(f"Minimum confidence required: {task.confidence_threshold:.2f}")

        strategy_override = execution_context.get_task_strategy(task.id) if execution_context else {}
        if strategy_override.get("strategy"):
            sections.append(f"Adaptive retry strategy: {strategy_override['strategy']}")
        if strategy_override.get("note"):
            sections.append(f"Retry note: {strategy_override['note']}")
        return "\n\n".join(sections)

    def _register_defaults(self) -> None:
        from nexus.agents.coding_agent import CodingAgent
        from nexus.agents.file_agent import FileAgent
        from nexus.agents.memory_agent import MemoryAgent
        from nexus.agents.research_agent import ResearchAgent
        from nexus.canary.canary_agent import CanaryAgent

        self.register("coding", CodingAgent)
        self.register("research", ResearchAgent)
        self.register("memory", MemoryAgent)
        self.register("file", FileAgent)
        self.register("canary", CanaryAgent)

    def _register_plugins(self) -> None:
        for spec in self._plugin_loader.discover():
            factory = spec.factory
            if factory is None:
                continue
            if spec.plugin_type == "agent":
                self.register(spec.name, factory)
            elif spec.plugin_type == "tool":
                self.register_tool(spec.name, factory)
            elif spec.plugin_type == "critic":
                self.register_critic(spec.name, factory)
            self._plugin_specs.append(
                {
                    "name": spec.name,
                    "type": spec.plugin_type,
                    "capabilities": list(spec.capabilities),
                    "source": spec.source,
                }
            )
