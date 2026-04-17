"""
AEON Mind Router - intelligent task routing with ReflectScore trust gating.
Classifies task complexity and intent, then routes to:
  - Local CompressX model via Ollama
  - Cloud model via OpenRouter or Anthropic API
  - Specialist agent
ReflectScore then decides whether to serve, warn, or block and reroute.
"""

import asyncio
from pathlib import Path
import re
from typing import Optional

from rich.console import Console
from rich.table import Table

from nexus.router.provider_runtime import (
    active_local_model_label,
    call_local_chat,
    call_openrouter_chat,
    log_token_usage,
    preferred_cloud_provider,
    retry_async,
)
from nexus.runtime.context_reducer import BaseContextReducer, ContextReductionResult, build_context_reducer

console = Console()

LOCAL_ROUTER_SYSTEM_PROMPT = (
    "You are NEXUS, a local repo-first AI coding assistant. "
    "Be concise, accurate, and action-oriented. Prefer repository facts, real commands, "
    "and honest uncertainty over generic claims. Do not invent frameworks, files, endpoints, "
    "or capabilities that are not present."
)

INTENT_KEYWORDS = {
    "coding": ["write code", "build", "implement", "debug", "fix bug", "function", "class", "script", "python", "rust", "javascript", "refactor", "test"],
    "research": ["search", "find", "what is", "explain", "summarize", "who is", "how does", "latest", "news", "research"],
    "memory": ["remember", "recall", "what did i", "last time", "history", "previous", "stored", "save this"],
    "file": ["read file", "write file", "open", "save", "delete", "list files", "directory", "folder", "create file"],
    "canary": ["canary", "leak", "rag", "canaryvaults", "canaryrag", "protect", "data leak", "monitor"],
    "design": ["design", "ui", "ux", "beautiful", "tailwind", "css", "frontend", "appearance", "style", "look"],
}

COMPLEXITY_HIGH_SIGNALS = [
    "architect",
    "design system",
    "production",
    "deploy",
    "scale",
    "explain in detail",
    "comprehensive",
    "full implementation",
    "entire",
    "complete project",
]

WORKSPACE_GROUNDING_SIGNALS = [
    "this repo",
    "this repository",
    "this codebase",
    "this project",
    "current project",
    "current repository",
    "workspace",
    "repo structure",
    "repository structure",
    "codebase structure",
    "in this repo",
    "in this repository",
    "in this project",
]

CAPABILITY_QUERY_SIGNALS = [
    "what can you build",
    "what you can build",
    "what can you do",
    "how can you help",
    "what do you do",
    "what are your capabilities",
    "show me what you can do",
    "best demo",
    "starter prompt",
    "where should i start",
    "what should i try first",
]

NEXUS_IDENTITY_SIGNALS = [
    "what is nexus",
    "how does nexus",
    "what can nexus",
    "nexus in one sentence",
    "this dashboard",
]

HIVE_ROUTE_SIGNALS = [
    "nexus hive",
    "/hive",
    "distributed intelligence",
    "distributed search",
    "swarm",
    "mesh",
    "parallelize",
    "parallel search",
    "zero cost",
    "donate idle compute",
]


class MindRouter:
    """
    AEON Mind Router.
    Routes tasks to the best executor, then lets ReflectScore decide whether to
    serve, warn on, or block and reroute the answer.
    """

    def __init__(self, *, context_reducer: BaseContextReducer | None = None):
        from nexus.config import config

        self.config = config
        self.stats = {"local": 0, "cloud": 0, "agent": 0, "hive": 0}
        self.reflect_stats = {"clean": 0, "warning": 0, "blocked": 0, "rerouted": 0}
        self._agents = {}
        self._hive_runtime = None
        self.context_reducer = context_reducer or build_context_reducer(
            enabled=config.context_reduction_enabled,
            backend=config.context_reduction_backend,
            threshold_chars=config.context_reduction_threshold_chars,
            target_chars=config.context_reduction_target_chars,
            model_name=config.context_reduction_model,
        )

    def _get_agent(self, name: str):
        """Lazy-load agents to avoid circular imports."""
        if name not in self._agents:
            if name == "coding":
                from nexus.agents.coding_agent import CodingAgent

                self._agents[name] = CodingAgent()
            elif name == "research":
                from nexus.agents.research_agent import ResearchAgent

                self._agents[name] = ResearchAgent()
            elif name == "memory":
                from nexus.agents.memory_agent import MemoryAgent

                self._agents[name] = MemoryAgent()
            elif name == "file":
                from nexus.agents.file_agent import FileAgent

                self._agents[name] = FileAgent()
            elif name == "canary":
                from nexus.canary.canary_agent import CanaryAgent

                self._agents[name] = CanaryAgent()
            elif name == "design":
                from nexus.agents.design_agent import DesignAgent

                self._agents[name] = DesignAgent()
        return self._agents[name]

    def _get_hive_runtime(self):
        """Lazy-load Hive runtime to avoid circular imports and cold-start cost."""
        if self._hive_runtime is None:
            from nexus.hive.runtime import HiveRuntime

            self._hive_runtime = HiveRuntime()
        return self._hive_runtime

    def classify_intent(self, task: str) -> Optional[str]:
        """Classify task intent based on keyword matching."""
        task_lower = task.lower()
        scores = {}
        for intent, keywords in INTENT_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in task_lower)
            if score > 0:
                scores[intent] = score
        return max(scores, key=scores.get) if scores else None

    def score_complexity(self, task: str) -> float:
        """Score task complexity from 0.0 (simple) to 1.0 (very complex)."""
        task_lower = task.lower()
        score = 0.0
        words = len(task.split())

        if words > 50:
            score += 0.3
        elif words > 20:
            score += 0.15

        for signal in COMPLEXITY_HIGH_SIGNALS:
            if signal in task_lower:
                score += 0.2
                break

        if "?" in task and words < 15:
            score -= 0.1

        return min(max(score, 0.0), 1.0)

    def _should_ground_in_workspace(self, task: str) -> bool:
        """Return True when the prompt is about this local repo/project, not the public web."""
        task_lower = task.lower()
        if any(signal in task_lower for signal in WORKSPACE_GROUNDING_SIGNALS + CAPABILITY_QUERY_SIGNALS + NEXUS_IDENTITY_SIGNALS):
            return True
        capability_patterns = (
            r"\bwhat (can|do)\s+(you|nexus)\b",
            r"\bhow can\s+(you|nexus)\b",
            r"\bwhat you can\b",
            r"\bwhere should i start\b",
            r"\bwhat should i try first\b",
        )
        return any(re.search(pattern, task_lower) for pattern in capability_patterns)

    def _should_route_to_hive(
        self,
        *,
        task: str,
        intent: Optional[str],
        complexity: float,
        workspace_grounded: bool,
    ) -> bool:
        """Return True when the experimental distributed Hive route is the best fit."""
        if not getattr(self.config, "hive_enabled", False):
            return False
        if workspace_grounded:
            return False
        if intent not in {None, "coding", "research", "design"}:
            return False
        task_lower = task.lower()
        explicit_signal = any(signal in task_lower for signal in HIVE_ROUTE_SIGNALS)
        implicit_signal = complexity >= max(self.config.routing_complexity_threshold, 0.45) and any(
            phrase in task_lower
            for phrase in ("subtask", "subtasks", "best answer", "multiple nodes", "many nodes", "volunteer compute")
        )
        return explicit_signal or implicit_signal

    def _workspace_root(self) -> Path:
        """Resolve the local project root for repository-grounded answers."""
        cwd = Path.cwd().resolve()
        if (cwd / "README.md").exists() or (cwd / "pyproject.toml").exists():
            return cwd
        return Path(__file__).resolve().parents[2]

    def _workspace_context(self) -> str:
        """Collect a small, stable snapshot of the local workspace for grounded prompts."""
        root = self._workspace_root()
        hidden_entries = {".git", ".venv", "node_modules", "__pycache__", "dist", "build"}
        entries = []
        try:
            for item in sorted(root.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower())):
                if item.name in hidden_entries:
                    continue
                prefix = "[DIR]" if item.is_dir() else "[FILE]"
                entries.append(f"{prefix} {item.name}")
                if len(entries) >= 12:
                    break
        except Exception:
            entries = []

        sections = [
            f"Workspace root: {root}",
            "Top-level entries:\n" + ("\n".join(entries) if entries else "Unavailable"),
            "Known product surfaces:\n- CLI runtime via main.py\n- FastAPI API via nexus/api.py\n- React dashboard via dashboard/src/App.jsx\n- Runtime explainability via nexus/runtime/insights.py",
        ]

        file_specs = [
            ("README.md", "README excerpt", 1600),
            ("pyproject.toml", "Python project metadata", 800),
            ("package.json", "Root workspace scripts", 500),
            ("dashboard/package.json", "Dashboard scripts", 500),
        ]
        for relative_path, label, limit in file_specs:
            excerpt = self._read_workspace_file(root / relative_path, limit)
            if excerpt:
                sections.append(f"{label} ({relative_path}):\n{excerpt}")

        return "\n\n".join(sections)

    def _read_workspace_file(self, path: Path, limit: int) -> str:
        """Read a small excerpt from a workspace file for local grounding."""
        try:
            if not path.exists():
                return ""
            return path.read_text(encoding="utf-8", errors="replace")[:limit]
        except Exception:
            return ""

    def _build_workspace_prompt(self, task: str, intent: Optional[str]) -> str:
        """Build a repository-grounded prompt for local capability or repo questions."""
        focus = (
            "Summarize what this local NEXUS project can do right now and suggest 2 or 3 strong next prompts or commands."
            if any(signal in task.lower() for signal in CAPABILITY_QUERY_SIGNALS)
            else "Answer using the local repository context and mention the most relevant files when helpful."
        )
        return (
            "You are answering questions about the local NEXUS repository in the current workspace.\n"
            "Use only the repository context below. Do not rely on web search or outside facts.\n"
            "Treat NEXUS as this local project, not any outside product or acronym.\n"
            "Do not mention alternate meanings of NEXUS.\n"
            "Do not write code unless the user explicitly asks for code.\n"
            "If the context is incomplete, say what is clear from the repo and what remains uncertain.\n"
            "Keep the answer concise, concrete, and grounded.\n\n"
            f"Detected intent: {intent or 'unknown'}\n"
            f"Instruction: {focus}\n\n"
            f"User question:\n{task}\n\n"
            f"Repository context:\n{self._workspace_context()}"
        )

    def _record_reflect_result(self, verdict: str, rerouted: bool):
        """Record the final ReflectScore outcome in router stats."""
        if verdict in self.reflect_stats:
            self.reflect_stats[verdict] += 1
        if rerouted:
            self.reflect_stats["rerouted"] += 1

    def _prepare_route_prompt(
        self,
        prompt: str,
        *,
        route: str,
        intent: Optional[str],
        workspace_grounded: bool,
    ) -> tuple[str, ContextReductionResult | None]:
        """Reduce oversized route prompts while keeping the original task for trust scoring."""
        if self.context_reducer is None:
            return prompt, None

        reduction = self.context_reducer.reduce(
            prompt,
            metadata={
                "scope": "router",
                "route": route,
                "intent": intent or "unknown",
                "workspace_grounded": workspace_grounded,
            },
        )
        if reduction.reduced:
            console.print(
                f"[dim]Context reduced {reduction.original_length} -> "
                f"{reduction.reduced_length} chars via {reduction.backend}[/dim]"
            )
            return reduction.text, reduction
        return prompt, None

    async def _call_agent_runtime(self, task: str, agent_name: str) -> str:
        """Run a specialist agent through the orchestrator instead of calling it directly."""
        from nexus.blueprint_generator import TaskBlueprint, WorkflowBlueprint
        from nexus.orchestrator import Orchestrator

        prototype = self._get_agent(agent_name)
        confidence_threshold = 0.55 if "reasoning" in getattr(prototype, "capabilities", ()) else 0.0
        retry_strategy = "tighten_prompt" if confidence_threshold > 0 else "repeat"

        blueprint = WorkflowBlueprint(
            goal=task,
            primary_intent=agent_name,
            tasks=[
                TaskBlueprint(
                    id=f"{agent_name}_1",
                    task_type=f"router_{agent_name}",
                    agent=agent_name,
                    instruction=task,
                    retries=1,
                    timeout_seconds=30,
                    output_key="router_response",
                    retry_strategy=retry_strategy,
                    confidence_threshold=confidence_threshold,
                    required_capabilities=list(getattr(prototype, "capabilities", ())),
                    candidate_agents=[agent_name],
                )
            ],
            metadata={"source": "mind_router"},
        )
        result = await Orchestrator().run_blueprint(blueprint)
        return result["final_output"]

    async def _call_hive(self, task: str, intent: Optional[str]) -> tuple[str, dict]:
        """Run the experimental Hive route and return the synthesized answer plus metadata."""
        runtime = self._get_hive_runtime()
        result = await runtime.demo(task, intent=intent or "coding")
        assembled = result.get("assembled_output") or (result.get("winner") or {}).get("output") or result.get("note") or ""
        return assembled, result

    async def route(self, task: str, force_agent: Optional[str] = None, return_meta: bool = False):
        """
        Route a task to the best executor and trust-gate the answer.

        Args:
            task: the user's task/query
            force_agent: optionally force a specific agent
            return_meta: return the full routing/trust metadata when True
        """
        result = await self.route_with_reflection(task, force_agent=force_agent)
        return result if return_meta else result["response"]

    async def route_with_reflection(self, task: str, force_agent: Optional[str] = None) -> dict:
        """Return the full routing result including ReflectScore trust metadata."""
        from nexus.reflect.reflect_score import ReflectScore

        scorer = ReflectScore()
        intent = force_agent or self.classify_intent(task)
        complexity = self.score_complexity(task)
        workspace_grounded = not force_agent and self._should_ground_in_workspace(task)

        console.print(
            f"[dim]Router: intent={intent}, complexity={complexity:.2f}, "
            f"workspace_grounded={workspace_grounded}[/dim]"
        )

        prepared_prompt = task
        context_reduction = None
        hive_details = None

        if workspace_grounded:
            initial_route = "local"
            console.print("[green]-> Routing to workspace-grounded local model[/green]")
            self.stats["local"] += 1
            prepared_prompt, context_reduction = self._prepare_route_prompt(
                self._build_workspace_prompt(task, intent),
                route=initial_route,
                intent=intent,
                workspace_grounded=workspace_grounded,
            )
            initial_response = await self._call_local(prepared_prompt)
        elif self._should_route_to_hive(
            task=task,
            intent=intent,
            complexity=complexity,
            workspace_grounded=workspace_grounded,
        ):
            initial_route = "hive"
            console.print("[bold cyan]-> Routing to NEXUS Hive[/bold cyan]")
            self.stats["hive"] += 1
            prepared_prompt, context_reduction = self._prepare_route_prompt(
                task,
                route=initial_route,
                intent=intent,
                workspace_grounded=workspace_grounded,
            )
            initial_response, hive_details = await self._call_hive(prepared_prompt, intent)
        elif intent and intent in ["coding", "research", "memory", "file", "canary", "design"]:
            initial_route = "agent"
            console.print(f"[cyan]-> Routing to {intent} agent[/cyan]")
            self.stats["agent"] += 1
            prepared_prompt, context_reduction = self._prepare_route_prompt(
                task,
                route=initial_route,
                intent=intent,
                workspace_grounded=workspace_grounded,
            )
            initial_response = await self._call_agent_runtime(prepared_prompt, intent)
        elif complexity < self.config.routing_complexity_threshold:
            initial_route = "local"
            console.print(f"[green]-> Routing to local model ({active_local_model_label(self.config)})[/green]")
            self.stats["local"] += 1
            prepared_prompt, context_reduction = self._prepare_route_prompt(
                task,
                route=initial_route,
                intent=intent,
                workspace_grounded=workspace_grounded,
            )
            initial_response = await self._call_local(prepared_prompt)
        else:
            initial_route = "cloud"
            cloud_provider = preferred_cloud_provider(self.config) or "cloud"
            console.print(f"[yellow]-> Routing to cloud model ({cloud_provider.title()})[/yellow]")
            self.stats["cloud"] += 1
            prepared_prompt, context_reduction = self._prepare_route_prompt(
                task,
                route=initial_route,
                intent=intent,
                workspace_grounded=workspace_grounded,
            )
            initial_response = await self._call_cloud(prepared_prompt)

        initial_assessment = await scorer.assess_response(task, initial_response)
        console.print(
            f"[dim]ReflectScore: {initial_assessment['score']:.2f} "
            f"({initial_assessment['verdict']} -> {initial_assessment['action']})[/dim]"
        )

        final_route = initial_route
        final_response = initial_response
        final_assessment = initial_assessment
        warning = initial_assessment["warning"]
        was_rerouted = False

        if initial_assessment["should_reroute"] and initial_route != "cloud":
            console.print(
                f"[red][WARN] ReflectScore blocked the initial answer "
                f"({initial_assessment['score']:.2f}). Re-routing to cloud.[/red]"
            )
            final_route = "cloud"
            was_rerouted = True
            final_response = await self._call_cloud(prepared_prompt)
            final_assessment = await scorer.assess_response(task, final_response)
            console.print(
                f"[dim]ReflectScore after re-route: {final_assessment['score']:.2f} "
                f"({final_assessment['verdict']} -> {final_assessment['action']})[/dim]"
            )
            if final_assessment["action"] == "serve":
                warning = (
                    f"ReflectScore blocked the first answer at {initial_assessment['score']:.2f} "
                    "and served the stronger model response instead."
                )
            elif final_assessment["action"] == "warn":
                warning = (
                    f"ReflectScore blocked the first answer at {initial_assessment['score']:.2f}. "
                    f"The stronger model response is still medium risk ({final_assessment['score']:.2f})."
                )
            else:
                warning = (
                    "ReflectScore blocked both the initial and stronger-model answers. "
                    "NEXUS withheld the response instead of showing a likely hallucination."
                )

        if final_assessment["action"] == "block":
            final_response = scorer.blocked_response(task, final_assessment["score"])

        self._record_reflect_result(final_assessment["verdict"], rerouted=was_rerouted)

        return {
            "response": final_response,
            "agent": intent if initial_route == "agent" else None,
            "intent": intent,
            "complexity": complexity,
            "initial_route": initial_route,
            "final_route": final_route,
            "was_rerouted": was_rerouted,
            "warning": warning,
            "reflect_score": final_assessment["score"],
            "reflect_verdict": final_assessment["verdict"],
            "reflect_action": final_assessment["action"],
            "initial_reflect_score": initial_assessment["score"],
            "initial_reflect_verdict": initial_assessment["verdict"],
            "workspace_grounded": workspace_grounded,
            "hive_details": hive_details,
            "context_reduction": context_reduction.to_dict() if context_reduction is not None else None,
        }

    async def _call_local(self, task: str) -> str:
        """Call the configured local runtime."""
        try:
            text, usage = await call_local_chat(
                config=self.config,
                prompt=task,
                system=LOCAL_ROUTER_SYSTEM_PROMPT,
            )
            log_token_usage(
                provider=usage.get("provider", "local"),
                model=usage.get("model", self.config.nexus_model),
                prompt=task,
                response_text=text,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
            )
            return text
        except Exception:
            console.print("[yellow]Local model is unavailable. Trying cloud fallback.[/yellow]")
            cloud_response = await self._call_cloud(task)
            if cloud_response.startswith("Cloud model error:"):
                return (
                    "No model backend is available. Start the configured local runtime "
                    "or configure OpenRouter or Anthropic for cloud fallback."
                )
            return cloud_response

    async def _call_cloud(self, task: str) -> str:
        """Call the configured cloud model as fallback."""
        if self.config.openrouter_api_key:
            try:
                text, usage = await call_openrouter_chat(
                    api_key=self.config.openrouter_api_key,
                    model=self.config.openrouter_model,
                    prompt=task,
                    base_url=self.config.openrouter_base_url,
                )
                log_token_usage(
                    provider="openrouter",
                    model=self.config.openrouter_model,
                    prompt=task,
                    response_text=text,
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                )
                return text
            except Exception:
                return (
                    "Cloud model error: OpenRouter request failed. "
                    "Please check your OPENROUTER_API_KEY, OPENROUTER_MODEL, and network connection."
                )

        if not self.config.anthropic_api_key:
            return "Cloud model error: configure OPENROUTER_API_KEY or ANTHROPIC_API_KEY."
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)

            async def operation() -> str:
                message = await asyncio.to_thread(
                    client.messages.create,
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    messages=[{"role": "user", "content": task}],
                )
                text = message.content[0].text
                usage = getattr(message, "usage", None)
                log_token_usage(
                    provider="anthropic",
                    model="claude-sonnet-4-6",
                    prompt=task,
                    response_text=text,
                    input_tokens=getattr(usage, "input_tokens", None),
                    output_tokens=getattr(usage, "output_tokens", None),
                )
                return text

            return await retry_async("anthropic", operation)
        except ImportError:
            return "Cloud model error: anthropic SDK not installed."
        except Exception:
            return (
                "Cloud model error: Anthropic request failed. "
                "Please check your ANTHROPIC_API_KEY and network connection."
            )

    def show_status(self):
        """Display routing and ReflectScore stats in a rich table."""
        table = Table(title="AEON Mind Router Stats")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")
        table.add_row("Local (CompressX)", str(self.stats["local"]))
        table.add_row(f"Cloud ({(preferred_cloud_provider(self.config) or 'none').title()})", str(self.stats["cloud"]))
        table.add_row("Agent", str(self.stats["agent"]))
        table.add_row("Hive", str(self.stats["hive"]))
        table.add_row("Reflect Clean", str(self.reflect_stats["clean"]))
        table.add_row("Reflect Warning", str(self.reflect_stats["warning"]))
        table.add_row("Reflect Blocked", str(self.reflect_stats["blocked"]))
        table.add_row("Reflect Re-routed", str(self.reflect_stats["rerouted"]))
        table.add_row(
            "Context Reduction",
            getattr(self.context_reducer, "backend_name", "custom") if self.context_reducer is not None else "disabled",
        )
        console.print(table)
