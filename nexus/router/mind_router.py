"""
AEON Mind Router - intelligent task routing with ReflectScore trust gating.
Classifies task complexity and intent, then routes to:
  - Local CompressX model via Ollama
  - Cloud model via Anthropic API
  - Specialist agent
ReflectScore then decides whether to serve, warn, or block and reroute.
"""

import asyncio
from typing import Optional

from rich.console import Console
from rich.table import Table

from nexus.router.provider_runtime import log_token_usage, retry_async

console = Console()

INTENT_KEYWORDS = {
    "coding": ["write code", "build", "implement", "debug", "fix bug", "function", "class", "script", "python", "rust", "javascript", "refactor", "test"],
    "research": ["search", "find", "what is", "explain", "summarize", "who is", "how does", "latest", "news", "research"],
    "memory": ["remember", "recall", "what did i", "last time", "history", "previous", "stored", "save this"],
    "file": ["read file", "write file", "open", "save", "delete", "list files", "directory", "folder", "create file"],
    "canary": ["canary", "leak", "rag", "canaryvaults", "canaryrag", "protect", "data leak", "monitor"],
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


class MindRouter:
    """
    AEON Mind Router.
    Routes tasks to the best executor, then lets ReflectScore decide whether to
    serve, warn on, or block and reroute the answer.
    """

    def __init__(self):
        from nexus.config import config

        self.config = config
        self.stats = {"local": 0, "cloud": 0, "agent": 0}
        self.reflect_stats = {"clean": 0, "warning": 0, "blocked": 0, "rerouted": 0}
        self._agents = {}

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
        return self._agents[name]

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

    def _record_reflect_result(self, verdict: str, rerouted: bool):
        """Record the final ReflectScore outcome in router stats."""
        if verdict in self.reflect_stats:
            self.reflect_stats[verdict] += 1
        if rerouted:
            self.reflect_stats["rerouted"] += 1

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

        console.print(f"[dim]Router: intent={intent}, complexity={complexity:.2f}[/dim]")

        if intent and intent in ["coding", "research", "memory", "file", "canary"]:
            initial_route = "agent"
            console.print(f"[cyan]-> Routing to {intent} agent[/cyan]")
            self.stats["agent"] += 1
            executor = self._get_agent(intent)
            initial_response = await executor.run(task)
        elif complexity < self.config.routing_complexity_threshold:
            initial_route = "local"
            console.print(f"[green]-> Routing to local model ({self.config.nexus_model})[/green]")
            self.stats["local"] += 1
            initial_response = await self._call_local(task)
        else:
            initial_route = "cloud"
            console.print("[yellow]-> Routing to cloud model (Anthropic)[/yellow]")
            self.stats["cloud"] += 1
            initial_response = await self._call_cloud(task)

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
            final_response = await self._call_cloud(task)
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
        }

    async def _call_local(self, task: str) -> str:
        """Call the local Ollama model."""
        try:
            import httpx

            async def operation() -> str:
                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.post(
                        f"{self.config.ollama_base_url}/api/generate",
                        json={"model": self.config.nexus_model, "prompt": task, "stream": False},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    text = payload.get("response", "No response from local model")
                    log_token_usage(
                        provider="ollama",
                        model=self.config.nexus_model,
                        prompt=task,
                        response_text=text,
                        input_tokens=payload.get("prompt_eval_count"),
                        output_tokens=payload.get("eval_count"),
                    )
                    return text

            return await retry_async("ollama", operation)
        except Exception:
            console.print("[yellow]Local model is unavailable. Trying cloud fallback.[/yellow]")
            cloud_response = await self._call_cloud(task)
            if cloud_response.startswith("Cloud model error:"):
                return (
                    "No model backend is available. Start Ollama with `ollama serve` "
                    "or install/configure Anthropic for cloud fallback."
                )
            return cloud_response

    async def _call_cloud(self, task: str) -> str:
        """Call the Anthropic cloud model as fallback."""
        if not self.config.anthropic_api_key:
            return "Cloud model error: ANTHROPIC_API_KEY is not configured."
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)

            async def operation() -> str:
                message = await asyncio.to_thread(
                    client.messages.create,
                    model="claude-sonnet-4-20250514",
                    max_tokens=2048,
                    messages=[{"role": "user", "content": task}],
                )
                text = message.content[0].text
                usage = getattr(message, "usage", None)
                log_token_usage(
                    provider="anthropic",
                    model="claude-sonnet-4-20250514",
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
        table.add_row("Cloud (Anthropic)", str(self.stats["cloud"]))
        table.add_row("Agent", str(self.stats["agent"]))
        table.add_row("Reflect Clean", str(self.reflect_stats["clean"]))
        table.add_row("Reflect Warning", str(self.reflect_stats["warning"]))
        table.add_row("Reflect Blocked", str(self.reflect_stats["blocked"]))
        table.add_row("Reflect Re-routed", str(self.reflect_stats["rerouted"]))
        console.print(table)
