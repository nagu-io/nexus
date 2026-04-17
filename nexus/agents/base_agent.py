"""
Base agent class. All NEXUS agents inherit from this.
Provides: run(), system_prompt, name, async HTTP client.
"""

import asyncio
from abc import ABC
from typing import Any

from rich.console import Console

from nexus.router.provider_runtime import call_local_chat, call_openrouter_chat, log_token_usage, retry_async

console = Console()


class BaseAgent(ABC):
    """Abstract base for all NEXUS agents."""

    name: str = "base"
    system_prompt: str = "You are a helpful AI assistant."
    capabilities: tuple[str, ...] = ("reasoning",)

    def __init__(self):
        from nexus.config import config
        self.config = config

    async def run(self, task: str) -> str:
        """Execute the task and return a response for text-oriented agents."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement run() or override act().")

    def tool_call(
        self,
        *,
        tool: str,
        action: str,
        arguments: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a structured tool request without breaking text-only agents."""
        return {
            "type": "tool_call",
            "tool": tool,
            "action": action,
            "arguments": dict(arguments or kwargs),
        }

    def normalize_tool_call(self, value: Any) -> dict[str, Any] | None:
        """Return a normalized tool request when an agent emitted one."""
        if not isinstance(value, dict):
            return None
        if value.get("type") != "tool_call":
            return None
        tool = value.get("tool") or value.get("tool_name")
        action = value.get("action")
        if not tool or not action:
            return None
        return {
            "type": "tool_call",
            "tool": str(tool),
            "action": str(action),
            "arguments": dict(value.get("arguments") or value.get("args") or {}),
        }

    def supports_capabilities(self, required_capabilities: list[str] | tuple[str, ...]) -> bool:
        """Return True when the agent advertises all required capabilities."""
        return all(capability in self.capabilities for capability in required_capabilities)

    async def think(self, task: str, memory: Any = None) -> dict[str, Any]:
        """Build a lightweight plan for the current task."""
        context = {}
        if memory and hasattr(memory, "agent_context"):
            context = memory.agent_context(self.name)
        return {
            "agent": self.name,
            "task": task,
            "context_keys": sorted(context.keys()),
        }

    async def act(self, task: str, memory: Any = None, thought: dict[str, Any] | None = None) -> str:
        """Execute the agent's primary action."""
        return await self.run(task)

    async def continue_after_tool(
        self,
        task: str,
        tool_result: dict[str, Any],
        memory: Any = None,
        thought: dict[str, Any] | None = None,
    ) -> str | dict[str, Any]:
        """Turn a tool result into a normal agent response by default."""
        if tool_result.get("ok", False):
            return tool_result.get("content") or tool_result.get("summary", "")
        return tool_result.get("summary", "Tool error: tool execution failed")

    async def observe(
        self,
        task: str,
        result: str,
        memory: Any = None,
        thought: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Inspect the outcome and decide whether it looks healthy."""
        result_text = result if isinstance(result, str) else str(result)
        lowered = result_text.lower()
        failure_markers = (
            "error:",
            "cloud error:",
            "failed:",
            "failed.",
            "tool error:",
            "access denied",
            "file not found",
            "invalid choice",
            "no model backend is available",
        )
        ok = not any(marker in lowered for marker in failure_markers)
        return {
            "ok": ok,
            "summary": result_text[:240],
        }

    async def reflect(
        self,
        task: str,
        result: str,
        observation: dict[str, Any],
        memory: Any = None,
        thought: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Turn observations into retry guidance for the orchestrator."""
        if observation.get("ok", False):
            return {
                "should_retry": False,
                "reason": "task completed",
            }
        return {
            "should_retry": True,
            "reason": observation.get("summary", "agent reported a recoverable failure"),
        }

    async def execute_cycle(self, task: str, memory: Any = None) -> dict[str, Any]:
        """
        Run the standard think -> act -> observe -> reflect lifecycle.

        Existing agents remain backward compatible because `act()` delegates to
        the current `run()` implementation unless a subclass overrides it.
        """
        thought = await self.think(task, memory=memory)
        if memory and hasattr(memory, "append_event"):
            memory.append_event("agent.think", self.name, {"task": task, "thought": thought})

        result = await self.act(task, memory=memory, thought=thought)
        if memory and hasattr(memory, "append_event"):
            memory.append_event("agent.act", self.name, {"task": task, "result": result})

        tool_request = self.normalize_tool_call(result)
        if tool_request is not None:
            if memory and hasattr(memory, "append_event"):
                memory.append_event(
                    "agent.tool_request",
                    self.name,
                    {"task": task, "tool_request": tool_request},
                )
            return {
                "agent": self.name,
                "thought": thought,
                "result": "",
                "tool_request": tool_request,
                "observation": {
                    "ok": False,
                    "summary": f"requested tool {tool_request['tool']}::{tool_request['action']}",
                    "tool_requested": True,
                },
                "reflection": {
                    "should_retry": False,
                    "reason": "awaiting tool execution",
                    "strategy": "tool_call",
                },
            }

        observation = await self.observe(task, result, memory=memory, thought=thought)
        if memory and hasattr(memory, "append_event"):
            memory.append_event(
                "agent.observe",
                self.name,
                {"task": task, "observation": observation},
            )

        reflection = await self.reflect(
            task,
            result,
            observation,
            memory=memory,
            thought=thought,
        )
        if memory and hasattr(memory, "append_event"):
            memory.append_event(
                "agent.reflect",
                self.name,
                {"task": task, "reflection": reflection},
            )

        return {
            "agent": self.name,
            "thought": thought,
            "result": result,
            "observation": observation,
            "reflection": reflection,
        }

    async def _call_local(self, prompt: str, system: str = None) -> str:
        """Call the configured local runtime."""
        system_prompt = system or self.system_prompt
        try:
            text, usage = await call_local_chat(
                config=self.config,
                prompt=prompt,
                system=system_prompt,
            )
            log_token_usage(
                provider=usage.get("provider", "local"),
                model=usage.get("model", self.config.nexus_model),
                prompt=prompt,
                response_text=text,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
            )
            return text
        except Exception:
            console.print(
                f"[yellow]{self.name} could not reach the local model runtime. "
                "Trying cloud fallback.[/yellow]"
            )
            cloud_response = await self._call_cloud(prompt, system)
            if cloud_response.startswith("Cloud error:"):
                return (
                    "No model backend is available. Start the configured local runtime "
                    "or configure OpenRouter or Anthropic for cloud fallback."
                )
            return cloud_response

    async def _call_cloud(self, prompt: str, system: str = None) -> str:
        """Call the configured cloud model provider."""
        system_prompt = system or self.system_prompt
        if self.config.openrouter_api_key:
            try:
                text, usage = await call_openrouter_chat(
                    api_key=self.config.openrouter_api_key,
                    model=self.config.openrouter_model,
                    prompt=prompt,
                    system=system_prompt,
                    base_url=self.config.openrouter_base_url,
                )
                log_token_usage(
                    provider="openrouter",
                    model=self.config.openrouter_model,
                    prompt=prompt,
                    response_text=text,
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                )
                return text
            except Exception:
                return (
                    "Cloud error: OpenRouter request failed. "
                    "Check your OPENROUTER_API_KEY, OPENROUTER_MODEL, and network connection."
                )

        if not self.config.anthropic_api_key:
            return "Cloud error: configure OPENROUTER_API_KEY or ANTHROPIC_API_KEY."
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)

            async def operation() -> str:
                message = await asyncio.to_thread(
                    client.messages.create,
                    model="claude-sonnet-4-20250514",
                    max_tokens=2048,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = message.content[0].text
                usage = getattr(message, "usage", None)
                log_token_usage(
                    provider="anthropic",
                    model="claude-sonnet-4-20250514",
                    prompt=prompt,
                    response_text=text,
                    input_tokens=getattr(usage, "input_tokens", None),
                    output_tokens=getattr(usage, "output_tokens", None),
                )
                return text

            return await retry_async("anthropic", operation)
        except ImportError:
            return "Cloud error: anthropic SDK not installed."
        except Exception:
            return (
                "Cloud error: Anthropic request failed. "
                "Check your ANTHROPIC_API_KEY and network connection."
            )
