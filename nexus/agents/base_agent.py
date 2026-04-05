"""
Base agent class. All NEXUS agents inherit from this.
Provides: run(), system_prompt, name, async HTTP client.
"""

import asyncio
from abc import ABC, abstractmethod

from rich.console import Console

from nexus.router.provider_runtime import log_token_usage, retry_async

console = Console()


class BaseAgent(ABC):
    """Abstract base for all NEXUS agents."""

    name: str = "base"
    system_prompt: str = "You are a helpful AI assistant."

    def __init__(self):
        from nexus.config import config
        self.config = config

    @abstractmethod
    async def run(self, task: str) -> str:
        """Execute the task and return a response."""
        pass

    async def _call_local(self, prompt: str, system: str = None) -> str:
        """Call local Ollama model."""
        import httpx

        system_prompt = system or self.system_prompt
        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        try:
            async def operation() -> str:
                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.post(
                        f"{self.config.ollama_base_url}/api/generate",
                        json={"model": self.config.nexus_model, "prompt": full_prompt, "stream": False},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    text = payload.get("response", "No response")
                    log_token_usage(
                        provider="ollama",
                        model=self.config.nexus_model,
                        prompt=full_prompt,
                        response_text=text,
                        input_tokens=payload.get("prompt_eval_count"),
                        output_tokens=payload.get("eval_count"),
                    )
                    return text

            return await retry_async("ollama", operation)
        except Exception:
            console.print(
                f"[yellow]{self.name} could not reach the local Ollama model. "
                "Trying cloud fallback.[/yellow]"
            )
            cloud_response = await self._call_cloud(prompt, system)
            if cloud_response.startswith("Cloud error:"):
                return (
                    "No model backend is available. Start Ollama with `ollama serve` "
                    "or install/configure Anthropic for cloud fallback."
                )
            return cloud_response

    async def _call_cloud(self, prompt: str, system: str = None) -> str:
        """Call Anthropic cloud model."""
        system_prompt = system or self.system_prompt
        if not self.config.anthropic_api_key:
            return "Cloud error: ANTHROPIC_API_KEY is not configured."
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
