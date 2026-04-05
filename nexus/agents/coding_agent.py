"""
CodingAgent — handles code generation, debugging, refactoring.
This is the Claude Code replacement inside NEXUS.
Runs locally via CompressX model. Falls back to cloud for complex tasks.
"""

from nexus.agents.base_agent import BaseAgent
from rich.console import Console
from rich.syntax import Syntax

console = Console()


class CodingAgent(BaseAgent):
    """
    Local coding agent powered by CompressX compressed model.
    Handles: code generation, debugging, refactoring, explanation.
    """

    name = "coding"
    system_prompt = """You are an expert software engineer. 
You write clean, correct, production-ready code.
You always include comments explaining what the code does.
You prefer simple solutions over complex ones.
When debugging, you identify the root cause before suggesting a fix.
Output only the code and a brief explanation. No fluff."""

    async def run(self, task: str) -> str:
        """Execute coding task."""
        console.print(f"[cyan]CodingAgent: {task[:60]}...[/cyan]")

        # Check if debugging request
        if any(w in task.lower() for w in ["debug", "fix", "error", "bug", "not working"]):
            response = await self._debug(task)
        elif any(w in task.lower() for w in ["explain", "what does", "how does"]):
            response = await self._explain(task)
        else:
            response = await self._generate(task)

        # Pretty print code blocks
        if "```" in response:
            console.print(Syntax(response, "python", theme="monokai"))

        return response

    async def _generate(self, task: str) -> str:
        """Generate code for a task."""
        prompt = f"Write code for the following task:\n\n{task}\n\nProvide complete, working code with comments."
        return await self._call_local(prompt)

    async def _debug(self, task: str) -> str:
        """Debug code or error."""
        prompt = f"Debug the following issue:\n\n{task}\n\nIdentify the root cause and provide the fix."
        return await self._call_local(prompt)

    async def _explain(self, task: str) -> str:
        """Explain code."""
        prompt = f"Explain the following clearly and concisely:\n\n{task}"
        return await self._call_local(prompt)
