"""
ResearchAgent — web search, summarization, fact-finding.
Uses DuckDuckGo search (no API key needed) + local model for summarization.
"""

import httpx
from nexus.agents.base_agent import BaseAgent
from rich.console import Console

console = Console()


class ResearchAgent(BaseAgent):
    """
    Research agent with web search and summarization.
    Uses DuckDuckGo for search. Summarizes with local model.
    """

    name = "research"
    system_prompt = """You are a research assistant.
You find accurate, relevant information and summarize it clearly.
You always cite where information comes from.
You distinguish between facts and opinions.
Keep summaries concise and actionable."""

    async def run(self, task: str) -> str:
        """Research a topic and return a summary."""
        console.print(f"[cyan]ResearchAgent: Searching for: {task[:60]}[/cyan]")

        # Search the web
        search_results = await self._search(task)

        if not search_results:
            return await self._call_local(task)

        # Summarize results
        context = "\n\n".join([
            f"Source: {r['title']}\n{r['snippet']}"
            for r in search_results[:5]
        ])
        prompt = f"Based on these search results, answer the question: {task}\n\nSearch results:\n{context}"
        return await self._call_local(prompt)

    async def _search(self, query: str) -> list:
        """Search DuckDuckGo Instant Answer API."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
                )
                data = r.json()
                results = []
                if data.get("AbstractText"):
                    results.append({"title": data.get("Heading", ""), "snippet": data["AbstractText"]})
                for topic in data.get("RelatedTopics", [])[:4]:
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append({"title": topic.get("FirstURL", ""), "snippet": topic["Text"]})
                return results
        except Exception:
            console.print("[yellow]Web search is unavailable right now. Falling back to model-only mode.[/yellow]")
            return []
