"""ResearchAgent — web search, page scraping, chunk selection, and summarization."""

from __future__ import annotations

import html
import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from nexus.agents.base_agent import BaseAgent
from rich.console import Console

try:  # pragma: no cover - dependency may be absent in test environments
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - fallback path is exercised instead
    BeautifulSoup = None

console = Console()


class ResearchAgent(BaseAgent):
    """
    Research agent with web search, page scraping, and summarization.
    Uses DuckDuckGo for search, scrapes top results, summarizes with local model.
    """

    name = "research"
    capabilities = ("reasoning", "web_search", "summarization", "web_scraping")
    system_prompt = """You are a research assistant.
You find accurate, relevant information and summarize it clearly.
You always cite where information comes from.
You distinguish between facts and opinions.
Keep summaries concise and actionable."""

    async def run(self, task: str) -> str:
        """Research a topic: search → scrape → summarize."""
        console.print(f"[cyan]ResearchAgent: Searching for: {task[:60]}[/cyan]")

        # Search the web
        search_results = await self._search(task)

        if not search_results:
            return await self._call_local(task)

        # Try to scrape actual page content from top results
        scraped_content = []
        urls_to_scrape = [
            r.get("url", "")
            for r in search_results[:3]
            if r.get("url", "").startswith("http")
        ]

        for url in urls_to_scrape[:3]:
            content = await self._scrape_page(url)
            if content:
                chunks = self._relevant_chunks(task, content)
                if chunks:
                    scraped_content.append(
                        f"Source: {url}\n" + "\n\n".join(chunks)
                    )

        # Build context from scraped pages (preferred) or search snippets (fallback)
        if scraped_content:
            context = "\n\n---\n\n".join(scraped_content)
            console.print(f"[green]ResearchAgent: Scraped {len(scraped_content)} pages[/green]")
        else:
            context = "\n\n".join([
                f"Source: {r['title']}\n{r['snippet']}"
                for r in search_results[:5]
            ])
            console.print("[yellow]ResearchAgent: Using search snippets (scraping unavailable)[/yellow]")

        prompt = (
            f"Based on these sources, answer the question: {task}\n\n"
            f"Sources:\n{context}\n\n"
            f"Provide a clear, well-cited answer."
        )
        return await self._call_local(prompt)

    async def _search(self, query: str) -> list:
        """Search the web and return result dictionaries with title/snippet/url."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                html_response = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "NEXUS-Research/1.0"},
                    follow_redirects=True,
                )
                html_response.raise_for_status()
                results = self._parse_duckduckgo_results(html_response.text)
                if results:
                    return results[:8]

                # Conservative fallback when HTML parsing yields nothing.
                r = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                )
                data = r.json()
                fallback = []
                if data.get("AbstractText"):
                    fallback.append(
                        {
                            "title": data.get("Heading", ""),
                            "snippet": data["AbstractText"],
                            "url": data.get("AbstractURL", ""),
                        }
                    )
                for topic in data.get("RelatedTopics", [])[:6]:
                    if isinstance(topic, dict) and "Text" in topic:
                        fallback.append(
                            {
                                "title": topic.get("FirstURL", ""),
                                "snippet": topic["Text"],
                                "url": topic.get("FirstURL", ""),
                            }
                        )
                return fallback
        except Exception:
            console.print("[yellow]Web search is unavailable right now. Falling back to model-only mode.[/yellow]")
            return []

    async def _scrape_page(self, url: str) -> str | None:
        """Fetch and extract main text content from a URL.

        Uses BeautifulSoup when available, otherwise falls back to regex-based
        extraction so the agent still works in constrained environments.
        """
        try:
            async with httpx.AsyncClient(
                timeout=10,
                follow_redirects=True,
                headers={"User-Agent": "NEXUS-Research/1.0 (local AI assistant)"},
            ) as client:
                r = await client.get(url)
                if r.status_code != 200:
                    return None
                content_type = r.headers.get("content-type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return None
                if "text/plain" in content_type:
                    return r.text
                return self._extract_text_from_html(r.text)
        except Exception:
            return None

    @staticmethod
    def _extract_text_from_html(page_html: str) -> str:
        """Extract readable text from HTML without external dependencies."""
        if BeautifulSoup is not None:
            soup = BeautifulSoup(page_html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()
            candidates = [
                soup.find("main"),
                soup.find("article"),
                max(
                    soup.find_all(["section", "div"], recursive=True),
                    key=lambda item: len(item.get_text(" ", strip=True)),
                    default=None,
                ),
                soup.body,
            ]
            container = next((candidate for candidate in candidates if candidate is not None), soup)
            text = container.get_text(" ", strip=True)
            return ResearchAgent._normalize_text(text)

        # Remove script, style, nav, footer, header tags and their content
        for tag in ["script", "style", "nav", "footer", "header", "aside", "noscript"]:
            page_html = re.sub(
                rf"<{tag}[^>]*>.*?</{tag}>",
                " ",
                page_html,
                flags=re.DOTALL | re.IGNORECASE,
            )
        # Remove all remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", page_html)
        text = html.unescape(text)
        return ResearchAgent._normalize_text(text)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        lines = [line.strip() for line in text.split(". ") if len(line.strip()) > 30]
        return ". ".join(lines[:120])

    def _parse_duckduckgo_results(self, page_html: str) -> list[dict]:
        if BeautifulSoup is not None:
            soup = BeautifulSoup(page_html, "html.parser")
            results = []
            for result in soup.select(".result")[:8]:
                link = result.select_one(".result__a")
                snippet = result.select_one(".result__snippet")
                url = self._clean_result_url(link.get("href", "")) if link else ""
                if link and url:
                    results.append(
                        {
                            "title": link.get_text(" ", strip=True),
                            "snippet": snippet.get_text(" ", strip=True) if snippet else "",
                            "url": url,
                        }
                    )
            return results

        results = []
        pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
            r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>|'
            r'<div[^>]+class="result__snippet"[^>]*>(?P<snippet_div>.*?)</div>',
            re.DOTALL,
        )
        for match in pattern.finditer(page_html):
            href = self._clean_result_url(match.group("href") or "")
            if not href:
                continue
            title = re.sub(r"<[^>]+>", " ", match.group("title") or "")
            snippet = re.sub(r"<[^>]+>", " ", (match.group("snippet") or match.group("snippet_div") or ""))
            results.append(
                {
                    "title": html.unescape(title).strip(),
                    "snippet": html.unescape(snippet).strip(),
                    "url": href,
                }
            )
            if len(results) >= 8:
                break
        return results

    def _clean_result_url(self, href: str) -> str:
        href = href or ""
        if href.startswith("//"):
            return f"https:{href}"
        if href.startswith("http://") or href.startswith("https://"):
            return href
        if "uddg=" in href:
            parsed = urlparse(href)
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            return unquote(target)
        return ""

    def _relevant_chunks(self, query: str, text: str, *, chunk_chars: int = 1600, limit: int = 3) -> list[str]:
        """Chunk long pages and keep the most query-relevant slices."""
        cleaned = text.strip()
        if not cleaned:
            return []
        if len(cleaned) <= chunk_chars:
            return [cleaned[:chunk_chars]]

        query_terms = {term.lower() for term in re.findall(r"\w+", query) if len(term) > 2}
        chunks = []
        for start in range(0, len(cleaned), chunk_chars):
            chunk = cleaned[start:start + chunk_chars]
            tokens = {term.lower() for term in re.findall(r"\w+", chunk) if len(term) > 2}
            overlap = len(query_terms & tokens)
            score = overlap + (2 if query.lower() in chunk.lower() else 0)
            chunks.append((score, chunk))
        chunks.sort(key=lambda item: item[0], reverse=True)
        best = [chunk for score, chunk in chunks if score > 0][:limit]
        return best or [cleaned[:chunk_chars]]
