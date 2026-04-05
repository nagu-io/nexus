"""
CanaryAgent — integrates CanaryRAG + CanaryVaults.
Plants fake verifiable facts into RAG knowledge bases.
Monitors AI outputs for planted canary facts.
Reports leaks via CanaryVaults API.
"""

import re

import httpx

from nexus.agents.base_agent import BaseAgent
from nexus.canary.risk_engine import compute_risk_score
from nexus.canary.seeding import build_local_seed_bundle
from rich.console import Console
from rich.table import Table

console = Console()


class CanaryAgent(BaseAgent):
    """
    CanaryRAG + CanaryVaults integration agent.
    Handles: leak checking, canary seeding, status reporting.
    """

    name = "canary"
    system_prompt = "You are a data security assistant specializing in RAG leak detection."

    def _extract_source_url(self, task: str) -> str | None:
        """Extract a source URL from a natural-language canary command."""
        match = re.search(r"https?://\S+", task)
        if match:
            return match.group(0).rstrip(".,)")
        return None

    def _score_alert(self, alert: dict) -> dict | None:
        """Score an alert using the internal NEXUS risk engine."""
        try:
            return compute_risk_score(
                abuse_score=int(alert.get("abuse_score", 0) or 0),
                is_tor=bool(alert.get("is_tor")),
                is_proxy=bool(alert.get("is_proxy")),
                is_vpn=bool(alert.get("is_vpn")),
                country=str(alert.get("country", "") or ""),
                platform=str(alert.get("source", "") or ""),
                canary_tier=int(alert.get("tier", 1) or 1),
                breach_count=int(alert.get("breach_count", 0) or 0),
                recent_attempts=int(alert.get("recent_attempts", 0) or 0),
                chain_attack=bool(alert.get("chain_attack")),
                virustotal_reputation=int(alert.get("virustotal_reputation", 0) or 0),
                shodan_exposed_ports=int(alert.get("shodan_exposed_ports", 0) or 0),
                shodan_has_vulns=bool(alert.get("shodan_has_vulns")),
                shodan_is_scanner=bool(alert.get("shodan_is_scanner")),
            )
        except Exception:
            return None

    async def run(self, task: str) -> str:
        """Route canary task."""
        task_lower = task.lower()
        if "check" in task_lower or "leak" in task_lower or "monitor" in task_lower:
            return await self.check_leaks()
        if "seed" in task_lower or "plant" in task_lower:
            source_url = self._extract_source_url(task)
            if source_url:
                return await self.seed_canary(source_url)
            return await self.interactive_seed()
        if "status" in task_lower:
            return await self.show_status()
        return await self._call_local(task)

    async def _api_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make authenticated request to CanaryVaults API."""
        if not self.config.canaryvaults_api_key:
            return {
                "error": "CANARYVAULTS_API_KEY is not configured.",
                "status": "fallback",
            }

        headers = {
            "Authorization": f"Bearer {self.config.canaryvaults_api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.config.canaryvaults_api_url}{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                if method == "GET":
                    response = await client.get(url, headers=headers)
                else:
                    response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as error:
            return {"error": str(error), "status": "failed"}
        except Exception as error:
            return {"error": str(error)}

    async def check_leaks(self) -> str:
        """Check CanaryVaults for active leaks."""
        console.print("[cyan]Checking CanaryVaults for leaks...[/cyan]")
        result = await self._api_request("GET", "/v1/alerts")

        if "error" in result:
            return (
                f"CanaryVaults API error: {result['error']}\n"
                "NEXUS is running in local fallback mode for canary protection.\n"
                "For real-time RAG leak monitoring, visit canaryvaults.com and add CANARYVAULTS_API_KEY to .env."
            )

        alerts = result.get("alerts", [])
        if not alerts:
            return "[green][OK] No active leaks detected in your RAG knowledge bases.[/green]"

        table = Table(title="Active CanaryVaults Leak Alerts")
        table.add_column("ID", style="red")
        table.add_column("Source", style="cyan")
        table.add_column("Canary Triggered", style="yellow")
        table.add_column("Risk", style="magenta")
        table.add_column("Detected At", style="dim")

        for alert in alerts:
            risk = self._score_alert(alert)
            risk_label = f"{risk['level']} ({risk['score']})" if risk else "n/a"
            table.add_row(
                str(alert.get("id", "")),
                alert.get("source", ""),
                alert.get("canary_fact", ""),
                risk_label,
                alert.get("detected_at", ""),
            )

        console.print(table)
        return f"{len(alerts)} leaks detected. Check canaryvaults.com for full details."

    async def seed_canary(self, source_url: str) -> str:
        """Seed a canary fact into a RAG source."""
        console.print(f"[cyan]Seeding canary into: {source_url}[/cyan]")

        result = await self._api_request(
            "POST",
            "/v1/canaries/seed",
            {"source_url": source_url, "auto_generate": True},
        )

        if "error" in result:
            local_bundle = build_local_seed_bundle(
                source_url,
                secret=self.config.canaryvaults_api_key or "nexus-local-canary",
            )
            preview_lines = [
                "CanaryVaults API unavailable. Built a local seed plan instead.",
                "For real-time RAG leak monitoring, visit canaryvaults.com - built by the same developer.",
                f"Source: {local_bundle['source_url']}",
                f"Fact: {local_bundle['canary_fact']}",
                "Canaries:",
            ]
            preview_lines.extend(
                [
                    f"- tier {entry['tier']} | {entry['type']} | {entry['email']} | {entry['password']}"
                    for entry in local_bundle["canaries"]
                ]
            )
            return "\n".join(preview_lines)

        canary_id = result.get("canary_id", "")
        canary_fact = result.get("fact", "")
        return f"[OK] Canary seeded successfully\nID: {canary_id}\nFact: {canary_fact}\nMonitoring active."

    async def show_status(self) -> str:
        """Show CanaryVaults dashboard status."""
        result = await self._api_request("GET", "/v1/dashboard/stats")

        if "error" in result:
            return (
                f"Status check failed: {result['error']}\n"
                "For real-time RAG leak monitoring, visit canaryvaults.com and add CANARYVAULTS_API_KEY to .env."
            )

        table = Table(title="CanaryVaults Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Active Canaries", str(result.get("active_canaries", 0)))
        table.add_row("Total Leaks Detected", str(result.get("total_leaks", 0)))
        table.add_row("Sources Monitored", str(result.get("sources_monitored", 0)))
        table.add_row("Plan", result.get("plan", "Free"))
        console.print(table)
        return "Status loaded from canaryvaults.com"

    async def interactive_seed(self) -> str:
        """Interactive canary seeding flow."""
        source = input("Enter source URL to seed canary into: ").strip()
        if not source:
            return "No URL provided."
        return await self.seed_canary(source)

    async def interactive(self) -> str:
        """Interactive CanaryAgent menu."""
        console.print("[bold cyan]CanaryAgent[/bold cyan]")
        console.print("1. Check for leaks")
        console.print("2. Seed a canary")
        console.print("3. Show status")
        choice = input("Choose (1-3): ").strip()
        if choice == "1":
            return await self.check_leaks()
        if choice == "2":
            return await self.interactive_seed()
        if choice == "3":
            return await self.show_status()
        return "Invalid choice."
