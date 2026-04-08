"""
NEXUS CLI — main entry point for all commands.
Usage: nexus [command] [options]
"""

import asyncio
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


NEXUS_BANNER = """
 _   _ ________   ___   _ ____  
| \\ | |  ____\\ \\ / / | | / ___| 
|  \\| | |__   \\ V /| | | \\___ \\ 
| . ` |  __|   > < | |_| |___) |
| |\\  | |____ / . \\|  _  |____/ 
|_| \\_|______/_/ \\_\\_| |_|      
Private AI Developer OS - v0.1.0
"""


@click.group()
def cli():
    """NEXUS — Your private AI developer OS."""
    pass


def _print_command_error(command_name: str, error: Exception, hint: str | None = None) -> None:
    """Render a human-readable command failure without a Python traceback."""
    body = f"{command_name} could not complete.\nReason: {error}"
    if hint:
        body += f"\nHint: {hint}"
    console.print(Panel(body, title="[bold red]NEXUS Error[/bold red]", style="red"))


def _probe_ollama() -> dict:
    """Return the current Ollama availability and installed model names."""
    import httpx
    from nexus.config import config

    try:
        response = httpx.get(f"{config.ollama_base_url}/api/tags", timeout=3)
        response.raise_for_status()
        models = [model["name"] for model in response.json().get("models", []) if model.get("name")]
        return {"online": True, "models": models, "error": None}
    except Exception as error:
        return {"online": False, "models": [], "error": str(error)}


def _runtime_checks() -> list[dict]:
    """Collect runtime dependency checks for status and doctor commands."""
    import importlib.util
    from nexus.config import config

    checks: list[dict] = []

    ollama_state = _probe_ollama()
    if not ollama_state["online"]:
        checks.append(
            {
                "level": "error",
                "name": "Ollama",
                "message": "not running - install from ollama.com then run: ollama serve",
            }
        )
    elif config.nexus_model in ollama_state["models"]:
        checks.append(
            {
                "level": "ok",
                "name": "Ollama",
                "message": f"running - default model ready: {config.nexus_model}",
            }
        )
    elif ollama_state["models"]:
        checks.append(
            {
                "level": "warn",
                "name": "Ollama",
                "message": (
                    f"running but default model '{config.nexus_model}' is missing - "
                    f"available: {', '.join(ollama_state['models'])}. Run: ollama pull {config.nexus_model}"
                ),
            }
        )
    else:
        checks.append(
            {
                "level": "warn",
                "name": "Ollama",
                "message": f"running but no models - run: ollama pull {config.nexus_model}",
            }
        )

    has_groq_sdk = importlib.util.find_spec("groq") is not None
    has_anthropic_sdk = importlib.util.find_spec("anthropic") is not None
    has_supabase_sdk = importlib.util.find_spec("supabase") is not None

    if config.groq_api_key and has_groq_sdk:
        checks.append(
            {
                "level": "ok",
                "name": "Groq API",
                "message": "configured - ReflectScore live scoring active",
            }
        )
    elif config.groq_api_key:
        checks.append(
            {
                "level": "warn",
                "name": "Groq API",
                "message": "key is set but the groq package is not installed - run: pip install groq",
            }
        )
    else:
        checks.append(
            {
                "level": "warn",
                "name": "Groq API",
                "message": "not set - ReflectScore will use heuristics only. Get a key at console.groq.com",
            }
        )

    if config.anthropic_api_key and has_anthropic_sdk:
        checks.append(
            {
                "level": "ok",
                "name": "Anthropic API",
                "message": "configured - cloud fallback active",
            }
        )
    elif config.anthropic_api_key:
        checks.append(
            {
                "level": "warn",
                "name": "Anthropic API",
                "message": "key is set but the anthropic package is not installed - run: pip install anthropic",
            }
        )
    else:
        checks.append(
            {
                "level": "warn",
                "name": "Anthropic API",
                "message": "not set - complex tasks will stay on local model",
            }
        )

    if config.supabase_url and config.supabase_key and has_supabase_sdk:
        checks.append(
            {
                "level": "ok",
                "name": "Supabase",
                "message": "configured - persistent memory active",
            }
        )
    elif config.supabase_url and config.supabase_key:
        checks.append(
            {
                "level": "warn",
                "name": "Supabase",
                "message": "credentials are set but the supabase package is not installed - run: pip install supabase",
            }
        )
    else:
        checks.append(
            {
                "level": "warn",
                "name": "Supabase",
                "message": "not set - memory will be session-only. Get a free project at supabase.com",
            }
        )

    if config.canaryvaults_api_key:
        checks.append(
            {
                "level": "ok",
                "name": "CanaryVaults",
                "message": "configured - real leak monitoring active",
            }
        )
    else:
        checks.append(
            {
                "level": "warn",
                "name": "CanaryVaults",
                "message": "not set - canary agent runs in local fallback mode",
            }
        )

    return checks


def _print_runtime_checks(checks: list[dict]) -> None:
    """Render runtime dependency checks in a terminal-safe format."""
    styles = {
        "ok": ("[OK]", "green"),
        "warn": ("[WARN]", "yellow"),
        "error": ("[ERR]", "red"),
    }
    for check in checks:
        label, color = styles[check["level"]]
        console.print(f"[{color}]{label} {check['name']}[/{color}] - {check['message']}")


def _summarize_runtime_checks(checks: list[dict]) -> tuple[int, int, int]:
    """Return ok, warn, and error counts."""
    ok_count = sum(1 for check in checks if check["level"] == "ok")
    warn_count = sum(1 for check in checks if check["level"] == "warn")
    error_count = sum(1 for check in checks if check["level"] == "error")
    return ok_count, warn_count, error_count


@cli.command()
@click.option(
    "--model",
    default=None,
    help="Compression source model alias or Hugging Face model id. Defaults to phi3:mini.",
)
@click.option("--bits", default=4, help="Quantization bits (4 or 8)")
def init(model, bits):
    """Initialize NEXUS: compress a model, benchmark it, and start serving."""
    try:
        console.print(NEXUS_BANNER, style="bold cyan")
        from nexus.compress.compressor import CompressXEngine
        from nexus.config import config
        from nexus.reflect.reflect_score import ReflectScore

        model_name = model or config.nexus_model
        ollama_state = _probe_ollama()

        if not ollama_state["online"]:
            raise RuntimeError(
                "Ollama is not running. Start it with `ollama serve`, "
                f"then install the launch model with `ollama pull {config.nexus_model}`."
            )
        if config.nexus_model not in ollama_state["models"]:
            raise RuntimeError(
                f"Ollama is running but the launch model '{config.nexus_model}' is missing. "
                f"Run: ollama pull {config.nexus_model}"
            )

        console.print(Panel(f"[bold]Initializing NEXUS[/bold]\nModel: {model_name}\nBits: {bits}", style="cyan"))
        if model_name != config.nexus_model:
            console.print(
                f"[yellow]NEXUS is currently optimized for the {config.nexus_model} launch path. "
                f"Using custom compression source: {model_name}[/yellow]"
            )

        engine = CompressXEngine()
        compressed_path = engine.compress(model_name, bits=bits)

        scorer = ReflectScore()
        result = asyncio.run(scorer.benchmark_model(compressed_path))

        console.print(Panel(
            f"[bold green]Model Ready[/bold green]\n"
            f"Accuracy: {result['accuracy']:.2%}\n"
            f"Warning Rate: {result['warning_rate']:.2%}\n"
            f"Hallucination Rate: {result['hallucination_rate']:.2%}\n"
            f"Benchmark Samples: {result['total']}\n"
            f"Benchmark Mode: {result.get('benchmark_mode', 'unknown')}\n"
            f"Compression Ratio: {result['compression_ratio']:.1f}x\n"
            f"Path: {compressed_path}",
            style="green"
        ))
        if result.get("benchmark_warning"):
            console.print(
                Panel(
                    result["benchmark_warning"],
                    title="[bold yellow]Benchmark Note[/bold yellow]",
                    style="yellow",
                )
            )
    except Exception as error:
        _print_command_error("nexus init", error, "Run `nexus doctor` first, then retry.")


@cli.command()
@click.argument("task", required=False)
@click.option("--agent", default=None, help="Force specific agent: coding, research, memory, file, canary")
def code(task, agent):
    """Run a coding task through the AEON Mind Router."""
    try:
        import asyncio
        from nexus.router.mind_router import MindRouter

        router = MindRouter()

        if not task:
            task = click.prompt("What do you want to build")

        console.print(f"[bold cyan]Routing task:[/bold cyan] {task}")
        result = asyncio.run(router.route(task, force_agent=agent, return_meta=True))
        if result["warning"]:
            warning_style = "yellow" if result["reflect_action"] != "block" else "red"
            console.print(Panel(result["warning"], title="[bold]ReflectScore[/bold]", style=warning_style))
        reduction = result.get("context_reduction")
        if reduction and reduction.get("reduced"):
            console.print(
                f"[dim]Context reduced {reduction['original_length']} -> "
                f"{reduction['reduced_length']} chars via {reduction['backend']}[/dim]"
            )
        route_label = result["final_route"]
        if result["was_rerouted"]:
            route_label = f"{result['initial_route']} -> {result['final_route']}"
        console.print(
            Panel(
                result["response"],
                title=f"[bold green]NEXUS Response[/bold green] ({route_label})",
                style="green",
            )
        )
    except Exception as error:
        _print_command_error("nexus code", error, "Check your local model or cloud fallback with `nexus doctor`.")


@cli.command()
@click.option("--voice", is_flag=True, help="Enable voice mode (Whisper + Coqui)")
@click.option("--dashboard", is_flag=True, help="Launch React dashboard")
def chat(voice, dashboard):
    """Start Jarvis — interactive chat interface."""
    try:
        import asyncio
        from nexus.jarvis.interface import JarvisInterface

        if dashboard:
            dashboard_dir = Path(__file__).resolve().parents[1] / "dashboard"
            subprocess.Popen([sys.executable, "-m", "uvicorn", "nexus.api:app", "--port", "8000"])
            console.print("[bold cyan]Dashboard API started on http://localhost:8000[/bold cyan]")
            if dashboard_dir.exists():
                try:
                    subprocess.Popen(["npm", "run", "dev", "--", "--port", "3000"], cwd=dashboard_dir)
                    console.print("[bold cyan]Dashboard UI starting on http://localhost:3000[/bold cyan]")
                except OSError as exc:
                    console.print(
                        f"[yellow]Could not start dashboard UI automatically: {exc}. "
                        "Run `npm run dev` in the dashboard folder if needed.[/yellow]"
                    )

        jarvis = JarvisInterface(voice_mode=voice)
        asyncio.run(jarvis.run())
    except Exception as error:
        _print_command_error("nexus chat", error, "Try `nexus doctor` or run `nexus chat` without `--voice` first.")


@cli.command()
@click.option("--live", is_flag=True, help="Score responses in real time")
@click.option("--compare", default=None, metavar="REFERENCE", help="Compare current model against a saved reference label (e.g. original)")
@click.option("--report", is_flag=True, help="Export benchmark report as JSON")
def reflect(live, compare, report):
    """Run ReflectScore hallucination benchmark."""
    try:
        import asyncio
        from nexus.reflect.reflect_score import ReflectScore

        scorer = ReflectScore()

        if live:
            console.print("[bold cyan]Live hallucination scoring enabled. Starting chat...[/bold cyan]")
            asyncio.run(scorer.live_mode())
        elif compare:
            results = asyncio.run(scorer.compare_models(reference_label=compare))
            scorer.display_comparison(results)
        elif report:
            asyncio.run(scorer.export_report())
        else:
            asyncio.run(scorer.run_benchmark())
    except Exception as error:
        _print_command_error("nexus reflect", error, "Make sure a model backend is available before benchmarking.")


@cli.command()
@click.option("--check", is_flag=True, help="Check for active leaks in your RAG")
@click.option("--seed", default=None, help="Seed a canary fact into a RAG source URL")
@click.option("--status", is_flag=True, help="Show CanaryVaults dashboard status")
def protect(check, seed, status):
    """CanaryRAG + CanaryVaults — protect your RAG from data leaks."""
    try:
        import asyncio
        from nexus.canary.canary_agent import CanaryAgent

        agent = CanaryAgent()
        result = None

        if check:
            result = asyncio.run(agent.check_leaks())
        elif seed:
            result = asyncio.run(agent.seed_canary(seed))
        elif status:
            result = asyncio.run(agent.show_status())
        else:
            result = asyncio.run(agent.interactive())

        if result:
            console.print(result)
    except Exception as error:
        _print_command_error("nexus protect", error, "Check your CanaryVaults settings or use local fallback mode.")


@cli.command()
def route():
    """Show AEON Mind Router status and routing stats."""
    try:
        from nexus.router.mind_router import MindRouter
        router = MindRouter()
        router.show_status()
    except Exception as error:
        _print_command_error("nexus route", error)


@cli.command()
def doctor():
    """Check all NEXUS runtime dependencies and explain what is missing."""
    console.print("[bold cyan]NEXUS Doctor - checking your setup...[/bold cyan]\n")
    checks = _runtime_checks()
    _print_runtime_checks(checks)

    ok_count, warn_count, error_count = _summarize_runtime_checks(checks)
    console.print(
        f"\n[bold]Result:[/bold] {ok_count} ok, {warn_count} warnings, {error_count} errors"
    )
    if error_count == 0:
        console.print("[bold green]NEXUS is ready to run.[/bold green]")
    else:
        console.print("[bold red]Fix the errors above before relying on local execution.[/bold red]")


@cli.command()
def status():
    """Show full NEXUS system status."""
    from nexus.config import config

    console.print(NEXUS_BANNER, style="bold cyan")
    _print_runtime_checks(_runtime_checks())

    console.print(f"\n[bold]Active model:[/bold] {config.nexus_model}")
    console.print(f"[bold]Data directory:[/bold] {config.data_dir}")
    console.print(f"[bold]Router complexity threshold:[/bold] {config.routing_complexity_threshold}")
    console.print(
        f"[bold]ReflectScore thresholds:[/bold] warn >= {config.reflect_warn_threshold}, "
        f"block >= {config.reflect_block_threshold}"
    )
    console.print("[bold]Setup check:[/bold] run `nexus doctor` for guided dependency fixes")


if __name__ == "__main__":
    cli()
