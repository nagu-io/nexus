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
from rich.table import Table

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
@click.option("--voice", is_flag=True, help="Talk to it: press the talk hotkey, speak, and hear the answer")
@click.option("--watch", is_flag=True, help="Also watch the screen and speak up when something needs you (it never acts on its own)")
@click.option("--goal", default=None, help="Do one request and exit")
@click.option("--dry-run", is_flag=True, help="Say what it would do without clicking or typing anything")
@click.option(
    "--backend",
    type=click.Choice(["local", "llamacpp", "ollama"]),
    default=None,
    help="local: your NEXUS Vision in this process (the default). llamacpp: a GGUF in llama-server. ollama: a model in Ollama",
)
@click.option("--model", default=None, help="Ollama vision model (implies --backend ollama); by default one is picked for this computer")
@click.option(
    "--confirm",
    type=click.Choice(["all", "risky"]),
    default=None,
    help="all: ask before every action (the default). risky: ask before typing and keys only",
)
@click.option("--teach", is_flag=True, help="Teach mode: do tasks yourself and save each step as training data for NEXUS Vision")
@click.option("--teach-dir", default=None, help="Where teach mode saves its examples (default ~/.nexus/live/teach)")
@click.option("--check", "check_only", is_flag=True, help="Check this computer, the vision model and the packages Live needs, then exit")
def live(voice, watch, goal, dry_run, backend, model, confirm, teach, teach_dir, check_only):
    """NEXUS Live: sees your screen, listens, talks, and uses the mouse and keyboard, all on this computer."""
    try:
        from nexus.live.app import LiveSession, checks, resolve_model, session_settings
        from nexus.live.settings import LiveSettings

        settings = session_settings(LiveSettings.from_env(), dry_run=dry_run, model=model, confirm=confirm, backend=backend)
        if check_only:
            styles = {"ok": ("[OK]", "green"), "warn": ("[WARN]", "yellow"), "error": ("[ERR]", "red")}
            for level, name, message in checks(settings):
                label, color = styles[level]
                console.print(f"[{color}]{label} {name}[/{color}] - ", end="")
                console.print(message, markup=False)
            return
        if teach:
            from nexus.live.teach import run_teach

            run_teach(settings, teach_dir, printer=lambda line: console.print(line, markup=False))
            return

        model_name, why = resolve_model(settings)
        session = LiveSession(settings, model_name, printer=lambda line: console.print(line, markup=False))
        console.print(
            Panel(
                f"Model: {model_name} ({why})\n"
                f"Keys: {session.keys_help()}\n"
                + ("Dry run: nothing will be clicked or typed.\n" if settings.dry_run else "")
                + "Throw the mouse into a screen corner to stop an action mid-way.",
                title="[bold cyan]NEXUS Live[/bold cyan]",
                style="cyan",
            )
        )
        session.start_hotkeys()
        session.note_console()  # this console is in front now: NEXUS never acts in it
        if watch:
            session.start_watching()
        try:
            if goal:
                result = session.handle(goal)
                console.print(f"[dim]Outcome: {result.outcome}[/dim]")
            elif voice:
                session.listen_loop()
            else:
                console.print("[dim]Type a request (or exit), then click the window it is for.[/dim]")
                session.type_loop()
        finally:
            session.close()
    except Exception as error:
        _print_command_error("nexus live", error, "Run `nexus live --check` to see what is missing.")


@cli.group()
def vision():
    """NEXUS Vision, your own model for NEXUS Live: fetch data, train, evaluate, download."""


_PASSTHROUGH = {"ignore_unknown_options": True, "allow_extra_args": True, "help_option_names": []}


@vision.command("fetch", context_settings=_PASSTHROUGH)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def vision_fetch(args):
    """Fetch OS-Atlas screenshots for training (--system windows --limit 4000 --out FOLDER; --help for all)."""
    from nexus.live.atlas import main

    raise SystemExit(main(list(args)))


@vision.command("train", context_settings=_PASSTHROUGH)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def vision_train(args):
    """Fine-tune NEXUS Vision on dataset folders (--data FOLDER --out FOLDER; --help for all)."""
    from nexus.live.finetune import main

    raise SystemExit(main(list(args)))


@vision.command("eval", context_settings=_PASSTHROUGH)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def vision_eval(args):
    """Test NEXUS Vision on held-out examples, against another model if you like (--help for all)."""
    from nexus.live.evaluate import main

    raise SystemExit(main(list(args)))


@vision.command("build", context_settings=_PASSTHROUGH)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def vision_build(args):
    """Fetch, train, evaluate and upload in one go, on this computer's graphics card (--help for all)."""
    from nexus.live.build import main

    raise SystemExit(main(list(args)))


@vision.command("merge")
@click.argument("adapter")
@click.argument("out")
def vision_merge(adapter, out):
    """Fold a NEXUS Vision adapter into its base model: one standalone folder, for GGUF or offline use."""
    try:
        from nexus.live.finetune import merge

        merge(Path(adapter), Path(out), printer=lambda line: console.print(line, markup=False))
    except Exception as error:
        _print_command_error("nexus vision merge", error, "It needs the vision extra: pip install -e .[vision]")


@vision.command("pull")
@click.argument("repo")
@click.option("--to", "target", default=None, help="Folder to put it in (default: NEXUS_LIVE_MODEL_DIR, ~/.nexus/live/nexus_vision)")
def vision_pull(repo, target):
    """Download your NEXUS Vision from your Hugging Face account, for the local backend."""
    try:
        import json

        from huggingface_hub import snapshot_download

        from nexus.live.local import card_problem, pull_model
        from nexus.live.settings import LiveSettings

        settings = LiveSettings.from_env()
        folder = Path(target or settings.model_dir).expanduser()
        console.print(f"Downloading {repo} into {folder}...")
        pull_model(repo, folder, snapshot_download)
        adapter = folder / "adapter_config.json"
        if adapter.exists():
            base = json.loads(adapter.read_text(encoding="utf-8")).get("base_model_name_or_path", "")
            if base:
                console.print(f"It runs on top of {base}: downloading that too (about 4.3 GB, once), so NEXUS Live works offline.")
                snapshot_download(base)
        problem = card_problem(folder, settings.coordinates)
        if problem:
            console.print(f"[yellow]Downloaded, but: {problem}[/yellow]")
        else:
            console.print("[green]Ready.[/green] Run `nexus live --check`, then `nexus live --voice`.")
    except Exception as error:
        _print_command_error(
            "nexus vision pull", error, "A private repo needs you logged in on this computer: run `hf auth login` first."
        )


@vision.command("share-teach")
@click.argument("repo")
@click.option("--from", "source", default=None, help="Your teach folder (default ~/.nexus/live/teach)")
def vision_share_teach(repo, source):
    """Upload your teach recordings to a PRIVATE dataset repo, for training in the cloud."""
    try:
        from huggingface_hub import HfApi

        from nexus.live.build import ensure_private
        from nexus.live.data import EXAMPLES_FILE
        from nexus.live.teach import default_folder

        folder = Path(source or default_folder()).expanduser()
        if not (folder / EXAMPLES_FILE).exists():
            raise click.ClickException(f"No recordings in {folder}. Record some with `nexus live --teach` first.")
        console.print(
            f"This uploads every screenshot in {folder} to {repo}, kept private. "
            "Look through the images folder first if anything private was on your screen."
        )
        if not click.confirm("Upload?", default=False):
            return
        api = HfApi()
        ensure_private(api, repo, "dataset")  # never into a repo that is already public
        api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="dataset", commit_message="NEXUS Live teach recordings")
        console.print(f"[green]Uploaded.[/green] Train with: --teach {repo}")
    except click.ClickException:
        raise
    except Exception as error:
        _print_command_error("nexus vision share-teach", error, "Log in first with `hf auth login`.")


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
@click.argument("task", required=False)
@click.option("--intent", default="coding", help="Hive task intent: coding, research, memory, design, canary")
def hive(task, intent):
    """Run the experimental NEXUS Hive distributed-search simulation."""
    try:
        from nexus.hive.runtime import HiveRuntime

        runtime = HiveRuntime()
        if not task:
            task = click.prompt("What should Hive search for")

        result = asyncio.run(runtime.demo(task, intent=intent))
        plan = result["plan"]
        winner = result.get("winner")

        console.print(
            Panel(
                f"[bold]Task[/bold]\n{task}\n\n"
                f"[bold]Intent[/bold] {intent}\n"
                f"[bold]Selected nodes[/bold] {', '.join(plan['selected_nodes']) or 'none'}\n"
                f"[bold]Canary coverage[/bold] {plan['canary_sample_size']} nodes\n"
                f"[bold]Responses[/bold] {result['responded_nodes']}",
                title="[bold cyan]NEXUS Hive[/bold cyan]",
                style="cyan",
            )
        )

        if winner:
            console.print(
                Panel(
                    f"[bold]Winner[/bold] {winner['node_id']}\n"
                    f"[bold]ReflectScore[/bold] {winner['reflect_score']:.2f} ({winner['reflect_verdict']})\n"
                    f"[bold]Network score[/bold] {winner['network_score']:.2f}\n"
                    f"[bold]Latency[/bold] {winner['latency_ms']:.0f} ms\n\n"
                    f"{winner['output']}",
                    title="[bold green]Consensus Winner[/bold green]",
                    style="green",
                )
            )
        elif result.get("note"):
            console.print(Panel(result["note"], title="[bold yellow]Hive Note[/bold yellow]", style="yellow"))

        if result.get("assembled_output"):
            console.print(
                Panel(
                    result["assembled_output"],
                    title="[bold blue]Hive Assembly[/bold blue]",
                    style="blue",
                )
            )

        failed_canaries = [item["node_id"] for item in result.get("canary_results", []) if not item.get("passed")]
        if result.get("canary_results"):
            console.print(
                Panel(
                    "All canary checks passed." if not failed_canaries else f"Canary failures: {', '.join(failed_canaries)}",
                    title="[bold magenta]Canary Checks[/bold magenta]",
                    style="magenta",
                )
            )

        table = Table(title="Hive Candidate Ranking")
        table.add_column("Node", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Reflect", style="yellow")
        table.add_column("Trust", style="magenta")
        table.add_column("Latency", style="white")
        for candidate in result["candidates"][:6]:
            table.add_row(
                candidate["node_id"],
                f"{candidate['network_score']:.2f}",
                f"{candidate['reflect_score']:.2f} {candidate['reflect_verdict']}",
                f"{candidate['trust_score']:.2f}",
                f"{candidate['latency_ms']:.0f} ms",
            )
        console.print(table)
    except Exception as error:
        _print_command_error("nexus hive", error, "Use `nexus status` to confirm Hive is enabled.")


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
    console.print(
        f"[bold]Hive:[/bold] {'enabled' if config.hive_enabled else 'disabled'} "
        f"(trust >= {config.hive_min_trust_score}, "
        f"replication {config.hive_replication_factor}, max nodes {config.hive_max_nodes})"
    )
    console.print("[bold]Setup check:[/bold] run `nexus doctor` for guided dependency fixes")


if __name__ == "__main__":
    cli()
