"""
Local entrypoint for the NEXUS agent compiler + runtime system.

Usage:
    python main.py "build a FastAPI auth service"
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nexus.blueprint_generator import BlueprintGenerator
from nexus.compiler.planner_engine import PlannerEngine
from nexus.intent_parser import IntentParser
from nexus.orchestrator import Orchestrator
from nexus.runtime.build_artifacts import BuildArtifactError, BuildArtifactMaterializer, MaterializationResult
from nexus.runtime.insights import RuntimeInsights
from nexus.runtime.project_mode import ProjectModeManager
from nexus.runtime.scaffold_runner import ScaffoldRunError, ScaffoldRunResult, ScaffoldRunner

console = Console()


async def _run_goal(
    goal: str,
    mode: str = "stable",
    *,
    project_mode: bool = False,
    project_dir: str | None = None,
) -> tuple[dict, dict, dict, dict]:
    parser = IntentParser()
    planner = PlannerEngine()
    generator = BlueprintGenerator()
    project_context = None
    project_manager = None
    if project_mode:
        project_manager = ProjectModeManager()
        project_context = project_manager.prepare(
            project_dir=project_dir or Path.cwd(),
            goal=goal,
            execution_mode=mode,
        )
    orchestrator = Orchestrator(
        execution_mode=mode,
        project_context=project_context,
        environment_memory=project_manager.environment_memory if project_manager else None,
    )

    intent = parser.parse(goal, project_context=project_context)
    plan = planner.plan(intent, project_context=project_context)
    blueprint = generator.generate(plan)
    result = await orchestrator.run_blueprint(blueprint)
    return intent.to_dict(), plan.to_dict(), blueprint.to_dict(), result


def _apply_shortcut(goal: str, args: argparse.Namespace) -> tuple[str, str | None]:
    if args.build:
        return f"Build a production-ready local implementation for this goal: {goal}", "build"
    if args.analyze:
        return f"Analyze this input carefully and return a concise summary with next steps: {goal}", "analyze"
    if args.react:
        return f"Create a React UI solution for this goal: {goal}", "react"
    return goal, None


def _percent(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{round(float(value) * 100)}%"


def _render_summary(
    *,
    user_goal: str,
    compiled_goal: str,
    shortcut: str | None,
    intent: dict,
    explanation: dict,
) -> Table:
    summary = explanation["summary"]
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="white")
    table.add_row("Goal", user_goal)
    if shortcut:
        table.add_row("Shortcut", shortcut)
        table.add_row("Compiled", compiled_goal)
    table.add_row("Intent", f"{intent['primary_intent']} ({intent['complexity']})")
    table.add_row("Mode", summary["execution_mode"] or "stable")
    if summary.get("project_mode"):
        table.add_row("Project", summary.get("project_root") or "project mode")
    table.add_row("Status", summary["status"])
    table.add_row("Confidence", _percent(summary["confidence"]))
    table.add_row("Retries", str(summary["retry_count"]))
    return table


def _render_plan(explanation: dict) -> Table:
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Task", style="white")
    table.add_column("Agent", style="magenta")
    table.add_column("Depends")
    table.add_column("Strategy")
    table.add_column("Min Conf", justify="right")
    for step in explanation["plan"]:
        table.add_row(
            step["task_id"],
            step.get("agent") or "-",
            ", ".join(step.get("depends_on", [])) or "-",
            step.get("retry_strategy") or "repeat",
            _percent(step.get("confidence_threshold")),
        )
    return table


def _render_explain_panel(explanation: dict, key: str, title: str) -> Panel:
    values = explanation.get(key, [])
    if not values:
        body = "No additional runtime notes."
    elif key == "strategies":
        body = "\n".join(
            f"- {item['task_id']} -> {item['strategy']} ({item.get('agent') or 'same agent'})"
            for item in values
        )
    else:
        body = "\n".join(f"- {item}" for item in values)
    return Panel(body, title=title, style="cyan")


def _render_materialization_panel(result: MaterializationResult) -> Panel:
    lines = [
        f"Output dir: {result.root_dir}",
        f"Files written: {len(result.files_written)}",
    ]
    if result.overwritten_files:
        lines.append(f"Overwritten: {len(result.overwritten_files)}")
    lines.extend(f"- {path.relative_to(result.root_dir)}" for path in result.files_written[:8])
    if len(result.files_written) > 8:
        lines.append(f"... and {len(result.files_written) - 8} more")
    return Panel("\n".join(lines), title="[bold green]Files Written[/bold green]", style="green")


def _render_run_panel(result: ScaffoldRunResult) -> Panel:
    lines = [
        f"URL: {result.url}",
        f"Health: {result.health_url}",
        f"PID: {result.pid}",
        f"Install: {'performed' if result.install_performed else 'reused existing node_modules'}",
        f"Logs: {result.stdout_log}",
        f"Metadata: {result.metadata_path}",
    ]
    return Panel("\n".join(lines), title="[bold green]App Running[/bold green]", style="green")


def _materialize_result(
    *,
    goal: str,
    final_output: str,
    output_dir: str | None,
    force: bool,
) -> MaterializationResult:
    materializer = BuildArtifactMaterializer()
    target_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else materializer.default_output_dir(goal=goal, output=final_output, base_dir=Path.cwd())
    )
    return materializer.materialize(
        output=final_output,
        target_dir=target_dir,
        overwrite=force,
    )


def _run_materialized_scaffold(
    *,
    root_dir: Path,
    preferred_port: int | None,
) -> ScaffoldRunResult:
    return ScaffoldRunner(root_dir).run(preferred_port=preferred_port)


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description="Run the NEXUS agent compiler + runtime system locally.")
    cli.add_argument(
        "--mode",
        choices=["stable", "explore"],
        default="stable",
        help="Execution policy mode for runtime adaptation",
    )
    cli.add_argument("--project-mode", action="store_true", help="Reuse persistent local project context across commands")
    cli.add_argument("--project-dir", help="Project directory to scan and reuse in project mode. Defaults to the current directory")
    cli.add_argument("--explain", action="store_true", help="Explain plan, runtime decisions, and final confidence")
    cli.add_argument("--write", action="store_true", help="Write generated scaffold files to disk when the result includes file blocks")
    cli.add_argument("--run", action="store_true", help="After materializing a scaffold, install dependencies and launch it locally")
    cli.add_argument("--run-port", type=int, help="Preferred port for --run. Defaults to the first free port starting at 3010")
    cli.add_argument("--output-dir", help="Target directory for --write output. Defaults to ./generated/<scaffold-name>")
    cli.add_argument("--force", action="store_true", help="Allow --write to overwrite existing files in the output directory")
    shortcuts = cli.add_mutually_exclusive_group()
    shortcuts.add_argument("--build", action="store_true", help="Tune the goal for build-and-implement workflows")
    shortcuts.add_argument("--analyze", action="store_true", help="Tune the goal for analysis and summarization workflows")
    shortcuts.add_argument("--react", action="store_true", help="Tune the goal for React UI generation workflows")
    cli.add_argument("goal", nargs="+", help="User goal to compile into a workflow")
    args = cli.parse_args(argv)

    user_goal = " ".join(args.goal).strip()
    compiled_goal, shortcut = _apply_shortcut(user_goal, args)
    project_mode = bool(args.project_mode or args.project_dir)
    intent, plan, blueprint, result = asyncio.run(
        _run_goal(
            compiled_goal,
            mode=args.mode,
            project_mode=project_mode,
            project_dir=args.project_dir,
        )
    )
    explanation = RuntimeInsights().explain_run(
        goal=user_goal,
        intent=intent,
        plan=plan,
        blueprint=blueprint,
        result=result,
    )

    console.print(
        Panel(
            _render_summary(
                user_goal=user_goal,
                compiled_goal=compiled_goal,
                shortcut=shortcut,
                intent=intent,
                explanation=explanation,
            ),
            title="[bold cyan]Run Summary[/bold cyan]",
            style="cyan",
        )
    )
    console.print(Panel(_render_plan(explanation), title="[bold cyan]Plan[/bold cyan]", style="cyan"))
    if args.explain:
        console.print(_render_explain_panel(explanation, "decisions", "[bold cyan]Decisions[/bold cyan]"))
        console.print(_render_explain_panel(explanation, "strategies", "[bold cyan]Strategies[/bold cyan]"))
        console.print(
            Panel(
                f"Trace: {explanation['result'].get('trace_path') or '--'}\n"
                f"Decision log: {explanation['result'].get('decision_log_path') or '--'}",
                title="[bold cyan]Artifacts[/bold cyan]",
                style="cyan",
            )
        )
    console.print(
        Panel(
            result["final_output"] or "Workflow finished without a final output.",
            title=f"[bold green]Result[/bold green] ({result['status']})",
            style="green" if result["status"] == "completed" else "red",
        )
    )

    materialization_failed = False
    run_failed = False
    materialization: MaterializationResult | None = None
    if args.write or args.run:
        if not result["final_output"]:
            console.print(
                Panel(
                    "Write or run mode was requested, but the run did not produce any final output to materialize.",
                    title="[bold red]Write Failed[/bold red]",
                    style="red",
                )
            )
            materialization_failed = True
        else:
            try:
                materialization = _materialize_result(
                    goal=user_goal,
                    final_output=result["final_output"],
                    output_dir=args.output_dir,
                    force=args.force,
                )
                console.print(_render_materialization_panel(materialization))
            except BuildArtifactError as error:
                console.print(
                    Panel(str(error), title="[bold red]Write Failed[/bold red]", style="red")
                )
                materialization_failed = True
    if args.run and not materialization_failed and materialization is not None:
        try:
            run_result = _run_materialized_scaffold(
                root_dir=materialization.root_dir,
                preferred_port=args.run_port,
            )
            console.print(_render_run_panel(run_result))
        except ScaffoldRunError as error:
            console.print(
                Panel(str(error), title="[bold red]Run Failed[/bold red]", style="red")
            )
            run_failed = True

    return 0 if result["status"] == "completed" and not materialization_failed and not run_failed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
