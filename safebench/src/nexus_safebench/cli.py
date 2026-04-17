"""CLI for NEXUS SafeBench."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nexus_safebench.dataset import DatasetError, load_dataset
from nexus_safebench.manifest import ManifestError, load_manifest
from nexus_safebench.runner import build_run_plan, run_dataset


console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@click.group()
def cli() -> None:
    """NEXUS SafeBench benchmark scaffold."""


def _resolve_manifest_path(value: str) -> Path:
    raw_path = Path(value)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(Path.cwd() / raw_path)
        candidates.append(PROJECT_ROOT / raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def _load_manifest_from_option(manifest_path: str):
    resolved = _resolve_manifest_path(manifest_path)
    manifest = load_manifest(resolved)
    return resolved, manifest


def _resolve_dataset_path(value: str, suite_id: str | None = None) -> Path:
    if value:
        return _resolve_manifest_path(value)
    if suite_id is None:
        raise click.ClickException("A dataset path or suite id is required.")
    return (PROJECT_ROOT / "datasets" / f"{suite_id}.json").resolve()


def _load_dataset_from_option(dataset_path: str, suite_id: str | None = None):
    resolved = _resolve_dataset_path(dataset_path, suite_id=suite_id)
    dataset = load_dataset(resolved)
    return resolved, dataset


def _render_manifest_table(manifest) -> Table:
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Suite")
    table.add_column("Objective")
    table.add_column("Metrics")
    table.add_column("Scenarios")
    for suite in manifest.suites:
        table.add_row(
            suite.id,
            suite.objective,
            ", ".join(suite.key_metrics),
            ", ".join(suite.scenario_types),
        )
    return table


@cli.command()
@click.option("--manifest", "manifest_path", default="manifests/core.json", show_default=True)
def overview(manifest_path: str) -> None:
    """Show the benchmark scaffold and available suites."""
    try:
        resolved, manifest = _load_manifest_from_option(manifest_path)
    except ManifestError as error:
        raise click.ClickException(str(error)) from error

    console.print(
        Panel(
            f"{manifest.description}\n\nManifest: {resolved}",
            title=f"{manifest.name} v{manifest.version}",
            border_style="cyan",
        )
    )
    console.print(_render_manifest_table(manifest))
    if manifest.roadmap:
        roadmap = "\n".join(f"- {item}" for item in manifest.roadmap)
        console.print(Panel(roadmap, title="Roadmap", border_style="magenta"))


@cli.command()
@click.option("--manifest", "manifest_path", default="manifests/core.json", show_default=True)
def validate(manifest_path: str) -> None:
    """Validate the SafeBench manifest."""
    try:
        resolved, manifest = _load_manifest_from_option(manifest_path)
    except ManifestError as error:
        raise click.ClickException(str(error)) from error

    console.print(
        f"[green][OK][/green] Manifest valid: {resolved}\n"
        f"Suites: {len(manifest.suites)} | Metrics: {len(manifest.metrics)}"
    )


@cli.command()
@click.option("--manifest", "manifest_path", default="manifests/core.json", show_default=True)
@click.option("--suite", "suite_id", default=None, help="Optional suite id to plan for.")
def plan(manifest_path: str, suite_id: str | None) -> None:
    """Generate a starter run plan from the manifest."""
    try:
        _, manifest = _load_manifest_from_option(manifest_path)
        run_plan = build_run_plan(manifest, suite_id=suite_id)
    except (ManifestError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    console.print_json(json.dumps(run_plan.to_dict(), indent=2))


@cli.command()
@click.option("--dataset", "dataset_path", default="", help="Optional dataset path. Defaults to datasets/<suite>.json")
@click.option("--suite", "suite_id", required=True, help="Suite id to inspect.")
def scenarios(dataset_path: str, suite_id: str) -> None:
    """List scenarios from a dataset."""
    try:
        resolved, dataset = _load_dataset_from_option(dataset_path, suite_id=suite_id)
    except DatasetError as error:
        raise click.ClickException(str(error)) from error

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Scenario")
    table.add_column("Type")
    table.add_column("Title")
    table.add_column("Tags")
    for scenario in dataset.scenarios:
        table.add_row(
            scenario.id,
            scenario.scenario_type,
            scenario.title,
            ", ".join(scenario.tags),
        )

    console.print(
        Panel(
            f"{dataset.description}\n\nDataset: {resolved}",
            title=f"{dataset.name} v{dataset.version}",
            border_style="cyan",
        )
    )
    console.print(table)


@cli.command()
@click.option("--manifest", "manifest_path", default="manifests/core.json", show_default=True)
@click.option("--dataset", "dataset_path", default="", help="Optional dataset path. Defaults to datasets/<suite>.json")
@click.option("--suite", "suite_id", required=True, help="Suite id to run.")
@click.option("--adapter", default="dry-run", show_default=True, help="Execution adapter to use.")
@click.option("--output-dir", "output_dir", default="runs", show_default=True, help="Directory for benchmark artifacts.")
def run(manifest_path: str, dataset_path: str, suite_id: str, adapter: str, output_dir: str) -> None:
    """Execute a benchmark dataset and emit results."""
    try:
        _, manifest = _load_manifest_from_option(manifest_path)
        _, dataset = _load_dataset_from_option(dataset_path, suite_id=suite_id)
        result = run_dataset(
            manifest,
            dataset,
            adapter=adapter,
            output_root=output_dir,
        )
    except (ManifestError, DatasetError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    console.print(
        Panel(
            f"Suite: {result.suite_id}\n"
            f"Adapter: {result.adapter}\n"
            f"Output: {result.output_dir}\n"
            f"Pass rate: {result.summary['pass_rate']:.2%}",
            title=f"Run {result.run_id}",
            border_style="green",
        )
    )
    console.print_json(json.dumps(result.summary, indent=2))
