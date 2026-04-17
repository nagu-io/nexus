"""Planning helpers for NEXUS SafeBench."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from nexus_safebench.dataset import validate_dataset
from nexus_safebench.models import (
    BenchmarkRunResult,
    DatasetDefinition,
    ManifestDefinition,
    MetricDefinition,
    RunPlan,
    ScenarioEvaluation,
    SuiteDefinition,
)
from nexus_safebench.scoring import evaluate_coding_honesty_scenario


def build_run_plan(manifest: ManifestDefinition, suite_id: str | None = None) -> RunPlan:
    selected_suites = _select_suites(manifest, suite_id)
    selected_metric_ids: set[str] = set()
    for suite in selected_suites:
        selected_metric_ids.update(suite.key_metrics)

    selected_metrics = [
        metric for metric in manifest.metrics
        if metric.id in selected_metric_ids
    ]
    notes = [
        "This scaffold generates a benchmark plan only.",
        "Execution harness, adapters, and scoring backends are the next implementation step.",
    ]
    return RunPlan(
        manifest_name=manifest.name,
        manifest_version=manifest.version,
        selected_suites=selected_suites,
        selected_metrics=selected_metrics,
        notes=notes,
    )


def _select_suites(manifest: ManifestDefinition, suite_id: str | None) -> list[SuiteDefinition]:
    if suite_id is None:
        return list(manifest.suites)

    matches = [suite for suite in manifest.suites if suite.id == suite_id]
    if matches:
        return matches

    known = ", ".join(sorted(suite.id for suite in manifest.suites))
    raise ValueError(f"Unknown suite '{suite_id}'. Known suites: {known}")


def run_dataset(
    manifest: ManifestDefinition,
    dataset: DatasetDefinition,
    *,
    adapter: str = "dry-run",
    output_root: str | Path = "runs",
) -> BenchmarkRunResult:
    validate_dataset(dataset, manifest=manifest)
    if adapter != "dry-run":
        raise ValueError(f"Unsupported adapter '{adapter}'. Only 'dry-run' is available right now.")

    suite = _select_suites(manifest, dataset.suite_id)[0]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root) / f"{run_id}-{dataset.suite_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scenario_results = [_evaluate_scenario(suite.id, scenario) for scenario in dataset.scenarios]
    summary = _build_summary(suite.id, adapter, dataset, scenario_results)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    result_lines = [json.dumps(result.to_dict()) for result in scenario_results]
    results_path = run_dir / "scenario_results.jsonl"
    results_path.write_text("\n".join(result_lines) + ("\n" if result_lines else ""), encoding="utf-8")

    dataset_snapshot_path = run_dir / "dataset_snapshot.json"
    dataset_snapshot_path.write_text(json.dumps(dataset.to_dict(), indent=2), encoding="utf-8")

    return BenchmarkRunResult(
        run_id=run_id,
        suite_id=suite.id,
        adapter=adapter,
        output_dir=str(run_dir.resolve()),
        summary=summary,
        scenario_results=scenario_results,
    )


def _evaluate_scenario(suite_id: str, scenario) -> ScenarioEvaluation:
    if suite_id == "coding_honesty":
        return evaluate_coding_honesty_scenario(scenario)
    raise ValueError(f"Scoring for suite '{suite_id}' is not implemented yet.")


def _build_summary(
    suite_id: str,
    adapter: str,
    dataset: DatasetDefinition,
    results: list[ScenarioEvaluation],
) -> dict:
    metric_names = sorted({metric_name for result in results for metric_name in result.metrics})
    metric_averages = {}
    for metric_name in metric_names:
        values = [result.metrics[metric_name] for result in results if metric_name in result.metrics]
        metric_averages[metric_name] = round(sum(values) / len(values), 4) if values else 0.0

    passed = sum(1 for result in results if result.status == "passed")
    failed = sum(1 for result in results if result.status != "passed")
    return {
        "suite_id": suite_id,
        "dataset_name": dataset.name,
        "dataset_version": dataset.version,
        "adapter": adapter,
        "total_scenarios": len(results),
        "passed_scenarios": passed,
        "failed_scenarios": failed,
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "metric_averages": metric_averages,
    }
