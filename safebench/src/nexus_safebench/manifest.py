"""Manifest loading and validation for NEXUS SafeBench."""

from __future__ import annotations

import json
from pathlib import Path

from nexus_safebench.models import ManifestDefinition, MetricDefinition, SuiteDefinition


class ManifestError(ValueError):
    """Raised when a SafeBench manifest is invalid."""


def load_manifest(path: str | Path) -> ManifestDefinition:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestError(f"Manifest not found: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ManifestError(f"Manifest is not valid JSON: {manifest_path}") from error

    metrics = [
        MetricDefinition(
            id=str(item["id"]),
            label=str(item["label"]),
            description=str(item["description"]),
        )
        for item in payload.get("metrics", [])
    ]
    suites = [
        SuiteDefinition(
            id=str(item["id"]),
            label=str(item["label"]),
            objective=str(item["objective"]),
            description=str(item["description"]),
            scenario_types=[str(value) for value in item.get("scenario_types", [])],
            key_metrics=[str(value) for value in item.get("key_metrics", [])],
            acceptance_criteria=[str(value) for value in item.get("acceptance_criteria", [])],
            seed_examples=[str(value) for value in item.get("seed_examples", [])],
        )
        for item in payload.get("suites", [])
    ]

    manifest = ManifestDefinition(
        name=str(payload["name"]),
        version=str(payload["version"]),
        description=str(payload["description"]),
        metrics=metrics,
        suites=suites,
        roadmap=[str(value) for value in payload.get("roadmap", [])],
    )
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: ManifestDefinition) -> None:
    if not manifest.suites:
        raise ManifestError("Manifest must define at least one suite.")
    if not manifest.metrics:
        raise ManifestError("Manifest must define at least one metric.")

    metric_ids = [metric.id for metric in manifest.metrics]
    if len(metric_ids) != len(set(metric_ids)):
        raise ManifestError("Metric ids must be unique.")

    suite_ids = [suite.id for suite in manifest.suites]
    if len(suite_ids) != len(set(suite_ids)):
        raise ManifestError("Suite ids must be unique.")

    known_metrics = set(metric_ids)
    for suite in manifest.suites:
        if not suite.key_metrics:
            raise ManifestError(f"Suite '{suite.id}' must reference at least one metric.")
        unknown_metrics = [metric_id for metric_id in suite.key_metrics if metric_id not in known_metrics]
        if unknown_metrics:
            joined = ", ".join(unknown_metrics)
            raise ManifestError(f"Suite '{suite.id}' references unknown metrics: {joined}")
