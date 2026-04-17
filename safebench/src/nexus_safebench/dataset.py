"""Scenario dataset loading for NEXUS SafeBench."""

from __future__ import annotations

import json
from pathlib import Path

from nexus_safebench.models import (
    ActionEvidence,
    ClaimEvidence,
    DatasetDefinition,
    DryRunTrace,
    ManifestDefinition,
    ScenarioDefinition,
)


class DatasetError(ValueError):
    """Raised when a SafeBench dataset is invalid."""


def load_dataset(path: str | Path) -> DatasetDefinition:
    dataset_path = Path(path)
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetError(f"Dataset not found: {dataset_path}") from error
    except json.JSONDecodeError as error:
        raise DatasetError(f"Dataset is not valid JSON: {dataset_path}") from error

    scenarios: list[ScenarioDefinition] = []
    suite_id = str(payload["suite_id"])
    for item in payload.get("scenarios", []):
        dry_run_payload = item.get("dry_run_trace")
        dry_run_trace = None
        if dry_run_payload is not None:
            dry_run_trace = DryRunTrace(
                response=str(dry_run_payload["response"]),
                expressed_uncertainty=bool(dry_run_payload.get("expressed_uncertainty", False)),
                uncertainty_appropriate=bool(dry_run_payload.get("uncertainty_appropriate", False)),
                claims=[
                    ClaimEvidence(
                        type=str(claim["type"]),
                        text=str(claim["text"]),
                        supported=bool(claim["supported"]),
                    )
                    for claim in dry_run_payload.get("claims", [])
                ],
                actions=[
                    ActionEvidence(
                        type=str(action["type"]),
                        target=str(action["target"]),
                        status=str(action["status"]),
                    )
                    for action in dry_run_payload.get("actions", [])
                ],
            )

        scenarios.append(
            ScenarioDefinition(
                id=str(item["id"]),
                suite_id=suite_id,
                title=str(item["title"]),
                scenario_type=str(item["scenario_type"]),
                prompt=str(item["prompt"]),
                description=str(item["description"]),
                tags=[str(value) for value in item.get("tags", [])],
                acceptance_thresholds={
                    str(key): float(value)
                    for key, value in item.get("acceptance_thresholds", {}).items()
                },
                dry_run_trace=dry_run_trace,
            )
        )

    dataset = DatasetDefinition(
        name=str(payload["name"]),
        suite_id=suite_id,
        version=str(payload["version"]),
        description=str(payload["description"]),
        scenarios=scenarios,
    )
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: DatasetDefinition, manifest: ManifestDefinition | None = None) -> None:
    if not dataset.scenarios:
        raise DatasetError("Dataset must define at least one scenario.")

    scenario_ids = [scenario.id for scenario in dataset.scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise DatasetError("Scenario ids must be unique.")

    for scenario in dataset.scenarios:
        if scenario.suite_id != dataset.suite_id:
            raise DatasetError(
                f"Scenario '{scenario.id}' suite_id '{scenario.suite_id}' does not match dataset suite '{dataset.suite_id}'."
            )
        if scenario.dry_run_trace is None:
            raise DatasetError(f"Scenario '{scenario.id}' must define a dry_run_trace in the current scaffold.")

    if manifest is not None and dataset.suite_id not in manifest.suite_ids():
        raise DatasetError(
            f"Dataset suite '{dataset.suite_id}' is not defined in manifest '{manifest.name}'."
        )
