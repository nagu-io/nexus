"""Typed benchmark models for NEXUS SafeBench."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class MetricDefinition:
    id: str
    label: str
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SuiteDefinition:
    id: str
    label: str
    objective: str
    description: str
    scenario_types: list[str] = field(default_factory=list)
    key_metrics: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    seed_examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ManifestDefinition:
    name: str
    version: str
    description: str
    metrics: list[MetricDefinition] = field(default_factory=list)
    suites: list[SuiteDefinition] = field(default_factory=list)
    roadmap: list[str] = field(default_factory=list)

    def metric_ids(self) -> set[str]:
        return {metric.id for metric in self.metrics}

    def suite_ids(self) -> set[str]:
        return {suite.id for suite in self.suites}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "suites": [suite.to_dict() for suite in self.suites],
            "roadmap": list(self.roadmap),
        }


@dataclass(frozen=True)
class ClaimEvidence:
    type: str
    text: str
    supported: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ActionEvidence:
    type: str
    target: str
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DryRunTrace:
    response: str
    expressed_uncertainty: bool
    uncertainty_appropriate: bool
    claims: list[ClaimEvidence] = field(default_factory=list)
    actions: list[ActionEvidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "response": self.response,
            "expressed_uncertainty": self.expressed_uncertainty,
            "uncertainty_appropriate": self.uncertainty_appropriate,
            "claims": [claim.to_dict() for claim in self.claims],
            "actions": [action.to_dict() for action in self.actions],
        }


@dataclass(frozen=True)
class ScenarioDefinition:
    id: str
    suite_id: str
    title: str
    scenario_type: str
    prompt: str
    description: str
    tags: list[str] = field(default_factory=list)
    acceptance_thresholds: dict[str, float] = field(default_factory=dict)
    dry_run_trace: DryRunTrace | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "suite_id": self.suite_id,
            "title": self.title,
            "scenario_type": self.scenario_type,
            "prompt": self.prompt,
            "description": self.description,
            "tags": list(self.tags),
            "acceptance_thresholds": dict(self.acceptance_thresholds),
            "dry_run_trace": self.dry_run_trace.to_dict() if self.dry_run_trace else None,
        }


@dataclass(frozen=True)
class DatasetDefinition:
    name: str
    suite_id: str
    version: str
    description: str
    scenarios: list[ScenarioDefinition] = field(default_factory=list)

    def scenario_ids(self) -> set[str]:
        return {scenario.id for scenario in self.scenarios}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "suite_id": self.suite_id,
            "version": self.version,
            "description": self.description,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }


@dataclass(frozen=True)
class RunPlan:
    manifest_name: str
    manifest_version: str
    selected_suites: list[SuiteDefinition]
    selected_metrics: list[MetricDefinition]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "manifest_name": self.manifest_name,
            "manifest_version": self.manifest_version,
            "selected_suites": [suite.to_dict() for suite in self.selected_suites],
            "selected_metrics": [metric.to_dict() for metric in self.selected_metrics],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ScenarioEvaluation:
    scenario_id: str
    suite_id: str
    title: str
    status: str
    metrics: dict[str, float]
    findings: list[str] = field(default_factory=list)
    response_preview: str = ""

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "suite_id": self.suite_id,
            "title": self.title,
            "status": self.status,
            "metrics": dict(self.metrics),
            "findings": list(self.findings),
            "response_preview": self.response_preview,
        }


@dataclass(frozen=True)
class BenchmarkRunResult:
    run_id: str
    suite_id: str
    adapter: str
    output_dir: str
    summary: dict
    scenario_results: list[ScenarioEvaluation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "suite_id": self.suite_id,
            "adapter": self.adapter,
            "output_dir": self.output_dir,
            "summary": dict(self.summary),
            "scenario_results": [result.to_dict() for result in self.scenario_results],
        }
