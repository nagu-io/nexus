"""Core data models for the experimental NEXUS Hive runtime."""

from __future__ import annotations

from dataclasses import dataclass


def clamp01(value: float) -> float:
    """Clamp a numeric score into the inclusive 0..1 range."""
    return max(0.0, min(float(value), 1.0))


@dataclass(slots=True, frozen=True)
class HiveNodeProfile:
    """One participating machine in the Hive."""

    node_id: str
    region: str = ""
    country: str = ""
    capabilities: tuple[str, ...] = ()
    avg_latency_ms: float = 250.0
    success_rate: float = 0.95
    canary_pass_rate: float = 1.0
    completed_tasks: int = 0
    idle_cpu_ratio: float = 1.0
    idle_gpu_ratio: float = 0.0
    recent_canary_failures: int = 0
    abuse_score: int = 0
    is_tor: bool = False
    is_proxy: bool = False
    is_vpn: bool = False
    breach_count: int = 0
    recent_attempts: int = 0
    chain_attack: bool = False
    virustotal_reputation: int = 0
    shodan_exposed_ports: int = 0
    shodan_has_vulns: bool = False
    shodan_is_scanner: bool = False

    def supports(self, required_capabilities: tuple[str, ...]) -> bool:
        """Return True when the node can satisfy the requested capability set."""
        if not required_capabilities:
            return True
        available = set(self.capabilities)
        return all(capability in available for capability in required_capabilities)


@dataclass(slots=True, frozen=True)
class HiveTaskRequest:
    """One distributed search request prepared for the Hive."""

    task_id: str
    prompt: str
    intent: str = "coding"
    required_capabilities: tuple[str, ...] = ()
    replication_factor: int = 6
    max_nodes: int = 12
    canary_fraction: float = 0.15
    privacy_mode: str = "signature_only"
    strategy: str = "parallel_search"
    latency_budget_ms: int = 5000


@dataclass(slots=True, frozen=True)
class HiveTaskEnvelope:
    """One privacy-preserving task package prepared for a node."""

    envelope_id: str
    node_id: str
    task_id: str
    task_signature: str
    prompt_digest: str
    masked_context: str
    sealed_payload: str
    privacy_mode: str
    capability_hints: tuple[str, ...] = ()
    is_canary: bool = False
    canary_id: str | None = None


@dataclass(slots=True, frozen=True)
class HiveCanaryChallenge:
    """A synthetic trust challenge mixed in beside real Hive work."""

    challenge_id: str
    node_id: str
    prompt: str
    expected_answer: str
    sealed_envelope: HiveTaskEnvelope


@dataclass(slots=True, frozen=True)
class HiveCanaryResult:
    """Evaluation result for one canary challenge."""

    node_id: str
    challenge_id: str
    passed: bool
    response: str
    score: float
    reason: str


@dataclass(slots=True, frozen=True)
class HiveCandidateResponse:
    """One candidate answer returned from a node."""

    node_id: str
    output: str
    latency_ms: float
    subtask_id: str | None = None


@dataclass(slots=True, frozen=True)
class HiveDispatchPlan:
    """Selected nodes plus dispatch metadata for a Hive run."""

    task: HiveTaskRequest
    selected_nodes: tuple[str, ...]
    standby_nodes: tuple[str, ...]
    canary_sample_size: int
    privacy_mode: str
    strategy: str
    rationale: str
    envelopes: tuple[HiveTaskEnvelope, ...] = ()
    canary_node_ids: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class HiveCandidateAssessment:
    """A scored candidate after ReflectScore and trust weighting."""

    node_id: str
    output: str
    latency_ms: float
    reflect_score: float
    reflect_verdict: str
    reflect_action: str
    trust_score: float
    latency_score: float
    network_score: float
    blocked: bool
    reason: str


@dataclass(slots=True, frozen=True)
class HiveAssemblyResult:
    """A synthesized answer built from the best Hive candidates."""

    output: str
    source_nodes: tuple[str, ...]
    confidence: float
    note: str | None = None


@dataclass(slots=True, frozen=True)
class HiveConsensusResult:
    """Final ranked network view of a Hive search."""

    winner: HiveCandidateAssessment | None
    ranked_candidates: tuple[HiveCandidateAssessment, ...]
    assembly_candidates: tuple[HiveCandidateAssessment, ...]
    blocked_nodes: tuple[str, ...]
    note: str | None = None
    assembled_output: str | None = None
    assembly_sources: tuple[str, ...] = ()
    canary_results: tuple[HiveCanaryResult, ...] = ()
