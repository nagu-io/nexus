"""Trust scoring for NEXUS Hive nodes."""

from __future__ import annotations

from dataclasses import dataclass

from nexus.canary.risk_engine import compute_risk_score
from nexus.hive.models import HiveNodeProfile, clamp01


@dataclass(slots=True, frozen=True)
class NodeTrustScore:
    """Structured trust score for one Hive node."""

    node_id: str
    score: float
    eligible: bool
    risk_level: str
    risk_score: int
    canary_component: float
    reliability_component: float
    capacity_component: float
    penalty: float
    rationale: str


class NodeTrustAssessor:
    """Combine canary history, runtime quality, and security risk into one score."""

    def __init__(self, *, min_trust_score: float = 0.45):
        self.min_trust_score = clamp01(min_trust_score)

    def evaluate(self, node: HiveNodeProfile) -> NodeTrustScore:
        """Return the current trust assessment for a node."""
        risk = compute_risk_score(
            abuse_score=node.abuse_score,
            is_tor=node.is_tor,
            is_proxy=node.is_proxy,
            is_vpn=node.is_vpn,
            country=node.country,
            platform="hive",
            canary_tier=3 if node.recent_canary_failures else 1,
            breach_count=node.breach_count,
            recent_attempts=node.recent_attempts,
            chain_attack=node.chain_attack,
            virustotal_reputation=node.virustotal_reputation,
            shodan_exposed_ports=node.shodan_exposed_ports,
            shodan_has_vulns=node.shodan_has_vulns,
            shodan_is_scanner=node.shodan_is_scanner,
        )

        canary_component = 0.5 * clamp01(node.canary_pass_rate)
        reliability_component = (
            0.25 * clamp01(node.success_rate)
            + 0.1 * clamp01(node.completed_tasks / 100.0)
        )
        capacity_component = 0.15 * max(
            clamp01(node.idle_cpu_ratio),
            clamp01(node.idle_gpu_ratio),
        )
        penalty = 0.3 * clamp01(risk["score"] / 100.0)
        penalty += min(max(node.recent_canary_failures, 0) * 0.08, 0.24)

        score = clamp01(canary_component + reliability_component + capacity_component - penalty)
        eligible = score >= self.min_trust_score
        rationale = (
            f"canary={canary_component:.2f}, reliability={reliability_component:.2f}, "
            f"capacity={capacity_component:.2f}, penalty={penalty:.2f}, risk={risk['level']}"
        )

        return NodeTrustScore(
            node_id=node.node_id,
            score=score,
            eligible=eligible,
            risk_level=str(risk["level"]),
            risk_score=int(risk["score"]),
            canary_component=canary_component,
            reliability_component=reliability_component,
            capacity_component=capacity_component,
            penalty=penalty,
            rationale=rationale,
        )
