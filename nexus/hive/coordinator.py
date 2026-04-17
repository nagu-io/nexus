"""Coordinator primitives for the experimental NEXUS Hive runtime."""

from __future__ import annotations

import asyncio

from nexus.hive.assembly import HiveResponseAssembler
from nexus.hive.models import (
    HiveCandidateAssessment,
    HiveCandidateResponse,
    HiveCanaryResult,
    HiveConsensusResult,
    HiveDispatchPlan,
    HiveNodeProfile,
    HiveTaskRequest,
    clamp01,
)
from nexus.hive.trust import NodeTrustAssessor
from nexus.reflect.reflect_score import ReflectScore


class HiveCoordinator:
    """
    Build dispatch plans and rank distributed candidates.

    This layer does not provide the peer-to-peer transport yet. It handles the
    logic NEXUS will need once nodes can advertise themselves and return work:
    selecting trustworthy nodes, injecting canary coverage, and letting
    ReflectScore judge which returned answer should win.
    """

    def __init__(
        self,
        *,
        trust_assessor: NodeTrustAssessor | None = None,
        reflect_scorer: ReflectScore | object | None = None,
        response_assembler: HiveResponseAssembler | None = None,
        assembly_width: int = 3,
    ):
        self.trust_assessor = trust_assessor or NodeTrustAssessor()
        self.reflect_scorer = reflect_scorer or ReflectScore()
        self.response_assembler = response_assembler or HiveResponseAssembler()
        self.assembly_width = max(1, int(assembly_width))

    def plan_task(self, task: HiveTaskRequest, nodes: list[HiveNodeProfile]) -> HiveDispatchPlan:
        """Pick the best nodes for one replicated Hive search."""
        supported = [node for node in nodes if node.supports(task.required_capabilities)]
        ranked = sorted(
            ((node, self.trust_assessor.evaluate(node)) for node in supported),
            key=lambda item: (
                -item[1].score,
                item[0].avg_latency_ms,
                item[0].node_id,
            ),
        )
        eligible = [item for item in ranked if item[1].eligible]

        desired_count = min(
            max(1, int(task.max_nodes)),
            max(1, int(task.replication_factor) * 2),
        )
        selected = tuple(node.node_id for node, _ in eligible[:desired_count])
        standby = tuple(node.node_id for node, _ in eligible[desired_count:desired_count + task.replication_factor])
        canary_sample_size = 0
        if selected:
            canary_sample_size = min(
                len(selected),
                max(1, round(len(selected) * clamp01(task.canary_fraction))),
            )

        if not supported:
            rationale = "No nodes matched the requested capability set."
        elif not selected:
            rationale = "Nodes matched capabilities, but all fell below the Hive trust floor."
        else:
            rationale = (
                f"Selected {len(selected)} trusted nodes from {len(supported)} capable nodes "
                f"for {task.strategy} with privacy mode {task.privacy_mode}."
            )

        return HiveDispatchPlan(
            task=task,
            selected_nodes=selected,
            standby_nodes=standby,
            canary_sample_size=canary_sample_size,
            privacy_mode=task.privacy_mode,
            strategy=task.strategy,
            rationale=rationale,
        )

    async def evaluate_candidates(
        self,
        task: HiveTaskRequest,
        candidates: list[HiveCandidateResponse],
        nodes: list[HiveNodeProfile],
        *,
        canary_results: tuple[HiveCanaryResult, ...] = (),
    ) -> HiveConsensusResult:
        """Run ReflectScore over returned candidates and rank the network result."""
        nodes_by_id = {node.node_id: node for node in nodes}
        failed_canary_nodes = {result.node_id for result in canary_results if not result.passed}
        assessments = await asyncio.gather(
            *[
                self._assess_candidate(
                    task=task,
                    candidate=candidate,
                    node=nodes_by_id.get(candidate.node_id),
                    failed_canary_nodes=failed_canary_nodes,
                )
                for candidate in candidates
            ]
        )
        ranked = tuple(
            sorted(
                assessments,
                key=lambda item: (
                    not item.blocked,
                    item.network_score,
                    -item.latency_ms,
                ),
                reverse=True,
            )
        )
        viable = [item for item in ranked if not item.blocked]
        winner = viable[0] if viable else None
        assembly_candidates = tuple(viable[: self.assembly_width])
        blocked_node_set = {item.node_id for item in ranked if item.blocked}
        blocked_node_set.update(failed_canary_nodes)
        blocked_nodes = tuple(sorted(blocked_node_set))
        note = None
        if winner is None:
            note = "Every Hive candidate was blocked by trust or ReflectScore gates."
        assembly = self.response_assembler.assemble(task=task, candidates=assembly_candidates)

        return HiveConsensusResult(
            winner=winner,
            ranked_candidates=ranked,
            assembly_candidates=assembly_candidates,
            blocked_nodes=blocked_nodes,
            note=note,
            assembled_output=assembly.output if assembly.output else None,
            assembly_sources=assembly.source_nodes,
            canary_results=canary_results,
        )

    async def _assess_candidate(
        self,
        *,
        task: HiveTaskRequest,
        candidate: HiveCandidateResponse,
        node: HiveNodeProfile | None,
        failed_canary_nodes: set[str],
    ) -> HiveCandidateAssessment:
        """Score one returned answer with network-aware trust weighting."""
        node_trust = self.trust_assessor.evaluate(node).score if node is not None else 0.0
        reflect = await self.reflect_scorer.assess_response(task.prompt, candidate.output)
        latency_score = self._latency_score(candidate.latency_ms, task.latency_budget_ms)
        quality_score = clamp01(1.0 - float(reflect["score"]))
        failed_canary = candidate.node_id in failed_canary_nodes
        blocked = bool(
            reflect["action"] == "block"
            or node_trust < self.trust_assessor.min_trust_score
            or failed_canary
        )
        network_score = (
            0.55 * quality_score
            + 0.25 * node_trust
            + 0.20 * latency_score
        )
        if blocked:
            network_score *= 0.2

        reason = (
            f"quality={quality_score:.2f}, trust={node_trust:.2f}, latency={latency_score:.2f}, "
            f"reflect={reflect['verdict']}"
        )
        if node is None:
            reason += ", node_unknown"
        if failed_canary:
            reason += ", canary_failed"

        return HiveCandidateAssessment(
            node_id=candidate.node_id,
            output=candidate.output,
            latency_ms=float(candidate.latency_ms),
            reflect_score=float(reflect["score"]),
            reflect_verdict=str(reflect["verdict"]),
            reflect_action=str(reflect["action"]),
            trust_score=node_trust,
            latency_score=latency_score,
            network_score=clamp01(network_score),
            blocked=blocked,
            reason=reason,
        )

    def _latency_score(self, latency_ms: float, latency_budget_ms: int) -> float:
        """Turn raw latency into a 0..1 preference score."""
        budget = max(1, int(latency_budget_ms))
        return clamp01(1.0 - (max(float(latency_ms), 0.0) / budget))
