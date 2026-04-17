"""Canary challenge injection and evaluation for NEXUS Hive."""

from __future__ import annotations

import hashlib

from nexus.hive.models import HiveCanaryChallenge, HiveCanaryResult, HiveNodeProfile, HiveTaskRequest
from nexus.hive.privacy import HiveEnvelopeBuilder
from nexus.hive.trust import NodeTrustAssessor


class HiveCanaryRuntime:
    """Generate and score canary tasks for Hive nodes."""

    def __init__(self, *, envelope_builder: HiveEnvelopeBuilder, trust_assessor: NodeTrustAssessor):
        self.envelope_builder = envelope_builder
        self.trust_assessor = trust_assessor

    def build_challenges(
        self,
        *,
        task: HiveTaskRequest,
        selected_nodes: list[HiveNodeProfile],
        sample_size: int,
    ) -> tuple[HiveCanaryChallenge, ...]:
        """Create canary challenges for the least-trusted selected nodes first."""
        if sample_size <= 0 or not selected_nodes:
            return ()

        ranked = sorted(
            selected_nodes,
            key=lambda node: (
                self.trust_assessor.evaluate(node).score,
                -node.avg_latency_ms,
                node.node_id,
            ),
        )
        challenges: list[HiveCanaryChallenge] = []
        for node in ranked[:sample_size]:
            challenge_id = hashlib.sha256(f"{task.task_id}:canary:{node.node_id}".encode("utf-8")).hexdigest()[:10]
            expected_answer = f"canary-pass-{challenge_id}"
            prompt = (
                "Canary verification task.\n"
                "Return the exact token below and nothing else.\n"
                f"Token: {expected_answer}"
            )
            draft = HiveCanaryChallenge(
                challenge_id=challenge_id,
                node_id=node.node_id,
                prompt=prompt,
                expected_answer=expected_answer,
                sealed_envelope=self.envelope_builder.build_task_envelope(task, node),
            )
            sealed = self.envelope_builder.build_canary_envelope(task, draft)
            challenges.append(
                HiveCanaryChallenge(
                    challenge_id=challenge_id,
                    node_id=node.node_id,
                    prompt=prompt,
                    expected_answer=expected_answer,
                    sealed_envelope=sealed,
                )
            )
        return tuple(challenges)

    def evaluate(self, *, challenge: HiveCanaryChallenge, response: str) -> HiveCanaryResult:
        """Judge a node's canary response."""
        clean = str(response or "").strip()
        passed = clean == challenge.expected_answer
        score = 1.0 if passed else 0.0
        reason = "matched expected token" if passed else "returned the wrong token"
        return HiveCanaryResult(
            node_id=challenge.node_id,
            challenge_id=challenge.challenge_id,
            passed=passed,
            response=clean,
            score=score,
            reason=reason,
        )
