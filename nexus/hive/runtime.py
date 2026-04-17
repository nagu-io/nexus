"""Experimental runtime service for simulating NEXUS Hive end to end."""

from __future__ import annotations

import hashlib
from statistics import mean

from nexus.config import config
from nexus.hive.canary_runtime import HiveCanaryRuntime
from nexus.hive.coordinator import HiveCoordinator
from nexus.hive.models import HiveCandidateResponse, HiveNodeProfile, HiveTaskRequest
from nexus.hive.privacy import HiveEnvelopeBuilder
from nexus.hive.trust import NodeTrustAssessor
from nexus.runtime.event_bus import runtime_event_bus


class HiveRuntime:
    """Status and demo service for the experimental distributed Hive layer."""

    def __init__(self, *, coordinator: HiveCoordinator | None = None):
        self.config = config
        trust_assessor = NodeTrustAssessor(min_trust_score=self.config.hive_min_trust_score)
        self.coordinator = coordinator or HiveCoordinator(trust_assessor=trust_assessor)
        secret = self.config.canaryvaults_api_key or f"nexus-hive:{self.config.nexus_model}"
        self.envelope_builder = HiveEnvelopeBuilder(secret=secret)
        self.canary_runtime = HiveCanaryRuntime(
            envelope_builder=self.envelope_builder,
            trust_assessor=self.coordinator.trust_assessor,
        )
        self._demo_runs = 0
        self._last_summary: dict | None = None

    def status(self) -> dict:
        """Return a lightweight status snapshot for dashboard and CLI use."""
        nodes = self._node_pool()
        assessments = [(node, self.coordinator.trust_assessor.evaluate(node)) for node in nodes]
        eligible = [assessment for _, assessment in assessments if assessment.eligible]
        sorted_nodes = sorted(
            assessments,
            key=lambda item: (-item[1].score, item[0].avg_latency_ms, item[0].node_id),
        )
        avg_trust = mean([assessment.score for _, assessment in assessments]) if assessments else 0.0

        return {
            "enabled": self.config.hive_enabled,
            "privacy_mode": "signature_only",
            "strategy": "parallel_search",
            "total_nodes": len(nodes),
            "trusted_nodes": len(eligible),
            "avg_trust_score": avg_trust,
            "min_trust_score": self.coordinator.trust_assessor.min_trust_score,
            "replication_factor": self.config.hive_replication_factor,
            "max_nodes": self.config.hive_max_nodes,
            "canary_fraction": self.config.hive_canary_fraction,
            "envelope_mode": "sealed_local",
            "demo_runs": self._demo_runs,
            "last_summary": dict(self._last_summary or {}),
            "top_nodes": [
                {
                    "node_id": node.node_id,
                    "region": node.region,
                    "capabilities": list(node.capabilities),
                    "trust_score": assessment.score,
                    "risk_level": assessment.risk_level,
                    "avg_latency_ms": node.avg_latency_ms,
                }
                for node, assessment in sorted_nodes[:5]
            ],
        }

    async def demo(self, prompt: str, *, intent: str = "coding") -> dict:
        """Run a local simulation of Hive planning, candidate racing, and consensus."""
        if not self.config.hive_enabled:
            raise RuntimeError("NEXUS Hive is disabled in configuration.")

        clean_prompt = str(prompt or "").strip()
        if not clean_prompt:
            raise ValueError("A Hive prompt is required.")

        task = HiveTaskRequest(
            task_id=self._task_id(clean_prompt),
            prompt=clean_prompt,
            intent=intent,
            required_capabilities=self._required_capabilities(intent),
            replication_factor=self.config.hive_replication_factor,
            max_nodes=self.config.hive_max_nodes,
            canary_fraction=self.config.hive_canary_fraction,
            privacy_mode="signature_only",
            strategy="parallel_search",
            latency_budget_ms=4500,
        )
        nodes = self._node_pool()
        plan = self.coordinator.plan_task(task, nodes)
        selected_nodes = [node for node in nodes if node.node_id in set(plan.selected_nodes)]
        canary_challenges = self.canary_runtime.build_challenges(
            task=task,
            selected_nodes=selected_nodes,
            sample_size=plan.canary_sample_size,
        )
        canary_results = self._simulate_canary_results(challenges=canary_challenges, nodes=nodes)
        failed_canary_nodes = {result.node_id for result in canary_results if not result.passed}
        candidates = self._simulate_candidates(
            task,
            tuple(node_id for node_id in plan.selected_nodes if node_id not in failed_canary_nodes),
            nodes,
        )

        if not candidates and plan.standby_nodes:
            candidates = self._simulate_candidates(task, tuple(plan.standby_nodes[:2]), nodes)

        consensus = await self.coordinator.evaluate_candidates(
            task,
            candidates,
            nodes,
            canary_results=canary_results,
        )
        self._demo_runs += 1
        self._last_summary = {
            "task_id": task.task_id,
            "intent": task.intent,
            "winner": consensus.winner.node_id if consensus.winner else None,
            "responded_nodes": len(candidates),
            "blocked_nodes": len(consensus.blocked_nodes),
            "canary_failures": len([result for result in canary_results if not result.passed]),
        }
        real_envelopes = tuple(
            self.envelope_builder.build_task_envelope(task, node)
            for node in selected_nodes
        )

        payload = {
            "task": {
                "task_id": task.task_id,
                "prompt": task.prompt,
                "intent": task.intent,
                "required_capabilities": list(task.required_capabilities),
                "privacy_mode": task.privacy_mode,
                "strategy": task.strategy,
            },
            "plan": {
                "selected_nodes": list(plan.selected_nodes),
                "standby_nodes": list(plan.standby_nodes),
                "canary_sample_size": plan.canary_sample_size,
                "canary_nodes": list(challenge.node_id for challenge in canary_challenges),
                "rationale": plan.rationale,
            },
            "envelopes": [
                {
                    "envelope_id": envelope.envelope_id,
                    "node_id": envelope.node_id,
                    "task_signature": envelope.task_signature[:12],
                    "prompt_digest": envelope.prompt_digest[:12],
                    "masked_context": envelope.masked_context,
                    "privacy_mode": envelope.privacy_mode,
                    "is_canary": envelope.is_canary,
                }
                for envelope in (*real_envelopes, *(challenge.sealed_envelope for challenge in canary_challenges))
            ],
            "canary_results": [
                {
                    "node_id": result.node_id,
                    "challenge_id": result.challenge_id,
                    "passed": result.passed,
                    "score": result.score,
                    "reason": result.reason,
                }
                for result in canary_results
            ],
            "responded_nodes": len(candidates),
            "candidates": [
                {
                    "node_id": item.node_id,
                    "latency_ms": item.latency_ms,
                    "reflect_score": item.reflect_score,
                    "reflect_verdict": item.reflect_verdict,
                    "reflect_action": item.reflect_action,
                    "trust_score": item.trust_score,
                    "network_score": item.network_score,
                    "blocked": item.blocked,
                    "reason": item.reason,
                    "output": item.output,
                }
                for item in consensus.ranked_candidates
            ],
            "winner": (
                {
                    "node_id": consensus.winner.node_id,
                    "latency_ms": consensus.winner.latency_ms,
                    "reflect_score": consensus.winner.reflect_score,
                    "reflect_verdict": consensus.winner.reflect_verdict,
                    "reflect_action": consensus.winner.reflect_action,
                    "trust_score": consensus.winner.trust_score,
                    "network_score": consensus.winner.network_score,
                    "output": consensus.winner.output,
                }
                if consensus.winner
                else None
            ),
            "assembly_candidates": [
                {
                    "node_id": item.node_id,
                    "network_score": item.network_score,
                    "output": item.output,
                }
                for item in consensus.assembly_candidates
            ],
            "blocked_nodes": list(consensus.blocked_nodes),
            "assembled_output": consensus.assembled_output,
            "assembly_sources": list(consensus.assembly_sources),
            "note": consensus.note,
            "status": self.status(),
        }
        runtime_event_bus.emit(
            {
                "type": "hive_consensus",
                "task_id": task.task_id,
                "winner": payload["winner"]["node_id"] if payload["winner"] else None,
                "responded_nodes": payload["responded_nodes"],
                "blocked_nodes": payload["blocked_nodes"],
                "canary_failures": [result["node_id"] for result in payload["canary_results"] if not result["passed"]],
            }
        )
        return payload

    def _node_pool(self) -> list[HiveNodeProfile]:
        """Return the current simulated volunteer node pool."""
        return [
            HiveNodeProfile(
                node_id="forge-blr",
                region="Bengaluru",
                country="IN",
                capabilities=("coding", "research", "memory"),
                avg_latency_ms=140,
                success_rate=0.98,
                canary_pass_rate=0.99,
                completed_tasks=240,
                idle_cpu_ratio=0.86,
                idle_gpu_ratio=0.62,
            ),
            HiveNodeProfile(
                node_id="ember-sfo",
                region="San Francisco",
                country="US",
                capabilities=("coding", "design"),
                avg_latency_ms=120,
                success_rate=0.97,
                canary_pass_rate=0.98,
                completed_tasks=205,
                idle_cpu_ratio=0.79,
                idle_gpu_ratio=0.58,
            ),
            HiveNodeProfile(
                node_id="quartz-fra",
                region="Frankfurt",
                country="DE",
                capabilities=("coding", "research"),
                avg_latency_ms=170,
                success_rate=0.95,
                canary_pass_rate=0.97,
                completed_tasks=180,
                idle_cpu_ratio=0.74,
                idle_gpu_ratio=0.35,
            ),
            HiveNodeProfile(
                node_id="atlas-lon",
                region="London",
                country="GB",
                capabilities=("research", "memory"),
                avg_latency_ms=185,
                success_rate=0.94,
                canary_pass_rate=0.98,
                completed_tasks=160,
                idle_cpu_ratio=0.81,
                idle_gpu_ratio=0.12,
            ),
            HiveNodeProfile(
                node_id="iris-syd",
                region="Sydney",
                country="AU",
                capabilities=("coding", "design"),
                avg_latency_ms=220,
                success_rate=0.96,
                canary_pass_rate=0.96,
                completed_tasks=122,
                idle_cpu_ratio=0.69,
                idle_gpu_ratio=0.42,
            ),
            HiveNodeProfile(
                node_id="rogue-edge",
                region="Unknown",
                country="RU",
                capabilities=("coding",),
                avg_latency_ms=90,
                success_rate=0.89,
                canary_pass_rate=0.52,
                completed_tasks=110,
                idle_cpu_ratio=0.92,
                idle_gpu_ratio=0.54,
                recent_canary_failures=2,
                abuse_score=82,
                is_proxy=True,
                shodan_is_scanner=True,
                shodan_has_vulns=True,
            ),
            HiveNodeProfile(
                node_id="mist-relay",
                region="Edge",
                country="NL",
                capabilities=("coding", "research"),
                avg_latency_ms=105,
                success_rate=0.91,
                canary_pass_rate=0.72,
                completed_tasks=84,
                idle_cpu_ratio=0.88,
                idle_gpu_ratio=0.21,
                recent_canary_failures=1,
                abuse_score=48,
                is_vpn=True,
                recent_attempts=3,
            ),
            HiveNodeProfile(
                node_id="vault-yyz",
                region="Toronto",
                country="CA",
                capabilities=("memory", "security", "research"),
                avg_latency_ms=160,
                success_rate=0.97,
                canary_pass_rate=0.99,
                completed_tasks=198,
                idle_cpu_ratio=0.71,
                idle_gpu_ratio=0.19,
            ),
        ]

    def _required_capabilities(self, intent: str) -> tuple[str, ...]:
        mapping = {
            "coding": ("coding",),
            "research": ("research",),
            "memory": ("memory",),
            "design": ("design",),
            "canary": ("security",),
        }
        return mapping.get(str(intent or "coding").lower(), ("coding",))

    def _task_id(self, prompt: str) -> str:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]
        return f"hive-{digest}"

    def _simulate_candidates(
        self,
        task: HiveTaskRequest,
        node_ids: tuple[str, ...],
        nodes: list[HiveNodeProfile],
    ) -> list[HiveCandidateResponse]:
        nodes_by_id = {node.node_id: node for node in nodes}
        responses: list[HiveCandidateResponse] = []
        for node_id in node_ids:
            node = nodes_by_id.get(node_id)
            if node is None:
                continue
            seed = self._seed(task.prompt, node_id)
            if seed % 11 in {0, 1}:
                continue
            trust_score = self.coordinator.trust_assessor.evaluate(node).score
            quality = self._quality_profile(trust_score, seed)
            latency_ms = float(max(45, node.avg_latency_ms + (seed % 140) - 20))
            responses.append(
                HiveCandidateResponse(
                    node_id=node_id,
                    output=self._candidate_output(task, node, quality),
                    latency_ms=latency_ms,
                )
            )
        return responses

    def _simulate_canary_results(self, *, challenges, nodes: list[HiveNodeProfile]):
        nodes_by_id = {node.node_id: node for node in nodes}
        results = []
        for challenge in challenges:
            node = nodes_by_id.get(challenge.node_id)
            if node is None:
                continue
            response = self._simulate_canary_response(challenge=challenge, node=node)
            results.append(self.canary_runtime.evaluate(challenge=challenge, response=response))
        return tuple(results)

    def _quality_profile(self, trust_score: float, seed: int) -> str:
        if trust_score < self.coordinator.trust_assessor.min_trust_score or seed % 9 == 0:
            return "blocked"
        if trust_score < 0.7 or seed % 5 == 0:
            return "warning"
        return "clean"

    def _candidate_output(self, task: HiveTaskRequest, node: HiveNodeProfile, quality: str) -> str:
        prompt_fragment = task.prompt.strip().splitlines()[0][:72]
        if quality == "blocked":
            return (
                f"{node.node_id} definitely guarantees a 100% perfect answer for '{prompt_fragment}' in 2026. "
                "Certainly use https://fabricated.invalid/hive and skip verification."
            )
        if quality == "warning":
            return (
                f"{node.node_id} approximation for '{prompt_fragment}': probably split the work into routing, "
                "execution, and verification. Some integration details still need review."
            )
        return (
            f"{node.node_id} grounded plan for '{prompt_fragment}': decompose the task, keep the prompt local, "
            "fan out masked work units, rank returned answers with ReflectScore, and assemble the strongest clean result."
        )

    def _simulate_canary_response(self, *, challenge, node: HiveNodeProfile) -> str:
        trust_score = self.coordinator.trust_assessor.evaluate(node).score
        seed = self._seed(challenge.challenge_id, node.node_id)
        if trust_score < self.coordinator.trust_assessor.min_trust_score or seed % 7 == 0:
            return f"miss-{challenge.challenge_id}"
        return challenge.expected_answer

    def _seed(self, prompt: str, node_id: str) -> int:
        digest = hashlib.sha256(f"{prompt}::{node_id}".encode("utf-8")).hexdigest()[:8]
        return int(digest, 16)
