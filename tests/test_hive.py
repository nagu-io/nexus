import asyncio
import unittest

from nexus.hive import (
    HiveCandidateResponse,
    HiveCanaryResult,
    HiveCoordinator,
    HiveNodeProfile,
    HiveTaskRequest,
    NodeTrustAssessor,
)


class StubReflectScore:
    async def assess_response(self, question: str, response: str) -> dict:
        if "bad" in response.lower():
            return {
                "score": 0.91,
                "verdict": "blocked",
                "action": "block",
                "warning": "blocked",
                "should_warn": False,
                "should_reroute": True,
            }
        if "uncertain" in response.lower():
            return {
                "score": 0.42,
                "verdict": "warning",
                "action": "warn",
                "warning": "warn",
                "should_warn": True,
                "should_reroute": False,
            }
        return {
            "score": 0.08,
            "verdict": "clean",
            "action": "serve",
            "warning": None,
            "should_warn": False,
            "should_reroute": False,
        }


class HiveTrustTests(unittest.TestCase):
    def test_canary_failures_and_risk_penalize_node_trust(self):
        assessor = NodeTrustAssessor()
        healthy = HiveNodeProfile(
            node_id="healthy",
            capabilities=("coding",),
            success_rate=0.98,
            canary_pass_rate=0.99,
            completed_tasks=120,
            idle_cpu_ratio=0.9,
        )
        risky = HiveNodeProfile(
            node_id="risky",
            capabilities=("coding",),
            success_rate=0.92,
            canary_pass_rate=0.55,
            completed_tasks=120,
            idle_cpu_ratio=0.9,
            recent_canary_failures=2,
            abuse_score=80,
            is_tor=True,
            shodan_is_scanner=True,
        )

        healthy_score = assessor.evaluate(healthy)
        risky_score = assessor.evaluate(risky)

        self.assertTrue(healthy_score.eligible)
        self.assertFalse(risky_score.eligible)
        self.assertGreater(healthy_score.score, risky_score.score)


class HiveCoordinatorTests(unittest.TestCase):
    def test_plan_task_selects_trusted_capable_nodes(self):
        coordinator = HiveCoordinator(reflect_scorer=StubReflectScore())
        task = HiveTaskRequest(
            task_id="auth",
            prompt="build auth",
            required_capabilities=("coding",),
            replication_factor=2,
            max_nodes=4,
            canary_fraction=0.25,
        )
        nodes = [
            HiveNodeProfile(
                node_id="top",
                capabilities=("coding",),
                success_rate=0.99,
                canary_pass_rate=0.99,
                completed_tasks=200,
                avg_latency_ms=90,
            ),
            HiveNodeProfile(
                node_id="mid",
                capabilities=("coding",),
                success_rate=0.94,
                canary_pass_rate=0.97,
                completed_tasks=60,
                avg_latency_ms=110,
            ),
            HiveNodeProfile(
                node_id="bad",
                capabilities=("coding",),
                success_rate=0.9,
                canary_pass_rate=0.5,
                recent_canary_failures=2,
                abuse_score=75,
                is_proxy=True,
                avg_latency_ms=40,
            ),
            HiveNodeProfile(
                node_id="design-only",
                capabilities=("design",),
                success_rate=0.99,
                canary_pass_rate=1.0,
                avg_latency_ms=50,
            ),
        ]

        plan = coordinator.plan_task(task, nodes)

        self.assertEqual(plan.selected_nodes, ("top", "mid"))
        self.assertEqual(plan.canary_sample_size, 1)
        self.assertNotIn("bad", plan.selected_nodes)

    def test_evaluate_candidates_prefers_clean_trusted_answer(self):
        coordinator = HiveCoordinator(
            reflect_scorer=StubReflectScore(),
            trust_assessor=NodeTrustAssessor(min_trust_score=0.45),
        )
        task = HiveTaskRequest(
            task_id="auth",
            prompt="build auth",
            required_capabilities=("coding",),
            latency_budget_ms=1000,
        )
        nodes = [
            HiveNodeProfile(
                node_id="trusted",
                capabilities=("coding",),
                success_rate=0.98,
                canary_pass_rate=0.99,
                completed_tasks=180,
                avg_latency_ms=140,
            ),
            HiveNodeProfile(
                node_id="fast-bad",
                capabilities=("coding",),
                success_rate=0.95,
                canary_pass_rate=0.98,
                completed_tasks=180,
                avg_latency_ms=50,
            ),
        ]
        candidates = [
            HiveCandidateResponse(node_id="fast-bad", output="bad fabricated answer", latency_ms=60),
            HiveCandidateResponse(node_id="trusted", output="clean grounded answer", latency_ms=180),
            HiveCandidateResponse(node_id="trusted", output="uncertain but usable answer", latency_ms=150),
        ]

        result = asyncio.run(coordinator.evaluate_candidates(task, candidates, nodes))

        self.assertIsNotNone(result.winner)
        self.assertEqual(result.winner.node_id, "trusted")
        self.assertEqual(result.blocked_nodes, ("fast-bad",))
        self.assertGreaterEqual(len(result.assembly_candidates), 2)
        self.assertEqual(result.ranked_candidates[0].reflect_action, "serve")

    def test_canary_failure_blocks_otherwise_clean_node(self):
        coordinator = HiveCoordinator(
            reflect_scorer=StubReflectScore(),
            trust_assessor=NodeTrustAssessor(min_trust_score=0.45),
        )
        task = HiveTaskRequest(
            task_id="auth",
            prompt="build auth",
            required_capabilities=("coding",),
            latency_budget_ms=1000,
        )
        nodes = [
            HiveNodeProfile(
                node_id="trusted",
                capabilities=("coding",),
                success_rate=0.98,
                canary_pass_rate=0.99,
                completed_tasks=180,
                avg_latency_ms=140,
            ),
        ]
        candidates = [
            HiveCandidateResponse(node_id="trusted", output="clean grounded answer", latency_ms=180),
        ]
        canary_results = (
            HiveCanaryResult(
                node_id="trusted",
                challenge_id="canary-1",
                passed=False,
                response="wrong-token",
                score=0.0,
                reason="returned the wrong token",
            ),
        )

        result = asyncio.run(coordinator.evaluate_candidates(task, candidates, nodes, canary_results=canary_results))

        self.assertIsNone(result.winner)
        self.assertIn("trusted", result.blocked_nodes)
        self.assertEqual(result.canary_results, canary_results)
        self.assertIn("no trusted candidates", result.assembled_output.lower())


if __name__ == "__main__":
    unittest.main()
