import asyncio
import unittest

from nexus.hive.coordinator import HiveCoordinator
from nexus.hive.runtime import HiveRuntime
from nexus.hive.trust import NodeTrustAssessor


class StubReflectScore:
    async def assess_response(self, question: str, response: str) -> dict:
        text = response.lower()
        if "definitely" in text or "100%" in text or "certainly" in text:
            return {
                "score": 0.82,
                "verdict": "blocked",
                "action": "block",
                "warning": "blocked",
                "should_warn": False,
                "should_reroute": True,
            }
        if "probably" in text or "approximation" in text:
            return {
                "score": 0.41,
                "verdict": "warning",
                "action": "warn",
                "warning": "warn",
                "should_warn": True,
                "should_reroute": False,
            }
        return {
            "score": 0.06,
            "verdict": "clean",
            "action": "serve",
            "warning": None,
            "should_warn": False,
            "should_reroute": False,
        }


class HiveRuntimeTests(unittest.TestCase):
    def _runtime(self) -> HiveRuntime:
        coordinator = HiveCoordinator(
            trust_assessor=NodeTrustAssessor(min_trust_score=0.45),
            reflect_scorer=StubReflectScore(),
        )
        return HiveRuntime(coordinator=coordinator)

    def test_status_reports_trusted_pool(self):
        runtime = self._runtime()

        status = runtime.status()

        self.assertTrue(status["enabled"])
        self.assertGreaterEqual(status["total_nodes"], status["trusted_nodes"])
        self.assertGreater(len(status["top_nodes"]), 0)

    def test_demo_returns_ranked_consensus(self):
        runtime = self._runtime()

        result = asyncio.run(runtime.demo("build me a full authentication system", intent="coding"))

        self.assertIn("plan", result)
        self.assertIn("candidates", result)
        self.assertGreater(len(result["plan"]["selected_nodes"]), 0)
        self.assertGreaterEqual(result["responded_nodes"], 1)
        self.assertIsNotNone(result["winner"])
        self.assertGreaterEqual(len(result["assembly_candidates"]), 1)
        self.assertGreaterEqual(result["status"]["demo_runs"], 1)
        self.assertTrue(result["assembled_output"])
        self.assertGreaterEqual(len(result["envelopes"]), len(result["plan"]["selected_nodes"]))
        self.assertEqual(len(result["canary_results"]), result["plan"]["canary_sample_size"])
        self.assertIn("masked_context", result["envelopes"][0])


if __name__ == "__main__":
    unittest.main()
