import asyncio
import unittest
from unittest.mock import patch

from nexus.router.mind_router import MindRouter
from nexus.runtime.context_reducer import ContextReductionResult


class StubReflectScore:
    async def assess_response(self, question: str, response: str) -> dict:
        return {
            "score": 0.05,
            "verdict": "clean",
            "action": "serve",
            "warning": None,
            "should_warn": False,
            "should_reroute": False,
        }

    def blocked_response(self, question: str, score: float) -> str:
        return f"blocked::{score}"


class TrackingMindRouter(MindRouter):
    def __init__(self, *, context_reducer=None):
        super().__init__(context_reducer=context_reducer)
        self.local_prompts: list[str] = []
        self.agent_calls: list[tuple[str, str]] = []
        self.cloud_prompts: list[str] = []
        self.hive_calls: list[tuple[str | None, str]] = []

    async def _call_local(self, task: str) -> str:
        self.local_prompts.append(task)
        return "local-grounded-response"

    async def _call_agent_runtime(self, task: str, agent_name: str) -> str:
        self.agent_calls.append((agent_name, task))
        return f"agent-response::{agent_name}"

    async def _call_cloud(self, task: str) -> str:
        self.cloud_prompts.append(task)
        return "cloud-response"

    async def _call_hive(self, task: str, intent: str | None) -> tuple[str, dict]:
        self.hive_calls.append((intent, task))
        return (
            "hive-synthesized-response",
            {
                "assembled_output": "hive-synthesized-response",
                "winner": {"node_id": "forge-blr"},
                "canary_results": [],
            },
        )


class MindRouterTests(unittest.TestCase):
    def test_repo_aware_question_uses_workspace_grounding(self):
        router = TrackingMindRouter()

        with patch("nexus.reflect.reflect_score.ReflectScore", StubReflectScore):
            result = asyncio.run(router.route_with_reflection("In this repository, what is NEXUS?"))

        self.assertEqual(result["initial_route"], "local")
        self.assertTrue(result["workspace_grounded"])
        self.assertEqual(router.agent_calls, [])
        self.assertEqual(len(router.local_prompts), 1)
        self.assertIn("Repository context:", router.local_prompts[0])
        self.assertIn("README excerpt", router.local_prompts[0])

    def test_capability_prompt_uses_workspace_grounding(self):
        router = TrackingMindRouter()

        with patch("nexus.reflect.reflect_score.ReflectScore", StubReflectScore):
            result = asyncio.run(router.route_with_reflection("what you can build"))

        self.assertEqual(result["initial_route"], "local")
        self.assertTrue(result["workspace_grounded"])
        self.assertEqual(router.agent_calls, [])
        self.assertIn("strong next prompts or commands", router.local_prompts[0])

    def test_explicit_coding_request_without_workspace_signal_stays_agent_routed(self):
        router = TrackingMindRouter()

        with patch("nexus.reflect.reflect_score.ReflectScore", StubReflectScore):
            result = asyncio.run(router.route_with_reflection("Build a FastAPI auth service"))

        self.assertEqual(result["initial_route"], "agent")
        self.assertFalse(result["workspace_grounded"])
        self.assertEqual(router.local_prompts, [])
        self.assertEqual(router.agent_calls, [("coding", "Build a FastAPI auth service")])

    def test_explicit_hive_prompt_routes_to_hive(self):
        router = TrackingMindRouter()

        with patch("nexus.reflect.reflect_score.ReflectScore", StubReflectScore):
            result = asyncio.run(router.route_with_reflection("/hive build me a full authentication system"))

        self.assertEqual(result["initial_route"], "hive")
        self.assertEqual(result["final_route"], "hive")
        self.assertEqual(router.hive_calls, [("coding", "/hive build me a full authentication system")])
        self.assertEqual(result["response"], "hive-synthesized-response")
        self.assertIsNotNone(result["hive_details"])

    def test_router_exposes_context_reduction_metadata_for_large_prompt(self):
        class StubReducer:
            backend_name = "stub"

            def reduce(self, text: str, *, metadata=None):
                return ContextReductionResult(
                    text="[ROUTER_REDUCED]",
                    reduced=True,
                    backend="stub",
                    strategy="unit_test",
                    original_length=len(text),
                    reduced_length=len("[ROUTER_REDUCED]"),
                    metadata=dict(metadata or {}),
                )

        router = TrackingMindRouter(context_reducer=StubReducer())

        with patch("nexus.reflect.reflect_score.ReflectScore", StubReflectScore):
            result = asyncio.run(router.route_with_reflection("Build a FastAPI auth service"))

        self.assertEqual(router.agent_calls, [("coding", "[ROUTER_REDUCED]")])
        self.assertIsNotNone(result["context_reduction"])
        self.assertEqual(result["context_reduction"]["backend"], "stub")
        self.assertEqual(result["context_reduction"]["metadata"]["scope"], "router")
        self.assertEqual(result["context_reduction"]["metadata"]["route"], "agent")


if __name__ == "__main__":
    unittest.main()
