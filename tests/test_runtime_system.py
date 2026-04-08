import asyncio
import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nexus.agents.base_agent import BaseAgent
from nexus.agents.coding_agent import CodingAgent
from nexus.blueprint_generator import BlueprintGenerator, TaskBlueprint, WorkflowBlueprint
from nexus.critics.base import BaseCritic
from nexus.critics.multi_critic import MultiCriticEvaluator
from nexus.compiler.planner_engine import PlannerEngine
from nexus.intent_parser import IntentParser
from nexus.memory.environment_memory import EnvironmentMemory
from nexus.memory.skill_memory import SkillMemory
from nexus.orchestrator import Orchestrator
from nexus.runtime.build_artifacts import BuildArtifactError, BuildArtifactMaterializer
from nexus.runtime.code_reader import CodeReader
from nexus.runtime.context_reducer import ContextReductionResult
from nexus.runtime.decision_cache import DecisionCache
from nexus.runtime.file_tool import FileTool
from nexus.runtime.insights import RuntimeInsights
from nexus.runtime.policy_engine import PolicyEngine
from nexus.runtime.project_mode import ProjectModeManager
from nexus.runtime.scaffold_runner import ScaffoldRunError, ScaffoldRunner
from nexus.runtime.strategy_engine import StrategyEngine
from nexus.runtime.terminal_tool import TerminalTool
from nexus.shared_memory import SharedMemory
from nexus.wiring_engine import WiringEngine


class EchoAgent(BaseAgent):
    name = "research"
    capabilities = ("reasoning", "summarization")

    async def run(self, task: str) -> str:
        return f"research-notes::{task}"


class PromptCapturingAgent(BaseAgent):
    name = "coding"
    capabilities = ("reasoning", "code_generation")

    async def run(self, task: str) -> str:
        return task


class FlakyAgent(BaseAgent):
    name = "coding"
    capabilities = ("reasoning", "code_generation")

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def run(self, task: str) -> str:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary failure")
        return f"completed::{task}"


class MemoryOnlyAgent(BaseAgent):
    name = "memory"
    capabilities = ("memory_write",)

    async def run(self, task: str) -> str:
        return f"stored::{task}"


class SlowAgent(BaseAgent):
    name = "coding"
    capabilities = ("reasoning", "code_generation")

    async def run(self, task: str) -> str:
        await asyncio.sleep(0.05)
        return "slow-result"


class TimedResearchAgent(BaseAgent):
    name = "research"
    capabilities = ("reasoning", "summarization")
    started_at: list[float] = []

    async def run(self, task: str) -> str:
        self.__class__.started_at.append(time.perf_counter())
        await asyncio.sleep(0.05)
        return "parallel-research"


class TimedMemoryReadAgent(BaseAgent):
    name = "memory"
    capabilities = ("memory_read",)
    started_at: list[float] = []

    async def run(self, task: str) -> str:
        self.__class__.started_at.append(time.perf_counter())
        await asyncio.sleep(0.05)
        return "parallel-memory"


class LongOutputAgent(BaseAgent):
    name = "research"
    capabilities = ("reasoning", "summarization")

    async def run(self, task: str) -> str:
        return "LONG_CONTEXT_BLOCK::" + ("abcdefghijklmnopqrstuvwxyz " * 600)


class StubContextReducer:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def reduce(self, text: str, *, metadata: dict | None = None) -> ContextReductionResult:
        self.calls.append({"text": text, "metadata": dict(metadata or {})})
        return ContextReductionResult(
            text="[REDUCED_PROMPT]",
            reduced=True,
            backend="stub",
            strategy="stub_reduce",
            original_length=len(text),
            reduced_length=len("[REDUCED_PROMPT]"),
            metadata=dict(metadata or {}),
        )


class ToolCallingAgent(BaseAgent):
    name = "coding"
    capabilities = ("reasoning", "code_generation")

    async def run(self, task: str):
        return self.tool_call(
            tool="file_tool",
            action="write_file",
            path="artifacts/note.txt",
            content="hello from nexus",
        )

    async def continue_after_tool(
        self,
        task: str,
        tool_result: dict,
        memory=None,
        thought: dict | None = None,
    ):
        if tool_result.get("ok"):
            return f"tool-finished::{tool_result['summary']}"
        return f"Tool error: {tool_result.get('summary')}"


class FixingLoopAgent(BaseAgent):
    name = "coding"
    capabilities = ("reasoning", "code_generation", "debugging")

    async def act(self, task: str, memory=None, thought: dict | None = None):
        return self.tool_call(
            tool="file_tool",
            action="write_file",
            path="main.py",
            content='print("broken"\n',
        )

    async def continue_after_tool(
        self,
        task: str,
        tool_result: dict,
        memory=None,
        thought: dict | None = None,
    ):
        if tool_result.get("tool") == "file_tool" and tool_result.get("ok"):
            return self.tool_call(
                tool="terminal_tool",
                action="run_command",
                command=["python", "main.py"],
                cwd=memory.get("workspace.root_dir"),
                timeout_seconds=30,
            )
        if tool_result.get("tool") == "terminal_tool" and not tool_result.get("ok") and not memory.get("workspace.fixed_once"):
            memory.put("workspace.fixed_once", True)
            return self.tool_call(
                tool="file_tool",
                action="edit_file",
                path="main.py",
                old_text='print("broken"\n',
                new_text='print("fixed")\n',
                replace_all=False,
            )
        if tool_result.get("tool") == "terminal_tool" and tool_result.get("ok"):
            return "fix-loop-complete"
        return f"Tool error: {tool_result.get('summary')}"


class LowThenHighConfidenceScorer:
    def __init__(self):
        self.calls = 0

    async def assess_response(self, question: str, response: str) -> dict:
        self.calls += 1
        low_confidence = "Retry note:" not in response
        score = 0.8 if low_confidence else 0.1
        action = "block" if score >= 0.6 else "serve"
        verdict = "blocked" if action == "block" else "clean"
        return {
            "score": score,
            "verdict": verdict,
            "action": action,
            "warning": None,
            "should_warn": False,
            "should_reroute": action == "block",
        }


class AlwaysHighConfidenceScorer:
    def __init__(self):
        self.calls = 0

    async def assess_response(self, question: str, response: str) -> dict:
        self.calls += 1
        return {
            "score": 0.05,
            "verdict": "clean",
            "action": "serve",
            "warning": None,
            "should_warn": False,
            "should_reroute": False,
        }


class FastPassCritic(BaseCritic):
    def __init__(self, name: str, score: float, weight: float):
        self.name = name
        self.score = score
        self.weight = weight

    async def evaluate(
        self,
        *,
        task,
        output: str,
        observation: dict,
        attempt: int,
        max_attempts: int,
    ):
        return self.assessment(score=self.score, reason=f"{self.name} passed")


class CountingExpensiveCritic(BaseCritic):
    name = "correctness"
    weight = 0.6
    cost_tier = "high"

    def __init__(self, score: float = 0.92):
        self.score = score
        self.calls = 0

    async def evaluate(
        self,
        *,
        task,
        output: str,
        observation: dict,
        attempt: int,
        max_attempts: int,
    ):
        self.calls += 1
        return self.assessment(score=self.score, reason="expensive correctness ran")


class IntentParserTests(unittest.TestCase):
    def test_parse_identifies_multi_agent_goal(self):
        parser = IntentParser()
        intent = parser.parse("Research FastAPI auth patterns and build the implementation in Python")

        self.assertEqual(intent.primary_intent, "coding")
        self.assertIn("research", intent.required_agents)
        self.assertIn("coding", intent.required_agents)

    def test_parse_marks_full_stack_goal_as_high_complexity(self):
        parser = IntentParser()
        intent = parser.parse("build a full stack login system with Express backend, API routes, and basic frontend form")

        self.assertEqual(intent.complexity, "high")


class CodeReaderTests(unittest.TestCase):
    def test_detects_frameworks_and_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "package.json").write_text(
                json.dumps(
                    {
                        "dependencies": {"react": "^18.0.0"},
                        "devDependencies": {"vite": "^5.0.0"},
                        "scripts": {"dev": "vite", "build": "vite build"},
                    }
                ),
                encoding="utf-8",
            )
            (temp_path / "src").mkdir()
            (temp_path / "src" / "App.jsx").write_text("export default function App() { return null; }\n", encoding="utf-8")

            context = CodeReader().scan(temp_path)

        self.assertIn("react", context["frameworks"])
        self.assertIn("vite", context["frameworks"])
        self.assertIn("javascript", context["languages"])
        self.assertIn("src/App.jsx", context["entrypoints"])
        self.assertTrue(context["project_signature"])


class PlannerEngineTests(unittest.TestCase):
    def test_reuses_skill_pattern_overrides(self):
        parser = IntentParser()
        intent = parser.parse("Build a local Python service safely")

        with tempfile.TemporaryDirectory() as temp_dir:
            skill_memory = SkillMemory(storage_path=Path(temp_dir) / "skill_memory.json")
            signature = skill_memory.signature_for_intent(intent)
            remembered_blueprint = WorkflowBlueprint(
                goal=intent.goal,
                primary_intent=intent.primary_intent,
                tasks=[
                    TaskBlueprint(
                        id="coding_1",
                        task_type="solution",
                        agent="coding",
                        instruction="Remembered solution task",
                        retry_strategy="switch_agent",
                        fallback="Use the remembered fallback path.",
                        confidence_threshold=0.72,
                        candidate_agents=["coding", "research"],
                        required_capabilities=["reasoning"],
                    )
                ],
                metadata={"plan_signature": signature, "complexity": intent.complexity, "required_agents": intent.required_agents},
            )
            skill_memory.record_workflow(remembered_blueprint, [{"agent": "coding", "attempts": 1}], "completed")

            planner = PlannerEngine(skill_memory=skill_memory)
            plan = planner.plan(intent)

        solution_task = next(task for task in plan.tasks if task.task_type == "solution")
        self.assertTrue(plan.metadata["skill_pattern_reused"])
        self.assertEqual(solution_task.retry_strategy, "switch_agent")
        self.assertAlmostEqual(solution_task.confidence_threshold, 0.72)

    def test_avoids_high_retry_patterns(self):
        parser = IntentParser()
        intent = parser.parse("Build a local Python service safely")

        with tempfile.TemporaryDirectory() as temp_dir:
            skill_memory = SkillMemory(storage_path=Path(temp_dir) / "skill_memory.json")
            signature = skill_memory.signature_for_intent(intent)
            remembered_blueprint = WorkflowBlueprint(
                goal=intent.goal,
                primary_intent=intent.primary_intent,
                tasks=[
                    TaskBlueprint(
                        id="coding_1",
                        task_type="solution",
                        agent="coding",
                        instruction="Remembered noisy solution task",
                        retry_strategy="repeat",
                        candidate_agents=["coding"],
                        required_capabilities=["reasoning", "code_generation"],
                    )
                ],
                metadata={"plan_signature": signature, "complexity": intent.complexity, "required_agents": intent.required_agents},
            )
            skill_memory.record_workflow(remembered_blueprint, [{"agent": "coding", "attempts": 4}], "completed")

            planner = PlannerEngine(skill_memory=skill_memory)
            plan = planner.plan(intent)

        solution_task = next(task for task in plan.tasks if task.task_type == "solution")
        self.assertEqual(plan.metadata["skill_pattern_reuse_mode"], "avoid_high_retry_pattern")
        self.assertEqual(solution_task.retry_strategy, "plan_modification")
        self.assertFalse(plan.metadata["skill_pattern_reused"])

    def test_full_stack_build_skips_research_context(self):
        parser = IntentParser()
        intent = parser.parse("build a full stack login system with Express backend, API routes, and basic frontend form")

        with tempfile.TemporaryDirectory() as temp_dir:
            planner = PlannerEngine(skill_memory=SkillMemory(storage_path=Path(temp_dir) / "skill_memory.json"))
            plan = planner.plan(intent)

        task_types = [task.task_type for task in plan.tasks]
        self.assertEqual(task_types[0], "solution")
        self.assertIn("test_generation", task_types)
        self.assertNotIn("research_context", task_types)

    def test_project_mode_applies_framework_context_and_error_learning(self):
        parser = IntentParser()
        project_context = {
            "enabled": True,
            "project_root": "D:/example-app",
            "project_signature": "project-123",
            "project_context": {
                "frameworks": ["react", "vite"],
                "languages": ["typescript"],
                "entrypoints": ["src/App.tsx"],
            },
            "user_preferences": {
                "preferred_frameworks": ["react"],
                "preferred_languages": ["typescript"],
            },
            "recent_goals": ["Build login form"],
            "successful_patterns": [
                {
                    "primary_intent": "coding",
                    "success_rate": 0.92,
                    "avg_retries": 0.5,
                    "avg_confidence": 0.84,
                    "best_agent_sequence": ["coding"],
                    "total_runs": 3,
                }
            ],
            "common_errors": [
                {
                    "failure_type": "import_error",
                    "summary": "Cannot resolve module",
                }
            ],
        }
        intent = parser.parse("Create a login component", project_context=project_context)

        with tempfile.TemporaryDirectory() as temp_dir:
            planner = PlannerEngine(skill_memory=SkillMemory(storage_path=Path(temp_dir) / "skill_memory.json"))
            plan = planner.plan(intent, project_context=project_context)

        solution_task = next(task for task in plan.tasks if task.task_type == "solution")
        self.assertIn("project_mode_active", plan.constraints)
        self.assertTrue(plan.metadata["project_mode"])
        self.assertEqual(plan.metadata["project_signature"], "project-123")
        self.assertEqual(solution_task.metadata["project_frameworks"], ["react", "vite"])
        self.assertEqual(solution_task.retry_strategy, "plan_modification")
        self.assertEqual(solution_task.agent, "coding")


class SkillMemoryTests(unittest.TestCase):
    def test_tracks_metrics_and_ranks_best_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_memory = SkillMemory(storage_path=Path(temp_dir) / "skill_memory.json")
            parser = IntentParser()

            target_intent = parser.parse("Build a Python service")
            better_intent = parser.parse("Build a Python service and save file")
            weaker_intent = parser.parse("Build a comprehensive production Python service")

            better_blueprint = WorkflowBlueprint(
                goal=better_intent.goal,
                primary_intent=better_intent.primary_intent,
                tasks=[
                    TaskBlueprint(
                        id="coding_1",
                        task_type="solution",
                        agent="coding",
                        instruction="Better pattern",
                        candidate_agents=["coding"],
                        required_capabilities=["reasoning"],
                    )
                ],
                metadata={
                    "plan_signature": skill_memory.signature_for_intent(better_intent),
                    "complexity": better_intent.complexity,
                    "required_agents": better_intent.required_agents,
                    "file_action": better_intent.metadata.get("file_action", "none"),
                    "memory_action": better_intent.metadata.get("memory_action", "none"),
                },
            )
            weaker_blueprint = WorkflowBlueprint(
                goal=weaker_intent.goal,
                primary_intent=weaker_intent.primary_intent,
                tasks=[
                    TaskBlueprint(
                        id="coding_1",
                        task_type="solution",
                        agent="coding",
                        instruction="Weaker pattern",
                        candidate_agents=["coding"],
                        required_capabilities=["reasoning"],
                    )
                ],
                metadata={
                    "plan_signature": skill_memory.signature_for_intent(weaker_intent),
                    "complexity": weaker_intent.complexity,
                    "required_agents": weaker_intent.required_agents,
                    "file_action": weaker_intent.metadata.get("file_action", "none"),
                    "memory_action": weaker_intent.metadata.get("memory_action", "none"),
                },
            )

            skill_memory.record_workflow(better_blueprint, [{"agent": "coding", "attempts": 1}], "completed")
            skill_memory.record_workflow(weaker_blueprint, [{"agent": "coding", "attempts": 3}], "failed")
            chosen = skill_memory.lookup(target_intent)

        self.assertIsNotNone(chosen)
        self.assertGreaterEqual(chosen["success_rate"], 1.0)
        self.assertLessEqual(chosen["avg_retries"], 1.0)
        self.assertEqual(chosen["best_agent_sequence"], ["coding"])


class BlueprintGeneratorTests(unittest.TestCase):
    def test_blueprint_orders_research_before_coding(self):
        parser = IntentParser()
        with tempfile.TemporaryDirectory() as temp_dir:
            planner = PlannerEngine(skill_memory=SkillMemory(storage_path=Path(temp_dir) / "skill_memory.json"))
            generator = BlueprintGenerator()

            intent = parser.parse("Research FastAPI auth patterns and build the implementation in Python")
            plan = planner.plan(intent)
            blueprint = generator.generate(plan)

        self.assertGreaterEqual(len(blueprint.tasks), 2)
        self.assertEqual(blueprint.tasks[0].agent, "research")
        self.assertEqual(blueprint.tasks[1].agent, "coding")
        self.assertIn(blueprint.tasks[0].id, blueprint.tasks[1].depends_on)


class WiringEngineTests(unittest.TestCase):
    def test_selects_agent_by_required_capability(self):
        blueprint = WorkflowBlueprint(goal="Store memory", primary_intent="memory", tasks=[])
        task = TaskBlueprint(
            id="memory_1",
            task_type="memory_store",
            agent="coding",
            instruction="Store the task result",
            candidate_agents=["coding", "memory"],
            required_capabilities=["memory_write"],
        )

        wiring = WiringEngine(auto_register=False)
        wiring.register("coding", PromptCapturingAgent)
        wiring.register("memory", MemoryOnlyAgent)

        with tempfile.TemporaryDirectory() as temp_dir:
            memory = SharedMemory(root_dir=Path(temp_dir))
            wired = wiring.wire_task(task, blueprint, memory)

        self.assertEqual(wired.agent.name, "memory")
        self.assertIn("memory_write", wired.selection_reason)


class EnvironmentMemoryTests(unittest.TestCase):
    def test_persists_preferences_patterns_and_common_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            storage_path = temp_path / "environment_memory.json"
            project_root = temp_path / "project"
            project_root.mkdir()
            project_context = {
                "project_root": str(project_root),
                "project_name": "project",
                "project_signature": "project-sig",
                "frameworks": ["react"],
                "languages": ["typescript"],
                "entrypoints": ["src/App.tsx"],
                "directories": ["src"],
                "files": [{"path": "src/App.tsx", "size": 42}],
                "scripts": {"dev": "vite"},
                "summary_text": "React project",
            }

            memory = EnvironmentMemory(storage_path=storage_path)
            session = memory.begin_project_session(
                project_context=project_context,
                goal="Improve the login screen",
                execution_mode="stable",
            )

            blueprint = WorkflowBlueprint(
                goal="Improve the login screen",
                primary_intent="coding",
                tasks=[
                    TaskBlueprint(
                        id="coding_1",
                        task_type="solution",
                        agent="coding",
                        instruction="Improve the login screen",
                    )
                ],
                metadata={"plan_signature": "pattern-1"},
            )
            memory.record_workflow(
                project_root=project_root,
                goal=blueprint.goal,
                blueprint=blueprint,
                executions=[
                    {
                        "task_id": "coding_1",
                        "agent": "coding",
                        "status": "failed",
                        "attempts": 2,
                        "output": "",
                        "observation": {"summary": "module not found", "failure_type": "import_error"},
                        "reflection": {},
                        "error": "module not found",
                    }
                ],
                status="failed",
                final_confidence=0.2,
                execution_mode="stable",
            )
            memory.record_workflow(
                project_root=project_root,
                goal=blueprint.goal,
                blueprint=blueprint,
                executions=[
                    {
                        "task_id": "coding_1",
                        "agent": "coding",
                        "status": "completed",
                        "attempts": 1,
                        "output": "done",
                        "observation": {"summary": "done"},
                        "reflection": {},
                    }
                ],
                status="completed",
                final_confidence=0.88,
                execution_mode="stable",
            )

            reloaded = EnvironmentMemory(storage_path=storage_path)
            persisted = reloaded.project_mode_context(project_root)

        self.assertTrue(session["enabled"])
        self.assertIn("react", persisted["user_preferences"]["preferred_frameworks"])
        self.assertIn("typescript", persisted["user_preferences"]["preferred_languages"])
        self.assertEqual(len(persisted["successful_patterns"]), 1)
        self.assertEqual(persisted["common_errors"][0]["failure_type"], "import_error")


class CodingAgentTests(unittest.TestCase):
    def test_observe_rejects_framework_mismatch(self):
        agent = CodingAgent()
        observation = asyncio.run(
            agent.observe(
                "build a full stack login system with Express backend and frontend form",
                "from flask import Flask\n\ndef login():\n    pass",
            )
        )

        self.assertFalse(observation["ok"])
        self.assertEqual(observation["failure_type"], "framework_mismatch")

    def test_generate_uses_express_login_scaffold_for_common_build_goal(self):
        agent = CodingAgent()
        result = asyncio.run(
            agent._generate(
                "build a full stack login system with Express backend, API routes, and basic frontend form"
            )
        )

        self.assertIn("const express = require(\"express\")", result)
        self.assertIn("router.post(\"/login\"", result)
        self.assertIn("<form id=\"login-form\">", result)
        self.assertIn("fetch(\"/api/login\"", result)

    def test_autonomous_act_emits_structured_tool_call(self):
        agent = CodingAgent()
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = SharedMemory(root_dir=Path(temp_dir) / "memory")
            memory.put("workspace.root_dir", str(Path(temp_dir) / "workspace"))
            memory.put("workspace.project_state", {"files": []})
            tool_request = asyncio.run(
                agent.act(
                    "Task type: solution\nWorkspace root: workspace\nBuild a full stack login system with Express backend, API routes, and basic frontend form",
                    memory=memory,
                )
            )

        self.assertIsInstance(tool_request, dict)
        self.assertEqual(tool_request["type"], "tool_call")
        self.assertEqual(tool_request["tool"], "file_tool")


class MultiCriticTests(unittest.TestCase):
    def test_combines_critic_scores(self):
        task = TaskBlueprint(
            id="coding_1",
            task_type="solution",
            agent="coding",
            instruction="Produce a trustworthy answer",
            confidence_threshold=0.6,
        )
        evaluator = MultiCriticEvaluator(reflect_scorer=LowThenHighConfidenceScorer())
        result = asyncio.run(
            evaluator.evaluate(
                task=task,
                output="A first pass answer",
                observation={"ok": True, "summary": "done"},
                attempt=1,
                max_attempts=2,
            )
        )

        self.assertIn("correctness", result["critic_scores"])
        self.assertIn("efficiency", result["critic_scores"])
        self.assertIn("safety", result["critic_scores"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["failure_type"], "low_confidence")

    def test_lazy_evaluation_reuses_cached_high_confidence_critic(self):
        expensive = CountingExpensiveCritic()
        evaluator = MultiCriticEvaluator(
            critics=[
                FastPassCritic(name="efficiency", score=0.95, weight=0.2),
                FastPassCritic(name="safety", score=0.96, weight=0.2),
                expensive,
            ]
        )
        task = TaskBlueprint(
            id="coding_1",
            task_type="solution",
            agent="coding",
            instruction="Produce code",
            confidence_threshold=0.6,
        )
        result = asyncio.run(
            evaluator.evaluate(
                task=task,
                output="A confident answer",
                observation={"ok": True, "summary": "done"},
                attempt=1,
                max_attempts=2,
                cache_hint={
                    "signature": "abc123",
                    "expected_confidence": 0.91,
                    "critic_scores": {"correctness": 0.92},
                },
            )
        )

        self.assertEqual(expensive.calls, 0)
        self.assertTrue(result["cached"])
        self.assertEqual(result["lazy_path"], "decision_cache")
        self.assertIn("correctness", result["skipped_critics"])


class DecisionCacheTests(unittest.TestCase):
    def test_reuses_high_confidence_successful_strategy(self):
        task = TaskBlueprint(
            id="coding_1",
            task_type="solution",
            agent="coding",
            instruction="Produce code",
        )
        blueprint = WorkflowBlueprint(
            goal="Ship code",
            primary_intent="coding",
            tasks=[task],
            metadata={"plan_signature": "sig-1"},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DecisionCache(storage_path=Path(temp_dir) / "decision_cache.json")
            cache.record(
                task=task,
                blueprint=blueprint,
                agent="coding",
                strategy="plan_modification",
                mode="stable",
                status="completed",
                attempts=1,
                evaluation={
                    "confidence": 0.91,
                    "critic_scores": {"correctness": 0.91, "efficiency": 0.95, "safety": 0.97},
                    "weights_used": {"correctness": 0.6, "efficiency": 0.2, "safety": 0.2},
                },
            )
            hit = cache.lookup(
                task=task,
                blueprint=blueprint,
                agent="coding",
                strategy="plan_modification",
                mode="stable",
                min_confidence=0.85,
            )

        self.assertIsNotNone(hit)
        self.assertGreaterEqual(hit["expected_confidence"], 0.9)
        self.assertEqual(hit["strategy"], "plan_modification")

    def test_stale_entries_invalidate_before_reuse(self):
        task = TaskBlueprint(
            id="coding_1",
            task_type="solution",
            agent="coding",
            instruction="Produce code",
        )
        blueprint = WorkflowBlueprint(
            goal="Ship code",
            primary_intent="coding",
            tasks=[task],
            metadata={"plan_signature": "sig-1"},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DecisionCache(storage_path=Path(temp_dir) / "decision_cache.json")
            cache.record(
                task=task,
                blueprint=blueprint,
                agent="coding",
                strategy="plan_modification",
                mode="stable",
                status="completed",
                attempts=1,
                evaluation={
                    "confidence": 0.93,
                    "critic_scores": {"correctness": 0.93},
                    "weights_used": {"correctness": 1.0},
                },
            )
            signature = cache.signature_for(
                task=task,
                blueprint=blueprint,
                agent="coding",
                strategy="plan_modification",
                mode="stable",
            )
            stale_at = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
            cache._data["entries"][signature]["updated_at"] = stale_at
            cache._data["entries"][signature]["last_used_at"] = stale_at
            cache._save()

            hit = cache.lookup(
                task=task,
                blueprint=blueprint,
                agent="coding",
                strategy="plan_modification",
                mode="stable",
                min_confidence=0.85,
            )
            summary = cache.summary(limit=3)

        self.assertIsNone(hit)
        self.assertEqual(summary["invalidated_entries"], 1)
        self.assertTrue(summary["top_entries"][0]["invalidated"])

    def test_reuses_similar_task_within_same_project(self):
        first_task = TaskBlueprint(
            id="coding_1",
            task_type="solution",
            agent="coding",
            instruction="Build React login form with client validation",
        )
        second_task = TaskBlueprint(
            id="coding_2",
            task_type="solution",
            agent="coding",
            instruction="Create React login form with validation",
        )
        blueprint = WorkflowBlueprint(
            goal="Ship UI",
            primary_intent="coding",
            tasks=[first_task],
            metadata={"plan_signature": "sig-2", "project_signature": "project-1"},
        )
        second_blueprint = WorkflowBlueprint(
            goal="Ship UI",
            primary_intent="coding",
            tasks=[second_task],
            metadata={"plan_signature": "sig-3", "project_signature": "project-1"},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DecisionCache(storage_path=Path(temp_dir) / "decision_cache.json")
            cache.record(
                task=first_task,
                blueprint=blueprint,
                agent="coding",
                strategy="plan_modification",
                mode="stable",
                status="completed",
                attempts=1,
                evaluation={
                    "confidence": 0.9,
                    "critic_scores": {"correctness": 0.9},
                    "weights_used": {"correctness": 1.0},
                },
            )
            hit = cache.lookup(
                task=second_task,
                blueprint=second_blueprint,
                agent="coding",
                strategy="plan_modification",
                mode="stable",
                min_confidence=0.85,
            )

        self.assertIsNotNone(hit)
        self.assertEqual(hit.get("match_type"), "similar")


class BuildArtifactMaterializerTests(unittest.TestCase):
    def test_extracts_and_writes_file_artifacts(self):
        materializer = BuildArtifactMaterializer()
        output = """Architecture Note:
- Sample

File Tree:
```text
sample-app/
|-- package.json
`-- src/
    `-- index.js
```

`package.json`
```json
{"name":"sample-app"}
```

`src/index.js`
```javascript
console.log("hello");
```
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            result = materializer.materialize(
                output=output,
                target_dir=Path(temp_dir) / "generated" / "sample-app",
            )

            self.assertEqual(len(result.files_written), 2)
            self.assertTrue((result.root_dir / "package.json").exists())
            self.assertEqual((result.root_dir / "src" / "index.js").read_text(encoding="utf-8"), 'console.log("hello");')

    def test_default_output_dir_uses_file_tree_root(self):
        materializer = BuildArtifactMaterializer()
        output = """File Tree:
```text
login-system/
|-- package.json
```
"""

        target_dir = materializer.default_output_dir(
            goal="build login system",
            output=output,
            base_dir=Path("D:/workspace"),
        )

        self.assertEqual(target_dir, Path("D:/workspace") / "generated" / "login-system")

    def test_refuses_to_overwrite_without_force(self):
        materializer = BuildArtifactMaterializer()
        output = """`package.json`
```json
{"name":"sample-app"}
```
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir) / "generated" / "sample-app"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "package.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(BuildArtifactError):
                materializer.materialize(output=output, target_dir=target_dir, overwrite=False)


class FileToolTests(unittest.TestCase):
    def test_read_write_and_edit_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = FileTool(allowed_roots=[Path(temp_dir)], log_path=Path(temp_dir) / "tool.log")

            write_result = tool.execute(
                {
                    "tool": "file_tool",
                    "action": "write_file",
                    "arguments": {"path": "notes/example.txt", "content": "hello"},
                }
            )
            edit_result = tool.execute(
                {
                    "tool": "file_tool",
                    "action": "edit_file",
                    "arguments": {"path": "notes/example.txt", "old_text": "hello", "new_text": "hello world"},
                }
            )
            read_result = tool.execute(
                {
                    "tool": "file_tool",
                    "action": "read_file",
                    "arguments": {"path": "notes/example.txt"},
                }
            )

        self.assertTrue(write_result["ok"])
        self.assertTrue(edit_result["ok"])
        self.assertTrue(read_result["ok"])
        self.assertEqual(read_result["content"], "hello world")

    def test_denies_paths_outside_allowed_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = FileTool(allowed_roots=[Path(temp_dir)], log_path=Path(temp_dir) / "tool.log")
            result = tool.execute(
                {
                    "tool": "file_tool",
                    "action": "read_file",
                    "arguments": {"path": "../secret.txt"},
                }
            )
            log_text = (Path(temp_dir) / "tool.log").read_text(encoding="utf-8")

        self.assertFalse(result["ok"])
        self.assertIn("Access denied", result["summary"])
        self.assertIn("\"ok\": false", log_text)


class TerminalToolTests(unittest.TestCase):
    def test_runs_allowed_command_inside_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = TerminalTool(allowed_roots=[Path(temp_dir)], log_path=Path(temp_dir) / "terminal.log")
            result = tool.execute(
                {
                    "tool": "terminal_tool",
                    "action": "run_command",
                    "arguments": {
                        "command": ["python", "-c", "print('terminal-ok')"],
                        "cwd": temp_dir,
                        "timeout_seconds": 30,
                    },
                }
            )
            log_text = (Path(temp_dir) / "terminal.log").read_text(encoding="utf-8")

        self.assertTrue(result["ok"])
        self.assertIn("terminal-ok", result["stdout"])
        self.assertIn("\"ok\": true", log_text)

    def test_blocks_disallowed_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = TerminalTool(allowed_roots=[Path(temp_dir)], log_path=Path(temp_dir) / "terminal.log")
            result = tool.execute(
                {
                    "tool": "terminal_tool",
                    "action": "run_command",
                    "arguments": {
                        "command": ["powershell", "-c", "Get-ChildItem"],
                        "cwd": temp_dir,
                    },
                }
            )

        self.assertFalse(result["ok"])
        self.assertIn("not allowed", result["summary"])


class ScaffoldRunnerTests(unittest.TestCase):
    def test_prepare_uses_dev_script_and_default_port_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "package.json").write_text(
                json.dumps(
                    {
                        "name": "sample-app",
                        "scripts": {"dev": "node backend/server.js"},
                    }
                ),
                encoding="utf-8",
            )
            runner = ScaffoldRunner(root)
            runner._npm_command = lambda: "npm"
            runner._port_open = lambda port: False

            plan = runner.prepare()

        self.assertEqual(plan.launch_script, "dev")
        self.assertEqual(plan.port, 3010)
        self.assertTrue(plan.install_required)

    def test_prepare_rejects_busy_preferred_port(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "package.json").write_text(
                json.dumps(
                    {
                        "name": "sample-app",
                        "scripts": {"dev": "node backend/server.js"},
                    }
                ),
                encoding="utf-8",
            )
            runner = ScaffoldRunner(root)
            runner._npm_command = lambda: "npm"
            runner._port_open = lambda port: port == 3012

            with self.assertRaises(ScaffoldRunError):
                runner.prepare(preferred_port=3012)


class PolicyEngineTests(unittest.TestCase):
    def test_execution_modes_shift_weights_and_targets(self):
        task = TaskBlueprint(
            id="coding_1",
            task_type="solution",
            agent="coding",
            instruction="Produce code",
            retries=2,
            timeout_seconds=10,
            confidence_threshold=0.6,
        )
        blueprint = WorkflowBlueprint(
            goal="Ship code",
            primary_intent="coding",
            tasks=[task],
            metadata={
                "skill_pattern_available": True,
                "skill_pattern_reused": True,
                "skill_pattern_success_rate": 0.9,
                "skill_pattern_avg_retries": 0.5,
            },
        )

        stable = PolicyEngine(mode="stable").build_profile(task=task, blueprint=blueprint, max_attempts=3)
        explore = PolicyEngine(mode="explore").build_profile(task=task, blueprint=blueprint, max_attempts=3)

        self.assertGreater(stable.critic_weights["safety"], explore.critic_weights["safety"])
        self.assertGreater(explore.critic_weights["efficiency"], stable.critic_weights["efficiency"])
        self.assertGreater(stable.confidence_target, explore.confidence_target)

    def test_repeated_failures_force_strategy_change(self):
        task = TaskBlueprint(
            id="coding_1",
            task_type="solution",
            agent="coding",
            instruction="Produce code",
            retries=2,
            timeout_seconds=10,
            confidence_threshold=0.6,
        )
        blueprint = WorkflowBlueprint(goal="Ship code", primary_intent="coding", tasks=[task])
        engine = PolicyEngine(mode="stable")
        profile = engine.build_profile(task=task, blueprint=blueprint, max_attempts=3)

        decision = engine.decide(
            task=task,
            blueprint=blueprint,
            profile=profile,
            attempts=2,
            elapsed_seconds=1.0,
            observation={"ok": False, "summary": "still low confidence", "failure_type": "low_confidence", "confidence": 0.33},
            evaluation={"ok": False, "score": 0.33, "failure_type": "low_confidence", "critic_scores": {"correctness": 0.33, "efficiency": 0.9, "safety": 1.0}},
            history=[
                {"failure_type": "low_confidence", "strategy": "plan_modification", "evaluation_score": 0.31},
                {"failure_type": "low_confidence", "strategy": "plan_modification", "evaluation_score": 0.32},
            ],
            current_strategy="plan_modification",
        )

        self.assertTrue(decision.should_retry)
        self.assertEqual(decision.action, "retry_with_strategy_change")
        self.assertTrue(decision.force_strategy_change)

    def test_improving_scores_continue_current_strategy(self):
        task = TaskBlueprint(
            id="coding_1",
            task_type="solution",
            agent="coding",
            instruction="Produce code",
            retries=2,
            timeout_seconds=10,
            confidence_threshold=0.6,
        )
        blueprint = WorkflowBlueprint(goal="Ship code", primary_intent="coding", tasks=[task])
        engine = PolicyEngine(mode="explore")
        profile = engine.build_profile(task=task, blueprint=blueprint, max_attempts=3)

        decision = engine.decide(
            task=task,
            blueprint=blueprint,
            profile=profile,
            attempts=2,
            elapsed_seconds=1.0,
            observation={"ok": False, "summary": "better but still low", "failure_type": "low_confidence", "confidence": 0.39},
            evaluation={"ok": False, "score": 0.39, "failure_type": "low_confidence", "critic_scores": {"correctness": 0.39, "efficiency": 0.95, "safety": 1.0}},
            history=[
                {"failure_type": "low_confidence", "strategy": "plan_modification", "evaluation_score": 0.25},
            ],
            current_strategy="plan_modification",
        )

        self.assertTrue(decision.should_retry)
        self.assertEqual(decision.action, "retry_current_strategy")
        self.assertTrue(decision.continue_current_strategy)


class StrategyEngineTests(unittest.TestCase):
    def test_low_confidence_prefers_agent_switch_when_available(self):
        task = TaskBlueprint(
            id="coding_1",
            task_type="solution",
            agent="coding",
            instruction="Produce code",
            retry_strategy="switch_agent",
            fallback_agent="research",
            candidate_agents=["coding", "research"],
            required_capabilities=["reasoning"],
        )
        decision = StrategyEngine().choose(
            task=task,
            current_agent="coding",
            attempts=1,
            max_attempts=2,
            observation={"ok": False, "summary": "confidence too low", "failure_type": "low_confidence"},
            evaluation={"failure_type": "low_confidence"},
            alternative_agents=["research"],
        )

        self.assertTrue(decision.should_retry)
        self.assertEqual(decision.strategy, "switch_agent")
        self.assertEqual(decision.next_agent, "research")

    def test_timeout_prefers_simplification(self):
        task = TaskBlueprint(
            id="coding_1",
            task_type="solution",
            agent="coding",
            instruction="Produce code",
            candidate_agents=["coding"],
        )
        decision = StrategyEngine().choose(
            task=task,
            current_agent="coding",
            attempts=1,
            max_attempts=2,
            observation={"ok": False, "summary": "task timed out", "failure_type": "timeout"},
            alternative_agents=[],
        )

        self.assertEqual(decision.strategy, "simplify_task")
        self.assertIn("Reduce scope", decision.note)


class SharedMemoryTests(unittest.TestCase):
    def test_shared_memory_is_communication_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = SharedMemory(root_dir=Path(temp_dir))

        self.assertFalse(hasattr(memory, "get_task_strategy"))
        self.assertFalse(hasattr(memory, "set_task_strategy"))
        self.assertFalse(hasattr(memory, "clear_task_strategy"))


class RuntimeInsightsTests(unittest.TestCase):
    def test_explain_run_and_overview_share_runtime_data(self):
        parser = IntentParser()
        intent = parser.parse("Build a local Python service")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            skill_memory = SkillMemory(storage_path=temp_path / "skill_memory.json")
            decision_cache = DecisionCache(storage_path=temp_path / "decision_cache.json")
            planner = PlannerEngine(skill_memory=skill_memory)
            plan = planner.plan(intent)
            blueprint = BlueprintGenerator().generate(plan)

            wiring = WiringEngine(auto_register=False)
            wiring.register("coding", PromptCapturingAgent)
            wiring.register("research", EchoAgent)

            orchestrator = Orchestrator(
                wiring_engine=wiring,
                shared_memory=SharedMemory(root_dir=temp_path / "memory"),
                reflect_scorer=AlwaysHighConfidenceScorer(),
                skill_memory=skill_memory,
                decision_cache=decision_cache,
                trace_dir=temp_path / "traces",
            )
            result = asyncio.run(orchestrator.run_blueprint(blueprint))

            insights = RuntimeInsights(
                trace_dir=temp_path / "traces",
                skill_memory=skill_memory,
                decision_cache=decision_cache,
            )
            explanation = insights.explain_run(
                goal=intent.goal,
                intent=intent.to_dict(),
                plan=plan.to_dict(),
                blueprint=blueprint.to_dict(),
                result=result,
            )
            overview = insights.overview(limit=5)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(explanation["summary"]["status"], "completed")
        self.assertGreaterEqual(len(explanation["plan"]), 1)
        self.assertEqual(overview["metrics"]["total_runs"], 1)
        self.assertGreaterEqual(len(overview["patterns"]), 1)
        self.assertGreaterEqual(overview["decision_cache"]["total_entries"], 1)


class OrchestratorTests(unittest.TestCase):
    def test_project_mode_uses_existing_project_context_and_persists_learning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            project_root = temp_path / "project"
            project_root.mkdir()
            (project_root / "package.json").write_text(
                json.dumps(
                    {
                        "dependencies": {"react": "^18.0.0"},
                        "devDependencies": {"vite": "^5.0.0"},
                        "scripts": {"dev": "vite"},
                    }
                ),
                encoding="utf-8",
            )
            (project_root / "src").mkdir()
            (project_root / "src" / "App.jsx").write_text("export default function App() { return null; }\n", encoding="utf-8")

            environment_memory = EnvironmentMemory(storage_path=temp_path / "environment_memory.json")
            project_context = ProjectModeManager(environment_memory=environment_memory).prepare(
                project_dir=project_root,
                goal="Improve the login UI",
                execution_mode="stable",
            )

            blueprint = WorkflowBlueprint(
                goal="Improve the login UI",
                primary_intent="coding",
                tasks=[
                    TaskBlueprint(
                        id="coding_1",
                        task_type="solution",
                        agent="coding",
                        instruction="Improve the login UI using the existing stack",
                        candidate_agents=["coding"],
                        required_capabilities=["reasoning", "code_generation"],
                    )
                ],
                metadata={
                    "plan_signature": "project-plan",
                    "project_mode": True,
                    "project_root": str(project_root),
                    "project_signature": project_context["project_signature"],
                    "project_frameworks": ["react", "vite"],
                    "project_languages": ["javascript"],
                },
            )

            memory = SharedMemory(root_dir=temp_path / "memory")
            wiring = WiringEngine(auto_register=False)
            wiring.register("coding", PromptCapturingAgent)
            orchestrator = Orchestrator(
                wiring_engine=wiring,
                shared_memory=memory,
                skill_memory=SkillMemory(storage_path=temp_path / "skill_memory.json"),
                decision_cache=DecisionCache(storage_path=temp_path / "decision_cache.json"),
                environment_memory=environment_memory,
                project_context=project_context,
                trace_dir=temp_path / "traces",
            )
            result = asyncio.run(orchestrator.run_blueprint(blueprint))
            persisted = environment_memory.project_mode_context(project_root)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(memory.get("workspace.root_dir"), str(project_root.resolve()))
        self.assertTrue(memory.get("project.mode"))
        self.assertIn("Project context:", result["final_output"])
        self.assertGreaterEqual(len(persisted["successful_patterns"]), 1)

    def test_shared_memory_feeds_dependency_output_into_next_task(self):
        blueprint = WorkflowBlueprint(
            goal="Research then synthesize",
            primary_intent="coding",
            tasks=[
                TaskBlueprint(
                    id="research_1",
                    task_type="research_context",
                    agent="research",
                    instruction="Gather context",
                    candidate_agents=["research"],
                    required_capabilities=["reasoning"],
                ),
                TaskBlueprint(
                    id="coding_2",
                    task_type="solution",
                    agent="coding",
                    instruction="Use the shared context to answer",
                    depends_on=["research_1"],
                    candidate_agents=["coding"],
                    required_capabilities=["reasoning", "code_generation"],
                ),
            ],
        )

        wiring = WiringEngine(auto_register=False)
        wiring.register("research", EchoAgent)
        wiring.register("coding", PromptCapturingAgent)

        with tempfile.TemporaryDirectory() as temp_dir:
            memory = SharedMemory(root_dir=Path(temp_dir))
            skill_memory = SkillMemory(storage_path=Path(temp_dir) / "skill_memory.json")
            orchestrator = Orchestrator(wiring_engine=wiring, shared_memory=memory, skill_memory=skill_memory)
            result = asyncio.run(orchestrator.run_blueprint(blueprint))

        self.assertEqual(result["status"], "completed")
        self.assertIn("research-notes::", result["final_output"])
        self.assertIn("Shared memory context:", result["final_output"])

    def test_context_reducer_compresses_large_dependency_prompt(self):
        blueprint = WorkflowBlueprint(
            goal="Condense large context before coding",
            primary_intent="coding",
            tasks=[
                TaskBlueprint(
                    id="research_1",
                    task_type="research_context",
                    agent="research",
                    instruction="Generate a very large context block",
                    candidate_agents=["research"],
                    required_capabilities=["reasoning"],
                ),
                TaskBlueprint(
                    id="coding_2",
                    task_type="solution",
                    agent="coding",
                    instruction="Use the large context to answer",
                    depends_on=["research_1"],
                    candidate_agents=["coding"],
                    required_capabilities=["reasoning", "code_generation"],
                ),
            ],
        )

        reducer = StubContextReducer()
        wiring = WiringEngine(auto_register=False)
        wiring.register("research", LongOutputAgent)
        wiring.register("coding", PromptCapturingAgent)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            memory = SharedMemory(root_dir=temp_path / "memory")
            orchestrator = Orchestrator(
                wiring_engine=wiring,
                shared_memory=memory,
                skill_memory=SkillMemory(storage_path=temp_path / "skill_memory.json"),
                decision_cache=DecisionCache(storage_path=temp_path / "decision_cache.json"),
                context_reducer=reducer,
                trace_dir=temp_path / "traces",
            )
            result = asyncio.run(orchestrator.run_blueprint(blueprint))

        self.assertEqual(result["status"], "completed")
        self.assertGreaterEqual(len(reducer.calls), 1)
        self.assertIn("[REDUCED_PROMPT]", result["final_output"])
        self.assertEqual(memory.get("context_reduction.last")["backend"], "stub")
        self.assertTrue(
            any(decision["decision_type"] == "context_reduction" for decision in result["trace"]["decisions"])
        )

    def test_retries_transient_agent_failures(self):
        blueprint = WorkflowBlueprint(
            goal="Recover from a transient failure",
            primary_intent="coding",
            tasks=[
                TaskBlueprint(
                    id="coding_1",
                    task_type="solution",
                    agent="coding",
                    instruction="Do the thing",
                    retries=1,
                    candidate_agents=["coding"],
                    required_capabilities=["reasoning", "code_generation"],
                )
            ],
        )

        wiring = WiringEngine(auto_register=False)
        wiring.register("coding", FlakyAgent)

        with tempfile.TemporaryDirectory() as temp_dir:
            memory = SharedMemory(root_dir=Path(temp_dir))
            skill_memory = SkillMemory(storage_path=Path(temp_dir) / "skill_memory.json")
            orchestrator = Orchestrator(wiring_engine=wiring, shared_memory=memory, skill_memory=skill_memory)
            result = asyncio.run(orchestrator.run_blueprint(blueprint))

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["executions"][0]["attempts"], 2)
        self.assertIn("completed::", result["final_output"])

    def test_executes_agent_tool_calls_and_records_trace(self):
        blueprint = WorkflowBlueprint(
            goal="Create a file with the tool layer",
            primary_intent="coding",
            tasks=[
                TaskBlueprint(
                    id="coding_1",
                    task_type="solution",
                    agent="coding",
                    instruction="Create the file using the file tool",
                    candidate_agents=["coding"],
                    required_capabilities=["reasoning", "code_generation"],
                )
            ],
        )

        wiring = WiringEngine(auto_register=False)
        wiring.register("coding", ToolCallingAgent)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            memory = SharedMemory(root_dir=temp_path / "memory")
            orchestrator = Orchestrator(
                wiring_engine=wiring,
                shared_memory=memory,
                file_tool=FileTool(
                    allowed_roots=[temp_path],
                    log_path=temp_path / "file_tool.log",
                ),
                skill_memory=SkillMemory(storage_path=temp_path / "skill_memory.json"),
                decision_cache=DecisionCache(storage_path=temp_path / "decision_cache.json"),
                trace_dir=temp_path / "traces",
            )
            result = asyncio.run(orchestrator.run_blueprint(blueprint))

            written_file = temp_path / "artifacts" / "note.txt"
            file_exists = written_file.exists()
            file_content = written_file.read_text(encoding="utf-8") if file_exists else ""
            trace = result["trace"]
            log_text = (temp_path / "file_tool.log").read_text(encoding="utf-8")

        self.assertEqual(result["status"], "completed")
        self.assertTrue(file_exists)
        self.assertEqual(file_content, "hello from nexus")
        self.assertIn("tool-finished::", result["final_output"])
        self.assertTrue(any(decision["decision_type"] == "tool_execution" for decision in trace["decisions"]))
        self.assertTrue(any(event["kind"] == "tool_executed" for event in trace["events"]))
        self.assertIn("\"action\": \"write_file\"", log_text)

    def test_terminal_error_feedback_triggers_fix_loop(self):
        blueprint = WorkflowBlueprint(
            goal="Write and fix a Python file iteratively",
            primary_intent="coding",
            tasks=[
                TaskBlueprint(
                    id="coding_1",
                    task_type="solution",
                    agent="coding",
                    instruction="Write code, run it, and fix failures",
                    candidate_agents=["coding"],
                    required_capabilities=["reasoning", "code_generation"],
                    retries=0,
                )
            ],
        )

        wiring = WiringEngine(auto_register=False)
        wiring.register("coding", FixingLoopAgent)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            memory = SharedMemory(root_dir=temp_path / "memory")
            orchestrator = Orchestrator(
                wiring_engine=wiring,
                shared_memory=memory,
                skill_memory=SkillMemory(storage_path=temp_path / "skill_memory.json"),
                decision_cache=DecisionCache(storage_path=temp_path / "decision_cache.json"),
                trace_dir=temp_path / "traces",
                workspace_base_dir=temp_path / "workspaces",
            )
            result = asyncio.run(orchestrator.run_blueprint(blueprint))
            workspace_root = Path(memory.get("workspace.root_dir"))
            main_file = workspace_root / "main.py"
            main_file_exists = main_file.exists()
            main_file_content = main_file.read_text(encoding="utf-8") if main_file_exists else ""
            trace = result["trace"]

        self.assertEqual(result["status"], "completed")
        self.assertTrue(main_file_exists)
        self.assertIn('print("fixed")\n', main_file_content)
        self.assertIn("NEXUS note:", main_file_content)
        self.assertTrue(result["documentation"]["readme_path"].endswith("README.md"))
        self.assertEqual(memory.get("workspace.last_terminal_error"), None)
        self.assertIn("fix-loop-complete", result["final_output"])
        self.assertTrue(any(decision["decision_type"] == "tool_execution" for decision in trace["decisions"]))

    def test_decision_cache_skips_expensive_correctness_on_repeat_run(self):
        blueprint = WorkflowBlueprint(
            goal="Generate a trustworthy answer",
            primary_intent="coding",
            tasks=[
                TaskBlueprint(
                    id="coding_1",
                    task_type="solution",
                    agent="coding",
                    instruction="Produce a trustworthy answer with confidence",
                    retries=0,
                    confidence_threshold=0.6,
                    candidate_agents=["coding"],
                    required_capabilities=["reasoning", "code_generation"],
                )
            ],
            metadata={"plan_signature": "cached-blueprint"},
        )

        wiring = WiringEngine(auto_register=False)
        wiring.register("coding", PromptCapturingAgent)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            scorer = AlwaysHighConfidenceScorer()
            decision_cache = DecisionCache(storage_path=temp_path / "decision_cache.json")

            orchestrator_first = Orchestrator(
                wiring_engine=wiring,
                shared_memory=SharedMemory(root_dir=temp_path / "memory1"),
                reflect_scorer=scorer,
                skill_memory=SkillMemory(storage_path=temp_path / "skill_memory.json"),
                decision_cache=decision_cache,
            )
            first = asyncio.run(orchestrator_first.run_blueprint(blueprint))
            first_calls = scorer.calls

            orchestrator_second = Orchestrator(
                wiring_engine=wiring,
                shared_memory=SharedMemory(root_dir=temp_path / "memory2"),
                reflect_scorer=scorer,
                skill_memory=SkillMemory(storage_path=temp_path / "skill_memory.json"),
                decision_cache=decision_cache,
            )
            second = asyncio.run(orchestrator_second.run_blueprint(blueprint))

        self.assertEqual(first["status"], "completed")
        self.assertEqual(second["status"], "completed")
        self.assertEqual(scorer.calls, first_calls)
        self.assertTrue(second["executions"][0]["observation"]["evaluation"]["cached"])

    def test_parallelizes_independent_context_tasks(self):
        TimedResearchAgent.started_at = []
        TimedMemoryReadAgent.started_at = []
        blueprint = WorkflowBlueprint(
            goal="Gather context from memory and research",
            primary_intent="coding",
            tasks=[
                TaskBlueprint(
                    id="memory_1",
                    task_type="memory_recall",
                    agent="memory",
                    instruction="Recall context",
                    candidate_agents=["memory"],
                    required_capabilities=["memory_read"],
                ),
                TaskBlueprint(
                    id="research_1",
                    task_type="research_context",
                    agent="research",
                    instruction="Gather context",
                    candidate_agents=["research"],
                    required_capabilities=["reasoning"],
                ),
                TaskBlueprint(
                    id="coding_2",
                    task_type="solution",
                    agent="coding",
                    instruction="Use both contexts",
                    depends_on=["memory_1", "research_1"],
                    candidate_agents=["coding"],
                    required_capabilities=["reasoning", "code_generation"],
                ),
            ],
        )

        wiring = WiringEngine(auto_register=False)
        wiring.register("memory", TimedMemoryReadAgent)
        wiring.register("research", TimedResearchAgent)
        wiring.register("coding", PromptCapturingAgent)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            orchestrator = Orchestrator(
                wiring_engine=wiring,
                shared_memory=SharedMemory(root_dir=temp_path / "memory"),
                skill_memory=SkillMemory(storage_path=temp_path / "skill_memory.json"),
                decision_log_path=temp_path / "decision_log.jsonl",
            )
            result = asyncio.run(orchestrator.run_blueprint(blueprint))
            decision_rows = [
                json.loads(line)
                for line in (temp_path / "decision_log.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(result["status"], "completed")
        self.assertTrue(
            any(
                row["decision_type"] == "parallel_batch" and row["metadata"].get("parallel")
                for row in decision_rows
            )
        )
        self.assertEqual(len(TimedResearchAgent.started_at), 1)
        self.assertEqual(len(TimedMemoryReadAgent.started_at), 1)
        self.assertLess(abs(TimedResearchAgent.started_at[0] - TimedMemoryReadAgent.started_at[0]), 0.03)

    def test_self_correction_retries_low_confidence_output(self):
        blueprint = WorkflowBlueprint(
            goal="Generate a trustworthy answer",
            primary_intent="coding",
            tasks=[
                TaskBlueprint(
                    id="coding_1",
                    task_type="solution",
                    agent="coding",
                    instruction="Produce a trustworthy answer",
                    retries=1,
                    retry_strategy="tighten_prompt",
                    confidence_threshold=0.6,
                    candidate_agents=["coding"],
                    required_capabilities=["reasoning", "code_generation"],
                )
            ],
        )

        wiring = WiringEngine(auto_register=False)
        wiring.register("coding", PromptCapturingAgent)

        with tempfile.TemporaryDirectory() as temp_dir:
            memory = SharedMemory(root_dir=Path(temp_dir))
            skill_memory = SkillMemory(storage_path=Path(temp_dir) / "skill_memory.json")
            orchestrator = Orchestrator(
                wiring_engine=wiring,
                shared_memory=memory,
                reflect_scorer=LowThenHighConfidenceScorer(),
                skill_memory=skill_memory,
            )
            result = asyncio.run(orchestrator.run_blueprint(blueprint))

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["executions"][0]["attempts"], 2)
        self.assertIn("Retry note:", result["final_output"])

    def test_timeout_uses_strategy_engine_simplification(self):
        blueprint = WorkflowBlueprint(
            goal="Handle a timeout",
            primary_intent="coding",
            tasks=[
                TaskBlueprint(
                    id="coding_1",
                    task_type="solution",
                    agent="coding",
                    instruction="Produce a result quickly",
                    retries=1,
                    timeout_seconds=0.01,
                    candidate_agents=["coding"],
                    required_capabilities=["reasoning", "code_generation"],
                )
            ],
        )

        wiring = WiringEngine(auto_register=False)
        wiring.register("coding", SlowAgent)

        with tempfile.TemporaryDirectory() as temp_dir:
            memory = SharedMemory(root_dir=Path(temp_dir))
            skill_memory = SkillMemory(storage_path=Path(temp_dir) / "skill_memory.json")
            orchestrator = Orchestrator(wiring_engine=wiring, shared_memory=memory, skill_memory=skill_memory)
            result = asyncio.run(orchestrator.run_blueprint(blueprint))

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["executions"][0]["reflection"]["strategy"], "simplify_task")

    def test_stable_mode_compacts_trace_payloads(self):
        blueprint = WorkflowBlueprint(
            goal="Trace compactly",
            primary_intent="coding",
            tasks=[
                TaskBlueprint(
                    id="coding_1",
                    task_type="solution",
                    agent="coding",
                    instruction="Produce a traced answer",
                    retries=0,
                    confidence_threshold=0.6,
                    candidate_agents=["coding"],
                    required_capabilities=["reasoning", "code_generation"],
                )
            ],
        )

        wiring = WiringEngine(auto_register=False)
        wiring.register("coding", PromptCapturingAgent)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            orchestrator = Orchestrator(
                wiring_engine=wiring,
                shared_memory=SharedMemory(root_dir=temp_path / "memory"),
                reflect_scorer=AlwaysHighConfidenceScorer(),
                skill_memory=SkillMemory(storage_path=temp_path / "skill_memory.json"),
                trace_dir=temp_path / "traces",
                execution_mode="stable",
            )
            result = asyncio.run(orchestrator.run_blueprint(blueprint))
            trace_payload = json.loads(Path(result["trace"]["trace_path"]).read_text(encoding="utf-8"))

        started_event = next(event for event in trace_payload["events"] if event["kind"] == "task_started")
        finished_event = next(event for event in trace_payload["events"] if event["kind"] == "task_finished")
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(started_event["input_preview"])
        self.assertIsNone(finished_event["output_preview"])
        self.assertEqual(finished_event["critic_scores"], {})

    def test_trace_and_decision_log_are_written(self):
        blueprint = WorkflowBlueprint(
            goal="Trace a trustworthy answer",
            primary_intent="coding",
            tasks=[
                TaskBlueprint(
                    id="coding_1",
                    task_type="solution",
                    agent="coding",
                    instruction="Produce a trustworthy traced answer",
                    retries=1,
                    retry_strategy="tighten_prompt",
                    confidence_threshold=0.6,
                    candidate_agents=["coding"],
                    required_capabilities=["reasoning", "code_generation"],
                )
            ],
        )

        wiring = WiringEngine(auto_register=False)
        wiring.register("coding", PromptCapturingAgent)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            memory = SharedMemory(root_dir=temp_path / "memory")
            skill_memory = SkillMemory(storage_path=temp_path / "skill_memory.json")
            trace_dir = temp_path / "traces"
            decision_log = temp_path / "decision_log.jsonl"
            orchestrator = Orchestrator(
                wiring_engine=wiring,
                shared_memory=memory,
                reflect_scorer=LowThenHighConfidenceScorer(),
                skill_memory=skill_memory,
                trace_dir=trace_dir,
                decision_log_path=decision_log,
                execution_mode="explore",
            )
            result = asyncio.run(orchestrator.run_blueprint(blueprint))

            trace_path = Path(result["trace"]["trace_path"])
            self.assertTrue(trace_path.exists())
            self.assertTrue(decision_log.exists())

            trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
            decision_rows = [
                json.loads(line)
                for line in decision_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(result["status"], "completed")
        self.assertTrue(any(event["kind"] == "task_started" for event in trace_payload["events"]))
        self.assertTrue(any(row["decision_type"] == "agent_selection" for row in decision_rows))
        self.assertTrue(any(row["decision_type"] == "evaluation" for row in decision_rows))
        self.assertTrue(any(row["decision_type"] == "policy" for row in decision_rows))
        self.assertTrue(
            any(
                row["decision_type"] == "policy" and row["metadata"].get("mode") == "explore"
                for row in decision_rows
            )
        )
        self.assertTrue(any(row["decision_type"] == "retry" for row in decision_rows))


if __name__ == "__main__":
    unittest.main()
