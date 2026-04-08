"""Tests for Tier 3 and completion-path upgrades."""

from __future__ import annotations

import asyncio
import tempfile
import textwrap
import unittest
from pathlib import Path

from nexus.agents.base_agent import BaseAgent
from nexus.blueprint_generator import TaskBlueprint, WorkflowBlueprint
from nexus.memory.skill_memory import SkillMemory
from nexus.orchestrator import Orchestrator
from nexus.runtime.decision_cache import DecisionCache
from nexus.runtime.doc_generator import DocGenerator
from nexus.runtime.event_bus import runtime_event_bus
from nexus.runtime.project_executor import ProjectExecutor
from nexus.shared_memory import SharedMemory
from nexus.wiring_engine import WiringEngine
from nexus.plugins.loader import PluginLoader


class StubFixAgent:
    async def _call_local(self, prompt: str) -> str:
        return '`main.py`\n```python\nprint("fixed")\n```\n'


class ToolEventAgent(BaseAgent):
    name = "coding"
    capabilities = ("reasoning", "code_generation")

    async def run(self, task: str):
        return self.tool_call(
            tool="file_tool",
            action="write_file",
            path="artifact.txt",
            content="runtime event bus",
        )

    async def continue_after_tool(self, task: str, tool_result: dict, memory=None, thought: dict | None = None):
        if tool_result.get("ok"):
            return "tool completed"
        return f"tool failed: {tool_result.get('summary')}"


class ProjectExecutorFixTests(unittest.TestCase):
    def test_fix_file_updates_only_targeted_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            executor = ProjectExecutor(workspace_base=workspace)
            root = executor.materialize(
                {
                    "main.py": 'print("broken"\n',
                    "helper.py": "VALUE = 1\n",
                },
                project_dir=workspace / "project",
            )
            error = f'Traceback (most recent call last):\n  File "{root / "main.py"}", line 1, in <module>\nSyntaxError: "(" was never closed'

            result = asyncio.run(
                executor.fix_file(
                    root,
                    error=error,
                    stdout="",
                    coding_agent=StubFixAgent(),
                    task="Repair the generated project",
                )
            )

            self.assertTrue(result["ok"])
            self.assertEqual((root / "main.py").read_text(encoding="utf-8").strip(), 'print("fixed")')
            self.assertEqual((root / "helper.py").read_text(encoding="utf-8"), "VALUE = 1\n")


class DocGeneratorTests(unittest.TestCase):
    def test_generate_writes_docs_and_annotations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source = workspace / "src" / "app.py"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("print('hello')\n", encoding="utf-8")

            blueprint = WorkflowBlueprint(
                goal="Create a tiny Python app",
                primary_intent="coding",
                tasks=[
                    TaskBlueprint(
                        id="coding_1",
                        task_type="solution",
                        agent="coding",
                        instruction="Create the app",
                    )
                ],
                metadata={"constraints": ["local_only"]},
            )

            result = DocGenerator().generate(
                workspace_root=workspace,
                blueprint=blueprint,
                executions=[{"task_id": "coding_1", "status": "completed", "attempts": 1}],
                trace_snapshot={"status": "completed", "decisions": [], "trace_path": "trace.json"},
                touched_files=["src/app.py"],
                managed_workspace=True,
            )

            self.assertTrue((workspace / "README.md").exists())
            self.assertTrue((workspace / "ARCHITECTURE.md").exists())
            annotated_paths = {str(Path(path).resolve()) for path in (result.annotated_files or [])}
            self.assertIn(str(source.resolve()), annotated_paths)
            self.assertTrue(source.read_text(encoding="utf-8").startswith("# NEXUS note:"))


class PluginLoaderTests(unittest.TestCase):
    def test_wiring_engine_auto_registers_local_plugins(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin_root = Path(temp_dir)

            self._write_plugin(
                plugin_root,
                "demo_agent_plugin",
                """
                from nexus.agents.base_agent import BaseAgent

                class DemoAgent(BaseAgent):
                    name = "plugin_agent"
                    capabilities = ("reasoning", "testing")

                    async def run(self, task: str) -> str:
                        return "plugin-agent-ok"
                """,
                """
                name: plugin_agent
                type: agent
                module: demo_agent_plugin
                entry: DemoAgent
                capabilities:
                  - reasoning
                  - testing
                """,
            )
            self._write_plugin(
                plugin_root,
                "demo_tool_plugin",
                """
                class DemoTool:
                    def execute(self, request):
                        return {
                            "ok": True,
                            "tool": "plugin_tool",
                            "summary": "plugin tool executed",
                        }
                """,
                """
                name: plugin_tool
                type: tool
                module: demo_tool_plugin
                entry: DemoTool
                capabilities:
                  - execution
                """,
            )
            self._write_plugin(
                plugin_root,
                "demo_critic_plugin",
                """
                from nexus.critics.base import BaseCritic

                class DemoCritic(BaseCritic):
                    name = "plugin_critic"
                    weight = 0.2

                    async def evaluate(self, *, task, output, observation, attempt, max_attempts):
                        return self.assessment(score=0.9, reason="plugin critic ok")
                """,
                """
                name: plugin_critic
                type: critic
                module: demo_critic_plugin
                entry: DemoCritic
                capabilities:
                  - evaluation
                """,
            )

            engine = WiringEngine(plugin_loader=PluginLoader(plugin_root=plugin_root))

            self.assertIn("plugin_agent", engine.available_agents())
            self.assertIn("plugin_tool", engine.available_tools())
            self.assertIn("plugin_critic", engine.available_critics())
            self.assertEqual(engine.resolve("plugin_agent").name, "plugin_agent")
            self.assertTrue(engine.resolve_tool("plugin_tool").execute({})["ok"])
            self.assertEqual(len(engine.resolve_critics()), 1)
            self.assertEqual(len(engine.discovered_plugins()), 3)

    def _write_plugin(self, root: Path, package_name: str, module_source: str, manifest: str) -> None:
        package_dir = root / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "__init__.py").write_text(textwrap.dedent(module_source).strip() + "\n", encoding="utf-8")
        (package_dir / "plugin.yaml").write_text(textwrap.dedent(manifest).strip() + "\n", encoding="utf-8")


class RuntimeEventBusTests(unittest.TestCase):
    def test_orchestrator_emits_live_runtime_events(self):
        async def _run() -> list[dict]:
            events: list[dict] = []

            async def collector(event: dict):
                events.append(event)

            runtime_event_bus.subscribe(collector)
            try:
                blueprint = WorkflowBlueprint(
                    goal="Write one artifact",
                    primary_intent="coding",
                    tasks=[
                        TaskBlueprint(
                            id="coding_1",
                            task_type="solution",
                            agent="coding",
                            instruction="Write one artifact",
                            retries=0,
                            candidate_agents=["coding"],
                            required_capabilities=["reasoning", "code_generation"],
                        )
                    ],
                )
                wiring = WiringEngine(auto_register=False)
                wiring.register("coding", ToolEventAgent)

                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    orchestrator = Orchestrator(
                        wiring_engine=wiring,
                        shared_memory=SharedMemory(root_dir=root / "memory"),
                        skill_memory=SkillMemory(storage_path=root / "skill_memory.json"),
                        decision_cache=DecisionCache(storage_path=root / "decision_cache.json"),
                        workspace_base_dir=root / "workspaces",
                        trace_dir=root / "traces",
                    )
                    await orchestrator.run_blueprint(blueprint)
                    await asyncio.sleep(0.05)
            finally:
                runtime_event_bus.unsubscribe(collector)
            return events

        events = asyncio.run(_run())
        event_types = {event["type"] for event in events}

        self.assertIn("agent_started", event_types)
        self.assertIn("tool_executed", event_types)
        self.assertIn("agent_output", event_types)
        self.assertIn("workflow_complete", event_types)


if __name__ == "__main__":
    unittest.main()
