import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from nexus.api import ChatRequest, _chat_via_workspace_execution
from nexus.blueprint_generator import TaskBlueprint, WorkflowBlueprint
from nexus.runtime.doc_generator import DocGenerator
from nexus.runtime.workspace import WorkspaceDirectory


class WorkspaceDocGeneratorTests(unittest.TestCase):
    def test_unmanaged_repo_docs_stay_inside_hidden_nexus_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text("# Real project docs\n", encoding="utf-8")

            blueprint = WorkflowBlueprint(
                goal="Inspect the repo safely",
                primary_intent="research",
                tasks=[
                    TaskBlueprint(
                        id="research_1",
                        task_type="research_context",
                        agent="research",
                        instruction="Inspect the repo safely",
                        candidate_agents=["research"],
                        required_capabilities=["reasoning"],
                    )
                ],
            )

            result = DocGenerator().generate(
                workspace_root=root,
                blueprint=blueprint,
                executions=[],
                trace_snapshot={"status": "completed", "decisions": []},
                managed_workspace=False,
            )

            self.assertEqual(readme.read_text(encoding="utf-8"), "# Real project docs\n")
            self.assertTrue(result.readme_path.endswith(".nexus\\docs\\README.nexus.md"))
            self.assertTrue(result.architecture_path.endswith(".nexus\\docs\\ARCHITECTURE.nexus.md"))
            self.assertTrue(Path(result.readme_path).exists())
            self.assertTrue(Path(result.architecture_path).exists())
            self.assertEqual(result.annotated_files, [])


class WorkspaceChatFormattingTests(unittest.TestCase):
    def test_workspace_chat_response_includes_execution_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request = ChatRequest(
                message="Add an auth page",
                session_id="workspace-test",
                workspace_root=str(root),
                workspace_mode=True,
                execution_mode="explore",
            )

            payload = {
                "workflow_id": "wf-1234",
                "status": "completed",
                "final_output": "Implemented the auth page.",
                "execution_mode": "explore",
                "final_confidence": 0.91,
                "documentation": {
                    "readme_path": str(root / ".nexus" / "docs" / "README.nexus.md"),
                    "architecture_path": str(root / ".nexus" / "docs" / "ARCHITECTURE.nexus.md"),
                },
                "touched_files": ["src/AuthPage.jsx", "src/routes.js"],
                "primary_intent": "coding",
                "blueprint": {},
            }

            with patch("nexus.api._execute_workspace_goal", new=AsyncMock(return_value=payload)):
                response = asyncio.run(_chat_via_workspace_execution(request))

            self.assertEqual(response["route"], "workspace")
            self.assertEqual(response["initial_route"], "workspace")
            self.assertEqual(response["agent"], "coding")
            self.assertEqual(response["workspace_root"], str(root.resolve()))
            self.assertEqual(response["execution"]["workflow_id"], "wf-1234")
            self.assertEqual(response["execution"]["execution_mode"], "explore")
            self.assertEqual(response["execution"]["touched_files"], ["src/AuthPage.jsx", "src/routes.js"])


class WorkspaceSnapshotTests(unittest.TestCase):
    def test_snapshot_skips_cache_and_binary_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "__pycache__").mkdir()
            (root / "src" / "math_utils.py").write_text(
                "def bump(value, delta=1):\n    return value * 2 + delta\n",
                encoding="utf-8",
            )
            (root / "__pycache__" / "math_utils.cpython-311.pyc").write_bytes(b"\x00pyc")
            (root / "diagram.png").write_bytes(b"\x89PNG\r\n\x1a\n")

            snapshot = WorkspaceDirectory.for_project(
                project_dir=root,
                workflow_id="wf-1234",
                goal="Fix the failing unit test",
            ).snapshot()

            self.assertEqual([entry["path"] for entry in snapshot["files"]], ["src/math_utils.py"])
            self.assertIn("return value * 2 + delta", snapshot["files"][0]["preview"])


if __name__ == "__main__":
    unittest.main()
