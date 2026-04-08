"""Tests for streaming executor, project executor, and git tool (Tier 1 upgrades)."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path

from nexus.runtime.executor import CodeExecutor, StreamEvent, AsyncStreamResult
from nexus.runtime.project_executor import ProjectExecutor, ProjectResult
from nexus.runtime.git_tool import GitTool, GitStatus


# ==================================================================
# Streaming Executor Tests
# ==================================================================

class StreamEventTests(unittest.TestCase):
    def test_stream_event_has_required_fields(self):
        event = StreamEvent(kind="output", data="hello")
        self.assertEqual(event.kind, "output")
        self.assertEqual(event.data, "hello")
        self.assertIsNone(event.exit_code)

    def test_exit_event_has_exit_code(self):
        event = StreamEvent(kind="exit", data="", exit_code=0)
        self.assertEqual(event.exit_code, 0)


class StreamingExecutorTests(unittest.TestCase):
    def test_streaming_valid_python(self):
        executor = CodeExecutor()
        code = "for i in range(3):\n    print(f'line {i}')"

        async def _test():
            stream = await executor.run_streaming(code, "python", timeout=15, idle_timeout=10)
            events = []
            async for event in stream:
                events.append(event)
            return events, stream.final

        events, final = asyncio.run(_test())

        output_events = [e for e in events if e.kind == "output"]
        exit_events = [e for e in events if e.kind == "exit"]
        self.assertEqual(len(exit_events), 1)
        self.assertEqual(exit_events[0].exit_code, 0)
        self.assertTrue(len(output_events) >= 3)

    def test_streaming_captures_errors(self):
        executor = CodeExecutor()
        code = "raise ValueError('boom')"

        async def _test():
            stream = await executor.run_streaming(code, "python", timeout=10, idle_timeout=10)
            events = []
            async for event in stream:
                events.append(event)
            return events, stream.final

        events, final = asyncio.run(_test())

        error_events = [e for e in events if e.kind == "error"]
        self.assertTrue(len(error_events) > 0)
        error_text = " ".join(e.data for e in error_events)
        self.assertIn("ValueError", error_text)

    def test_streaming_unsupported_language(self):
        executor = CodeExecutor()

        async def _test():
            stream = await executor.run_streaming("code", "cobol")
            events = []
            async for event in stream:
                events.append(event)
            return events

        events = asyncio.run(_test())
        self.assertTrue(any(e.kind == "error" for e in events))
        self.assertTrue(any(e.kind == "exit" for e in events))

    def test_async_stream_result_collects_events(self):
        async def _gen():
            yield StreamEvent(kind="output", data="hello")
            yield StreamEvent(kind="exit", data="", exit_code=0)

        async def _test():
            result = AsyncStreamResult(_gen())
            events = []
            async for event in result:
                events.append(event)
            return events

        events = asyncio.run(_test())
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].data, "hello")


# ==================================================================
# Project Executor Tests
# ==================================================================

class ProjectExecutorMaterializeTests(unittest.TestCase):
    def test_writes_file_tree_to_disk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = ProjectExecutor(workspace_base=Path(temp_dir))
            file_tree = {
                "main.py": "print('hello')",
                "lib/utils.py": "def add(a, b): return a + b",
                "README.md": "# Test Project",
            }
            root = executor.materialize(file_tree)

            self.assertTrue((root / "main.py").exists())
            self.assertTrue((root / "lib/utils.py").exists())
            self.assertTrue((root / "README.md").exists())
            self.assertEqual((root / "main.py").read_text(), "print('hello')")

    def test_uses_explicit_project_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "my_project"
            executor = ProjectExecutor(workspace_base=Path(temp_dir))
            root = executor.materialize({"app.py": "pass"}, project_dir)

            self.assertEqual(root, project_dir)
            self.assertTrue((root / "app.py").exists())


class ProjectExecutorDetectionTests(unittest.TestCase):
    def test_detects_npm_from_package_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = ProjectExecutor(workspace_base=Path(temp_dir))
            root = Path(temp_dir) / "project"
            root.mkdir()
            (root / "package.json").write_text('{"name": "test"}')

            pm = executor.detect_package_manager(root)
            self.assertEqual(pm, "npm")

    def test_detects_pip_from_requirements(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = ProjectExecutor(workspace_base=Path(temp_dir))
            root = Path(temp_dir) / "project"
            root.mkdir()
            (root / "requirements.txt").write_text("requests\n")

            pm = executor.detect_package_manager(root)
            self.assertEqual(pm, "pip")

    def test_returns_none_for_no_deps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = ProjectExecutor(workspace_base=Path(temp_dir))
            root = Path(temp_dir) / "project"
            root.mkdir()

            pm = executor.detect_package_manager(root)
            self.assertIsNone(pm)


class ProjectExecutorRunTests(unittest.TestCase):
    def test_builds_and_runs_python_project(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = ProjectExecutor(workspace_base=Path(temp_dir))
            file_tree = {
                "main.py": "print('project works')",
            }
            result = executor.build_and_run(file_tree)

            self.assertTrue(result.success, f"Expected success but got: {result.summary}")
            self.assertIn("project works", result.run_result.stdout)
            self.assertEqual(result.files_written, ["main.py"])

    def test_detects_python_entry_point(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = ProjectExecutor(workspace_base=Path(temp_dir))
            root = executor.materialize({"app.py": "print('app')"})

            entry = executor._detect_entry_point(root)
            self.assertEqual(entry, "app.py")

    def test_handles_no_entry_point(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = ProjectExecutor(workspace_base=Path(temp_dir))
            root = executor.materialize({"data.txt": "just data"})

            result = executor.run_project(root)
            self.assertFalse(result.success)
            self.assertIn("No entry point", result.stderr)


class ProjectExecutorErrorDetectionTests(unittest.TestCase):
    def test_identify_python_error_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = ProjectExecutor(workspace_base=Path(temp_dir))
            stderr = f'''Traceback (most recent call last):
  File "{temp_dir}/project/main.py", line 3, in <module>
    raise ValueError("broken")
ValueError: broken'''
            result = executor.identify_failing_file(Path(temp_dir) / "project", stderr)
            self.assertEqual(result, "main.py")

    def test_identify_node_error_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = ProjectExecutor(workspace_base=Path(temp_dir))
            stderr = f"at Object.<anonymous> ({temp_dir}/project/index.js:5:3)"
            result = executor.identify_failing_file(Path(temp_dir) / "project", stderr)
            self.assertEqual(result, "index.js")


class ProjectExecutorToolInterfaceTests(unittest.TestCase):
    def test_tool_dispatch_build_and_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = ProjectExecutor(workspace_base=Path(temp_dir))
            result = executor.execute({
                "arguments": {
                    "action": "build_and_run",
                    "file_tree": {"main.py": "print('tool dispatch works')"},
                }
            })
            self.assertTrue(result["ok"])
            self.assertIn("tool dispatch works", result["stdout"])

    def test_tool_dispatch_empty_tree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = ProjectExecutor(workspace_base=Path(temp_dir))
            result = executor.execute({
                "arguments": {"action": "build_and_run", "file_tree": {}}
            })
            self.assertFalse(result["ok"])


# ==================================================================
# Git Tool Tests
# ==================================================================

class GitToolTests(unittest.TestCase):
    def test_init_creates_repo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            project.mkdir()
            (project / "README.md").write_text("# Hello")

            git = GitTool()
            created = git.init(project)

            self.assertTrue(created)
            self.assertTrue((project / ".git").exists())

    def test_init_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            project.mkdir()
            (project / "README.md").write_text("# Hello")

            git = GitTool()
            git.init(project)
            created_again = git.init(project)
            self.assertFalse(created_again)

    def test_status_reports_clean_after_init(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            project.mkdir()
            (project / "file.txt").write_text("initial")

            git = GitTool()
            git.init(project)
            st = git.status(project)

            self.assertTrue(st.initialized)
            self.assertTrue(st.clean)
            self.assertEqual(st.commit_count, 1)

    def test_status_detects_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            project.mkdir()
            (project / "file.txt").write_text("initial")

            git = GitTool()
            git.init(project)
            (project / "file.txt").write_text("modified")

            st = git.status(project)
            self.assertFalse(st.clean)
            self.assertTrue(len(st.changed_files) > 0)

    def test_checkpoint_and_rollback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            project.mkdir()
            (project / "file.txt").write_text("v1")

            git = GitTool()
            git.init(project)

            # Checkpoint v1
            sha1 = git.checkpoint(project, "version 1")
            self.assertTrue(len(sha1) > 6)

            # Modify to v2
            (project / "file.txt").write_text("v2")
            sha2 = git.checkpoint(project, "version 2")
            self.assertNotEqual(sha1, sha2)
            self.assertEqual((project / "file.txt").read_text(), "v2")

            # Rollback to v1
            git.rollback(project, steps=1)
            self.assertEqual((project / "file.txt").read_text(), "v1")

    def test_diff_shows_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            project.mkdir()
            (project / "file.txt").write_text("original")

            git = GitTool()
            git.init(project)
            (project / "file.txt").write_text("changed")

            diff_text = git.diff(project)
            self.assertIn("changed", diff_text)

    def test_log_returns_commits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            project.mkdir()
            (project / "file.txt").write_text("start")

            git = GitTool()
            git.init(project)

            for i in range(3):
                (project / "file.txt").write_text(f"v{i}")
                git.checkpoint(project, f"commit {i}")

            log = git.log(project, limit=5)
            self.assertEqual(len(log), 4)  # 1 init + 3 checkpoints
            self.assertTrue(all("sha" in entry for entry in log))

    def test_status_on_non_repo_returns_uninitialized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "not_a_repo"
            project.mkdir()

            git = GitTool()
            st = git.status(project)

            self.assertFalse(st.initialized)

    def test_allowed_roots_security(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            allowed = Path(temp_dir) / "allowed"
            forbidden = Path(temp_dir) / "forbidden"
            allowed.mkdir()
            forbidden.mkdir()

            git = GitTool(allowed_roots=[allowed])
            git.init(allowed)  # Should work

            with self.assertRaises(PermissionError):
                git.init(forbidden)


class GitToolDispatchTests(unittest.TestCase):
    def test_dispatch_init(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            project.mkdir()
            (project / "file.txt").write_text("x")

            git = GitTool()
            result = git.execute({
                "arguments": {"action": "init", "project_dir": str(project)}
            })
            self.assertTrue(result["ok"])

    def test_dispatch_checkpoint_and_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            project.mkdir()
            (project / "file.txt").write_text("x")

            git = GitTool()
            git.init(project)

            (project / "file.txt").write_text("y")
            result = git.execute({
                "arguments": {"action": "checkpoint", "project_dir": str(project), "message": "test"}
            })
            self.assertTrue(result["ok"])
            self.assertIn("sha", result)

            status_result = git.execute({
                "arguments": {"action": "status", "project_dir": str(project)}
            })
            self.assertTrue(status_result["ok"])
            self.assertTrue(status_result["clean"])


if __name__ == "__main__":
    unittest.main()
