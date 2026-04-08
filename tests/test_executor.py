"""Tests for nexus.runtime.executor — the autonomous code execution loop."""

import asyncio
import tempfile
import unittest
from pathlib import Path

from nexus.agents.base_agent import BaseAgent
from nexus.runtime.executor import CodeExecutor, ExecutionResult


class FixingAgent(BaseAgent):
    """Mock agent that returns corrected code on demand."""

    name = "coding"
    capabilities = ("reasoning", "code_generation", "debugging")

    def __init__(self, fixed_code: str):
        super().__init__()
        self._fixed_code = fixed_code
        self.call_count = 0

    async def run(self, task: str) -> str:
        return self._fixed_code

    async def _call_local(self, prompt: str, system: str = None) -> str:
        self.call_count += 1
        return f"```python\n{self._fixed_code}\n```"


class NeverFixAgent(BaseAgent):
    """Mock agent that always returns the same broken code."""

    name = "coding"
    capabilities = ("reasoning", "code_generation")

    def __init__(self):
        super().__init__()
        self.call_count = 0

    async def run(self, task: str) -> str:
        return "broken"

    async def _call_local(self, prompt: str, system: str = None) -> str:
        self.call_count += 1
        return '```python\nprint("still broken"\n```'


class FailingCallAgent(BaseAgent):
    """Mock agent whose _call_local raises an exception."""

    name = "coding"
    capabilities = ("reasoning", "code_generation")

    async def run(self, task: str) -> str:
        return ""

    async def _call_local(self, prompt: str, system: str = None) -> str:
        raise RuntimeError("model unavailable")


class ExecutionResultTests(unittest.TestCase):
    def test_to_dict_round_trips(self):
        result = ExecutionResult(
            code='print("hello")',
            language="python",
            exit_code=0,
            stdout="hello\n",
            stderr="",
            success=True,
            duration_seconds=0.05,
        )
        payload = result.to_dict()
        self.assertEqual(payload["exit_code"], 0)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["language"], "python")

    def test_error_summary_uses_last_stderr_line(self):
        result = ExecutionResult(
            code="x",
            language="python",
            exit_code=1,
            stdout="",
            stderr="Traceback (most recent call last):\n  File ...\nNameError: name 'x' is not defined",
            success=False,
            duration_seconds=0.01,
        )
        self.assertIn("NameError", result.error_summary)

    def test_error_summary_success(self):
        result = ExecutionResult(
            code="x",
            language="python",
            exit_code=0,
            stdout="ok",
            stderr="",
            success=True,
            duration_seconds=0.01,
        )
        self.assertEqual(result.error_summary, "success")


class CodeExecutorRunTests(unittest.TestCase):
    def test_run_valid_python_succeeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
            result = executor.run('print("hello nexus")', "python")

        self.assertTrue(result.success)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hello nexus", result.stdout)
        self.assertEqual(result.language, "python")

    def test_run_broken_python_captures_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
            result = executor.run('print("broken"', "python")

        self.assertFalse(result.success)
        self.assertNotEqual(result.exit_code, 0)
        self.assertTrue(result.stderr)

    def test_run_timeout_returns_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
            result = executor.run(
                "import time; time.sleep(10)",
                "python",
                timeout=1,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, -1)
        self.assertIn("timed out", result.stderr)

    def test_run_unsupported_language(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
            result = executor.run("some code", "brainfuck")

        self.assertFalse(result.success)
        self.assertIn("Unsupported language", result.stderr)

    def test_run_captures_multiline_output(self):
        code = 'for i in range(5): print(f"line {i}")'
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
            result = executor.run(code, "python")

        self.assertTrue(result.success)
        self.assertIn("line 0", result.stdout)
        self.assertIn("line 4", result.stdout)

    def test_run_logs_to_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "test.jsonl"
            executor = CodeExecutor(log_path=log_path)
            executor.run('print("logged")', "python")

            self.assertTrue(log_path.exists())
            import json
            with open(log_path, "r") as f:
                entry = json.loads(f.readline())
            self.assertTrue(entry["success"])
            self.assertEqual(entry["language"], "python")

    def test_run_with_cwd(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
            result = executor.run(
                "import os; print(os.getcwd())",
                "python",
                cwd=temp_dir,
            )

        self.assertTrue(result.success)
        self.assertIn(Path(temp_dir).name, result.stdout)


class CodeExecutorRunAndFixTests(unittest.TestCase):
    def test_returns_immediately_on_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
            result = asyncio.run(
                executor.run_and_fix(
                    task="print hello",
                    code='print("hello")',
                    language="python",
                    max_retries=3,
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.attempt, 1)

    def test_fixes_broken_code_with_agent(self):
        agent = FixingAgent('print("fixed")')
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
            result = asyncio.run(
                executor.run_and_fix(
                    task="print something",
                    code='print("broken"',  # syntax error
                    language="python",
                    max_retries=3,
                    coding_agent=agent,
                )
            )

        self.assertTrue(result.success)
        self.assertGreater(result.attempt, 1)
        self.assertGreater(agent.call_count, 0)

    def test_exhausts_retries_when_fix_fails(self):
        agent = NeverFixAgent()
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
            result = asyncio.run(
                executor.run_and_fix(
                    task="print something",
                    code='print("broken"',
                    language="python",
                    max_retries=2,
                    coding_agent=agent,
                )
            )

        self.assertFalse(result.success)
        self.assertEqual(result.attempt, 2)

    def test_returns_without_retry_when_no_agent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
            result = asyncio.run(
                executor.run_and_fix(
                    task="print something",
                    code='print("broken"',
                    language="python",
                    max_retries=3,
                    coding_agent=None,
                )
            )

        self.assertFalse(result.success)
        self.assertEqual(result.attempt, 1)

    def test_handles_agent_call_failure(self):
        agent = FailingCallAgent()
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
            result = asyncio.run(
                executor.run_and_fix(
                    task="print something",
                    code='print("broken"',
                    language="python",
                    max_retries=3,
                    coding_agent=agent,
                )
            )

        self.assertFalse(result.success)
        # When the agent call fails, we get back attempt 1 with no further retries
        self.assertEqual(result.attempt, 1)


class CodeExecutorToolInterfaceTests(unittest.TestCase):
    def test_execute_succeeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
            result = executor.execute({
                "arguments": {
                    "code": 'print("tool test")',
                    "language": "python",
                }
            })

        self.assertTrue(result["ok"])
        self.assertIn("tool test", result["stdout"])
        self.assertEqual(result["tool"], "code_executor")

    def test_execute_fails_on_empty_code(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
            result = executor.execute({"arguments": {"code": ""}})

        self.assertFalse(result["ok"])
        self.assertIn("No code", result["summary"])

    def test_execute_captures_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
            result = executor.execute({
                "arguments": {
                    "code": "raise ValueError('boom')",
                    "language": "python",
                }
            })

        self.assertFalse(result["ok"])
        self.assertIn("boom", result.get("stderr", ""))


class CodeExtractTests(unittest.TestCase):
    def test_extracts_fenced_python_block(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
        response = '```python\nprint("fixed")\n```'
        code = executor._extract_code(response, "python")
        self.assertEqual(code, 'print("fixed")')

    def test_extracts_generic_fenced_block(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
        response = '```\nprint("generic")\n```'
        code = executor._extract_code(response, "python")
        self.assertEqual(code, 'print("generic")')

    def test_returns_none_for_empty_markdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = CodeExecutor(log_path=Path(temp_dir) / "test.jsonl")
        code = executor._extract_code("# Just a heading", "python")
        self.assertIsNone(code)


if __name__ == "__main__":
    unittest.main()
