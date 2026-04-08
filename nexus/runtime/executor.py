"""Autonomous code execution with self-correcting retry loop.

This is the missing piece that closes the write → run → error → fix cycle.
CodeExecutor runs arbitrary code in a subprocess, captures structured output,
and optionally feeds errors back through a coding agent for autonomous repair.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nexus.runtime.event_bus import runtime_event_bus


@dataclass
class ExecutionResult:
    """Structured result from a single code execution."""

    code: str
    language: str
    exit_code: int
    stdout: str
    stderr: str
    success: bool
    duration_seconds: float
    attempt: int = 1
    temp_dir: str = ""
    fix_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def error_summary(self) -> str:
        """One-line summary useful for log lines and trace metadata."""
        if self.success:
            return "success"
        text = self.stderr.strip() or self.stdout.strip()
        first_line = text.split("\n")[-1] if text else f"exit code {self.exit_code}"
        return first_line[:200]


@dataclass
class StreamEvent:
    """A single event emitted during streaming execution."""

    kind: str  # "output", "error", "timeout", "exit"
    data: str
    timestamp: float = field(default_factory=time.monotonic)
    exit_code: int | None = None


class AsyncStreamResult:
    """Async-iterable wrapper for streaming execution output.

    Usage::

        stream = await executor.run_streaming(code)
        async for event in stream:
            print(event.kind, event.data)
        final = stream.final  # ExecutionResult
    """

    def __init__(self, generator):
        self._gen = generator
        self.final: ExecutionResult | None = None
        self._events: list[StreamEvent] = []

    def __aiter__(self):
        return self

    async def __anext__(self) -> StreamEvent:
        try:
            event = await self._gen.__anext__()
            self._events.append(event)
            # Check if the generator stashed a final result
            if hasattr(self._gen, "_final_result"):
                self.final = self._gen._final_result
            return event
        except StopAsyncIteration:
            if hasattr(self._gen, "_final_result"):
                self.final = self._gen._final_result
            raise


# Module name → pip package mapping for common mismatches
MODULE_TO_PIP: dict[str, str] = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "bs4": "beautifulsoup4",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "gi": "PyGObject",
    "attr": "attrs",
    "dotenv": "python-dotenv",
    "jose": "python-jose",
    "jwt": "PyJWT",
    "magic": "python-magic",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "wx": "wxPython",
    "serial": "pyserial",
    "usb": "pyusb",
    "Crypto": "pycryptodome",
    "lxml": "lxml",
    "zmq": "pyzmq",
    "psycopg2": "psycopg2-binary",
    "MySQLdb": "mysqlclient",
}


class CodeExecutor:
    """Run code in a subprocess, capture output, and optionally self-correct.

    Supports Python, JavaScript, and TypeScript.  Each execution writes to an
    isolated temp directory, runs the appropriate interpreter, and returns a
    structured ``ExecutionResult``.

    The ``run_and_fix`` method connects this to a coding agent for the
    autonomous write → run → error → fix loop.
    """

    LANGUAGE_CONFIG: dict[str, dict[str, Any]] = {
        "python": {
            "extension": ".py",
            "commands": [["python", "{file}"], ["py", "{file}"]],
        },
        "javascript": {
            "extension": ".js",
            "commands": [["node", "{file}"]],
        },
        "typescript": {
            "extension": ".ts",
            "commands": [
                ["npx", "tsx", "{file}"],
                ["npx", "ts-node", "{file}"],
            ],
        },
    }

    def __init__(
        self,
        *,
        log_path: Path | None = None,
        max_output_bytes: int = 8000,
        auto_install_deps: bool = False,
    ):
        from nexus.config import config

        self._config = config
        self._max_output_bytes = max_output_bytes
        self.auto_install_deps = bool(auto_install_deps)
        runtime_log_dir = config.data_dir / "runtime_logs"
        runtime_log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path) if log_path else runtime_log_dir / "executor.jsonl"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        code: str,
        language: str = "python",
        *,
        timeout: int = 30,
        cwd: Path | str | None = None,
    ) -> ExecutionResult:
        """Write *code* to a temp file, execute it, and return structured output."""
        language = language.lower().strip()
        lang_config = self.LANGUAGE_CONFIG.get(language)
        if lang_config is None:
            return ExecutionResult(
                code=code,
                language=language,
                exit_code=1,
                stdout="",
                stderr=f"Unsupported language: {language}. Supported: {', '.join(self.LANGUAGE_CONFIG)}",
                success=False,
                duration_seconds=0.0,
            )

        work_dir = Path(tempfile.mkdtemp(prefix="nexus_exec_"))
        file_path = work_dir / f"main{lang_config['extension']}"
        file_path.write_text(code, encoding="utf-8")

        command = self._resolve_command(lang_config["commands"], str(file_path))
        if command is None:
            return ExecutionResult(
                code=code,
                language=language,
                exit_code=127,
                stdout="",
                stderr=f"No suitable interpreter found for {language}. Checked: {lang_config['commands']}",
                success=False,
                duration_seconds=0.0,
                temp_dir=str(work_dir),
            )

        run_cwd = Path(cwd).resolve() if cwd else work_dir
        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=run_cwd,
                text=True,
                capture_output=True,
                timeout=timeout,
                shell=False,
                env=self._build_env(),
            )
            elapsed = time.monotonic() - start
            result = ExecutionResult(
                code=code,
                language=language,
                exit_code=completed.returncode,
                stdout=completed.stdout[-self._max_output_bytes:],
                stderr=completed.stderr[-self._max_output_bytes:],
                success=completed.returncode == 0,
                duration_seconds=round(elapsed, 3),
                temp_dir=str(work_dir),
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            result = ExecutionResult(
                code=code,
                language=language,
                exit_code=-1,
                stdout="",
                stderr=f"Execution timed out after {timeout}s",
                success=False,
                duration_seconds=round(elapsed, 3),
                temp_dir=str(work_dir),
            )

        self._log_execution(result)
        return result

    async def run_streaming(
        self,
        code: str,
        language: str = "python",
        *,
        timeout: int = 60,
        idle_timeout: int = 15,
        cwd: Path | str | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> AsyncStreamResult:
        """Run code and yield output lines in real-time as ``StreamEvent`` objects.

        Unlike ``run()``, this method uses ``asyncio.subprocess`` so callers
        can process stdout/stderr lines the instant they arrive — ideal for
        long-running scripts, servers, or build tools.

        Parameters
        ----------
        code : str
            Source code to execute.
        language : str
            Language key (``python``, ``javascript``, ``typescript``).
        timeout : int
            Hard wall-clock timeout for the entire execution (seconds).
        idle_timeout : int
            Kill the process if no output arrives within this many seconds.
        cwd : Path | str | None
            Working directory for the subprocess.

        Returns
        -------
        AsyncStreamResult
            An async-iterable wrapper.  Use ``async for event in executor.run_streaming(…):``.
            After iteration, ``result.final`` contains the full ``ExecutionResult``.
        """
        language = language.lower().strip()
        lang_config = self.LANGUAGE_CONFIG.get(language)
        if lang_config is None:
            async def _unsupported():
                yield StreamEvent(kind="error", data=f"Unsupported language: {language}")
                yield StreamEvent(kind="exit", data="", exit_code=1)
            return AsyncStreamResult(_unsupported())

        work_dir = Path(tempfile.mkdtemp(prefix="nexus_stream_"))
        file_path = work_dir / f"main{lang_config['extension']}"
        file_path.write_text(code, encoding="utf-8")

        command = self._resolve_command(lang_config["commands"], str(file_path))
        if command is None:
            async def _no_interpreter():
                yield StreamEvent(kind="error", data=f"No interpreter found for {language}")
                yield StreamEvent(kind="exit", data="", exit_code=127)
            return AsyncStreamResult(_no_interpreter())

        run_cwd = str(Path(cwd).resolve()) if cwd else str(work_dir)
        executor_ref = self

        async def _stream():
            stdout_lines: list[str] = []
            stderr_lines: list[str] = []
            start = time.monotonic()
            last_activity = start

            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=run_cwd,
                env=executor_ref._build_env(),
            )

            async def _read_stream(stream, kind, collector, queue):
                nonlocal last_activity
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace").rstrip("\n\r")
                    collector.append(decoded)
                    last_activity = time.monotonic()
                    await queue.put(StreamEvent(kind=kind, data=decoded))
                await queue.put(None)

            async def _watchdog():
                nonlocal last_activity
                while proc.returncode is None:
                    await asyncio.sleep(1)
                    elapsed = time.monotonic() - start
                    idle = time.monotonic() - last_activity
                    if elapsed > timeout:
                        proc.kill()
                        return "timeout"
                    if idle > idle_timeout:
                        proc.kill()
                        return "idle_timeout"
                return None

            queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
            watchdog_task = asyncio.create_task(_watchdog())
            stdout_task = asyncio.create_task(_read_stream(proc.stdout, "output", stdout_lines, queue))
            stderr_task = asyncio.create_task(_read_stream(proc.stderr, "error", stderr_lines, queue))

            readers_finished = 0
            while readers_finished < 2:
                event = await queue.get()
                if event is None:
                    readers_finished += 1
                    continue
                runtime_event_bus.emit(
                    {
                        "type": "execution_output",
                        "kind": event.kind,
                        "data": event.data,
                        **dict(event_context or {}),
                    }
                )
                yield event

            watchdog_reason = await watchdog_task
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            await proc.wait()

            elapsed = time.monotonic() - start
            exit_code = proc.returncode if proc.returncode is not None else -1

            if watchdog_reason:
                timeout_event = StreamEvent(
                    kind="timeout",
                    data=f"Process killed: {watchdog_reason} after {elapsed:.1f}s",
                    exit_code=exit_code,
                )
                runtime_event_bus.emit(
                    {
                        "type": "execution_output",
                        "kind": timeout_event.kind,
                        "data": timeout_event.data,
                        **dict(event_context or {}),
                    }
                )
                yield timeout_event

            exit_event = StreamEvent(kind="exit", data="", exit_code=exit_code)
            runtime_event_bus.emit(
                {
                    "type": "execution_output",
                    "kind": exit_event.kind,
                    "data": exit_event.data,
                    "exit_code": exit_code,
                    **dict(event_context or {}),
                }
            )
            yield exit_event

            # Build and log the final result
            result = ExecutionResult(
                code=code,
                language=language,
                exit_code=exit_code,
                stdout="\n".join(stdout_lines)[-executor_ref._max_output_bytes:],
                stderr="\n".join(stderr_lines)[-executor_ref._max_output_bytes:],
                success=exit_code == 0 and watchdog_reason is None,
                duration_seconds=round(elapsed, 3),
                temp_dir=str(work_dir),
                metadata={"watchdog_reason": watchdog_reason} if watchdog_reason else {},
            )
            executor_ref._log_execution(result)
            # Stash result on the generator for AsyncStreamResult to pick up
            _stream._final_result = result  # type: ignore[attr-defined]

        gen = _stream()
        return AsyncStreamResult(gen)

    async def run_and_fix(
        self,
        task: str,
        code: str,
        language: str = "python",
        *,
        max_retries: int = 3,
        timeout: int = 30,
        coding_agent: Any = None,
        auto_install_deps: bool | None = None,
    ) -> ExecutionResult:
        """Run code, and if it fails, use a coding agent to fix and retry.

        This is the autonomous loop:
            1. Run the code
            2. If success → return
            2a. If failure is a missing dependency → auto-install and retry
            3. Build a fix prompt with the error context
            4. Ask the coding agent for a corrected version
            5. Extract the code from the agent's response
            6. Go to 1, up to *max_retries* total attempts

        If no *coding_agent* is provided, runs once without retries.
        """
        fix_history: list[dict[str, Any]] = []
        current_code = code
        _installed_deps: set[str] = set()
        allow_auto_install = self.auto_install_deps if auto_install_deps is None else auto_install_deps

        for attempt in range(1, max_retries + 1):
            result = self.run(current_code, language, timeout=timeout)
            result.attempt = attempt
            result.fix_history = list(fix_history)

            if result.success:
                return result

            # --- Dependency auto-resolution (before agent fix) ---
            if allow_auto_install:
                missing = self._detect_missing_deps(result.stderr, language)
                new_deps = missing - _installed_deps
                if new_deps:
                    installed = self._install_deps(new_deps, language)
                    _installed_deps.update(installed)
                    if installed:
                        fix_history.append({
                            "attempt": attempt,
                            "error": result.error_summary,
                            "fix_status": f"auto-installed: {', '.join(installed)}",
                        })
                        continue  # Retry with same code after installing

            if coding_agent is None:
                return result

            if attempt == max_retries:
                # Exhausted all retries
                return result

            # Build fix prompt and ask the agent
            fix_prompt = self._build_fix_prompt(
                task=task,
                code=current_code,
                language=language,
                stderr=result.stderr,
                stdout=result.stdout,
                exit_code=result.exit_code,
                attempt=attempt,
                max_retries=max_retries,
            )

            try:
                agent_response = await coding_agent._call_local(fix_prompt)
            except Exception as error:
                fix_history.append({
                    "attempt": attempt,
                    "error": result.error_summary,
                    "fix_status": f"agent call failed: {error}",
                })
                return result

            fixed_code = self._extract_code(agent_response, language)
            if not fixed_code or fixed_code.strip() == current_code.strip():
                fix_history.append({
                    "attempt": attempt,
                    "error": result.error_summary,
                    "fix_status": "agent returned identical or empty code",
                })
                return result

            fix_history.append({
                "attempt": attempt,
                "error": result.error_summary,
                "fix_status": "applied fix",
                "code_changed": True,
            })
            current_code = fixed_code

        return result  # type: ignore[possibly-undefined]

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute a structured tool request (same interface as TerminalTool/FileTool).

        This lets the orchestrator dispatch ``code_executor`` tool calls from agents.
        """
        arguments = dict(request.get("arguments") or request.get("args") or {})
        action = str(request.get("action") or arguments.get("action") or "run").strip().lower()
        code = arguments.get("code", "")
        language = arguments.get("language", "python")
        timeout = int(arguments.get("timeout", 30) or 30)

        if not code.strip():
            return {
                "ok": False,
                "tool": "code_executor",
                "action": "run",
                "summary": "No code provided to execute.",
                "error": "empty code",
            }

        if action == "run_and_fix":
            return {
                "ok": False,
                "tool": "code_executor",
                "action": action,
                "summary": "run_and_fix requires async orchestration. Use execute_async instead.",
                "error": "async action required",
            }

        result = self.run(code, language, timeout=timeout)
        return {
            "ok": result.success,
            "tool": "code_executor",
            "action": action,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_seconds": result.duration_seconds,
            "summary": (
                f"Code executed successfully ({result.duration_seconds}s)"
                if result.success
                else f"Code failed (exit {result.exit_code}): {result.error_summary}"
            ),
        }

    async def execute_async(self, request: dict[str, Any]) -> dict[str, Any]:
        """Async tool entry point for streaming or autonomous execution."""
        arguments = dict(request.get("arguments") or request.get("args") or {})
        action = str(request.get("action") or arguments.get("action") or "run").strip().lower()
        code = arguments.get("code", "")
        language = arguments.get("language", "python")
        timeout = int(arguments.get("timeout", 30) or 30)

        if not code.strip():
            return {
                "ok": False,
                "tool": "code_executor",
                "action": action,
                "summary": "No code provided to execute.",
                "error": "empty code",
            }

        if action == "run_streaming":
            stream = await self.run_streaming(
                code,
                language,
                timeout=timeout,
                idle_timeout=int(arguments.get("idle_timeout", 15) or 15),
                cwd=arguments.get("cwd"),
                event_context={
                    "tool": "code_executor",
                    "action": action,
                    "language": language,
                },
            )
            events: list[dict[str, Any]] = []
            async for event in stream:
                events.append(
                    {
                        "kind": event.kind,
                        "data": event.data,
                        "exit_code": event.exit_code,
                    }
                )
            final = stream.final or ExecutionResult(
                code=code,
                language=language,
                exit_code=1,
                stdout="",
                stderr="Streaming execution did not produce a final result.",
                success=False,
                duration_seconds=0.0,
            )
            return {
                "ok": final.success,
                "tool": "code_executor",
                "action": action,
                "exit_code": final.exit_code,
                "stdout": final.stdout,
                "stderr": final.stderr,
                "events": events,
                "summary": (
                    f"Streaming execution succeeded ({final.duration_seconds}s)"
                    if final.success
                    else f"Streaming execution failed (exit {final.exit_code}): {final.error_summary}"
                ),
            }

        if action == "run_and_fix":
            coding_agent = arguments.get("coding_agent")
            result = await self.run_and_fix(
                task=arguments.get("task", "Fix the provided code"),
                code=code,
                language=language,
                max_retries=int(arguments.get("max_retries", 3) or 3),
                timeout=timeout,
                coding_agent=coding_agent,
                auto_install_deps=arguments.get("auto_install_deps"),
            )
            return {
                "ok": result.success,
                "tool": "code_executor",
                "action": action,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "fix_history": list(result.fix_history),
                "summary": (
                    f"Code fixed and executed successfully ({result.attempt} attempt(s))"
                    if result.success
                    else f"Code still failing after {result.attempt} attempt(s): {result.error_summary}"
                ),
            }

        return self.execute(request)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_command(
        self,
        command_templates: list[list[str]],
        file_path: str,
    ) -> list[str] | None:
        """Find the first available interpreter from the candidate list."""
        for template in command_templates:
            executable = template[0]
            # On Windows, check .cmd variants for npm-based tools
            candidates = [executable]
            if os.name == "nt" and executable in ("npx", "npm", "pnpm"):
                candidates.insert(0, f"{executable}.cmd")

            for candidate in candidates:
                resolved = shutil.which(candidate)
                if resolved:
                    return [resolved] + [
                        file_path if part == "{file}" else part
                        for part in template[1:]
                    ]
        return None

    def _build_env(self) -> dict[str, str]:
        """Build a clean environment for subprocess execution."""
        env = os.environ.copy()
        # Prevent interactive prompts
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.setdefault("NODE_NO_WARNINGS", "1")
        return env

    def _build_fix_prompt(
        self,
        *,
        task: str,
        code: str,
        language: str,
        stderr: str,
        stdout: str,
        exit_code: int,
        attempt: int,
        max_retries: int,
    ) -> str:
        """Build a focused prompt for the coding agent to fix the error."""
        return (
            f"You are fixing broken {language} code. This is attempt {attempt} of {max_retries}.\n"
            f"Return ONLY the corrected code in a single fenced code block.\n"
            f"Do not explain the fix. Do not add commentary outside the code block.\n\n"
            f"Original task:\n{task}\n\n"
            f"Failing code:\n```{language}\n{code}\n```\n\n"
            f"Exit code: {exit_code}\n\n"
            f"stderr:\n```\n{stderr[-3000:]}\n```\n\n"
            f"stdout:\n```\n{stdout[-1000:]}\n```\n\n"
            f"Return the complete fixed code in a single ```{language} code block."
        )

    def _extract_code(self, response: str, language: str) -> str | None:
        """Extract the first fenced code block matching the language."""
        # Try language-specific fence first
        patterns = [
            rf"```{re.escape(language)}\s*\n(.*?)```",
            r"```\w*\s*\n(.*?)```",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()

        # If the entire response looks like code (no markdown), use it directly
        stripped = response.strip()
        if stripped and "```" not in stripped and not stripped.startswith("#"):
            return stripped

        return None

    def _log_execution(self, result: ExecutionResult) -> None:
        """Append a structured log entry for every execution."""
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "language": result.language,
            "exit_code": result.exit_code,
            "success": result.success,
            "duration_seconds": result.duration_seconds,
            "attempt": result.attempt,
            "error_summary": result.error_summary,
            "stdout_length": len(result.stdout),
            "stderr_length": len(result.stderr),
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except OSError:
            pass  # Never crash on log failure

    @staticmethod
    def _detect_missing_deps(stderr: str, language: str) -> set[str]:
        """Parse stderr for missing dependency errors.

        Returns a set of package names to install.
        """
        missing: set[str] = set()

        if language == "python":
            # ModuleNotFoundError: No module named 'foo'
            for match in re.finditer(
                r"ModuleNotFoundError: No module named ['\"](\w+)['\"]", stderr
            ):
                module = match.group(1)
                # Map to pip package name (or use module name as-is)
                missing.add(MODULE_TO_PIP.get(module, module))

            # ImportError: cannot import name 'X' from 'Y'
            for match in re.finditer(
                r"ImportError:.*from ['\"](\w+)['\"]", stderr
            ):
                module = match.group(1)
                missing.add(MODULE_TO_PIP.get(module, module))

        elif language in ("javascript", "typescript"):
            # Error: Cannot find module 'foo'
            for match in re.finditer(
                r"Cannot find module ['\"]([^'\"]+)['\"]", stderr
            ):
                pkg = match.group(1)
                # Skip relative imports
                if not pkg.startswith(".") and not pkg.startswith("/"):
                    # Get the base package name (e.g., @scope/name or name)
                    parts = pkg.split("/")
                    if parts[0].startswith("@") and len(parts) > 1:
                        missing.add(f"{parts[0]}/{parts[1]}")
                    else:
                        missing.add(parts[0])

        return missing

    @staticmethod
    def _install_deps(packages: set[str], language: str) -> set[str]:
        """Attempt to install missing packages. Returns the set that was installed."""
        installed: set[str] = set()

        if language == "python":
            for pkg in packages:
                try:
                    subprocess.run(
                        [shutil.which("pip") or "pip", "install", pkg],
                        capture_output=True,
                        timeout=60,
                    )
                    installed.add(pkg)
                except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                    pass

        elif language in ("javascript", "typescript"):
            pkg_list = list(packages)
            if pkg_list:
                npm = shutil.which("npm.cmd") if os.name == "nt" else shutil.which("npm")
                if npm:
                    try:
                        subprocess.run(
                            [npm, "install", "--save"] + pkg_list,
                            capture_output=True,
                            timeout=120,
                        )
                        installed.update(pkg_list)
                    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                        pass

        return installed
