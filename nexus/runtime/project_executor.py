"""Multi-file project execution with dependency management and error-targeted fixes.

ProjectExecutor bridges the gap between single-file CodeExecutor runs and real
multi-file projects.  It can:

1. Materialize a ``{path: content}`` file tree to disk.
2. Auto-detect and install dependencies (npm, pip).
3. Run the project and capture output (or stream it).
4. On failure, identify the failing file and route it back to CodingAgent.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexus.runtime.build_artifacts import BuildArtifactMaterializer
from nexus.runtime.executor import CodeExecutor, ExecutionResult
from nexus.runtime.event_bus import runtime_event_bus
from nexus.runtime.scaffold_runner import ScaffoldRunError, ScaffoldRunner


@dataclass
class ProjectResult:
    """Structured result from a multi-file project execution."""

    project_dir: str
    files_written: list[str]
    install_ok: bool
    install_output: str
    run_result: ExecutionResult | None
    package_manager: str | None = None
    fix_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.install_ok and self.run_result is not None and self.run_result.success

    @property
    def summary(self) -> str:
        if self.success:
            return f"Project built and ran successfully ({len(self.files_written)} files)"
        if not self.install_ok:
            return f"Dependency install failed: {self.install_output[:200]}"
        if self.run_result:
            return f"Project run failed: {self.run_result.error_summary}"
        return "Project was written but not executed"


class ProjectExecutor:
    """Build, install, and run multi-file projects with autonomous error repair.

    Works with the existing ``BuildArtifactMaterializer`` file-tree format
    and ``ScaffoldRunner`` patterns but adds the missing execution + fix loop.
    """

    def __init__(
        self,
        *,
        executor: CodeExecutor | None = None,
        workspace_base: Path | None = None,
    ):
        from nexus.config import config

        self._executor = executor or CodeExecutor()
        self._artifact_materializer = BuildArtifactMaterializer()
        self._workspace_base = Path(workspace_base) if workspace_base else config.data_dir / "projects"
        self._workspace_base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def materialize(
        self,
        file_tree: dict[str, str],
        project_dir: Path | str | None = None,
    ) -> Path:
        """Write a ``{relative_path: content}`` dict to disk.

        Returns the project root directory.
        """
        if project_dir is None:
            project_dir = Path(tempfile.mkdtemp(prefix="nexus_project_", dir=self._workspace_base))
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in file_tree.items():
            safe_rel_path = self._safe_relative_path(rel_path)
            file_path = project_dir / safe_rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

        return project_dir

    def materialize_output(
        self,
        output: str,
        project_dir: Path | str | None = None,
    ) -> Path:
        """Materialize path-tagged fenced file blocks."""
        artifacts = self._artifact_materializer.extract(output)
        return self.materialize(
            {artifact.relative_path.as_posix(): artifact.content for artifact in artifacts},
            project_dir=project_dir,
        )

    def detect_package_manager(self, project_dir: Path) -> str | None:
        """Detect which package manager the project uses."""
        if (project_dir / "package-lock.json").exists():
            return "npm"
        if (project_dir / "pnpm-lock.yaml").exists():
            return "pnpm"
        if (project_dir / "yarn.lock").exists():
            return "yarn"
        if (project_dir / "package.json").exists():
            return "npm"
        if (project_dir / "requirements.txt").exists():
            return "pip"
        if (project_dir / "pyproject.toml").exists():
            return "pip"
        return None

    def install(self, project_dir: Path) -> tuple[bool, str, str | None]:
        """Auto-detect package manager and install dependencies.

        Returns ``(success, output, package_manager)``.
        """
        pm = self.detect_package_manager(project_dir)
        if pm is None:
            return (True, "No dependencies to install", None)

        commands = {
            "npm": ["npm", "install", "--no-audit", "--no-fund"],
            "pnpm": ["pnpm", "install", "--no-frozen-lockfile"],
            "yarn": ["yarn", "install"],
            "pip": ["pip", "install", "-r", "requirements.txt"],
        }
        cmd = commands.get(pm)
        if cmd is None:
            return (True, f"Unknown package manager: {pm}", pm)

        # For pip with pyproject.toml, use `pip install .`
        if pm == "pip" and not (project_dir / "requirements.txt").exists():
            cmd = ["pip", "install", "-e", "."]

        # Resolve executable (handles .cmd on Windows)
        executable = cmd[0]
        if os.name == "nt" and executable in ("npm", "pnpm", "yarn"):
            resolved = shutil.which(f"{executable}.cmd") or shutil.which(executable)
        else:
            resolved = shutil.which(executable)

        if not resolved:
            return (False, f"{executable} not found on PATH", pm)

        cmd[0] = resolved
        try:
            result = subprocess.run(
                cmd,
                cwd=project_dir,
                text=True,
                capture_output=True,
                timeout=120,
                shell=False,
            )
            output = result.stdout + result.stderr
            return (result.returncode == 0, output[-4000:], pm)
        except subprocess.TimeoutExpired:
            return (False, "Dependency install timed out (120s)", pm)
        except Exception as error:
            return (False, f"Install error: {error}", pm)

    async def run(
        self,
        project_dir: Path,
        *,
        entry_point: str | None = None,
        timeout: int = 30,
        workflow_id: str | None = None,
        task_id: str | None = None,
    ) -> ExecutionResult:
        """Run the project's entry point and return structured output.

        Auto-detects the entry point from package.json scripts, main.py, etc.
        """
        project_dir = Path(project_dir)
        entry = entry_point or self._detect_entry_point(project_dir)

        if entry is None:
            return ExecutionResult(
                code="",
                language="unknown",
                exit_code=1,
                stdout="",
                stderr="No entry point detected. Add a main.py, index.js, or package.json scripts.start",
                success=False,
                duration_seconds=0.0,
                temp_dir=str(project_dir),
            )

        if entry.startswith("npm:"):
            return await self._run_with_scaffold_runner(
                project_dir,
                timeout=timeout,
                workflow_id=workflow_id,
                task_id=task_id,
            )

        # Determine language and build command
        language, command = self._build_run_command(project_dir, entry)

        if command is None:
            return ExecutionResult(
                code=f"entry: {entry}",
                language=language,
                exit_code=127,
                stdout="",
                stderr=f"Could not resolve interpreter for entry point: {entry}",
                success=False,
                duration_seconds=0.0,
                temp_dir=str(project_dir),
            )

        return await self._run_command_async(
            command,
            cwd=project_dir,
            language=language,
            entry=entry,
            timeout=timeout,
            workflow_id=workflow_id,
            task_id=task_id,
        )

    def run_project(
        self,
        project_dir: Path,
        *,
        entry_point: str | None = None,
        timeout: int = 30,
    ) -> ExecutionResult:
        """Synchronous compatibility wrapper used by the existing tests/CLI."""
        return asyncio.run(self.run(project_dir, entry_point=entry_point, timeout=timeout))

    def build_and_run(
        self,
        file_tree: dict[str, str],
        *,
        project_dir: Path | str | None = None,
        timeout: int = 30,
    ) -> ProjectResult:
        """Materialize → install → run in one call."""
        root = self.materialize(file_tree, project_dir)
        files = sorted(file_tree.keys())

        install_ok, install_output, pm = self.install(root)
        if not install_ok:
            return ProjectResult(
                project_dir=str(root),
                files_written=files,
                install_ok=False,
                install_output=install_output,
                run_result=None,
                package_manager=pm,
            )

        run_result = self.run_project(root, timeout=timeout)
        return ProjectResult(
            project_dir=str(root),
            files_written=files,
            install_ok=True,
            install_output=install_output,
            run_result=run_result,
            package_manager=pm,
        )

    def identify_failing_file(
        self,
        project_dir: Path,
        stderr: str,
    ) -> str | None:
        """Parse error output to figure out which file caused the failure.

        Looks for common error patterns:
        - Python: ``File "path/to/file.py", line N``
        - Node/JS: ``at Object.<anonymous> (path/to/file.js:N:N)``
        - ``Error in ./path/to/file``
        """
        project_str = str(Path(project_dir)).replace("\\", "/")
        patterns = [
            r'File "([^"]+)", line \d+',
            r'at (?:Object\.|Module\.)?\S+ \(([^:)]+):\d+:\d+\)',
            r'Error in \./([^\s]+)',
            r'Module not found.*?in ([^\s]+)',
            r'(/[^\s:]+\.\w+):\d+:\d+',
        ]
        for pattern in patterns:
            match = re.search(pattern, stderr)
            if match:
                file_path = match.group(1)
                # Convert to relative path within project
                file_path = file_path.replace("\\", "/")
                if project_str in file_path:
                    return file_path.split(project_str + "/", 1)[-1]
                return Path(file_path).name
        return None

    async def build_and_run_async(
        self,
        file_tree: dict[str, str],
        *,
        project_dir: Path | str | None = None,
        timeout: int = 30,
        coding_agent: Any = None,
        max_fix_attempts: int = 2,
        workflow_id: str | None = None,
        task_id: str | None = None,
    ) -> ProjectResult:
        """Async project execution path with targeted file repair."""
        root = self.materialize(file_tree, project_dir)
        files = sorted(file_tree.keys())
        install_ok, install_output, pm = await asyncio.to_thread(self.install, root)
        if not install_ok:
            return ProjectResult(
                project_dir=str(root),
                files_written=files,
                install_ok=False,
                install_output=install_output,
                run_result=None,
                package_manager=pm,
            )

        fix_history: list[dict[str, Any]] = []
        run_result = await self.run(
            root,
            timeout=timeout,
            workflow_id=workflow_id,
            task_id=task_id,
        )
        attempts = 0
        while not run_result.success and coding_agent is not None and attempts < max_fix_attempts:
            attempts += 1
            fix_result = await self.fix_file(
                root,
                error=run_result.stderr,
                stdout=run_result.stdout,
                coding_agent=coding_agent,
                task="Repair the generated project",
                workflow_id=workflow_id,
                task_id=task_id,
            )
            fix_history.append(fix_result)
            if not fix_result.get("ok"):
                break
            run_result = await self.run(
                root,
                timeout=timeout,
                workflow_id=workflow_id,
                task_id=task_id,
            )

        return ProjectResult(
            project_dir=str(root),
            files_written=files,
            install_ok=True,
            install_output=install_output,
            run_result=run_result,
            package_manager=pm,
            fix_history=fix_history,
        )

    async def fix_file(
        self,
        project_dir: Path,
        *,
        error: str,
        coding_agent: Any,
        stdout: str = "",
        task: str = "",
        workflow_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Identify and repair only the failing file inside the workspace."""
        project_dir = Path(project_dir)
        failing_file = self.identify_failing_file(project_dir, error) or self._default_fix_target(project_dir)
        if not failing_file:
            return {
                "ok": False,
                "summary": "Could not identify a target file to fix.",
            }

        target_path = project_dir / failing_file
        if not target_path.exists():
            return {
                "ok": False,
                "summary": f"Target file does not exist: {failing_file}",
                "file": failing_file,
            }

        original = target_path.read_text(encoding="utf-8", errors="replace")
        prompt = (
            "You are fixing exactly one file in a generated project.\n"
            "Return ONLY the full corrected contents for the requested file in a single fenced code block.\n"
            "Do not rename files. Do not return explanations.\n\n"
            f"Goal:\n{task or 'Repair the project'}\n\n"
            f"Target file: {failing_file}\n\n"
            f"Current contents:\n```{target_path.suffix.lstrip('.') or 'text'}\n{original}\n```\n\n"
            f"stderr:\n```\n{error[-3000:]}\n```\n\n"
            f"stdout:\n```\n{stdout[-1200:]}\n```\n"
        )
        response = await coding_agent._call_local(prompt)
        updated = self._extract_file_content(response, failing_file, original)
        if not updated or updated.strip() == original.strip():
            return {
                "ok": False,
                "summary": f"No code change produced for {failing_file}",
                "file": failing_file,
            }

        target_path.write_text(updated, encoding="utf-8")
        runtime_event_bus.emit(
            {
                "type": "fix_applied",
                "workflow_id": workflow_id,
                "task_id": task_id,
                "file": failing_file,
                "summary": f"Applied targeted fix to {failing_file}",
            }
        )
        return {
            "ok": True,
            "summary": f"Applied targeted fix to {failing_file}",
            "file": failing_file,
        }

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        """Tool dispatch interface for the orchestrator."""
        args = dict(request.get("arguments") or request.get("args") or {})
        action = args.get("action", "build_and_run")

        if action == "build_and_run":
            file_tree = args.get("file_tree", {})
            if not file_tree:
                return {
                    "ok": False,
                    "tool": "project_executor",
                    "action": action,
                    "summary": "No file_tree provided.",
                }
            result = self.build_and_run(file_tree, timeout=int(args.get("timeout", 30)))
            return {
                "ok": result.success,
                "tool": "project_executor",
                "action": action,
                "project_dir": result.project_dir,
                "files_written": result.files_written,
                "install_ok": result.install_ok,
                "stdout": result.run_result.stdout if result.run_result else "",
                "stderr": result.run_result.stderr if result.run_result else result.install_output,
                "summary": result.summary,
            }

        return {
            "ok": False,
            "tool": "project_executor",
            "action": action,
            "summary": f"Unknown action: {action}",
        }

    async def execute_async(self, request: dict[str, Any]) -> dict[str, Any]:
        """Async tool dispatch for orchestrator-driven project workflows."""
        args = dict(request.get("arguments") or request.get("args") or {})
        action = args.get("action", "build_and_run")

        if action == "materialize":
            if args.get("output"):
                root = await asyncio.to_thread(self.materialize_output, args["output"], args.get("project_dir"))
            else:
                root = await asyncio.to_thread(self.materialize, args.get("file_tree", {}), args.get("project_dir"))
            return {
                "ok": True,
                "tool": "project_executor",
                "action": action,
                "project_dir": str(root),
                "summary": f"Materialized project to {root}",
            }

        if action == "install":
            root = Path(args.get("project_dir") or ".")
            ok, output, pm = await asyncio.to_thread(self.install, root)
            return {
                "ok": ok,
                "tool": "project_executor",
                "action": action,
                "project_dir": str(root),
                "package_manager": pm,
                "stdout": output if ok else "",
                "stderr": "" if ok else output,
                "summary": (
                    output[:240]
                    if output[:240]
                    else "Dependencies installed"
                    if ok
                    else "Dependency install failed"
                ),
            }

        if action == "run":
            root = Path(args.get("project_dir") or ".")
            result = await self.run(
                root,
                entry_point=args.get("entry_point"),
                timeout=int(args.get("timeout", 30) or 30),
                workflow_id=args.get("workflow_id"),
                task_id=args.get("task_id"),
            )
            return {
                "ok": result.success,
                "tool": "project_executor",
                "action": action,
                "project_dir": str(root),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "metadata": dict(result.metadata),
                "summary": (
                    f"Project running on port {result.metadata.get('port')}"
                    if result.success and result.metadata.get("port")
                    else f"Project run succeeded ({result.duration_seconds}s)"
                    if result.success
                    else f"Project run failed: {result.error_summary}"
                ),
            }

        if action == "fix_file":
            result = await self.fix_file(
                Path(args.get("project_dir") or "."),
                error=str(args.get("error", "")),
                stdout=str(args.get("stdout", "")),
                coding_agent=args.get("coding_agent"),
                task=str(args.get("task", "")),
                workflow_id=args.get("workflow_id"),
                task_id=args.get("task_id"),
            )
            return {
                "tool": "project_executor",
                "action": action,
                **result,
            }

        if action == "build_and_run":
            file_tree = args.get("file_tree")
            if not file_tree and args.get("output"):
                root = self.materialize_output(args["output"], args.get("project_dir"))
                install_ok, install_output, pm = await asyncio.to_thread(self.install, root)
                run_result = (
                    await self.run(
                        root,
                        timeout=int(args.get("timeout", 30) or 30),
                        workflow_id=args.get("workflow_id"),
                        task_id=args.get("task_id"),
                    )
                    if install_ok
                    else None
                )
                result = ProjectResult(
                    project_dir=str(root),
                    files_written=[path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()],
                    install_ok=install_ok,
                    install_output=install_output,
                    run_result=run_result,
                    package_manager=pm,
                )
            else:
                result = await self.build_and_run_async(
                    file_tree or {},
                    project_dir=args.get("project_dir"),
                    timeout=int(args.get("timeout", 30) or 30),
                    coding_agent=args.get("coding_agent"),
                    max_fix_attempts=int(args.get("max_fix_attempts", 2) or 2),
                    workflow_id=args.get("workflow_id"),
                    task_id=args.get("task_id"),
                )
            return {
                "ok": result.success,
                "tool": "project_executor",
                "action": action,
                "project_dir": result.project_dir,
                "files_written": result.files_written,
                "install_ok": result.install_ok,
                "stdout": result.run_result.stdout if result.run_result else "",
                "stderr": result.run_result.stderr if result.run_result else result.install_output,
                "metadata": dict((result.run_result.metadata if result.run_result else {}) or {}),
                "fix_history": list(result.fix_history),
                "summary": result.summary,
            }

        return self.execute(request)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_entry_point(self, project_dir: Path) -> str | None:
        """Find the most likely entry point for the project."""
        # Check package.json for start/dev scripts
        pkg_json = project_dir / "package.json"
        if pkg_json.exists():
            try:
                pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
                scripts = pkg.get("scripts", {})
                if "start" in scripts:
                    return "npm:start"
                if "dev" in scripts:
                    return "npm:dev"
                main = pkg.get("main")
                if main:
                    return main
            except (json.JSONDecodeError, OSError):
                pass

        # Check common entry point files
        candidates = [
            "main.py", "app.py", "server.py", "run.py",
            "index.js", "main.js", "server.js", "app.js",
            "index.ts", "main.ts", "server.ts", "app.ts",
            "src/index.js", "src/main.js", "src/index.ts", "src/main.ts",
            "src/App.jsx", "src/App.tsx",
        ]
        for candidate in candidates:
            if (project_dir / candidate).exists():
                return candidate

        return None

    def _build_run_command(
        self, project_dir: Path, entry: str
    ) -> tuple[str, list[str] | None]:
        """Build the shell command list for the detected entry point."""
        if entry.startswith("npm:"):
            script = entry[4:]
            npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
            if npm:
                return ("javascript", [npm, "run", script])
            return ("javascript", None)

        ext = Path(entry).suffix.lower()
        lang_map = {
            ".py": ("python", [["python", entry], ["py", entry]]),
            ".js": ("javascript", [["node", entry]]),
            ".jsx": ("javascript", [["node", entry]]),
            ".ts": ("typescript", [["npx", "tsx", entry], ["npx", "ts-node", entry]]),
            ".tsx": ("typescript", [["npx", "tsx", entry]]),
        }

        if ext not in lang_map:
            return ("unknown", None)

        language, command_templates = lang_map[ext]
        resolved = self._executor._resolve_command(command_templates, str(project_dir / entry))
        return (language, resolved)

    def _safe_relative_path(self, raw_path: str) -> Path:
        normalized = str(raw_path).strip().replace("\\", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        relative = Path(normalized)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe file path: {raw_path}")
        return relative

    async def _run_command_async(
        self,
        command: list[str],
        *,
        cwd: Path,
        language: str,
        entry: str,
        timeout: int,
        workflow_id: str | None,
        task_id: str | None,
    ) -> ExecutionResult:
        start = time.monotonic()
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._executor._build_env(),
        )
        queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()

        async def pump(stream, kind: str, bucket: list[str]) -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip("\n\r")
                bucket.append(decoded)
                await queue.put((kind, decoded))
            await queue.put(None)

        stdout_task = asyncio.create_task(pump(proc.stdout, "output", stdout_lines))
        stderr_task = asyncio.create_task(pump(proc.stderr, "error", stderr_lines))
        finished = 0
        timed_out = False
        deadline = asyncio.create_task(asyncio.sleep(timeout))

        while finished < 2:
            queue_task = asyncio.create_task(queue.get())
            done, pending = await asyncio.wait({queue_task, deadline}, return_when=asyncio.FIRST_COMPLETED)
            if deadline in done:
                timed_out = True
                proc.kill()
                queue_task.cancel()
                break
            payload = queue_task.result()
            if payload is None:
                finished += 1
                continue
            kind, text = payload
            runtime_event_bus.emit(
                {
                    "type": "execution_output",
                    "kind": kind,
                    "data": text,
                    "workflow_id": workflow_id,
                    "task_id": task_id,
                    "tool": "project_executor",
                }
            )

        if not deadline.done():
            deadline.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        await proc.wait()
        elapsed = time.monotonic() - start

        stderr = "\n".join(stderr_lines)[-8000:]
        stdout = "\n".join(stdout_lines)[-8000:]
        if timed_out:
            stderr = (stderr + "\n" if stderr else "") + f"Project execution timed out after {timeout}s"
        return ExecutionResult(
            code=f"entry: {entry}",
            language=language,
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout,
            stderr=stderr,
            success=(proc.returncode == 0 and not timed_out),
            duration_seconds=round(elapsed, 3),
            temp_dir=str(cwd),
        )

    async def _run_with_scaffold_runner(
        self,
        project_dir: Path,
        *,
        timeout: int,
        workflow_id: str | None,
        task_id: str | None,
    ) -> ExecutionResult:
        start = time.monotonic()
        try:
            launched = await asyncio.wait_for(asyncio.to_thread(ScaffoldRunner(project_dir).run), timeout=timeout)
            stdout_text = Path(launched.stdout_log).read_text(encoding="utf-8", errors="replace")[-4000:]
            stderr_text = Path(launched.stderr_log).read_text(encoding="utf-8", errors="replace")[-4000:]
            for line in stdout_text.splitlines()[-20:]:
                runtime_event_bus.emit(
                    {
                        "type": "execution_output",
                        "kind": "output",
                        "data": line,
                        "workflow_id": workflow_id,
                        "task_id": task_id,
                        "tool": "project_executor",
                    }
                )
            for line in stderr_text.splitlines()[-20:]:
                runtime_event_bus.emit(
                    {
                        "type": "execution_output",
                        "kind": "error",
                        "data": line,
                        "workflow_id": workflow_id,
                        "task_id": task_id,
                        "tool": "project_executor",
                    }
                )
            return ExecutionResult(
                code="entry: npm server",
                language="javascript",
                exit_code=0,
                stdout=stdout_text,
                stderr=stderr_text,
                success=True,
                duration_seconds=round(time.monotonic() - start, 3),
                temp_dir=str(project_dir),
                metadata={
                    "port": launched.port,
                    "url": launched.url,
                    "health_url": launched.health_url,
                    "pid": launched.pid,
                    "stdout_log": str(launched.stdout_log),
                    "stderr_log": str(launched.stderr_log),
                },
            )
        except (asyncio.TimeoutError, ScaffoldRunError) as error:
            return ExecutionResult(
                code="entry: npm server",
                language="javascript",
                exit_code=-1,
                stdout="",
                stderr=str(error),
                success=False,
                duration_seconds=round(time.monotonic() - start, 3),
                temp_dir=str(project_dir),
            )

    def _extract_file_content(self, response: str, failing_file: str, fallback: str) -> str:
        artifacts = self._artifact_materializer.extract(response)
        for artifact in artifacts:
            if artifact.relative_path.as_posix() == failing_file or artifact.relative_path.name == Path(failing_file).name:
                return artifact.content
        extracted = self._executor._extract_code(response, Path(failing_file).suffix.lstrip(".") or "text")
        return extracted or fallback

    def _default_fix_target(self, project_dir: Path) -> str | None:
        for candidate in self._detect_entry_point(project_dir), "main.py", "app.py", "index.js", "server.js":
            if candidate and (project_dir / candidate).exists():
                return candidate
        for path in project_dir.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".js", ".ts", ".jsx", ".tsx"}:
                return path.relative_to(project_dir).as_posix()
        return None
