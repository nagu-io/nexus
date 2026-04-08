"""Install and launch generated scaffold projects locally."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


class ScaffoldRunError(RuntimeError):
    """Raised when a generated scaffold cannot be launched safely."""


@dataclass
class ScaffoldRunPlan:
    """Prepared launch plan for a materialized scaffold."""

    root_dir: Path
    port: int
    url: str
    health_url: str
    install_required: bool
    npm_command: str
    launch_script: str
    stdout_log: Path
    stderr_log: Path
    metadata_path: Path


@dataclass
class ScaffoldRunResult:
    """Runtime details for a launched scaffold."""

    root_dir: Path
    port: int
    url: str
    health_url: str
    pid: int
    stdout_log: Path
    stderr_log: Path
    metadata_path: Path
    install_performed: bool


class ScaffoldRunner:
    """Prepare, install, and launch generated local scaffold projects."""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir).expanduser().resolve()

    def prepare(self, preferred_port: int | None = None) -> ScaffoldRunPlan:
        """Build a launch plan for an npm-based generated project."""
        package_json = self.root_dir / "package.json"
        if not package_json.exists():
            raise ScaffoldRunError(
                f"No package.json was found in {self.root_dir}. Auto-run currently supports npm-based scaffolds."
            )

        manifest = json.loads(package_json.read_text(encoding="utf-8"))
        scripts = dict(manifest.get("scripts", {}))
        launch_script = "dev" if "dev" in scripts else "start" if "start" in scripts else ""
        if not launch_script:
            raise ScaffoldRunError(
                f"{package_json} does not define a dev or start script, so NEXUS does not know how to launch it."
            )

        npm_command = self._npm_command()
        port = self._pick_port(preferred_port)
        runtime_dir = self.root_dir / ".nexus"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        return ScaffoldRunPlan(
            root_dir=self.root_dir,
            port=port,
            url=f"http://127.0.0.1:{port}",
            health_url=f"http://127.0.0.1:{port}/health",
            install_required=not (self.root_dir / "node_modules").exists(),
            npm_command=npm_command,
            launch_script=launch_script,
            stdout_log=runtime_dir / "run.stdout.log",
            stderr_log=runtime_dir / "run.stderr.log",
            metadata_path=runtime_dir / "run.json",
        )

    def run(self, preferred_port: int | None = None) -> ScaffoldRunResult:
        """Install dependencies if needed, launch the scaffold, and wait until it is reachable."""
        plan = self.prepare(preferred_port=preferred_port)
        if plan.install_required:
            self._install_dependencies(plan)
        return self._launch(plan)

    def _install_dependencies(self, plan: ScaffoldRunPlan) -> None:
        install_command = [plan.npm_command, "install"]
        completed = subprocess.run(
            install_command,
            cwd=plan.root_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ScaffoldRunError(
                "Dependency installation failed.\n"
                f"stdout:\n{completed.stdout[-1200:]}\n"
                f"stderr:\n{completed.stderr[-1200:]}"
            )

    def _launch(self, plan: ScaffoldRunPlan) -> ScaffoldRunResult:
        env = os.environ.copy()
        env["PORT"] = str(plan.port)

        with plan.stdout_log.open("w", encoding="utf-8") as stdout_handle, plan.stderr_log.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            process = subprocess.Popen(
                [plan.npm_command, "run", plan.launch_script],
                cwd=plan.root_dir,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                stdin=subprocess.DEVNULL,
                text=True,
                creationflags=self._creationflags(),
                start_new_session=os.name != "nt",
            )

        self._wait_until_ready(process, plan)
        result = ScaffoldRunResult(
            root_dir=plan.root_dir,
            port=plan.port,
            url=plan.url,
            health_url=plan.health_url,
            pid=process.pid,
            stdout_log=plan.stdout_log,
            stderr_log=plan.stderr_log,
            metadata_path=plan.metadata_path,
            install_performed=plan.install_required,
        )
        plan.metadata_path.write_text(
            json.dumps(
                {
                    **asdict(result),
                    "root_dir": str(result.root_dir),
                    "stdout_log": str(result.stdout_log),
                    "stderr_log": str(result.stderr_log),
                    "metadata_path": str(result.metadata_path),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "launch_script": plan.launch_script,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result

    def _wait_until_ready(self, process: subprocess.Popen[str], plan: ScaffoldRunPlan) -> None:
        deadline = time.time() + 25
        while time.time() < deadline:
            if process.poll() is not None:
                stderr_text = plan.stderr_log.read_text(encoding="utf-8", errors="replace") if plan.stderr_log.exists() else ""
                stdout_text = plan.stdout_log.read_text(encoding="utf-8", errors="replace") if plan.stdout_log.exists() else ""
                raise ScaffoldRunError(
                    "The generated scaffold exited before it became ready.\n"
                    f"stdout:\n{stdout_text[-1200:]}\n"
                    f"stderr:\n{stderr_text[-1200:]}"
                )
            if self._port_open(plan.port):
                return
            time.sleep(0.4)
        raise ScaffoldRunError(
            f"The generated scaffold did not become reachable on port {plan.port}. "
            f"Check {plan.stdout_log} and {plan.stderr_log} for details."
        )

    def _pick_port(self, preferred_port: int | None) -> int:
        if preferred_port is not None:
            if self._port_open(preferred_port):
                raise ScaffoldRunError(f"Port {preferred_port} is already in use.")
            return preferred_port

        for port in range(3010, 3040):
            if not self._port_open(port):
                return port
        raise ScaffoldRunError("Could not find a free local port in the range 3010-3039.")

    def _port_open(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.3)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    def _npm_command(self) -> str:
        candidates = ["npm.cmd", "npm"] if os.name == "nt" else ["npm"]
        for candidate in candidates:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        raise ScaffoldRunError("npm was not found on PATH, so the generated scaffold cannot be launched.")

    def _creationflags(self) -> int:
        if os.name != "nt":
            return 0
        return subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
