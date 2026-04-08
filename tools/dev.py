"""Run the NEXUS API and dashboard together from the repo root."""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _npm_command() -> str | None:
    candidates = ["npm.cmd", "npm"] if os.name == "nt" else ["npm"]
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _check_prerequisites(frontend_port: int) -> list[str]:
    issues: list[str] = []
    if _npm_command() is None:
        issues.append("npm was not found on PATH")
    vite_path = ROOT / "dashboard" / "node_modules" / "vite" / "bin" / "vite.js"
    if not vite_path.exists():
        issues.append("dashboard dependencies are missing; run `npm run dashboard:install`")
    if _port_open(8000):
        issues.append("port 8000 is already in use")
    if _port_open(frontend_port):
        issues.append(f"port {frontend_port} is already in use")
    return issues


def _spawn_processes(frontend_port: int) -> tuple[subprocess.Popen[str], subprocess.Popen[str]]:
    backend_cmd = [sys.executable, "-m", "uvicorn", "nexus.api:app", "--port", "8000"]
    npm = _npm_command()
    if npm is None:
        raise RuntimeError("npm was not found on PATH")
    frontend_cmd = [npm, "run", "dev", "--prefix", "dashboard", "--", "--host", "0.0.0.0", "--port", str(frontend_port)]

    frontend_env = os.environ.copy()
    frontend_env["VITE_API_URL"] = "http://localhost:8000"

    backend = subprocess.Popen(backend_cmd, cwd=ROOT, text=True)
    try:
        frontend = subprocess.Popen(frontend_cmd, cwd=ROOT, env=frontend_env, text=True)
    except Exception:
        if backend.poll() is None:
            backend.terminate()
            try:
                backend.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend.kill()
        raise
    return backend, frontend


def _terminate(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 5
    for process in processes:
        while process.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            process.kill()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the NEXUS backend and dashboard together.")
    parser.add_argument("--frontend-port", type=int, default=3000, help="Port for the Vite dashboard")
    parser.add_argument("--check", action="store_true", help="Validate local prerequisites without starting processes")
    args = parser.parse_args(argv)

    issues = _check_prerequisites(args.frontend_port)
    if issues:
        print("NEXUS dev environment is not ready:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Starting NEXUS local dev environment")
    print("- API: http://localhost:8000")
    print(f"- Dashboard: http://localhost:{args.frontend_port}")

    if args.check:
        print("Prerequisites look good.")
        return 0

    backend, frontend = _spawn_processes(args.frontend_port)
    processes = [backend, frontend]
    try:
        while True:
            exited = next((process for process in processes if process.poll() is not None), None)
            if exited is not None:
                code = exited.returncode or 0
                if code != 0:
                    print(f"A dev process exited early with code {code}.")
                return code
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping NEXUS dev environment...")
        return 0
    finally:
        _terminate(processes)


if __name__ == "__main__":
    raise SystemExit(main())
