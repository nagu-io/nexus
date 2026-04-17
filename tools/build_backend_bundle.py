"""Build the packaged NEXUS backend sidecar for the Electron desktop app."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESKTOP_DIR = ROOT / "desktop"
ARTIFACT_ROOT = DESKTOP_DIR / ".artifacts"
BACKEND_DIST_ROOT = ARTIFACT_ROOT / "backend"
PYINSTALLER_WORK = ARTIFACT_ROOT / "pyinstaller-work"
PYINSTALLER_SPEC = ARTIFACT_ROOT / "pyinstaller-spec"
ENTRYPOINT = DESKTOP_DIR / "backend_entry.py"


def main() -> int:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    BACKEND_DIST_ROOT.mkdir(parents=True, exist_ok=True)

    if shutil.which(sys.executable) is None:
        raise SystemExit("Python executable is not available for backend bundling.")

    _run_pyinstaller()
    _write_manifest()
    print(f"Backend bundle ready at {BACKEND_DIST_ROOT / 'nexus-backend'}")
    return 0


def _run_pyinstaller() -> None:
    if BACKEND_DIST_ROOT.exists():
        shutil.rmtree(BACKEND_DIST_ROOT)
    if PYINSTALLER_WORK.exists():
        shutil.rmtree(PYINSTALLER_WORK)
    if PYINSTALLER_SPEC.exists():
        shutil.rmtree(PYINSTALLER_SPEC)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        "nexus-backend",
        "--distpath",
        str(BACKEND_DIST_ROOT),
        "--workpath",
        str(PYINSTALLER_WORK),
        "--specpath",
        str(PYINSTALLER_SPEC),
        "--paths",
        str(ROOT),
        "--collect-submodules",
        "uvicorn",
        "--collect-submodules",
        "nexus",
        str(ENTRYPOINT),
    ]

    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit("PyInstaller backend bundling failed.")


def _write_manifest() -> None:
    executable_name = "nexus-backend.exe" if sys.platform.startswith("win") else "nexus-backend"
    payload = {
      "entrypoint": str(ENTRYPOINT),
      "dist_root": str(BACKEND_DIST_ROOT / "nexus-backend"),
      "executable": executable_name,
      "python": sys.executable,
    }
    manifest_path = BACKEND_DIST_ROOT / "backend-bundle.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
