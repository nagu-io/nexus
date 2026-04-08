"""Git integration for autonomous checkpoint, rollback, and diff.

Every fix attempt can be checkpointed so regressions are rollbackable.
The orchestrator auto-checkpoints before every fix cycle and rolls back
if a fix makes things worse.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GitStatus:
    """Snapshot of a git repository's current state."""

    initialized: bool
    clean: bool
    branch: str
    changed_files: list[str] = field(default_factory=list)
    commit_count: int = 0
    head_sha: str = ""
    head_message: str = ""


class GitTool:
    """Safe git operations for autonomous workflows.

    Provides ``init``, ``checkpoint``, ``rollback``, ``diff``, and ``status``
    methods that the orchestrator uses to protect against regressions during
    the write → run → error → fix loop.
    """

    def __init__(self, *, allowed_roots: list[Path] | None = None):
        self._allowed_roots = [Path(r).resolve() for r in allowed_roots] if allowed_roots else None
        self._git = self._find_git()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init(self, project_dir: Path) -> bool:
        """Initialize a git repo if one doesn't exist. Returns True if initialized."""
        project_dir = Path(project_dir)
        self._check_allowed(project_dir)

        if (project_dir / ".git").exists():
            return False

        self._run(["init"], cwd=project_dir)
        self._run(["add", "-A"], cwd=project_dir)
        self._run(["commit", "-m", "nexus: initial checkpoint", "--allow-empty"], cwd=project_dir)
        return True

    def checkpoint(self, project_dir: Path, message: str = "nexus: checkpoint") -> str:
        """Stage all changes and commit. Returns the commit SHA."""
        project_dir = Path(project_dir)
        self._check_allowed(project_dir)
        self._ensure_repo(project_dir)

        self._run(["add", "-A"], cwd=project_dir)

        # Check if there's anything to commit
        status = self._run(["status", "--porcelain"], cwd=project_dir)
        if not status.strip():
            # Nothing to commit — return current HEAD
            return self._run(["rev-parse", "HEAD"], cwd=project_dir).strip()

        self._run(["commit", "-m", message], cwd=project_dir)
        return self._run(["rev-parse", "HEAD"], cwd=project_dir).strip()

    def rollback(self, project_dir: Path, steps: int = 1) -> str:
        """Roll back to a previous commit. Returns the new HEAD SHA."""
        project_dir = Path(project_dir)
        self._check_allowed(project_dir)
        self._ensure_repo(project_dir)

        self._run(["reset", "--hard", f"HEAD~{steps}"], cwd=project_dir)
        return self._run(["rev-parse", "HEAD"], cwd=project_dir).strip()

    def rollback_to(self, project_dir: Path, target: str) -> str:
        """Roll back working tree changes to a specific commit SHA."""
        project_dir = Path(project_dir)
        self._check_allowed(project_dir)
        self._ensure_repo(project_dir)

        self._run(["reset", "--hard", target], cwd=project_dir)
        return self._run(["rev-parse", "HEAD"], cwd=project_dir).strip()

    def diff(self, project_dir: Path, staged: bool = False) -> str:
        """Return the diff of working (or staged) changes."""
        project_dir = Path(project_dir)
        self._check_allowed(project_dir)

        cmd = ["diff"]
        if staged:
            cmd.append("--staged")
        return self._run(cmd, cwd=project_dir)

    def status(self, project_dir: Path) -> GitStatus:
        """Return a structured snapshot of the repo state."""
        project_dir = Path(project_dir)
        self._check_allowed(project_dir)

        if not (project_dir / ".git").exists():
            return GitStatus(
                initialized=False,
                clean=True,
                branch="",
            )

        porcelain = self._run(["status", "--porcelain"], cwd=project_dir)
        changed = [line[3:] for line in porcelain.strip().splitlines() if line.strip()]

        branch = self._run(["branch", "--show-current"], cwd=project_dir).strip()

        try:
            head_sha = self._run(["rev-parse", "HEAD"], cwd=project_dir).strip()
            head_message = self._run(["log", "-1", "--format=%s"], cwd=project_dir).strip()
            commit_count = int(self._run(["rev-list", "--count", "HEAD"], cwd=project_dir).strip())
        except Exception:
            head_sha = ""
            head_message = ""
            commit_count = 0

        return GitStatus(
            initialized=True,
            clean=len(changed) == 0,
            branch=branch or "main",
            changed_files=changed,
            commit_count=commit_count,
            head_sha=head_sha,
            head_message=head_message,
        )

    def log(self, project_dir: Path, limit: int = 10) -> list[dict[str, str]]:
        """Return recent commits as a list of ``{sha, message, date}`` dicts."""
        project_dir = Path(project_dir)
        self._check_allowed(project_dir)
        self._ensure_repo(project_dir)

        raw = self._run(
            ["log", f"-{limit}", "--format=%H|||%s|||%aI"],
            cwd=project_dir,
        )
        entries = []
        for line in raw.strip().splitlines():
            parts = line.split("|||")
            if len(parts) == 3:
                entries.append({"sha": parts[0], "message": parts[1], "date": parts[2]})
        return entries

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        """Tool dispatch interface for the orchestrator."""
        args = dict(request.get("arguments") or request.get("args") or {})
        action = args.get("action", "status")
        project_dir = Path(args.get("project_dir") or args.get("cwd") or ".")

        try:
            if action == "init":
                created = self.init(project_dir)
                return {
                    "ok": True,
                    "tool": "git_tool",
                    "action": action,
                    "summary": f"Git repo {'initialized' if created else 'already exists'}",
                }
            elif action == "checkpoint":
                message = args.get("message", "nexus: checkpoint")
                sha = self.checkpoint(project_dir, message)
                return {
                    "ok": True,
                    "tool": "git_tool",
                    "action": action,
                    "sha": sha,
                    "summary": f"Checkpointed at {sha[:8]}",
                }
            elif action == "rollback":
                steps = int(args.get("steps", 1))
                sha = self.rollback(project_dir, steps)
                return {
                    "ok": True,
                    "tool": "git_tool",
                    "action": action,
                    "sha": sha,
                    "summary": f"Rolled back to {sha[:8]}",
                }
            elif action == "rollback_to":
                target = str(args.get("target") or "").strip()
                if not target:
                    raise ValueError("target commit is required for rollback_to")
                sha = self.rollback_to(project_dir, target)
                return {
                    "ok": True,
                    "tool": "git_tool",
                    "action": action,
                    "sha": sha,
                    "summary": f"Rolled back to {sha[:8]}",
                }
            elif action == "diff":
                diff_text = self.diff(project_dir, staged=args.get("staged", False))
                return {
                    "ok": True,
                    "tool": "git_tool",
                    "action": action,
                    "diff": diff_text[:8000],
                    "summary": f"Diff: {len(diff_text)} chars",
                }
            elif action == "status":
                st = self.status(project_dir)
                return {
                    "ok": True,
                    "tool": "git_tool",
                    "action": action,
                    "initialized": st.initialized,
                    "clean": st.clean,
                    "branch": st.branch,
                    "changed_files": st.changed_files,
                    "summary": (
                        f"{'Clean' if st.clean else f'{len(st.changed_files)} changed'} "
                        f"on {st.branch} ({st.commit_count} commits)"
                        if st.initialized
                        else "Not a git repository"
                    ),
                }
            else:
                return {
                    "ok": False,
                    "tool": "git_tool",
                    "action": action,
                    "summary": f"Unknown git action: {action}",
                }
        except Exception as error:
            return {
                "ok": False,
                "tool": "git_tool",
                "action": action,
                "summary": f"Git error: {error}",
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_git(self) -> str:
        """Locate the git executable."""
        git = shutil.which("git")
        if git:
            return git
        # Common fallback locations on Windows
        if os.name == "nt":
            for candidate in [
                r"C:\Program Files\Git\bin\git.exe",
                r"C:\Program Files (x86)\Git\bin\git.exe",
            ]:
                if os.path.isfile(candidate):
                    return candidate
        return "git"  # Let subprocess fail with a clear error

    def _run(self, args: list[str], cwd: Path) -> str:
        """Run a git command and return stdout."""
        result = subprocess.run(
            [self._git] + args,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=30,
            shell=False,
            env=self._build_env(),
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout

    def _build_env(self) -> dict[str, str]:
        """Build env with git identity for automated commits."""
        env = os.environ.copy()
        env.setdefault("GIT_AUTHOR_NAME", "NEXUS")
        env.setdefault("GIT_AUTHOR_EMAIL", "nexus@local")
        env.setdefault("GIT_COMMITTER_NAME", "NEXUS")
        env.setdefault("GIT_COMMITTER_EMAIL", "nexus@local")
        return env

    def _check_allowed(self, path: Path) -> None:
        """Verify path is within allowed roots (security)."""
        if not self._allowed_roots:
            return
        resolved = path.resolve()
        if not any(resolved == root or root in resolved.parents for root in self._allowed_roots):
            raise PermissionError(f"Path {path} is outside allowed roots")

    def _ensure_repo(self, project_dir: Path) -> None:
        """Ensure the directory is a git repository."""
        if not (project_dir / ".git").exists():
            self.init(project_dir)
