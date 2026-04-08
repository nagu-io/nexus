"""Offline project scanner for local project intelligence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any


class CodeReader:
    """Scan a local project directory and summarize its stack and structure."""

    IGNORED_DIRS = {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        "coverage",
        ".next",
        ".turbo",
    }
    LANGUAGE_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".json": "json",
        ".html": "html",
        ".css": "css",
        ".md": "markdown",
        ".toml": "toml",
        ".yml": "yaml",
        ".yaml": "yaml",
    }

    def scan(self, project_dir: str | Path) -> dict[str, Any]:
        """Return a compact project summary for planning and runtime prompts."""
        root_dir = Path(project_dir).expanduser()
        if not root_dir.exists():
            raise FileNotFoundError(f"Project directory does not exist: {root_dir}")
        if root_dir.is_file():
            root_dir = root_dir.parent
        root_dir = root_dir.resolve()

        package_json = self._read_package_json(root_dir / "package.json")
        pyproject = self._read_pyproject(root_dir / "pyproject.toml")
        requirements = self._read_requirements(root_dir / "requirements.txt")

        dependencies = set(package_json["dependencies"]) | set(package_json["dev_dependencies"]) | set(requirements)
        frameworks = self._detect_frameworks(
            root_dir=root_dir,
            dependencies=dependencies,
            package_json=package_json,
            pyproject=pyproject,
        )
        files, directories, language_counts, entrypoints = self._scan_tree(root_dir)
        languages = sorted(language_counts, key=lambda name: (-language_counts[name], name))
        scripts = dict(package_json["scripts"])
        package_manager = self._detect_package_manager(root_dir)

        payload = {
            "project_root": str(root_dir),
            "project_name": root_dir.name,
            "project_signature": self._signature_for(
                root_dir=root_dir,
                frameworks=frameworks,
                languages=languages,
                files=files,
                dependencies=sorted(dependencies),
            ),
            "frameworks": frameworks,
            "languages": languages,
            "language_counts": language_counts,
            "package_manager": package_manager,
            "scripts": scripts,
            "entrypoints": entrypoints[:10],
            "directories": directories[:20],
            "files": files[:40],
            "dependencies": sorted(dependencies)[:60],
            "summary_text": self._build_summary_text(
                root_dir=root_dir,
                frameworks=frameworks,
                languages=languages,
                entrypoints=entrypoints,
            ),
        }
        return payload

    def _scan_tree(self, root_dir: Path) -> tuple[list[dict[str, Any]], list[str], dict[str, int], list[str]]:
        files: list[dict[str, Any]] = []
        directories: set[str] = set()
        language_counts: dict[str, int] = {}
        entrypoints: list[str] = []

        for path in sorted(root_dir.rglob("*")):
            relative_parts = path.relative_to(root_dir).parts
            if any(part in self.IGNORED_DIRS for part in relative_parts):
                continue
            if path.is_dir():
                directories.add(path.relative_to(root_dir).as_posix())
                continue
            relative = path.relative_to(root_dir).as_posix()
            suffix = path.suffix.lower()
            language = self.LANGUAGE_EXTENSIONS.get(suffix)
            if language:
                language_counts[language] = language_counts.get(language, 0) + 1
            if self._is_entrypoint(relative):
                entrypoints.append(relative)
            if len(files) < 120:
                files.append(
                    {
                        "path": relative,
                        "size": path.stat().st_size,
                    }
                )

        return files, sorted(directories), language_counts, entrypoints

    def _detect_frameworks(
        self,
        *,
        root_dir: Path,
        dependencies: set[str],
        package_json: dict[str, Any],
        pyproject: dict[str, Any],
    ) -> list[str]:
        package_names = {name.lower() for name in dependencies}
        frameworks = set()

        dependency_rules = {
            "react": {"react"},
            "vite": {"vite"},
            "express": {"express"},
            "nextjs": {"next"},
            "fastapi": {"fastapi"},
            "flask": {"flask"},
            "django": {"django"},
            "pytest": {"pytest"},
            "tailwind": {"tailwindcss"},
        }
        for framework, signals in dependency_rules.items():
            if package_names & signals:
                frameworks.add(framework)

        if (root_dir / "manage.py").exists():
            frameworks.add("django")
        if (root_dir / "next.config.js").exists() or (root_dir / "next.config.mjs").exists():
            frameworks.add("nextjs")
        if (root_dir / "vite.config.js").exists() or (root_dir / "vite.config.ts").exists():
            frameworks.add("vite")
        if (root_dir / "tailwind.config.js").exists() or (root_dir / "tailwind.config.ts").exists():
            frameworks.add("tailwind")
        if pyproject.get("project", {}).get("dependencies"):
            dependencies_text = " ".join(pyproject["project"]["dependencies"]).lower()
            if "fastapi" in dependencies_text:
                frameworks.add("fastapi")
            if "flask" in dependencies_text:
                frameworks.add("flask")
            if "django" in dependencies_text:
                frameworks.add("django")
            if "pytest" in dependencies_text:
                frameworks.add("pytest")
        if package_json["scripts"].get("dev") and "vite" in package_json["scripts"]["dev"]:
            frameworks.add("vite")

        return sorted(frameworks)

    def _detect_package_manager(self, root_dir: Path) -> str | None:
        if (root_dir / "pnpm-lock.yaml").exists():
            return "pnpm"
        if (root_dir / "package-lock.json").exists():
            return "npm"
        if (root_dir / "yarn.lock").exists():
            return "yarn"
        return None

    def _read_package_json(self, package_path: Path) -> dict[str, Any]:
        if not package_path.exists():
            return {"dependencies": {}, "dev_dependencies": {}, "scripts": {}}
        try:
            payload = json.loads(package_path.read_text(encoding="utf-8"))
        except Exception:
            return {"dependencies": {}, "dev_dependencies": {}, "scripts": {}}
        return {
            "dependencies": dict(payload.get("dependencies") or {}),
            "dev_dependencies": dict(payload.get("devDependencies") or {}),
            "scripts": dict(payload.get("scripts") or {}),
        }

    def _read_pyproject(self, pyproject_path: Path) -> dict[str, Any]:
        if not pyproject_path.exists():
            return {}
        try:
            return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _read_requirements(self, requirements_path: Path) -> list[str]:
        if not requirements_path.exists():
            return []
        entries = []
        for line in requirements_path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            name = item.split("==", 1)[0].split(">=", 1)[0].split("[", 1)[0].strip()
            if name:
                entries.append(name)
        return entries

    def _signature_for(
        self,
        *,
        root_dir: Path,
        frameworks: list[str],
        languages: list[str],
        files: list[dict[str, Any]],
        dependencies: list[str],
    ) -> str:
        payload = {
            "root": str(root_dir),
            "frameworks": frameworks,
            "languages": languages[:4],
            "files": [file_info["path"] for file_info in files[:25]],
            "dependencies": dependencies[:25],
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _build_summary_text(
        self,
        *,
        root_dir: Path,
        frameworks: list[str],
        languages: list[str],
        entrypoints: list[str],
    ) -> str:
        framework_text = ", ".join(frameworks) if frameworks else "no explicit framework detected"
        language_text = ", ".join(languages[:4]) if languages else "unknown languages"
        entrypoint_text = ", ".join(entrypoints[:5]) if entrypoints else "no obvious entrypoints"
        return (
            f"Project {root_dir.name} at {root_dir} uses {framework_text}. "
            f"Primary languages: {language_text}. "
            f"Likely entrypoints: {entrypoint_text}."
        )

    def _is_entrypoint(self, relative_path: str) -> bool:
        lower = relative_path.lower()
        return any(
            lower.endswith(candidate)
            for candidate in (
                "main.py",
                "app.py",
                "server.js",
                "server.ts",
                "src/app.jsx",
                "src/app.tsx",
                "src/main.jsx",
                "src/main.tsx",
                "index.js",
                "index.ts",
                "index.html",
            )
        )
