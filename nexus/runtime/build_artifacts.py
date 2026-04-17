"""Helpers for turning generated scaffold output into real files on disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re


class BuildArtifactError(ValueError):
    """Raised when generated output cannot be materialized safely."""


@dataclass
class BuildArtifact:
    """One extracted file artifact from a generated response."""

    relative_path: Path
    content: str
    language: str = ""


@dataclass
class MaterializationResult:
    """Summary of files written to disk."""

    root_dir: Path
    files_written: list[Path]
    overwritten_files: list[Path]


class BuildArtifactMaterializer:
    """Extract fenced file blocks and write them into a target directory."""

    FILE_BLOCK_PATTERN = re.compile(
        r"(?ms)^(?:`(?P<path_backtick>[^`\r\n]+)`|\*\*(?P<path_bold>[^*\r\n]+)\*\*)\s*\r?\n```(?P<language>[^\r\n]*)\r?\n(?P<content>.*?)\r?\n```"
    )
    FENCED_PATH_BLOCK_PATTERN = re.compile(
        r"(?ms)^```[^\r\n]*\r?\n(?P<path>[^\r\n]+)\r?\n```\s*\r?\n```(?P<language>[^\r\n]*)\r?\n(?P<content>.*?)\r?\n```"
    )
    FILE_TREE_PATTERN = re.compile(
        r"(?ms)^File Tree:\s*\r?\n```text\r?\n(?P<tree>.*?)\r?\n```"
    )

    def extract(self, output: str) -> list[BuildArtifact]:
        """Extract path-tagged fenced code blocks from a model response."""
        normalized_output = self._normalize_file_block_formats(output or "")
        artifacts: list[BuildArtifact] = []
        for match in self.FILE_BLOCK_PATTERN.finditer(normalized_output):
            raw_path = match.group("path_backtick") or match.group("path_bold") or ""
            relative_path = self._normalize_relative_path(raw_path)
            artifacts.append(
                BuildArtifact(
                    relative_path=relative_path,
                    content=match.group("content"),
                    language=(match.group("language") or "").strip(),
                )
            )
        return artifacts

    def _normalize_file_block_formats(self, output: str) -> str:
        def replace_fenced_path(match: re.Match[str]) -> str:
            raw_path = (match.group("path") or "").strip()
            try:
                self._normalize_relative_path(raw_path)
            except BuildArtifactError:
                return match.group(0)
            language = (match.group("language") or "").strip()
            content = match.group("content")
            return f"`{raw_path}`\n```{language}\n{content}\n```"

        return self.FENCED_PATH_BLOCK_PATTERN.sub(replace_fenced_path, output)

    def default_output_dir(self, *, goal: str, output: str, base_dir: Path) -> Path:
        """Choose a predictable default target directory for a build scaffold."""
        root_name = self._root_name_from_file_tree(output) or self._slugify(goal)
        return base_dir / "generated" / root_name

    def materialize(
        self,
        *,
        output: str,
        target_dir: Path,
        overwrite: bool = False,
    ) -> MaterializationResult:
        """Write extracted file artifacts into the chosen target directory."""
        artifacts = self.extract(output)
        if not artifacts:
            raise BuildArtifactError(
                "No file artifacts were found in the result. Use --write only when the run returns fenced source files."
            )

        resolved_target = Path(target_dir).expanduser().resolve()
        existing_files: list[Path] = []
        for artifact in artifacts:
            destination = resolved_target / artifact.relative_path
            if destination.exists():
                existing_files.append(destination)

        if existing_files and not overwrite:
            sample = existing_files[0]
            raise BuildArtifactError(
                f"Refusing to overwrite existing files in {resolved_target}. "
                f"First conflict: {sample}. Re-run with --force or choose --output-dir."
            )

        written_files: list[Path] = []
        for artifact in artifacts:
            destination = resolved_target / artifact.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(artifact.content, encoding="utf-8")
            written_files.append(destination)

        return MaterializationResult(
            root_dir=resolved_target,
            files_written=written_files,
            overwritten_files=existing_files,
        )

    def _root_name_from_file_tree(self, output: str) -> str | None:
        """Extract the top-level folder name from the rendered file tree when present."""
        match = self.FILE_TREE_PATTERN.search(output or "")
        if not match:
            return None

        for raw_line in match.group("tree").splitlines():
            cleaned = raw_line.strip()
            if not cleaned:
                continue
            cleaned = cleaned.lstrip("|`- ")
            cleaned = cleaned.rstrip("/")
            if cleaned:
                return self._slugify(cleaned)
        return None

    def _normalize_relative_path(self, raw_path: str) -> Path:
        normalized = (raw_path or "").strip().replace("\\", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        posix_path = PurePosixPath(normalized)
        if not posix_path.parts or posix_path.is_absolute() or ".." in posix_path.parts:
            raise BuildArtifactError(f"Unsafe generated file path: {raw_path}")
        return Path(*posix_path.parts)

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
        return slug[:60] or "nexus-build"
