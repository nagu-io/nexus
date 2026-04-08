"""Prompt-sized context reduction helpers for long NEXUS agent runs."""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextReductionResult:
    """Structured output from one context-reduction pass."""

    text: str
    reduced: bool
    backend: str
    strategy: str
    original_length: int
    reduced_length: int
    omitted_sections: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reduced": self.reduced,
            "backend": self.backend,
            "strategy": self.strategy,
            "original_length": self.original_length,
            "reduced_length": self.reduced_length,
            "omitted_sections": self.omitted_sections,
            "metadata": dict(self.metadata),
        }


class BaseContextReducer(ABC):
    """Base interface for reducers that shrink oversized agent context to text."""

    backend_name = "base"

    def __init__(
        self,
        *,
        threshold_chars: int = 12000,
        target_chars: int = 6000,
    ):
        self.threshold_chars = max(128, int(threshold_chars))
        self.target_chars = max(96, int(target_chars))
        if self.target_chars >= self.threshold_chars:
            self.target_chars = max(64, self.threshold_chars - 32)

    def reduce(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ContextReductionResult:
        payload = str(text or "")
        extra = dict(metadata or {})
        if len(payload) <= self.threshold_chars:
            return ContextReductionResult(
                text=payload,
                reduced=False,
                backend=self.backend_name,
                strategy="pass_through",
                original_length=len(payload),
                reduced_length=len(payload),
                metadata=extra,
            )

        reduced_text, reduction_meta = self._reduce_impl(payload, metadata=extra)
        merged_meta = dict(extra)
        merged_meta.update(reduction_meta)
        final_text = self._hard_cap(reduced_text, self.target_chars)
        return ContextReductionResult(
            text=final_text,
            reduced=final_text != payload,
            backend=self.backend_name,
            strategy=str(merged_meta.get("strategy") or "reduced"),
            original_length=len(payload),
            reduced_length=len(final_text),
            omitted_sections=int(merged_meta.get("omitted_sections", 0) or 0),
            metadata=merged_meta,
        )

    @abstractmethod
    def _reduce_impl(
        self,
        text: str,
        *,
        metadata: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Return reduced text plus metadata about the reduction."""

    def _hard_cap(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        if limit <= 128:
            return text[:limit]
        marker_template = "\n\n[... {count} chars omitted ...]\n\n"
        available = limit - len(marker_template.format(count=0))
        if available <= 64:
            return text[:limit]
        head_budget = max(48, int(available * 0.68))
        tail_budget = max(24, available - head_budget)
        omitted = max(0, len(text) - head_budget - tail_budget)
        marker = marker_template.format(count=omitted)
        available = limit - len(marker)
        if available <= 64:
            return text[:limit]
        head_budget = max(48, int(available * 0.68))
        tail_budget = max(24, available - head_budget)
        head = text[:head_budget].rstrip()
        tail = text[-tail_budget:].lstrip()
        capped = f"{head}{marker}{tail}"
        return capped[:limit]


class HeuristicContextReducer(BaseContextReducer):
    """Reduce long context by preserving high-signal sections and trimming lists."""

    backend_name = "heuristic"
    _CRITICAL_PREFIXES = (
        "workflow goal:",
        "primary intent:",
        "task type:",
        "task:",
        "fallback strategy:",
        "minimum confidence required:",
        "adaptive retry strategy:",
        "retry note:",
    )
    _HIGH_PREFIXES = (
        "terminal feedback:",
        "shared memory context:",
        "common project errors:",
        "constraints:",
    )
    _MEDIUM_PREFIXES = (
        "project context:",
        "project summary:",
        "project files:",
        "workspace files:",
        "recent project goals:",
        "user preferences:",
        "last command:",
    )

    def _reduce_impl(
        self,
        text: str,
        *,
        metadata: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        sections = [section.strip() for section in text.split("\n\n") if section.strip()]
        if not sections:
            return self._hard_cap(text, self.target_chars), {"strategy": "hard_cap"}

        items: list[dict[str, Any]] = []
        for index, section in enumerate(sections):
            priority = self._priority(section, index)
            items.append(
                {
                    "index": index,
                    "priority": priority,
                    "full": section,
                    "compressed": self._compress_section(section, priority),
                }
            )

        reserve_for_note = min(180, max(80, self.target_chars // 3))
        section_budget = max(64, self.target_chars - reserve_for_note)
        selected: dict[int, str] = {}
        used = 0
        for minimum in (90, 70, 40, 10):
            for item in items:
                if item["priority"] < minimum or item["index"] in selected:
                    continue
                candidate = item["full"] if item["priority"] >= 90 else item["compressed"]
                projected = used + len(candidate) + 2
                if projected <= section_budget or item["priority"] >= 90:
                    selected[item["index"]] = candidate
                    used = projected

        if not selected:
            reduced = self._hard_cap(text, self.target_chars)
            return reduced, {"strategy": "hard_cap"}

        omitted_sections = max(0, len(sections) - len(selected))
        reduced = self._assemble_output(items, selected, omitted_sections)
        while len(reduced) > self.target_chars:
            droppable = [
                item
                for item in reversed(items)
                if item["index"] in selected and item["priority"] < 90
            ]
            if not droppable:
                break
            victim = droppable[0]
            selected.pop(victim["index"], None)
            omitted_sections += 1
            reduced = self._assemble_output(items, selected, omitted_sections)
        return reduced, {
            "strategy": "priority_trim",
            "omitted_sections": omitted_sections,
            "selected_sections": len(selected),
            "total_sections": len(sections),
        }

    def _priority(self, section: str, index: int) -> int:
        header = section.splitlines()[0].strip().lower()
        if index < 4:
            return 95
        if header.startswith(self._CRITICAL_PREFIXES):
            return 95
        if header.startswith(self._HIGH_PREFIXES):
            return 75
        if header.startswith(self._MEDIUM_PREFIXES):
            return 45
        return 15

    def _compress_section(self, section: str, priority: int) -> str:
        if len(section) <= 900 and priority >= 70:
            return section

        lines = [line.rstrip() for line in section.splitlines()]
        if not lines:
            return section
        header = lines[0]
        body = lines[1:]

        if header.lower().startswith(("workspace files:", "project files:", "recent project goals:")):
            kept = body[:8]
            omitted = max(0, len(body) - len(kept))
            if omitted:
                kept.append(f"... {omitted} more line(s) omitted")
            return "\n".join([header, *kept]).strip()

        if header.lower().startswith("shared memory context:"):
            return self._hard_cap(section, min(1500, max(900, self.target_chars // 3)))

        if header.lower().startswith(("terminal feedback:", "common project errors:")):
            return self._hard_cap(section, min(1200, max(700, self.target_chars // 4)))

        if not body:
            return self._hard_cap(section, 320)

        compact = "\n".join([header, *body[:6]])
        if len(compact) > 800:
            return self._hard_cap(compact, 800)
        if len(body) > 6:
            compact += "\n... section trimmed"
        return compact

    def _assemble_output(
        self,
        items: list[dict[str, Any]],
        selected: dict[int, str],
        omitted_sections: int,
    ) -> str:
        ordered_sections = [selected[item["index"]] for item in items if item["index"] in selected]
        note = "Context reducer note: long context was summarized to fit the runtime prompt budget."
        if omitted_sections:
            note += f" Omitted {omitted_sections} lower-priority section(s)."
        return "\n\n".join([note, *ordered_sections])


def build_context_reducer(
    *,
    enabled: bool,
    backend: str = "heuristic",
    threshold_chars: int = 12000,
    target_chars: int = 6000,
    model_name: str = "",
) -> BaseContextReducer | None:
    """Build the configured reducer, falling back safely when needed."""
    if not enabled:
        return None

    selected_backend = str(backend or "heuristic").strip().lower()
    if selected_backend == "mamba":
        mamba_available = importlib.util.find_spec("mamba_ssm") is not None
        if not mamba_available or not model_name.strip():
            selected_backend = "heuristic"

    if selected_backend != "heuristic":
        selected_backend = "heuristic"

    return HeuristicContextReducer(
        threshold_chars=threshold_chars,
        target_chars=target_chars,
    )
