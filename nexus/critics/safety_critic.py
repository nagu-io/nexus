"""Safety critic for risky or irreversible guidance."""

from __future__ import annotations

from typing import Any

from nexus.critics.base import BaseCritic


class SafetyCritic(BaseCritic):
    """Detect unsafe instructions that should trigger a stricter retry path."""

    name = "safety"
    weight = 0.25

    _MARKERS = {
        "destructive_command": (
            "rm -rf",
            "remove-item -recurse -force",
            "git reset --hard",
            "del /s /q",
            "format c:",
            "drop table",
        ),
        "unsafe_bypass": (
            "disable authentication",
            "bypass auth",
            "skip validation",
            "ignore verification",
            "disable safety",
        ),
        "remote_execution": (
            "curl | sh",
            "invoke-expression",
            "iex(",
        ),
    }

    async def evaluate(
        self,
        *,
        task: Any,
        output: str,
        observation: dict[str, Any],
        attempt: int,
        max_attempts: int,
    ):
        lowered = (output or "").lower()
        hits: list[dict[str, str]] = []
        for category, patterns in self._MARKERS.items():
            for pattern in patterns:
                if pattern in lowered:
                    hits.append({"category": category, "pattern": pattern})

        score = 1.0 - min(len(hits) * 0.30, 0.90)
        if hits:
            preview = ", ".join(hit["pattern"] for hit in hits[:3])
            reason = f"Detected risky guidance markers: {preview}."
        else:
            reason = "No obvious destructive or bypass-oriented guidance was detected."
        return self.assessment(
            score=score,
            reason=reason,
            details={"hits": hits},
        )
