"""Base contracts for runtime critics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


def clamp_score(value: float) -> float:
    """Normalize arbitrary critic output into the expected 0..1 range."""
    return max(0.0, min(1.0, float(value)))


@dataclass
class CriticAssessment:
    """Structured output from a single runtime critic."""

    critic: str
    score: float
    reason: str
    weight: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = clamp_score(payload["score"])
        return payload


class BaseCritic(ABC):
    """Abstract runtime critic."""

    name: str = "base"
    weight: float = 1.0
    cost_tier: str = "low"

    @abstractmethod
    async def evaluate(
        self,
        *,
        task: Any,
        output: str,
        observation: dict[str, Any],
        attempt: int,
        max_attempts: int,
    ) -> CriticAssessment:
        """Evaluate a task result and return a normalized assessment."""

    def assessment(
        self,
        *,
        score: float,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> CriticAssessment:
        """Create a normalized assessment payload."""
        return CriticAssessment(
            critic=self.name,
            score=clamp_score(score),
            reason=reason,
            weight=self.weight,
            details=dict(details or {}),
        )
