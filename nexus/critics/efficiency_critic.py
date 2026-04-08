"""Efficiency critic for runtime retries and response shape."""

from __future__ import annotations

from typing import Any

from nexus.critics.base import BaseCritic


class EfficiencyCritic(BaseCritic):
    """Penalize outputs that look expensive to maintain or repeatedly recover."""

    name = "efficiency"
    weight = 0.15

    async def evaluate(
        self,
        *,
        task: Any,
        output: str,
        observation: dict[str, Any],
        attempt: int,
        max_attempts: int,
    ):
        if not output:
            return self.assessment(
                score=0.0,
                reason="No output was produced, so the task made no efficient progress.",
                details={"word_count": 0, "attempt_penalty": 1.0, "length_penalty": 0.0},
            )

        word_count = len(output.split())
        attempt_penalty = min(max(attempt - 1, 0) * 0.18, 0.54)
        if word_count > 900:
            length_penalty = 0.35
        elif word_count > 500:
            length_penalty = 0.20
        elif word_count > 250:
            length_penalty = 0.10
        else:
            length_penalty = 0.0

        score = 1.0 - attempt_penalty - length_penalty
        reasons = []
        if attempt_penalty:
            reasons.append(f"retry pressure added a {attempt_penalty:.2f} penalty")
        if length_penalty:
            reasons.append(f"response length ({word_count} words) added a {length_penalty:.2f} penalty")
        if not reasons:
            reasons.append("output length and retry count look efficient")
        return self.assessment(
            score=score,
            reason="; ".join(reasons).capitalize() + ".",
            details={
                "word_count": word_count,
                "attempt_penalty": attempt_penalty,
                "length_penalty": length_penalty,
            },
        )
