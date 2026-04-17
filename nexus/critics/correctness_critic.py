"""Correctness critic built on top of ReflectScore."""

from __future__ import annotations

from typing import Any

from nexus.critics.base import BaseCritic
from nexus.reflect.reflect_score import ReflectScore


class CorrectnessCritic(BaseCritic):
    """Estimate factual/grounded quality using the existing ReflectScore layer."""

    name = "correctness"
    weight = 0.6
    cost_tier = "high"

    def __init__(self, reflect_scorer: ReflectScore | Any | None = None):
        self.reflect_scorer = reflect_scorer or ReflectScore()

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
                reason="No output was produced, so correctness could not be established.",
                details={"reflect_score": 1.0, "verdict": "missing_output", "action": "retry"},
            )

        assessment = await self.reflect_scorer.assess_response(task.instruction, output)
        reflect_risk = float(assessment.get("score", 0.5) or 0.5)
        score = 1.0 - reflect_risk
        verdict = assessment.get("verdict", "unknown")
        action = assessment.get("action", "serve")
        tool_verified = self._tool_verified_success(observation)
        if tool_verified:
            score = max(score, 0.8)
            reason = (
                "Successful workspace validation raised the correctness floor above "
                f"ReflectScore's {verdict} risk ({reflect_risk:.2f})."
            )
        else:
            reason = f"ReflectScore reported {verdict} risk ({reflect_risk:.2f}) for this output."
        return self.assessment(
            score=score,
            reason=reason,
            details={
                "reflect_score": reflect_risk,
                "verdict": verdict,
                "action": action,
                "tool_verified": tool_verified,
                "should_warn": bool(assessment.get("should_warn", False)),
                "should_reroute": bool(assessment.get("should_reroute", False)),
            },
        )

    def _tool_verified_success(self, observation: dict[str, Any]) -> bool:
        if not observation.get("ok", False):
            return False
        tool_actions = observation.get("tool_actions") or []
        return any(
            action.get("tool") == "terminal_tool" and action.get("ok", False)
            for action in tool_actions
        )
