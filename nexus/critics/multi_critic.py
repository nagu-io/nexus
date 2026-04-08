"""Aggregate runtime evaluation across multiple critics."""

from __future__ import annotations

import asyncio
from typing import Any

from nexus.critics.base import BaseCritic, CriticAssessment, clamp_score
from nexus.critics.correctness_critic import CorrectnessCritic
from nexus.critics.efficiency_critic import EfficiencyCritic
from nexus.critics.safety_critic import SafetyCritic


class MultiCriticEvaluator:
    """Combine critic opinions into one structured execution evaluation."""

    def __init__(
        self,
        critics: list[BaseCritic] | None = None,
        reflect_scorer: Any | None = None,
    ):
        self.critics = list(
            critics
            or [
                CorrectnessCritic(reflect_scorer=reflect_scorer),
                EfficiencyCritic(),
                SafetyCritic(),
            ]
        )

    async def evaluate(
        self,
        *,
        task: Any,
        output: str,
        observation: dict[str, Any],
        attempt: int,
        max_attempts: int,
        weights: dict[str, float] | None = None,
        confidence_target: float | None = None,
        cache_hint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        threshold = float(
            confidence_target
            if confidence_target is not None
            else getattr(task, "confidence_threshold", 0.0) or 0.0
        )
        cheap_critics = [critic for critic in self.critics if getattr(critic, "cost_tier", "low") != "high"]
        expensive_critics = [critic for critic in self.critics if getattr(critic, "cost_tier", "low") == "high"]

        cheap_assessments = await self._evaluate_group(
            cheap_critics,
            task=task,
            output=output,
            observation=observation,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        weight_map = self._weight_map(weights)

        early_failure = self._early_failure_from_cheap(
            cheap_assessments,
            observation=observation,
            output=output,
            threshold=threshold,
        )
        if early_failure is not None:
            return self._build_result(
                assessments=cheap_assessments,
                task=task,
                observation=observation,
                output=output,
                threshold=threshold,
                weight_map=weight_map,
                skipped_critics=[critic.name for critic in expensive_critics],
                lazy_path="early_failure",
                cached=False,
                override_failure_type=early_failure,
            )

        if expensive_critics and self._can_use_cache(cache_hint, cheap_assessments, threshold):
            cached_assessments = self._cached_assessments(expensive_critics, cache_hint)
            all_assessments = cheap_assessments + cached_assessments
            return self._build_result(
                assessments=all_assessments,
                task=task,
                observation=observation,
                output=output,
                threshold=threshold,
                weight_map=weight_map,
                evaluated_critics=[assessment.critic for assessment in cheap_assessments],
                skipped_critics=[assessment.critic for assessment in cached_assessments],
                lazy_path="decision_cache",
                cached=True,
                cache_hint=cache_hint,
            )

        expensive_assessments = await self._evaluate_group(
            expensive_critics,
            task=task,
            output=output,
            observation=observation,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        assessments = cheap_assessments + expensive_assessments
        return self._build_result(
            assessments=assessments,
            task=task,
            observation=observation,
            output=output,
            threshold=threshold,
            weight_map=weight_map,
            evaluated_critics=[assessment.critic for assessment in assessments],
            skipped_critics=[],
            lazy_path="full",
            cached=False,
        )

    async def _evaluate_group(
        self,
        critics: list[BaseCritic],
        *,
        task: Any,
        output: str,
        observation: dict[str, Any],
        attempt: int,
        max_attempts: int,
    ) -> list[CriticAssessment]:
        if not critics:
            return []
        return await asyncio.gather(
            *[
                critic.evaluate(
                    task=task,
                    output=output,
                    observation=observation,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                for critic in critics
            ]
        )

    def _weight_map(self, weights: dict[str, float] | None) -> dict[str, float]:
        return {
            critic.name: max(float((weights or {}).get(critic.name, critic.weight)), 0.0)
            for critic in self.critics
        }

    def _build_result(
        self,
        *,
        assessments: list[CriticAssessment],
        task: Any,
        observation: dict[str, Any],
        output: str,
        threshold: float,
        weight_map: dict[str, float],
        evaluated_critics: list[str] | None = None,
        skipped_critics: list[str] | None = None,
        lazy_path: str,
        cached: bool,
        cache_hint: dict[str, Any] | None = None,
        override_failure_type: str | None = None,
    ) -> dict[str, Any]:
        total_weight = sum(weight_map.values()) or 1.0
        weighted_score = sum(
            assessment.score * weight_map.get(assessment.critic, 0.0)
            for assessment in assessments
        )
        combined_score = clamp_score(weighted_score / total_weight)
        critic_scores = {assessment.critic: assessment.score for assessment in assessments}

        safety_score = critic_scores.get("safety", 1.0)
        efficiency_score = critic_scores.get("efficiency", 1.0)
        correctness_score = critic_scores.get("correctness", combined_score)
        failure_type = override_failure_type or self._failure_type(
            observation=observation,
            output=output,
            threshold=threshold,
            combined_score=combined_score,
            correctness_score=correctness_score,
            efficiency_score=efficiency_score,
            safety_score=safety_score,
        )
        dominant = min(assessments, key=lambda item: item.score)
        evaluation_ok = failure_type is None

        summary_prefix = (
            "Decision cache reused expensive critic scores. "
            if cached
            else "Cheap critics short-circuited evaluation. "
            if lazy_path == "early_failure"
            else ""
        )
        summary = summary_prefix + (
            f"Combined critic score {combined_score:.2f} passed."
            if evaluation_ok
            else (
                f"Combined critic score {combined_score:.2f} failed due to "
                f"{failure_type}: {dominant.reason}"
            )
        )

        correctness_details = next(
            (assessment.details for assessment in assessments if assessment.critic == "correctness"),
            {},
        )
        return {
            "ok": evaluation_ok,
            "score": combined_score,
            "confidence": combined_score,
            "threshold": threshold,
            "failure_type": failure_type,
            "summary": summary,
            "dominant_critic": dominant.critic,
            "critic_scores": critic_scores,
            "weights_used": weight_map,
            "evaluated_critics": list(evaluated_critics or [assessment.critic for assessment in assessments]),
            "skipped_critics": list(skipped_critics or []),
            "lazy_path": lazy_path,
            "cached": cached,
            "cache_signature": (cache_hint or {}).get("signature"),
            "critics": [assessment.to_dict() for assessment in assessments],
            "legacy_reflect_score": correctness_details.get("reflect_score"),
        }

    def _can_use_cache(
        self,
        cache_hint: dict[str, Any] | None,
        cheap_assessments: list[CriticAssessment],
        threshold: float,
    ) -> bool:
        if not cache_hint:
            return False
        cheap_scores = {assessment.critic: assessment.score for assessment in cheap_assessments}
        if cheap_scores.get("safety", 1.0) < 0.60:
            return False
        if cheap_scores.get("efficiency", 1.0) < 0.45:
            return False
        expected_confidence = float(
            cache_hint.get("decayed_confidence", cache_hint.get("expected_confidence", 0.0)) or 0.0
        )
        return expected_confidence >= max(threshold, 0.70)

    def _cached_assessments(
        self,
        critics: list[BaseCritic],
        cache_hint: dict[str, Any],
    ) -> list[CriticAssessment]:
        cached_scores = dict(cache_hint.get("critic_scores", {}))
        expected_confidence = float(
            cache_hint.get("decayed_confidence", cache_hint.get("expected_confidence", 0.0)) or 0.0
        )
        assessments: list[CriticAssessment] = []
        for critic in critics:
            score = float(cached_scores.get(critic.name, expected_confidence))
            assessments.append(
                CriticAssessment(
                    critic=critic.name,
                    score=score,
                    reason="Reused a cached high-confidence critic score from a matching successful task.",
                    weight=critic.weight,
                    details={
                        "cached": True,
                        "signature": cache_hint.get("signature"),
                    },
                )
            )
        return assessments

    def _early_failure_from_cheap(
        self,
        assessments: list[CriticAssessment],
        *,
        observation: dict[str, Any],
        output: str,
        threshold: float,
    ) -> str | None:
        critic_scores = {assessment.critic: assessment.score for assessment in assessments}
        if not output:
            return "missing_output"
        if critic_scores.get("safety", 1.0) < 0.35:
            return "safety_risk"
        if critic_scores.get("efficiency", 1.0) < 0.30:
            return "inefficient"
        if not observation.get("ok", True) and threshold <= 0:
            return observation.get("failure_type", "runtime_error")
        return None

    def _failure_type(
        self,
        *,
        observation: dict[str, Any],
        output: str,
        threshold: float,
        combined_score: float,
        correctness_score: float,
        efficiency_score: float,
        safety_score: float,
    ) -> str | None:
        if not output:
            return "missing_output"
        if safety_score < 0.35:
            return "safety_risk"
        if threshold > 0 and combined_score < threshold:
            return "low_confidence"
        if correctness_score < 0.40:
            return "logic_error"
        if efficiency_score < 0.30:
            return "inefficient"
        if not observation.get("ok", True):
            return observation.get("failure_type", "runtime_error")
        return None
