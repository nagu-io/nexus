"""Internal evaluation helpers for ReflectScore."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

REFUSAL_PHRASES = (
    "i do not know",
    "i don't know",
    "not enough information",
    "insufficient information",
    "not in the provided context",
    "cannot determine",
    "can't determine",
    "unknown",
)


class AutoEvaluator:
    """Lightweight correctness evaluator for benchmark rows."""

    def lexical_similarity(self, left: str, right: str) -> float:
        token_pattern = re.compile(r"[A-Za-z0-9_\.]+")
        left_tokens = {token.lower() for token in token_pattern.findall(left)}
        right_tokens = {token.lower() for token in token_pattern.findall(right)}
        if not left_tokens or not right_tokens:
            jaccard = 0.0
        else:
            union = len(left_tokens | right_tokens)
            jaccard = len(left_tokens & right_tokens) / union if union else 0.0
        sequence_ratio = SequenceMatcher(None, left.lower(), right.lower()).ratio()
        return max(jaccard, sequence_ratio)

    def keyword_match(self, answer: str, keywords: list[str]) -> bool:
        answer_lower = answer.lower()
        return any(keyword and keyword.lower() in answer_lower for keyword in keywords)

    def is_refusal(self, answer: str) -> bool:
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)

    def is_correct(self, answer: str, ground_truth: str, keywords: list[str], unanswerable: bool = False) -> bool:
        if not answer:
            return False
        if unanswerable or ground_truth == "NOT_IN_CONTEXT":
            return self.is_refusal(answer)
        if self.keyword_match(answer=answer, keywords=keywords):
            return True
        return self.lexical_similarity(answer, ground_truth) >= 0.72
