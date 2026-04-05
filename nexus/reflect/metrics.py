"""Internal benchmark metrics for ReflectScore."""

from __future__ import annotations


def _to_list(results) -> list[dict]:
    return list(results)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() == "true"
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(value)


def hallucination_rate(results, evaluator=None):
    rows = _to_list(results)
    if not rows:
        return None
    incorrect = sum(1 for row in rows if not _as_bool(row.get("is_correct", False)))
    return incorrect / len(rows)


def grounding_score(results, evaluator=None):
    rows = [row for row in _to_list(results) if row.get("category") == "code"]
    if not rows:
        return None
    grounded = 0
    for row in rows:
        answer = str(row.get("answer", ""))
        file_reference = str(row.get("file_reference", ""))
        keywords = row.get("keywords", []) or []
        primary_keyword = keywords[0] if keywords else ""
        if file_reference and file_reference in answer:
            grounded += 1
        elif primary_keyword and primary_keyword in answer:
            grounded += 1
    return grounded / len(rows)


def refusal_accuracy(results, evaluator=None):
    rows = [row for row in _to_list(results) if bool(row.get("unanswerable"))]
    if not rows:
        return None
    correct = sum(1 for row in rows if _as_bool(row.get("is_correct", False)))
    return correct / len(rows)


def mean_latency(results, evaluator=None):
    rows = _to_list(results)
    if not rows:
        return None
    values = [float(row.get("response_time_seconds", 0.0) or 0.0) for row in rows]
    return float(sum(values) / len(values))
