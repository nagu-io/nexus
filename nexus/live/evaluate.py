"""How well a vision model does NEXUS Live's job, on examples it was not trained on.

    python -m nexus.live.evaluate --data data/vision --model nexus_vision --compare Qwen/Qwen3-VL-2B-Instruct

Each held-out example is put to the model exactly as NEXUS Live would put it,
and its answer is graded against the recorded step:
- a click is right when it is the same kind of click and lands inside the
  element's box (OS-Atlas), or within 3% of the screen of where you clicked (teach);
- typing must match the text, keys must be the same keys, a scroll must go the
  same way, and wait, answer and done must be the same action;
- an answer that is not a step NEXUS Live can take counts as wrong, and is
  counted separately as invalid: that is what makes a run stop.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from nexus.live.data import Example, split
from nexus.live.model import POINTER_ACTIONS, DecisionError, VisionModel, parse_decision

NEAR = 30.0  # in 0 to 1000 units: within 3% of the screen of the recorded click


@dataclass
class Grade:
    id: str
    source: str
    expected: str
    answered: str  # the action it chose, or "invalid"
    correct: bool
    in_box: bool | None = None
    distance: float | None = None
    seconds: float = 0.0
    error: str = ""
    raw: str = ""  # the start of what the model said, to see why an answer was wrong
    tokens: int | None = None  # how many tokens it generated (local models)
    raw_tokens: str = ""  # those tokens with the special ones left in: an answer of only <|im_end|> is not a failure


SHOWN = 3  # invalid answers and errors printed per model, so a run's log says what went wrong
RAW_KEPT = 300


def grade(example: Example, raw: str, seconds: float = 0.0) -> Grade:
    """How the model's raw answer compares with the recorded step."""
    target = parse_decision(dict(example.target))
    result = Grade(example.id, example.source, target.action, "invalid", False, seconds=round(seconds, 3), raw=raw[:RAW_KEPT])
    try:
        decision = parse_decision(raw)
    except DecisionError as error:
        result.error = str(error)
        return result
    result.answered = decision.action
    if target.action in POINTER_ACTIONS:
        if decision.action in POINTER_ACTIONS:
            result.distance = round(math.hypot(decision.x - target.x, decision.y - target.y), 1)
            if example.box is not None:
                left, top, right, bottom = example.box
                result.in_box = left <= decision.x <= right and top <= decision.y <= bottom
        landed = result.in_box if example.box is not None else (result.distance is not None and result.distance <= NEAR)
        result.correct = decision.action == target.action and bool(landed)
    elif target.action == "type":
        result.correct = decision.action == "type" and decision.text.strip() == target.text.strip()
    elif target.action in ("press", "hotkey"):
        result.correct = decision.action in ("press", "hotkey") and sorted(decision.keys) == sorted(target.keys)
    elif target.action == "scroll":
        result.correct = decision.action == "scroll" and (decision.amount > 0) == (target.amount > 0)
    else:
        result.correct = decision.action == target.action
    return result


def summarize(grades: list[Grade]) -> dict[str, Any]:
    """The share right overall and by action, how often answers were invalid, and where clicks landed."""
    total = len(grades)
    if not total:
        return {"examples": 0}
    by_action: dict[str, dict[str, Any]] = {}
    for item in grades:
        entry = by_action.setdefault(item.expected, {"examples": 0, "right": 0})
        entry["examples"] += 1
        entry["right"] += int(item.correct)
    for entry in by_action.values():
        entry["share_right"] = round(entry["right"] / entry["examples"], 4)
    boxed = [item for item in grades if item.in_box is not None]
    distances = [item.distance for item in grades if item.distance is not None]
    return {
        "examples": total,
        "share_right": round(sum(item.correct for item in grades) / total, 4),
        "share_invalid": round(sum(item.answered == "invalid" for item in grades) / total, 4),
        "by_action": by_action,
        "clicks_in_box": round(sum(bool(item.in_box) for item in boxed) / len(boxed), 4) if boxed else None,
        "median_click_distance": round(statistics.median(distances), 1) if distances else None,
        "seconds_per_step": round(sum(item.seconds for item in grades) / total, 3),
    }


def held_out(items: list[tuple[Path, Example]], eval_fraction: float) -> list[tuple[Path, Example]]:
    """The examples ``split`` keeps out of training, for the same fraction: what the model never saw."""
    by_example = {id(example): (folder, example) for folder, example in items}
    _, held = split([example for _, example in items], eval_fraction)
    return [by_example[id(example)] for example in held]


def evaluate(
    model: VisionModel,
    items: list[tuple[Path, Example]],
    limit: int | None = None,
    printer: Callable[[str], None] = print,
) -> tuple[dict[str, Any], list[Grade]]:
    grades: list[Grade] = []
    chosen = items if limit is None else items[:limit]
    load = getattr(model, "load", None)
    if callable(load):
        try:
            load()  # before the clock starts: seconds per step is answering, not loading
        except Exception:
            pass  # the first question fails the same way, and is graded with the reason
    for number, (folder, example) in enumerate(chosen, start=1):
        image = (Path(folder) / example.image).read_bytes()
        conversation = [(who, what) for who, what in example.conversation]
        started = time.monotonic()
        try:
            raw = model.ask(example.request, list(example.steps), conversation, image, example.coordinates)
        except Exception as error:  # one failure is graded, not fatal
            raw = ""
            failure = f"{type(error).__name__}: {error}"
        else:
            failure = ""
        result = grade(example, raw, time.monotonic() - started)
        if failure:
            result.error = failure
        generation = getattr(model, "last_generation", None) or {}
        if not failure and generation:
            result.tokens = generation.get("new_tokens")
            result.raw_tokens = str(generation.get("raw", ""))[:RAW_KEPT]
        grades.append(result)
        if result.answered == "invalid" and sum(item.answered == "invalid" for item in grades) <= SHOWN:
            generated = f" | {result.tokens} new tokens: {result.raw_tokens[:120]!r}" if result.tokens is not None else ""
            printer(f"    {example.id}: {result.error[:200]} | said: {raw[:160]!r}{generated}")
        if number % 25 == 0 or number == len(chosen):
            right = sum(item.correct for item in grades)
            printer(f"  {number} of {len(chosen)}: {right} right")
    return summarize(grades), grades


def eval_fraction_of(model_dir: Path, default: float = 0.1) -> float:
    """The held-out fraction the model in ``model_dir`` was trained with, so evaluation uses the same split."""
    try:
        summary = json.loads((Path(model_dir) / "training.json").read_text(encoding="utf-8"))
        return float(summary["options"]["eval_fraction"])
    except (OSError, ValueError, KeyError, TypeError):
        return default


def report_table(results: dict[str, dict[str, Any]]) -> str:
    """The summaries side by side, as a Markdown table."""
    names = list(results)
    rows = [("Steps right", "share_right"), ("Invalid answers", "share_invalid"), ("Clicks inside the box", "clicks_in_box"),
            ("Median click distance (of 1000)", "median_click_distance"), ("Seconds per step", "seconds_per_step")]
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, key in rows:
        cells = []
        for name in names:
            value = results[name].get(key)
            if value is None:
                cells.append("n/a")
            elif key.startswith("share") or key == "clicks_in_box":
                cells.append(f"{value:.1%}")
            else:
                cells.append(f"{value:g}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    from nexus.live.finetune import load_items
    from nexus.live.local import LocalVision

    parser = argparse.ArgumentParser(prog="nexus vision eval", description="Evaluate NEXUS Vision on held-out NEXUS Live examples.")
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, default=Path("nexus_vision"), help="the NEXUS Vision folder")
    parser.add_argument("--compare", help="also evaluate this model (a folder or Hub name), such as the base model")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--all", action="store_true", help="every example, not just the held-out ones")
    parser.add_argument("--out", type=Path, help="write the report here as JSON")
    args = parser.parse_args(argv)

    items, _ = load_items(args.data)
    if not args.all:
        items = held_out(items, eval_fraction_of(args.model))
    print(f"Evaluating on {min(len(items), args.limit)} examples.")
    results: dict[str, dict[str, Any]] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    candidates: list[tuple[str, VisionModel]] = [("NEXUS Vision", LocalVision(args.model))]
    if args.compare:
        candidates.append((args.compare, LocalVision(args.compare, check_card=False)))
    for name, model in candidates:
        print(f"{name}:")
        summary, grades = evaluate(model, items, args.limit)
        results[name] = summary
        details[name] = [asdict(item) for item in grades]
        if hasattr(model, "close"):
            model.close()
    print(report_table(results))
    if args.out:
        args.out.write_text(json.dumps({"results": results, "grades": details}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
