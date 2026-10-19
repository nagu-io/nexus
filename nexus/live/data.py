"""Training data for NEXUS Vision: a screenshot, a request, the steps so far, and the right next step.

A dataset is a folder with examples.jsonl and an images/ folder. Each example
is turned into the exact conversation NEXUS Live gives the model at run time
(``to_messages``), with the right step as the answer it should learn to give.

Two sources fill it:
- your own demonstrations, recorded by ``nexus live --teach`` (nexus.live.teach);
- OS-Atlas desktop grounding data (Apache-2.0): screenshots of Windows, macOS and
  Linux, each with the elements on it and their boxes (``atlas_elements``).
  nexus.live.atlas fetches just the screenshots it needs from the Hub.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from nexus.live.model import (
    POINTER_ACTIONS,
    PROMPT_VERSION,
    Decision,
    DecisionError,
    chat_messages,
    parse_decision,
    request_text,
    system_prompt,
    target_json,
)
from nexus.live.screen import NORMALIZED_SPAN

EXAMPLES_FILE = "examples.jsonl"
IMAGES_DIR = "images"


@dataclass
class Example:
    id: str
    source: str  # "teach" or "os-atlas"
    group: str  # examples of one group (a teach session, a screenshot) stay on one side of the split
    image: str  # relative to the dataset folder
    request: str
    target: dict[str, Any]  # the right next step, as NEXUS Live's answer format
    steps: list[str] = field(default_factory=list)
    conversation: list[list[str]] = field(default_factory=list)
    box: list[float] | None = None  # the element's box, 0 to 1000, for grading where a click lands
    coordinates: str = "normalized"
    prompt_version: int = PROMPT_VERSION

    def check(self) -> None:
        """Raises ValueError when this example would teach something NEXUS Live cannot do."""
        if not self.request.strip():
            raise ValueError(f"{self.id}: no request")
        try:
            decision = parse_decision(dict(self.target))
        except DecisionError as error:
            raise ValueError(f"{self.id}: {error}") from None
        if self.coordinates == "normalized" and decision.action in POINTER_ACTIONS:
            if not (0 <= decision.x <= NORMALIZED_SPAN and 0 <= decision.y <= NORMALIZED_SPAN):
                raise ValueError(f"{self.id}: the click is off the screenshot")
        if self.box is not None:
            left, top, right, bottom = self.box
            if not (0 <= left < right <= NORMALIZED_SPAN and 0 <= top < bottom <= NORMALIZED_SPAN):
                raise ValueError(f"{self.id}: the box is not on the screenshot")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "Example":
        return cls(**json.loads(line))


def write_examples(folder: Path, examples: Iterable[Example], append: bool = False) -> int:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    checked = list(examples)
    for example in checked:  # all or nothing: a bad example leaves the file as it was
        example.check()
    with open(folder / EXAMPLES_FILE, "a" if append else "w", encoding="utf-8") as handle:
        for example in checked:
            handle.write(example.to_json() + "\n")
    return len(checked)


def read_examples(path: Path) -> list[Example]:
    path = Path(path)
    if path.is_dir():
        path = path / EXAMPLES_FILE
    with open(path, encoding="utf-8") as handle:
        return [Example.from_json(line) for line in handle if line.strip()]


def to_messages(example: Example, image: Any, with_answer: bool = True) -> list[dict[str, Any]]:
    """The conversation for ``example``: what NEXUS Live asks, and (to learn from) the right answer."""
    conversation = [(who, what) for who, what in example.conversation]
    messages = chat_messages(
        system_prompt(example.coordinates),
        request_text(example.request, list(example.steps), conversation),
        image,
    )
    if with_answer:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target_json(example.target)}]})
    return messages


def split(examples: list[Example], eval_fraction: float = 0.1) -> tuple[list[Example], list[Example]]:
    """Train and eval sets, by group, so no session or screenshot is on both sides."""
    train: list[Example] = []
    held_out: list[Example] = []
    for example in examples:
        digest = hashlib.sha256(example.group.encode("utf-8")).digest()
        (held_out if digest[0] / 256 < eval_fraction else train).append(example)
    return train, held_out


def save_screenshot(data: bytes, target: Path, max_width: int) -> tuple[int, int]:
    """Save an image as PNG, shrunk to ``max_width`` as NEXUS Live shrinks the screen. Returns its size."""
    from PIL import Image  # type: ignore

    with Image.open(io.BytesIO(data)) as picture:
        picture = picture.convert("RGB")
        if picture.width > max_width:
            picture = picture.resize((max_width, max(1, round(picture.height * max_width / picture.width))), Image.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".part")
        picture.save(temporary, format="PNG")
        os.replace(temporary, target)
        return picture.size


# OS-Atlas
# The smallest side of a box worth teaching, as a share of the screenshot (about 4 pixels at 1280).
MIN_SIDE = 0.003
REQUEST_TEMPLATES = ("Click {q}", "Click on {q}", "Select {q}", "Find {q} and click it")
VERBS = frozenset(
    "click tap open select press choose close find check uncheck toggle search switch turn enable disable "
    "show hide expand collapse sort filter add remove delete save download upload play pause view".split()
)


@dataclass(frozen=True)
class AtlasElement:
    instruction: str
    box: tuple[float, float, float, float]  # left, top, right, bottom, 0 to 1 of the screenshot, clipped to it


def clean_box(raw: Any) -> tuple[float, float, float, float] | None:
    """An OS-Atlas box ([left, top, right, bottom], 0 to 1) clipped to the screenshot, or None if it is unusable.

    Some boxes run past the edge (a right of 1.09): those are clipped, unless most
    of the element is off the screenshot, in which case the instruction may mean
    something that cannot be seen. That same rule turns away boxes given as 0 to
    1000 instead of 0 to 1, which are almost all off a 0 to 1 screenshot.
    """
    try:
        left, top, right, bottom = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        return None
    if right <= left or bottom <= top:
        return None
    clipped = (max(0.0, left), max(0.0, top), min(1.0, right), min(1.0, bottom))
    width, height = clipped[2] - clipped[0], clipped[3] - clipped[1]
    if width < MIN_SIDE or height < MIN_SIDE:
        return None
    if width * height < (right - left) * (bottom - top) / 2 - 1e-12:  # at least half of it on the screenshot
        return None
    return clipped


def atlas_elements(annotations: Any) -> dict[str, list[AtlasElement]]:
    """The usable elements of OS-Atlas annotations, by screenshot name.

    Reads both layouts it uses: the desktop files group the elements under each
    screenshot ({img_filename, elements: [{instruction, bbox}]}), the others have
    one element per entry ({img_filename, instruction, bbox}). An instruction
    that names two elements on one screenshot is left out, as it is ambiguous.
    """
    if isinstance(annotations, dict):
        annotations = annotations.get("data") or annotations.get("annotations") or []
    by_image: dict[str, list[AtlasElement]] = {}
    seen: dict[str, dict[str, int]] = {}
    for entry in annotations if isinstance(annotations, list) else []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("img_filename") or entry.get("image")
        if not isinstance(name, str) or not name.strip():
            continue
        raw_elements = entry.get("elements", [entry])
        for element in raw_elements if isinstance(raw_elements, list) else []:
            if not isinstance(element, dict):
                continue
            instruction = " ".join(str(element.get("instruction") or "").split())
            box = clean_box(element.get("bbox"))
            counts = seen.setdefault(name, {})
            key = instruction.lower()
            counts[key] = counts.get(key, 0) + 1
            if instruction and len(instruction) <= 200 and box is not None:
                by_image.setdefault(name, []).append(AtlasElement(instruction, box))
    for name, elements in list(by_image.items()):
        unique = [element for element in elements if seen[name][element.instruction.lower()] == 1]
        if unique:
            by_image[name] = unique
        else:
            del by_image[name]
    return by_image


def choose(
    by_image: dict[str, list[AtlasElement]], limit: int | None, per_image: int = 3, seed: int = 7
) -> list[tuple[str, list[AtlasElement]]]:
    """Up to ``limit`` elements, at most ``per_image`` from each screenshot, the same ones for the same seed."""
    rng = random.Random(seed)
    names = sorted(by_image)
    rng.shuffle(names)
    chosen: list[tuple[str, list[AtlasElement]]] = []
    count = 0
    for name in names:
        if limit is not None and count >= limit:
            break
        elements = sorted(by_image[name], key=lambda element: (element.instruction, element.box))
        rng.shuffle(elements)
        picked = elements[: per_image if limit is None else min(per_image, limit - count)]
        chosen.append((name, picked))
        count += len(picked)
    return chosen


def request_for(instruction: str, rng: random.Random | None = None) -> str:
    """The request a user would make for an OS-Atlas instruction, which is often just the element's label."""
    text = " ".join(instruction.split()).strip()
    if not text:
        return ""
    if text.split()[0].lower() in VERBS or text.lower().startswith("go to "):
        return text[0].upper() + text[1:]
    template = rng.choice(REQUEST_TEMPLATES) if rng else REQUEST_TEMPLATES[0]
    return template.format(q=f'"{text}"')


def atlas_examples(
    chosen: list[tuple[str, list[AtlasElement]]],
    read_image: Callable[[str], bytes | None],
    out_folder: Path,
    max_width: int = 1280,
    seed: int = 7,
    done_share: float = 0.0,
) -> list[Example]:
    """Examples for the chosen elements: click the middle of each.

    ``read_image`` gives a screenshot's bytes by its name (None if missing).
    With ``done_share``, that share of them also gets a second example: the click
    already taken, and the answer "done", so the model learns to stop.
    """
    rng = random.Random(seed)
    images_out = Path(out_folder) / IMAGES_DIR
    images_out.mkdir(parents=True, exist_ok=True)
    examples: list[Example] = []
    for name, elements in chosen:
        stem = hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
        target_image = images_out / f"atlas-{stem}.png"
        if not target_image.exists():
            data = read_image(name)
            if data is None:
                continue
            try:
                save_screenshot(data, target_image, max_width)
            except Exception:  # a broken or unreadable image: skip it
                continue
        for index, element in enumerate(elements):
            left, top, right, bottom = element.box
            x = round((left + right) / 2 * NORMALIZED_SPAN)
            y = round((top + bottom) / 2 * NORMALIZED_SPAN)
            request = request_for(element.instruction, rng)
            common = {
                "source": "os-atlas",
                "group": f"atlas-{stem}",
                "image": f"{IMAGES_DIR}/{target_image.name}",
                "request": request,
            }
            examples.append(
                Example(
                    id=f"atlas-{stem}-{index}",
                    target={"thought": "", "say": "", "action": "click", "x": x, "y": y},
                    box=[round(value * NORMALIZED_SPAN, 1) for value in element.box],
                    **common,
                )
            )
            if done_share and rng.random() < done_share:
                examples.append(
                    Example(
                        id=f"atlas-{stem}-{index}-done",
                        target={"thought": "", "say": "Done.", "action": "done"},
                        steps=[Decision("click", x=x, y=y).describe()],
                        **common,
                    )
                )
    for example in examples:
        example.check()
    return examples


def folder_reader(root: Path) -> Callable[[str], bytes | None]:
    """Reads screenshots from a folder OS-Atlas images were unzipped into, by name or, failing that, file name."""
    root = Path(root)
    index: dict[str, Path] = {}

    def read(name: str) -> bytes | None:
        direct = root / name
        if direct.is_file():
            return direct.read_bytes()
        if not index:
            for folder, _, files in os.walk(root):
                for file_name in files:
                    index.setdefault(file_name, Path(folder) / file_name)
        found = index.get(Path(name).name)
        return found.read_bytes() if found else None

    return read


def from_os_atlas(
    annotations: Path,
    images: Path | Callable[[str], bytes | None],
    out_folder: Path,
    limit: int | None = None,
    per_image: int = 3,
    max_width: int = 1280,
    seed: int = 7,
    done_share: float = 0.0,
) -> list[Example]:
    """Examples from an OS-Atlas annotation file and its screenshots (a folder, or a reader function)."""
    entries = json.loads(Path(annotations).read_text(encoding="utf-8"))
    chosen = choose(atlas_elements(entries), limit, per_image, seed)
    reader = images if callable(images) else folder_reader(Path(images))
    return atlas_examples(chosen, reader, out_folder, max_width=max_width, seed=seed, done_share=done_share)
