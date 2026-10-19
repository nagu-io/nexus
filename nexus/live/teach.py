"""Teach mode: you do a task, and NEXUS Live writes down each step as a training example.

For every click, scroll, key or run of typing you make, it keeps the screenshot
from just before it and the step itself, in NEXUS Live's own answer format,
together with the request you are carrying out and the steps so far. Finishing
the session (the stop hotkey) adds a last example: on the final screen, "done".
That is the data that makes NEXUS Vision good at your apps.

It records nothing in windows on the blocked list (passwords, banking,
terminals...) and nothing while paused. Typed text is stored as you typed it,
so before a session is saved you are shown everything you typed and can drop
any of it. The examples stay on this computer unless you upload them.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from nexus.live.data import IMAGES_DIR, Example, write_examples
from nexus.live.model import Decision
from nexus.live.screen import NORMALIZED_SPAN, Shot
from nexus.live.settings import LiveSettings, window_matches

MODIFIERS = ("ctrl", "alt", "shift")
# A step is matched to the screenshot taken at least this long before it.
BEFORE = 0.15
DOUBLE_CLICK_SECONDS = 0.5  # what Windows counts as a double click by default
DOUBLE_CLICK_DISTANCE = 6  # pixels
TYPING_PAUSE = 2.0  # seconds without a key that end a run of typing
SCROLL_PAUSE = 1.5  # seconds between wheel turns that make them separate scrolls
MAX_SCROLL = 20


@dataclass
class _Pending:
    """A run of typing not yet written down: its text, and the screen before its first key."""

    text: str = ""
    shot: Shot | None = None
    title: str | None = None
    last_key: float = 0.0


@dataclass
class Step:
    target: dict[str, Any]
    shot: Shot
    title: str | None
    at: float
    point: tuple[int, int] | None = None


@dataclass
class TeachSession:
    request: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    steps: list[Step] = field(default_factory=list)

    def typed(self) -> list[tuple[int, str]]:
        """(step number, text) for every piece of typing, for the review before saving."""
        return [(index, step.target["text"]) for index, step in enumerate(self.steps) if step.target["action"] == "type"]


def _hotkey_keys(hotkey: str) -> frozenset[str]:
    return frozenset(part.strip().strip("<>").lower() for part in hotkey.split("+") if part.strip())


class TeachRecorder:
    def __init__(
        self,
        settings: LiveSettings,
        capture: Callable[[], Shot],
        title_reader: Callable[[], str | None],
        clock: Callable[[], float] = time.monotonic,
        interval: float = 0.3,
    ):
        self.settings = settings
        self.capture = capture
        self.title_reader = title_reader
        self.clock = clock
        self.interval = interval
        self.session: TeachSession | None = None
        self.paused = False
        self._shots: "deque[tuple[float, Shot]]" = deque(maxlen=10)
        self._pending = _Pending()
        self._held: set[str] = set()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._ignored_hotkeys = {
            _hotkey_keys(hotkey)
            for hotkey in (settings.hotkey_stop, settings.hotkey_talk, settings.hotkey_yes, settings.hotkey_no)
        }

    # The screen, every ``interval`` seconds, so each step has the screen from just before it.
    def snap(self) -> None:
        shot = self.capture()
        with self._lock:
            now = self.clock()
            self._shots.append((now, shot))
            if self._pending.text and now - self._pending.last_key >= TYPING_PAUSE:
                self._flush_typing()

    def run_snapshots(self) -> None:
        while not self._stop.is_set():
            try:
                self.snap()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def _shot_before(self, moment: float) -> Shot | None:
        for taken, shot in reversed(self._shots):
            if taken <= moment - BEFORE:
                return shot
        return self._shots[0][1] if self._shots else None

    def start(self, request: str) -> TeachSession:
        with self._lock:
            self.session = TeachSession(" ".join(request.split()))
            self._pending = _Pending()
            self._held.clear()
            self._shots.clear()
            self.paused = False
            self._stop.clear()
        return self.session

    def _usable(self, title: str | None) -> bool:
        if self.session is None or self.paused:
            return False
        return not (title and window_matches(title, self.settings.blocked_windows))

    def _add(self, target: dict[str, Any], shot: Shot | None, title: str | None, point: tuple[int, int] | None = None) -> None:
        if shot is None or self.session is None:
            return
        target = {"thought": "", "say": "", **target}
        self.session.steps.append(Step(target, shot, title, self.clock(), point))

    def _flush_typing(self) -> None:
        pending = self._pending
        if pending.text and pending.shot is not None:
            self._add({"action": "type", "text": pending.text}, pending.shot, pending.title)
        self._pending = _Pending()

    def _last_step(self) -> Step | None:
        return self.session.steps[-1] if self.session and self.session.steps else None

    # Mouse
    def on_click(self, x: int, y: int, button: str, pressed: bool) -> None:
        if not pressed:
            return
        with self._lock:
            title = self.title_reader()
            if not self._usable(title):
                self._pending = _Pending()
                return
            self._flush_typing()
            moment = self.clock()
            shot = self._shot_before(moment)
            if shot is None:
                return
            point = self._normalized(x, y, shot)
            if point is None:
                return
            last = self._last_step()
            if (
                button == "left"
                and last is not None
                and last.target.get("action") == "click"
                and last.point is not None
                and moment - last.at <= DOUBLE_CLICK_SECONDS
                and abs(last.point[0] - x) <= DOUBLE_CLICK_DISTANCE
                and abs(last.point[1] - y) <= DOUBLE_CLICK_DISTANCE
            ):
                last.target["action"] = "double_click"
                return
            action = {"left": "click", "right": "right_click"}.get(button)
            if action is None:
                return
            self._add({"action": action, "x": point[0], "y": point[1]}, shot, title, point=(x, y))

    def on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        notches = max(-MAX_SCROLL, min(MAX_SCROLL, int(round(dy))))
        if not notches:
            return
        with self._lock:
            title = self.title_reader()
            if not self._usable(title):
                return
            self._flush_typing()
            moment = self.clock()
            last = self._last_step()
            if (
                last is not None
                and last.target.get("action") == "scroll"
                and (last.target["amount"] > 0) == (notches > 0)
                and moment - last.at <= SCROLL_PAUSE
            ):
                last.target["amount"] = max(-MAX_SCROLL, min(MAX_SCROLL, last.target["amount"] + notches))
                last.at = moment
                return
            shot = self._shot_before(moment)
            if shot is None:
                return
            target: dict[str, Any] = {"action": "scroll", "amount": notches}
            point = self._normalized(x, y, shot)
            if point is not None:
                target.update(x=point[0], y=point[1])
            self._add(target, shot, title)

    # Keyboard: ``key`` is a character ("a", "A", "1") or a key name ("enter", "ctrl", "f5").
    def on_press(self, key: str) -> None:
        with self._lock:
            name = key.lower() if len(key) > 1 else key
            if name in MODIFIERS:
                self._held.add(name)
                return
            if frozenset(self._held | {name.lower()}) in self._ignored_hotkeys:
                return
            title = self.title_reader()
            if not self._usable(title):
                self._pending = _Pending()
                return
            moment = self.clock()
            if self._held - {"shift"}:
                self._flush_typing()
                keys = [modifier for modifier in MODIFIERS if modifier in self._held] + [name.lower()]
                self._add({"action": "hotkey", "keys": keys}, self._shot_before(moment), title)
                return
            if len(key) == 1 or name == "space":
                character = " " if name == "space" else key
                if not self._pending.text:
                    self._pending = _Pending(shot=self._shot_before(moment), title=title)
                self._pending.text += character
                self._pending.last_key = moment
                return
            if name == "backspace" and self._pending.text:
                self._pending.text = self._pending.text[:-1]
                self._pending.last_key = moment
                return
            self._flush_typing()
            self._add({"action": "press", "keys": [name]}, self._shot_before(moment), title)

    def on_release(self, key: str) -> None:
        with self._lock:
            self._held.discard(key.lower() if len(key) > 1 else key)

    def _normalized(self, x: int, y: int, shot: Shot) -> tuple[int, int] | None:
        """A pixel on the screen as 0 to 1000 across the screenshot, or None if it is off the main monitor."""
        inside_x = x - shot.left
        inside_y = y - shot.top
        if not (0 <= inside_x < shot.screen_width and 0 <= inside_y < shot.screen_height):
            return None
        return (
            round(inside_x / shot.screen_width * NORMALIZED_SPAN),
            round(inside_y / shot.screen_height * NORMALIZED_SPAN),
        )

    def stop(self) -> TeachSession | None:
        """End the session: the typing so far becomes a step, and the last screen gets "done"."""
        with self._lock:
            self._stop.set()
            self._flush_typing()
            session = self.session
            if session is None:
                return None
            final = self._shots[-1][1] if self._shots else None
            if session.steps and final is not None and self._usable(self.title_reader()):
                self._add({"action": "done", "say": "Done."}, final, None)
            self.session = None
            return session


def save_session(session: TeachSession, folder: Path, drop: Sequence[int] = ()) -> list[Example]:
    """Write a session's examples (and their screenshots) into the dataset ``folder``, leaving out the steps in ``drop``."""
    folder = Path(folder).expanduser()
    kept = [step for index, step in enumerate(session.steps) if index not in set(drop)]
    if not any(step.target["action"] != "done" for step in kept):
        return []
    (folder / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    examples: list[Example] = []
    done_so_far: list[str] = []
    for index, step in enumerate(kept):
        image_name = f"teach-{session.id}-{index:03d}.png"
        examples.append(
            Example(
                id=f"teach-{session.id}-{index:03d}",
                source="teach",
                group=f"teach-{session.id}",
                image=f"{IMAGES_DIR}/{image_name}",
                request=session.request,
                target=dict(step.target),
                steps=list(done_so_far),
            )
        )
        done_so_far.append(describe(step.target))
    for example in examples:
        example.check()
    for example, step in zip(examples, kept):
        (folder / example.image).write_bytes(step.shot.image)
    write_examples(folder, examples, append=True)
    return examples


def describe(target: dict[str, Any]) -> str:
    """The step as NEXUS Live describes it to the model afterwards (Decision.describe)."""
    decision = Decision(
        action=target["action"],
        thought=target.get("thought", ""),
        x=target.get("x"),
        y=target.get("y"),
        text=target.get("text", ""),
        keys=tuple(target.get("keys", ())),
        amount=int(target.get("amount", 0) or 0),
        seconds=float(target.get("seconds", 0) or 0),
    )
    return decision.describe()


def pynput_key_name(key: Any) -> str | None:
    """A pynput key as on_press wants it: the character, or the key's name.

    With Ctrl held, Windows reports letters as control characters (Ctrl+S as
    "\\x13"); those are read back as the letter, from the key code when there is one.
    """
    character = getattr(key, "char", None)
    if character and ord(character[0]) >= 32:
        return character
    code = getattr(key, "vk", None)
    if isinstance(code, int):
        if 65 <= code <= 90:
            return chr(code).lower()
        if 48 <= code <= 57:
            return chr(code)
    if character and len(character) == 1 and 1 <= ord(character) <= 26:
        return chr(ord(character) + 96)
    name = getattr(key, "name", None)
    if not name:
        return None
    name = name.lower()
    for modifier in MODIFIERS:
        if name.startswith(modifier):  # ctrl_l, alt_r, alt_gr, shift_r
            return modifier
    if name.startswith("cmd"):
        return "win"
    return {"page_up": "pageup", "page_down": "pagedown", "return": "enter", "escape": "esc"}.get(name, name)


def default_folder() -> Path:
    return Path.home() / ".nexus" / "live" / "teach"


def run_teach(
    settings: LiveSettings,
    folder: Path | None = None,
    read: Callable[[str], str] = input,
    printer: Callable[[str], None] = print,
) -> int:
    """Teach mode in the console: for each request you type, record what you do until the stop hotkey."""
    from pynput import keyboard, mouse  # type: ignore

    from nexus.live.executor import foreground_title
    from nexus.live.screen import capture
    from nexus.live.voice import hotkey_words

    folder = Path(folder or default_folder()).expanduser()
    recorder = TeachRecorder(settings, lambda: capture(settings.screen_max_width, optimize=False), foreground_title)
    finished = threading.Event()

    def toggle_pause() -> None:
        recorder.paused = not recorder.paused
        printer("(paused)" if recorder.paused else "(recording again)")

    hotkeys = keyboard.GlobalHotKeys({settings.hotkey_stop: finished.set, settings.hotkey_talk: toggle_pause})
    keys = keyboard.Listener(
        on_press=lambda key: _forward(recorder.on_press, key),
        on_release=lambda key: _forward(recorder.on_release, key),
    )
    pointer = mouse.Listener(
        on_click=lambda x, y, button, pressed: recorder.on_click(int(x), int(y), getattr(button, "name", ""), pressed),
        on_scroll=lambda x, y, dx, dy: recorder.on_scroll(int(x), int(y), dx, dy),
    )
    capture(settings.screen_max_width)  # makes this process DPI aware before the listeners read positions
    saved = 0
    hotkeys.start()
    keys.start()
    pointer.start()
    try:
        printer(f"Teach mode. Examples are saved in {folder}.")
        while True:
            request = read("Request to teach (Enter to finish): ").strip()
            if not request:
                break
            finished.clear()
            recorder.start(request)
            snapshots = threading.Thread(target=recorder.run_snapshots, name="nexus-teach-snapshots", daemon=True)
            snapshots.start()
            printer(
                f"Recording. Do it now, the way you want NEXUS to. {hotkey_words(settings.hotkey_talk)} pauses, "
                f"{hotkey_words(settings.hotkey_stop)} finishes."
            )
            finished.wait()
            session = recorder.stop()
            snapshots.join(timeout=2)
            if session is None or not session.steps:
                printer("Nothing was recorded.")
                continue
            drop = _review_typing(session, read, printer)
            examples = save_session(session, folder, drop)
            saved += len(examples)
            printer(f"Saved {len(examples)} examples ({len(session.steps) - len(drop)} steps).")
    finally:
        pointer.stop()
        keys.stop()
        hotkeys.stop()
    printer(f"{saved} examples saved in {folder} this time.")
    return saved


def _forward(handler: Callable[[str], None], key: Any) -> None:
    name = pynput_key_name(key)
    if name:
        handler(name)


def _review_typing(session: TeachSession, read: Callable[[str], str], printer: Callable[[str], None]) -> list[int]:
    """Show everything typed in the session and return the steps the user wants left out."""
    typed = session.typed()
    if not typed:
        return []
    printer("You typed:")
    for number, (_, text) in enumerate(typed, start=1):
        printer(f"  {number}. {text!r}")
    answer = read("Numbers to leave out (such as 2 3), or Enter to keep them all: ").strip()
    drop: list[int] = []
    for part in answer.replace(",", " ").split():
        if part.isdigit() and 1 <= int(part) <= len(typed):
            drop.append(typed[int(part) - 1][0])
    return drop
