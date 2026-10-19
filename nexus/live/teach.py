"""Teach mode: you do a task, and NEXUS Live writes down each step as a training example.

For every click, scroll, key or run of typing you make, it keeps the screenshot
from just before it and the step itself, in NEXUS Live's own answer format,
together with the request you are carrying out and the steps so far. Finishing
the session (the stop hotkey) adds a last example: on the final screen, "done".
That is the data that makes NEXUS Vision good at your apps.

It records nothing in windows on the blocked list (passwords, banking,
terminals...), in its own console, in a window whose title cannot be read, and
nothing while paused. Those windows are blacked out wherever they show in the
screenshots (on Windows and Linux; a Mac cannot say where its windows are).
Keys NEXUS Live never presses (the Windows key, Caps Lock, Print Screen...) are
not steps. Typed text is stored as you typed it, so before a session is saved
you are shown everything you typed and can drop any of it. The examples stay
on this computer unless you upload them.
"""

from __future__ import annotations

import ctypes
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from nexus.live.data import IMAGES_DIR, Example, write_examples
from nexus.live.model import KEY_NAMES, Decision, DecisionError, clean, parse_decision
from nexus.live.policy import Policy
from nexus.live.screen import NORMALIZED_SPAN
from nexus.live.settings import LiveSettings, blocked_by

MODIFIERS = ("ctrl", "alt", "shift")
HELD_KEYS = MODIFIERS + ("win", "altgr")  # followed while held down; never steps of their own
INTERVAL = 0.1  # seconds between screenshots
# How long a step takes to show on the screen. The screenshot for the next step
# must be taken at least this long after it, or it may not show what it did
# (a menu that has not opened yet): the next step is then kept only as history.
# A typed character shows almost at once; a click or a key like Enter can open
# or close whole windows.
SETTLE = 0.15
SETTLE_TYPED = 0.05
DOUBLE_CLICK_SECONDS = 0.5  # what Windows counts as a double click by default
DOUBLE_CLICK_DISTANCE = 6  # pixels
TYPING_PAUSE = 2.0  # seconds without a key that end a run of typing
SCROLL_PAUSE = 1.5  # seconds between wheel turns that make them separate scrolls
MAX_SCROLL = 20
WHEEL_DELTA = 120  # Windows' units in one wheel notch
SNAPSHOTS = 20  # screenshots kept, about two seconds' worth
# A modifier the listener saw go down but that the keyboard says is up, this long after: its release was
# never heard (Win+L and Ctrl+Alt+Del let go of the keys on another screen).
STUCK_AFTER = 1.0


@dataclass
class _Snapshot:
    at: float  # when the grab started
    shot: Any
    title: str | None  # the window in front for the whole grab; None if it changed or could not be read
    problem: str = ""  # why no step may use it, when that is not the window in front


@dataclass
class _Pending:
    """A run of typing not yet written down: its text, and the screen before its first key."""

    text: str = ""
    shot: Any = None
    why: str = ""
    title: str | None = None
    last_key: float = 0.0


@dataclass
class Step:
    target: dict[str, Any]
    shot: Any  # the screen just before it, or None when none can go with it (``why`` says why)
    title: str | None
    at: float
    point: tuple[int, int] | None = None
    why: str = ""
    described: str = ""  # how it reads in the history of later examples, when describe() would get it wrong


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


class WheelNotches:
    """Wheel movement in whole notches, from Windows' raw deltas (120 to a notch).

    A precision touchpad sends many small deltas. pynput divides each by 120
    rounding down, which loses every small turn up and makes every small turn
    down a whole notch; here they add up first.
    """

    def __init__(self) -> None:
        self.total = 0

    def add(self, delta: int) -> int:
        if not delta:
            return 0
        if self.total and (self.total > 0) != (delta > 0):
            self.total = 0  # turned the other way: start again
        self.total += delta
        notches = int(self.total / WHEEL_DELTA)  # whole notches, toward zero
        self.total -= notches * WHEEL_DELTA
        return notches


class TeachRecorder:
    def __init__(
        self,
        settings: LiveSettings,
        capture: Callable[[], Any],
        title_reader: Callable[[], str | None],
        clock: Callable[[], float] = time.monotonic,
        interval: float = INTERVAL,
        layout_reader: Callable[[], list | None] | None = None,
        cover: Callable[[Any, list, Callable[[str | None], bool]], bool] | None = None,
        window_reader: Callable[[tuple[int, int]], str | None] | None = None,
        key_reader: Callable[[], frozenset[str] | None] | None = None,
        layout_required: bool = False,
    ):
        self.settings = settings
        self.capture = capture
        self.title_reader = title_reader
        self.clock = clock
        self.interval = interval
        self.layout_reader = layout_reader
        self.cover = cover
        self.window_reader = window_reader  # None: the window under a click is taken to be the one in front
        self.key_reader = key_reader
        # Where the windows on the screen can be read (Windows, Linux), a screenshot whose blocked
        # windows could not be blacked out goes with no step.
        self.layout_required = layout_required
        self.policy = Policy(settings)
        self._pressed_at: dict[str, float] = {}
        self.session: TeachSession | None = None
        self.paused = False
        self.own_titles: set[str] = set()  # NEXUS Live's own console
        self._shots: "deque[_Snapshot]" = deque(maxlen=SNAPSHOTS)
        self._pending = _Pending()
        self._held: set[str] = set()
        self._quiet_from = float("-inf")  # when the last step has had time to show
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._ignored_hotkeys = {
            _hotkey_keys(hotkey)
            for hotkey in (settings.hotkey_stop, settings.hotkey_talk, settings.hotkey_yes, settings.hotkey_no)
        }

    # The screen, every ``interval`` seconds, so each step has the screen from just before it.
    def snap(self) -> None:
        before = self.title_reader()
        started = self.clock()
        shot = self.capture()
        after = self.title_reader()
        problem = ""
        if self.layout_reader is not None and self.cover is not None:
            layout = self.layout_reader()  # a failure here, or in cover, keeps the screenshot out altogether
            if layout is not None:
                self.cover(shot, layout, self.hidden)
            elif self.layout_required:
                problem = "the windows on the screen could not be read, so private ones could not be blacked out in its screenshot"
        with self._lock:
            self._shots.append(_Snapshot(started, shot, before if before == after else None, problem))
            if self._pending.text and self.clock() - self._pending.last_key >= TYPING_PAUSE:
                self._flush_typing()

    def run_snapshots(self) -> None:
        while not self._stop.is_set():
            try:
                self.snap()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def hidden(self, title: str | None) -> bool:
        """Whether the window titled ``title`` is one teach mode never records (and blacks out in screenshots)."""
        return bool(title) and (title in self.own_titles or blocked_by(title, self.settings) is not None)

    def _usable(self, title: str | None) -> bool:
        """Whether a step in the window titled ``title`` may be recorded.

        Never while paused, never in a blocked window or NEXUS Live's own, and
        never when the title cannot be read: then it could be any of those.
        """
        if self.session is None or self.paused:
            return False
        return bool(title) and not self.hidden(title)

    def _shot_for(self, moment: float) -> tuple[Any, str]:
        """The screenshot for a step at ``moment``: the newest taken before it. None, and why, when it will not do."""
        for snapshot in reversed(self._shots):
            if snapshot.at >= moment:
                continue
            if snapshot.at < self._quiet_from:
                return None, "too soon after the step before it for a screenshot to show that step"
            if snapshot.problem:
                return None, snapshot.problem
            if not self._usable(snapshot.title):
                return None, "its screenshot was taken while a window that is not recorded was in front"
            return snapshot.shot, ""
        return None, "there was no screenshot from before it"

    def _geometry(self) -> Any:
        """The latest screenshot, for its monitor's size and place."""
        return self._shots[-1].shot if self._shots else None

    def start(self, request: str) -> TeachSession:
        with self._lock:
            self.session = TeachSession(" ".join(clean(request).split()))
            self._pending = _Pending()
            self._held.clear()
            self._shots.clear()
            self._quiet_from = float("-inf")
            self.paused = False
            self._stop.clear()
        return self.session

    def _add(self, target: dict[str, Any], shot: Any, title: str | None, point: tuple[int, int] | None = None,
             why: str = "", described: str = "") -> None:
        if self.session is None:
            return
        target = {"thought": "", "say": "", **target}
        if target["action"] in ("press", "hotkey") and shot is not None:
            # Keys NEXUS Live refuses at run time (Ctrl+Esc, Ctrl+Shift+Esc) would teach a step that always fails.
            try:
                refusal = self.policy.key_refusal(parse_decision(dict(target)))
            except DecisionError:
                refusal = ""
            if refusal:
                shot, why = None, f"NEXUS Live never takes this step: {refusal}"
        self.session.steps.append(Step(target, shot, title, self.clock(), point, why, described))

    def _flush_typing(self) -> None:
        pending = self._pending
        if pending.text:
            self._add({"action": "type", "text": pending.text}, pending.shot, pending.title, why=pending.why)
        self._pending = _Pending()

    def _last_step(self) -> Step | None:
        return self.session.steps[-1] if self.session and self.session.steps else None

    def _held_modifiers(self) -> list[str]:
        return [key for key in HELD_KEYS if key in self._held]

    def _forget_stuck(self, moment: float) -> None:
        """Let go of modifiers the keyboard says are up whose release the listener never heard."""
        if self.key_reader is None or not self._held:
            return
        actual = self.key_reader()
        if actual is None:
            return
        for key in list(self._held):
            # A fresh press may not be visible yet: only one held a while is let go of.
            if key not in actual and moment - self._pressed_at.get(key, moment) > STUCK_AFTER:
                self._held.discard(key)

    def _lands_unrecorded(self, x: int, y: int) -> bool:
        """Whether a click or scroll at (x, y) lands in a window that is not recorded.

        It goes to the window under the pointer, which need not be the one in
        front. One whose title cannot be read could be any window: not recorded.
        """
        if self.window_reader is None:
            return False
        under = self.window_reader((x, y))
        return under is None or self.hidden(under)

    # Mouse
    def on_click(self, x: int, y: int, button: str, pressed: bool) -> None:
        if not pressed:
            return
        with self._lock:
            title = self.title_reader()
            if not self._usable(title) or self._lands_unrecorded(x, y):
                self._pending = _Pending()
                return
            self._flush_typing()
            moment = self.clock()
            self._forget_stuck(moment)
            geometry = self._geometry()
            if geometry is None:
                return
            point = self._normalized(x, y, geometry)
            if point is None:
                return
            last = self._last_step()
            held = self._held_modifiers()
            if (
                button == "left"
                and not held
                and last is not None
                and last.target.get("action") == "click"
                and last.point is not None
                and moment - last.at <= DOUBLE_CLICK_SECONDS
                and abs(last.point[0] - x) <= DOUBLE_CLICK_DISTANCE
                and abs(last.point[1] - y) <= DOUBLE_CLICK_DISTANCE
            ):
                last.target["action"] = "double_click"
                self._quiet_from = moment + SETTLE
                return
            action = {"left": "click", "right": "right_click"}.get(button)
            if action is None:
                return
            target = {"action": action, "x": point[0], "y": point[1]}
            if held:
                # Ctrl+click or Shift+click selects what a plain click would not: not a step NEXUS Live can take.
                self._add(target, None, title, (x, y), "a click with keys held down is not a step NEXUS Live can take",
                          f"{'+'.join(held)}+{describe(target)}")
            else:
                shot, why = self._shot_for(moment)
                self._add(target, shot, title, (x, y), why)
            self._quiet_from = moment + SETTLE

    def on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        notches = max(-MAX_SCROLL, min(MAX_SCROLL, int(round(dy))))
        if not notches:
            return
        with self._lock:
            title = self.title_reader()
            if not self._usable(title) or self._lands_unrecorded(x, y):
                return
            self._flush_typing()
            moment = self.clock()
            self._forget_stuck(moment)
            held = self._held_modifiers()
            last = self._last_step()
            if (
                not held
                and last is not None
                and last.target.get("action") == "scroll"
                and not last.described
                and (last.target["amount"] > 0) == (notches > 0)
                and moment - last.at <= SCROLL_PAUSE
            ):
                last.target["amount"] = max(-MAX_SCROLL, min(MAX_SCROLL, last.target["amount"] + notches))
                last.at = moment
                self._quiet_from = moment + SETTLE
                return
            geometry = self._geometry()
            if geometry is None:
                return
            target: dict[str, Any] = {"action": "scroll", "amount": notches}
            point = self._normalized(x, y, geometry)
            if point is not None:
                target.update(x=point[0], y=point[1])
            if held:
                # Ctrl and the wheel zoom instead of scrolling.
                self._add(target, None, title, why="a scroll with keys held down is not a step NEXUS Live can take",
                          described=f"{'+'.join(held)}+{describe(target)}")
            else:
                shot, why = self._shot_for(moment)
                self._add(target, shot, title, why=why)
            self._quiet_from = moment + SETTLE

    # Keyboard: ``key`` is a character ("a", "A", "1") or a key name ("enter", "ctrl", "f5"); ``char`` is
    # the character the key made, if it made one.
    def on_press(self, key: str, char: str | None = None) -> None:
        with self._lock:
            name = key.lower() if len(key) > 1 else key
            if name in HELD_KEYS:
                self._held.add(name)
                self._pressed_at[name] = self.clock()
                return
            self._forget_stuck(self.clock())
            if frozenset(self._held | {name.lower()}) in self._ignored_hotkeys:
                return
            if "win" in self._held:
                return  # the Windows key opens system menus: NEXUS Live never uses it, so nothing with it is a step
            title = self.title_reader()
            if not self._usable(title):
                self._pending = _Pending()
                return
            moment = self.clock()
            made = char if isinstance(char, str) and len(char) == 1 and char.isprintable() else None
            if made is not None and ("altgr" in self._held or {"ctrl", "alt"} <= self._held):
                self._type(made, moment, title)  # Ctrl+Alt is AltGr on many keyboards: what it makes (₹ € @) is typing
                return
            modifiers = [modifier for modifier in MODIFIERS if modifier in self._held]
            if self._held & {"ctrl", "alt"} or (modifiers and len(key) > 1 and name != "space"):
                # A shortcut: Ctrl or Alt with any key, or Shift with a named key (Shift+Tab, Shift+Enter).
                if name.lower() not in KEY_NAMES:
                    return
                self._flush_typing()
                shot, why = self._shot_for(moment)
                self._add({"action": "hotkey", "keys": modifiers + [name.lower()]}, shot, title, why=why)
                self._quiet_from = moment + SETTLE
                return
            if len(key) == 1 or name == "space":
                self._type(" " if name == "space" else key, moment, title)
                return
            if name == "backspace" and self._pending.text:
                self._pending.text = self._pending.text[:-1]
                self._pending.last_key = moment
                self._quiet_from = moment + SETTLE_TYPED
                return
            if name not in KEY_NAMES:
                return  # Caps Lock, Print Screen, the volume keys...: not keys NEXUS Live presses
            self._flush_typing()
            shot, why = self._shot_for(moment)
            self._add({"action": "press", "keys": [name]}, shot, title, why=why)
            self._quiet_from = moment + SETTLE

    def _type(self, character: str, moment: float, title: str | None) -> None:
        if not self._pending.text:
            shot, why = self._shot_for(moment)
            self._pending = _Pending(shot=shot, why=why, title=title)
        self._pending.text += character
        self._pending.last_key = moment
        self._quiet_from = moment + SETTLE_TYPED

    def on_release(self, key: str) -> None:
        with self._lock:
            self._held.discard(key.lower() if len(key) > 1 else key)

    def _normalized(self, x: int, y: int, shot: Any) -> tuple[int, int] | None:
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
            if session.steps and self._usable(self.title_reader()):
                final, _ = self._shot_for(self.clock() + 1.0)  # the newest screenshot, if it shows the last step
                if final is not None:
                    self._add({"action": "done", "say": "Done."}, final, None)
            self.session = None
            return session


def save_session(
    session: TeachSession, folder: Path, drop: Sequence[int] = (), printer: Callable[[str], None] | None = None
) -> list[Example]:
    """Write a session's examples (and their screenshots) into the dataset ``folder``, leaving out the steps in ``drop``.

    Each step is checked on its own: one that cannot be an example (no
    screenshot fits it, or it fails the dataset's checks) is still written into
    the history of the examples after it, since it happened, and the rest of
    the session is kept.
    """
    folder = Path(folder).expanduser()
    dropped = set(drop)
    kept = [step for index, step in enumerate(session.steps) if index not in dropped]
    examples: list[Example] = []
    shots: list[Any] = []
    skipped: dict[str, int] = {}
    history: list[str] = []
    for index, step in enumerate(kept):
        if step.shot is None:
            reason = step.why or "no screenshot"
            skipped[reason] = skipped.get(reason, 0) + 1
        else:
            example = Example(
                id=f"teach-{session.id}-{index:03d}",
                source="teach",
                group=f"teach-{session.id}",
                image=f"{IMAGES_DIR}/teach-{session.id}-{index:03d}.png",
                request=session.request,
                target=dict(step.target),
                steps=list(history),
            )
            try:
                example.check()
            except ValueError as error:
                skipped[str(error)] = skipped.get(str(error), 0) + 1
            else:
                examples.append(example)
                shots.append(step.shot)
        history.append(clean(step.described or describe(step.target)))  # as NEXUS Live would put it to the model
    if printer is not None:
        for reason, count in skipped.items():
            printer(f"  {count} step{'s' if count > 1 else ''} kept only as history for the steps after: {reason}.")
    if not any(example.target["action"] != "done" for example in examples):
        return []
    (folder / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    for example, shot in zip(examples, shots):
        (folder / example.image).write_bytes(shot.image)
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


# Windows' key codes for the punctuation keys (a US layout's), which come with no character
# while Ctrl is held: Ctrl+- and Ctrl+= zoom.
OEM_KEYS = {0xBA: ";", 0xBB: "=", 0xBC: ",", 0xBD: "-", 0xBE: ".", 0xBF: "/", 0xC0: "`", 0xDB: "[", 0xDC: "\\", 0xDD: "]", 0xDE: "'"}


def pynput_key_name(key: Any) -> str | None:
    """A pynput key as on_press wants it: the character, or the key's name.

    With Ctrl held, Windows reports letters as control characters (Ctrl+S as
    "\\x13") or no character at all; those are read back from the key code.
    """
    character = getattr(key, "char", None)
    if character and len(character) == 1 and character.isprintable():
        return character
    code = getattr(key, "vk", None)
    if isinstance(code, int):
        if 65 <= code <= 90:
            return chr(code).lower()
        if 48 <= code <= 57:
            return chr(code)
        if code in OEM_KEYS:
            return OEM_KEYS[code]
    if character and len(character) == 1 and 1 <= ord(character) <= 26:
        return chr(ord(character) + 96)
    name = getattr(key, "name", None)
    if not name:
        return None
    name = name.lower()
    if name == "alt_gr":
        return "altgr"
    for modifier in MODIFIERS:
        if name.startswith(modifier):  # ctrl_l, alt_r, shift_r
            return modifier
    if name.startswith("cmd"):
        return "win"
    return {"page_up": "pageup", "page_down": "pagedown", "return": "enter", "escape": "esc"}.get(name, name)


WM_MOUSEWHEEL = 0x020A


def wheel_filter(recorder: TeachRecorder) -> Callable[[int, Any], bool]:
    """A pynput ``win32_event_filter`` that scrolls by whole notches of raw wheel movement (see WheelNotches)."""
    notches = WheelNotches()

    def event_filter(message: int, data: Any) -> bool:
        if message != WM_MOUSEWHEEL:
            return True
        try:
            count = notches.add(ctypes.c_short((int(data.mouseData) >> 16) & 0xFFFF).value)
            if count:
                recorder.on_scroll(int(data.pt.x), int(data.pt.y), 0, count)
        except Exception:
            pass
        return False  # pynput's own scroll, each small delta rounded down, is not used

    return event_filter


def default_folder() -> Path:
    return Path.home() / ".nexus" / "live" / "teach"


def run_teach(
    settings: LiveSettings,
    folder: Path | None = None,
    read: Callable[[str], str] = input,
    printer: Callable[[str], None] = print,
) -> int:
    """Teach mode in the console: for each request you type, record what you do until the stop hotkey."""
    import platform

    from pynput import keyboard, mouse  # type: ignore

    from nexus.live.executor import foreground_title, held_modifiers, window_at, window_layout
    from nexus.live.screen import cover_windows, grab
    from nexus.live.voice import hotkey_filter, hotkey_words

    folder = Path(folder or default_folder()).expanduser()
    on_mac = platform.system() == "Darwin"  # where the windows on the screen cannot be read
    recorder = TeachRecorder(
        settings, lambda: grab(settings.screen_max_width), foreground_title,
        layout_reader=window_layout, cover=cover_windows, window_reader=None if on_mac else window_at,
        key_reader=held_modifiers, layout_required=not on_mac,
    )
    finished = threading.Event()
    on_windows = platform.system() == "Windows"

    def toggle_pause() -> None:
        recorder.paused = not recorder.paused
        printer("(paused)" if recorder.paused else "(recording again)")

    callbacks = {settings.hotkey_stop: finished.set, settings.hotkey_talk: toggle_pause}
    hotkeys: Any = None
    hotkeys = keyboard.GlobalHotKeys(
        callbacks, **({"win32_event_filter": hotkey_filter(callbacks, lambda: hotkeys)} if on_windows else {})
    )
    keys = keyboard.Listener(
        on_press=lambda key: _forward_press(recorder, key),
        on_release=lambda key: _forward(recorder.on_release, key),
    )
    pointer = mouse.Listener(
        on_click=lambda x, y, button, pressed: recorder.on_click(int(x), int(y), getattr(button, "name", ""), pressed),
        # On Windows the wheel comes through wheel_filter instead, in whole notches.
        on_scroll=None if on_windows else (lambda x, y, dx, dy: recorder.on_scroll(int(x), int(y), dx, dy)),
        **({"win32_event_filter": wheel_filter(recorder)} if on_windows else {}),
    )
    grab(settings.screen_max_width)  # makes this process DPI aware before the listeners read positions
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
            console = foreground_title()  # the request was typed here: this console is never recorded
            if console:
                recorder.own_titles.add(console)
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
            try:
                examples = save_session(session, folder, drop, printer=printer)
            except Exception as error:  # a session that cannot be saved never ends teach mode
                printer(f"Could not save this session ({type(error).__name__}: {error}).")
                continue
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


def _forward_press(recorder: TeachRecorder, key: Any) -> None:
    name = pynput_key_name(key)
    if name:
        character = getattr(key, "char", None)
        recorder.on_press(name, character if isinstance(character, str) else None)


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
