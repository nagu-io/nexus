"""The hands: moving the mouse, clicking and typing, through pyautogui.

pyautogui's fail-safe stays on: throw the mouse into a corner of the screen and
the next action stops with an error.
"""

from __future__ import annotations

import ctypes
import platform
import shutil
import subprocess
import time
from typing import Any, Callable

from nexus.live.model import Decision
from nexus.live.screen import Shot

# A wheel notch, in the units each system's scroll call takes.
WHEEL_NOTCH = 120 if platform.system() == "Windows" else 1


def _windows_title() -> str | None:
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    handle = user32.GetForegroundWindow()
    if not handle:
        return None
    length = user32.GetWindowTextLengthW(handle)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(handle, buffer, length + 1)
    return buffer.value


def _mac_title() -> str | None:
    script = 'tell application "System Events" to get name of first application process whose frontmost is true'
    out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=3)
    return out.stdout.strip() or None


def _x11_title() -> str | None:
    if shutil.which("xdotool"):
        out = subprocess.run(["xdotool", "getactivewindow", "getwindowname"], capture_output=True, text=True, timeout=3)
        return out.stdout.strip() if out.returncode == 0 else None
    from Xlib import X, display  # type: ignore

    screen = display.Display()
    root = screen.screen().root
    active = root.get_full_property(screen.intern_atom("_NET_ACTIVE_WINDOW"), X.AnyPropertyType)
    if not active or not active.value or not active.value[0]:
        return None
    window = screen.create_resource_object("window", active.value[0])
    name = window.get_full_property(screen.intern_atom("_NET_WM_NAME"), 0) or window.get_full_property(X.WM_NAME, 0)
    if not name:
        return ""
    value = name.value
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def foreground_title() -> str | None:
    """The title of the window in front, or None when it cannot be read."""
    try:
        system = platform.system()
        if system == "Windows":
            return _windows_title()
        if system == "Darwin":
            return _mac_title()
        return _x11_title()
    except Exception:
        return None


class Executor:
    def __init__(
        self,
        dry_run: bool = False,
        backend: Any = None,
        title_reader: Callable[[], str | None] | None = None,
        clipboard: Any = None,
    ):
        self.dry_run = dry_run
        self._backend = backend
        self._title_reader = title_reader or foreground_title
        self._clipboard = clipboard

    @property
    def backend(self) -> Any:
        if self._backend is None:
            import pyautogui  # type: ignore

            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.05
            self._backend = pyautogui
        return self._backend

    def foreground_title(self) -> str | None:
        return self._title_reader()

    def mouse_point(self, point: tuple[int, int], shot: Shot) -> tuple[int, int]:
        """A pixel on the screenshot's monitor, in the coordinates the mouse uses.

        They are the same unless Windows scales the display and this process
        could not be made aware of it; then the mouse sees a smaller screen.
        """
        width, height = self.backend.size()
        if (width, height) == (shot.screen_width, shot.screen_height) or not (shot.screen_width and shot.screen_height):
            return point
        return int(point[0] * width / shot.screen_width), int(point[1] * height / shot.screen_height)

    def point_at(self, point: tuple[int, int]) -> None:
        """Move the pointer to ``point`` without clicking, so the user sees where a click would land."""
        if not self.dry_run:
            self.backend.moveTo(point[0], point[1], duration=0.25)

    def _paste(self, text: str) -> None:
        # pyautogui types only plain ASCII; anything else (Telugu, emoji,
        # accents) goes through the clipboard, which is then put back.
        clipboard = self._clipboard
        if clipboard is None:
            import pyperclip  # type: ignore

            clipboard = pyperclip
        try:
            previous = clipboard.paste()
        except Exception:
            previous = None
        clipboard.copy(text)
        try:
            self.backend.hotkey("command" if platform.system() == "Darwin" else "ctrl", "v")
            time.sleep(0.15)
        finally:
            if previous is not None:
                clipboard.copy(previous)

    def run(self, decision: Decision, point: tuple[int, int] | None) -> str:
        """Carry out ``decision`` (at ``point``, for the actions that need one) and say what was done."""
        description = decision.describe()
        if self.dry_run:
            return f"(dry run, nothing done) {description}"
        backend = self.backend
        action = decision.action
        if action in ("click", "double_click", "right_click"):
            if point is None:
                raise ValueError(f"{action} needs a position")
            {"click": backend.click, "double_click": backend.doubleClick, "right_click": backend.rightClick}[action](
                point[0], point[1]
            )
        elif action == "type":
            if decision.text.isascii():
                backend.write(decision.text, interval=0.02)
            else:
                self._paste(decision.text)
        elif action == "press":
            backend.press(decision.keys[0])
        elif action == "hotkey":
            backend.hotkey(*decision.keys)
        elif action == "scroll":
            if point is not None:
                backend.moveTo(point[0], point[1], duration=0.1)
            backend.scroll(decision.amount * WHEEL_NOTCH)
        else:
            raise ValueError(f"{action} is not something the hands do")
        return description
