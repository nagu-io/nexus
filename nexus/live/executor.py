"""The hands: moving the mouse, clicking and typing, through pyautogui.

pyautogui's fail-safe stays on: throw the mouse into a corner of the screen and
the next action stops with an error.

It also answers the questions the safety checks ask about the computer: which
window is in front, which window is under a point, where the pointer is, and
whether a modifier key (Ctrl, Alt, Shift, the Windows key) is held down.
"""

from __future__ import annotations

import ctypes
import platform
import shutil
import subprocess
import threading
import time
from itertools import groupby
from typing import Any, Callable

from nexus.live.model import Decision
from nexus.live.screen import Shot

# A wheel notch, in the units each system's scroll call takes.
WHEEL_NOTCH = 120 if platform.system() == "Windows" else 1

# Windows' own surfaces that have no title, by window class.
SHELL_WINDOWS = {
    "Shell_TrayWnd": "the taskbar",
    "Shell_SecondaryTrayWnd": "the taskbar",
    "Progman": "the desktop",
    "WorkerW": "the desktop",
}


class Interrupted(RuntimeError):
    """A step stopped part-way through: the stop key, or a key held down while typing."""


# --- Windows ---------------------------------------------------------------
# NEXUS Live keeps its own handle on user32, so the argument types set here never
# change the functions pyautogui and pynput call through ctypes.windll.

_USER32: Any = None


def _user32() -> Any:
    global _USER32
    if _USER32 is None:
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)  # type: ignore[attr-defined]
        handle = ctypes.c_void_p
        user32.GetForegroundWindow.restype = handle
        user32.GetForegroundWindow.argtypes = []
        user32.GetWindowTextLengthW.argtypes = [handle]
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = [handle, ctypes.c_wchar_p, ctypes.c_int]
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.GetClassNameW.argtypes = [handle, ctypes.c_wchar_p, ctypes.c_int]
        user32.GetClassNameW.restype = ctypes.c_int
        user32.WindowFromPoint.argtypes = [wintypes.POINT]
        user32.WindowFromPoint.restype = handle
        user32.GetAncestor.argtypes = [handle, ctypes.c_uint]
        user32.GetAncestor.restype = handle
        user32.GetWindowThreadProcessId.argtypes = [handle, ctypes.POINTER(ctypes.c_ulong)]
        user32.GetWindowThreadProcessId.restype = ctypes.c_ulong
        user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        user32.GetAsyncKeyState.restype = ctypes.c_short
        user32.SendInput.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]
        user32.SendInput.restype = ctypes.c_uint
        user32.IsWindowVisible.argtypes = [handle]
        user32.IsWindowVisible.restype = ctypes.c_bool
        user32.IsIconic.argtypes = [handle]
        user32.IsIconic.restype = ctypes.c_bool
        user32.GetWindowRect.argtypes = [handle, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowRect.restype = ctypes.c_bool
        user32.EnumWindows.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        user32.EnumWindows.restype = ctypes.c_bool
        _USER32 = user32
    return _USER32


def _window_text(user32: Any, handle: Any) -> str:
    length = user32.GetWindowTextLengthW(handle)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(handle, buffer, length + 1)
    return buffer.value


def _class_name(user32: Any, handle: Any) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(handle, buffer, 256)
    return buffer.value


def _process(user32: Any, handle: Any) -> int:
    process = ctypes.c_ulong(0)
    user32.GetWindowThreadProcessId(handle, ctypes.byref(process))
    return process.value


def _windows_title() -> str | None:
    user32 = _user32()
    handle = user32.GetForegroundWindow()
    if not handle:
        return None
    return _window_text(user32, handle)


GA_ROOT = 2


def _windows_window_at(x: int, y: int) -> str | None:
    from ctypes import wintypes

    user32 = _user32()
    handle = user32.WindowFromPoint(wintypes.POINT(x, y))
    if not handle:
        return None
    root = user32.GetAncestor(handle, GA_ROOT) or handle
    title = _window_text(user32, root)
    if title:
        return title
    shell = SHELL_WINDOWS.get(_class_name(user32, root))
    if shell:
        return shell
    front = user32.GetForegroundWindow()
    if front and _process(user32, front) == _process(user32, root):
        # A menu, list or pop-up of the app in front, which have no titles of their own: part of that window.
        return _window_text(user32, front)
    return None  # an untitled window of some other program: it cannot be checked


# The virtual key codes of the modifiers.
_WINDOWS_MODIFIERS = {"shift": (0x10,), "ctrl": (0x11,), "alt": (0x12,), "win": (0x5B, 0x5C)}


def _windows_held() -> frozenset[str]:
    state = _user32().GetAsyncKeyState
    return frozenset(name for name, codes in _WINDOWS_MODIFIERS.items() if any(state(code) & 0x8000 for code in codes))


DWMWA_EXTENDED_FRAME_BOUNDS = 9
DWMWA_CLOAKED = 14


def _windows_layout() -> list[tuple[str | None, tuple[int, int, int, int]]]:
    from ctypes import wintypes

    user32 = _user32()
    dwmapi = ctypes.WinDLL("dwmapi")  # type: ignore[attr-defined]
    dwmapi.DwmGetWindowAttribute.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
    dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long
    found: list[tuple[str | None, tuple[int, int, int, int]]] = []

    @ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)  # type: ignore[attr-defined]
    def each(handle: Any, _: Any) -> int:
        if not user32.IsWindowVisible(handle) or user32.IsIconic(handle):
            return 1
        cloaked = ctypes.c_int(0)  # on another desktop, or a suspended app: not on the screen
        if dwmapi.DwmGetWindowAttribute(handle, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked)) == 0 and cloaked.value:
            return 1
        rect = wintypes.RECT()
        if dwmapi.DwmGetWindowAttribute(handle, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect)) != 0:
            user32.GetWindowRect(handle, ctypes.byref(rect))
        if rect.right > rect.left and rect.bottom > rect.top:
            found.append((_window_text(user32, handle) or None, (rect.left, rect.top, rect.right, rect.bottom)))
        return 1  # go on to the next window

    user32.EnumWindows(each, None)  # top window first
    return found


# SendInput's structures, with fixed-size fields so their layout is Windows' own
# on any 64-bit system (and can be checked in tests on any of them).
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004


class _KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_uint16),
        ("wScan", ctypes.c_uint16),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_int32),
        ("dy", ctypes.c_int32),
        ("mouseData", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("mi", _MouseInput), ("ki", _KeyboardInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("union", _InputUnion)]


def unicode_inputs(text: str) -> Any:
    """The SendInput events that type ``text`` as characters: a press and a release per UTF-16 unit.

    Typed this way, text goes into the window whatever the keyboard layout, and
    never through the clipboard.
    """
    encoded = text.encode("utf-16-le")
    units = [int.from_bytes(encoded[offset : offset + 2], "little") for offset in range(0, len(encoded), 2)]
    events = (_Input * (2 * len(units)))()
    for index, unit in enumerate(units):
        for half, flags in enumerate((KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)):
            event = events[2 * index + half]
            event.type = INPUT_KEYBOARD
            event.union.ki = _KeyboardInput(0, unit, flags, 0, 0)
    return events


def _send_unicode(text: str) -> None:
    events = unicode_inputs(text)
    sent = _user32().SendInput(len(events), events, ctypes.sizeof(_Input))
    if sent != len(events):
        raise OSError(f"Windows took {sent} of {len(events)} key events while typing")


# --- Mac -------------------------------------------------------------------


def _mac_title() -> str | None:
    script = 'tell application "System Events" to get name of first application process whose frontmost is true'
    out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=3)
    return out.stdout.strip() or None


def _mac_held() -> frozenset[str]:
    import Quartz  # type: ignore

    flags = Quartz.CGEventSourceFlagsState(Quartz.kCGEventSourceStateHIDSystemState)
    masks = {
        "shift": Quartz.kCGEventFlagMaskShift,
        "ctrl": Quartz.kCGEventFlagMaskControl,
        "alt": Quartz.kCGEventFlagMaskAlternate,
        "command": Quartz.kCGEventFlagMaskCommand,
    }
    return frozenset(name for name, mask in masks.items() if flags & mask)


# --- Linux (X11) -----------------------------------------------------------


def _x11_name(screen: Any, window: Any) -> str | None:
    from Xlib import Xatom  # type: ignore

    name = window.get_full_property(screen.intern_atom("_NET_WM_NAME"), 0) or window.get_full_property(Xatom.WM_NAME, 0)
    if not name:
        return None
    value = name.value
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def _x11_title() -> str | None:
    if shutil.which("xdotool"):
        out = subprocess.run(["xdotool", "getactivewindow", "getwindowname"], capture_output=True, text=True, timeout=3)
        return out.stdout.strip() if out.returncode == 0 else None
    from Xlib import X, display  # type: ignore

    screen = display.Display()
    try:
        root = screen.screen().root
        active = root.get_full_property(screen.intern_atom("_NET_ACTIVE_WINDOW"), X.AnyPropertyType)
        if not active or not active.value or not active.value[0]:
            return None
        return _x11_name(screen, screen.create_resource_object("window", active.value[0])) or ""
    finally:
        screen.close()


def _x11_window_at(x: int, y: int) -> str | None:
    from Xlib import X, display  # type: ignore

    screen = display.Display()
    try:
        root = screen.screen().root
        stacking = root.get_full_property(screen.intern_atom("_NET_CLIENT_LIST_STACKING"), X.AnyPropertyType)
        if stacking is not None and len(stacking.value):
            windows = [screen.create_resource_object("window", wid) for wid in stacking.value]
        else:  # no window manager: the top-level windows themselves, bottom first
            windows = list(root.query_tree().children)
        for window in reversed(windows):
            try:
                if window.get_attributes().map_state != X.IsViewable:
                    continue
                geometry = window.get_geometry()
                origin = root.translate_coords(window, 0, 0)
            except Exception:  # closed in the meantime
                continue
            if origin.x <= x < origin.x + geometry.width and origin.y <= y < origin.y + geometry.height:
                return _x11_name(screen, window)
        return None
    finally:
        screen.close()


def _x11_layout() -> list[tuple[str | None, tuple[int, int, int, int]]]:
    from Xlib import X, display  # type: ignore

    screen = display.Display()
    try:
        root = screen.screen().root
        stacking = root.get_full_property(screen.intern_atom("_NET_CLIENT_LIST_STACKING"), X.AnyPropertyType)
        if stacking is not None and len(stacking.value):
            windows = [screen.create_resource_object("window", wid) for wid in stacking.value]
        else:
            windows = list(root.query_tree().children)
        found = []
        for window in reversed(windows):
            try:
                if window.get_attributes().map_state != X.IsViewable:
                    continue
                geometry = window.get_geometry()
                origin = root.translate_coords(window, 0, 0)
                name = _x11_name(screen, window)
            except Exception:
                continue
            found.append((name, (origin.x, origin.y, origin.x + geometry.width, origin.y + geometry.height)))
        return found
    finally:
        screen.close()


# X's modifier slots (Shift, Lock, Control, Mod1 to Mod5) that are keys held down, not locks.
_X11_MODIFIERS = {0: "shift", 2: "ctrl", 3: "alt", 6: "win", 7: "altgr"}


def _x11_held() -> frozenset[str]:
    from Xlib import display  # type: ignore

    screen = display.Display()
    try:
        keymap = screen.query_keymap()
        mapping = screen.get_modifier_mapping()
    finally:
        screen.close()
    held = set()
    for slot, name in _X11_MODIFIERS.items():
        for keycode in mapping[slot]:
            if keycode and keymap[keycode // 8] & (1 << (keycode % 8)):
                held.add(name)
    return frozenset(held)


# --- The questions, on whatever system this is -------------------------------


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


def window_at(point: tuple[int, int]) -> str | None:
    """The title of the window under ``point`` (in the mouse's coordinates), or None when it cannot be told.

    On Windows a menu or list with no title of its own counts as the window in
    front when it belongs to the same program. A Mac cannot be asked, so there
    it is always None.
    """
    try:
        system = platform.system()
        if system == "Windows":
            return _windows_window_at(int(point[0]), int(point[1]))
        if system == "Darwin":
            return None
        return _x11_window_at(int(point[0]), int(point[1]))
    except Exception:
        return None


def window_layout() -> list[tuple[str | None, tuple[int, int, int, int]]] | None:
    """The windows on the screen, the top one first, with their titles and places; None when that cannot be read.

    Teach mode blacks out the ones it never records (password managers,
    banking, terminals) in its screenshots. A Mac cannot be asked: None.
    """
    try:
        system = platform.system()
        if system == "Windows":
            return _windows_layout()
        if system == "Darwin":
            return None
        return _x11_layout()
    except Exception:
        return None


def held_modifiers() -> frozenset[str] | None:
    """The modifier keys held down right now (ctrl, alt, shift, win...), or None when that cannot be read."""
    try:
        system = platform.system()
        if system == "Windows":
            return _windows_held()
        if system == "Darwin":
            return _mac_held()
        return _x11_held()
    except Exception:
        return None


def _off_the_corners(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """(x, y), moved one pixel in from a corner of the screen.

    The corners are pyautogui's fail-safe: with the pointer on one, the next
    action stops as if the user had thrown the mouse there to stop NEXUS.
    """
    if x in (0, width - 1) and y in (0, height - 1):
        x = 1 if x == 0 else width - 2
        y = 1 if y == 0 else height - 2
    return x, y


class Executor:
    def __init__(
        self,
        dry_run: bool = False,
        backend: Any = None,
        title_reader: Callable[[], str | None] | None = None,
        clipboard: Any = None,
        window_reader: Callable[[tuple[int, int]], str | None] | None = None,
        key_reader: Callable[[], frozenset[str] | None] | None = None,
        unicode_typer: Callable[[str], None] | None = None,
        system: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.dry_run = dry_run
        self._backend = backend
        self._title_reader = title_reader or foreground_title
        self._clipboard = clipboard
        self._window_reader = window_reader or window_at
        self._key_reader = key_reader or held_modifiers
        self._unicode_typer = unicode_typer or _send_unicode
        self.system = system or platform.system()
        self._sleep = sleep
        self._clock = clock

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

    def window_at(self, point: tuple[int, int]) -> str | None:
        """The title of the window a click at ``point`` would land in (None: cannot be told)."""
        return self._window_reader(point)

    def pointer(self) -> tuple[int, int]:
        """Where the pointer is now, in the mouse's coordinates."""
        x, y = self.backend.position()
        return int(x), int(y)

    def mouse_point(self, point: tuple[int, int], shot: Shot) -> tuple[int, int]:
        """A pixel on the screenshot's monitor, in the coordinates the mouse uses.

        They are the same unless Windows scales the display and this process
        could not be made aware of it; then the mouse sees a smaller screen, and
        the position within the monitor is scaled to it. (Elsewhere the mouse's
        screen is all the monitors together, so its size says nothing of scaling.)
        """
        width, height = self.backend.size()
        x, y = point
        if (
            self.system == "Windows"
            and (width, height) != (shot.screen_width, shot.screen_height)
            and shot.screen_width
            and shot.screen_height
        ):
            x = shot.left + int((x - shot.left) * width / shot.screen_width)
            y = shot.top + int((y - shot.top) * height / shot.screen_height)
        return _off_the_corners(x, y, width, height)

    def point_at(self, point: tuple[int, int]) -> None:
        """Move the pointer to ``point`` without clicking, so the user sees where a click would land."""
        if not self.dry_run:
            self.backend.moveTo(point[0], point[1], duration=0.25)

    def wait_for_release(self, timeout: float = 2.0, poll: float = 0.05) -> bool:
        """Wait until no modifier key is held down; False if one still is after ``timeout`` seconds.

        The yes hotkey is pressed with Ctrl and Alt: a step carried out while
        they are still down would type Ctrl+Alt+letters and Ctrl+Alt+click
        instead of the step that was approved. Where the keys cannot be read,
        a short pause gives the hand time to come off them.
        """
        deadline = self._clock() + timeout
        while True:
            held = self._key_reader()
            if held is None:
                self._sleep(0.5)
                return True
            if not held:
                return True
            if self._clock() >= deadline:
                return False
            self._sleep(poll)

    def _check(self, stop: threading.Event | None) -> None:
        if stop is not None and stop.is_set():
            raise Interrupted("stopped")
        held = self._key_reader()
        if held:
            raise Interrupted(f"{'+'.join(sorted(held))} held down")

    def _paste(self, text: str) -> None:
        # Typing other than plain ASCII on a Mac or Linux goes through the clipboard,
        # which keeps the text afterwards. The old content is never put back: an app
        # busy enough to read the clipboard after that would paste it (often a
        # password just copied) instead of the text.
        clipboard = self._clipboard
        if clipboard is None:
            import pyperclip  # type: ignore

            clipboard = pyperclip
        clipboard.copy(text)
        self.backend.hotkey("command" if self.system == "Darwin" else "ctrl", "v")

    def _type(self, text: str, stop: threading.Event | None) -> None:
        """Type ``text``, a character at a time, stopping at once for the stop key or a key held down.

        A modifier held down while typing would turn the letters into
        shortcuts, so typing stops instead of carrying on.
        """
        if self.system != "Windows" and not text.isascii():
            self._check(stop)
            self._paste(text)  # in one piece: two pastes in a row could race each other
            return
        for plain, characters in groupby(text, key=str.isascii):
            if plain:
                for letter in characters:
                    self._check(stop)
                    self.backend.write(letter)
            else:  # Telugu, accents, emoji: straight in as characters, never through the clipboard
                self._check(stop)
                self._unicode_typer("".join(characters))

    def run(self, decision: Decision, point: tuple[int, int] | None, stop: threading.Event | None = None) -> str:
        """Carry out ``decision`` (at ``point``, for the actions that need one) and say what was done.

        Raises Interrupted when typing is cut short by ``stop`` or a key held down.
        """
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
            self._type(decision.text, stop)
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
