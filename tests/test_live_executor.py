"""Tests for nexus.live.executor: the hands, driven through a fake pyautogui."""

import ctypes
import importlib.util
import platform
import sys
import threading
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import nexus.live.executor as executor_module
from nexus.live.executor import WHEEL_NOTCH, Executor, Interrupted
from nexus.live.model import Decision
from nexus.live.screen import Shot, Window

# A 1920 by 1080 monitor, seen by the model as 1280 by 720.
SHOT = Shot(image=b"png", width=1280, height=720, screen_width=1920, screen_height=1080)


class FakeBackend:
    """Records what pyautogui would have been asked to do; ``fail_on`` names a call that raises."""

    def __init__(self, size=(1920, 1080), fail_on=None):
        self.screen = size
        self.fail_on = fail_on
        self.calls = []

    def _record(self, *call):
        self.calls.append(call)
        if call[0] == self.fail_on:
            raise RuntimeError(f"{call[0]} failed")

    def size(self):
        self._record("size")
        return self.screen

    def moveTo(self, x, y, duration=None):
        self._record("moveTo", x, y, duration)

    def click(self, x, y):
        self._record("click", x, y)

    def doubleClick(self, x, y):
        self._record("doubleClick", x, y)

    def rightClick(self, x, y):
        self._record("rightClick", x, y)

    def write(self, text, interval=None):
        self._record("write", text, interval)

    def press(self, key):
        self._record("press", key)

    def hotkey(self, *keys):
        self._record("hotkey", *keys)

    def scroll(self, clicks):
        self._record("scroll", clicks)


class FakeClipboard:
    """pyperclip, writing its copies into ``log`` so their order against the key presses can be checked."""

    def __init__(self, log, content="what the user copied", fail_paste=False):
        self.log = log
        self.content = content
        self.fail_paste = fail_paste

    def paste(self):
        if self.fail_paste:
            raise RuntimeError("no clipboard here")
        return self.content

    def copy(self, text):
        self.content = text
        self.log.append(("copy", text))


class MousePointTests(unittest.TestCase):
    def test_same_size_is_unchanged(self):
        backend = FakeBackend((1920, 1080))
        self.assertEqual(Executor(backend=backend, system="Windows").mouse_point((960, 540), SHOT), (960, 540))
        self.assertEqual(backend.calls, [("size",)])

    def test_scaled_display_on_windows(self):
        executor = Executor(backend=FakeBackend((1536, 864)), system="Windows")  # 125% without DPI awareness
        self.assertEqual(executor.mouse_point((960, 540), SHOT), (768, 432))
        self.assertEqual(executor.mouse_point((1918, 1078), SHOT), (1534, 862))

    def test_width_and_height_scale_separately(self):
        executor = Executor(backend=FakeBackend((1000, 500)), system="Windows")
        self.assertEqual(executor.mouse_point((100, 100), Shot(b"", 100, 100, 2000, 2000)), (50, 25))
        self.assertEqual(executor.mouse_point((6, 9), Shot(b"", 100, 100, 2000, 2000)), (3, 2))

    def test_only_the_position_within_the_monitor_is_scaled(self):
        # A monitor that starts at x=1920 keeps its start: only the offset inside it shrinks.
        executor = Executor(backend=FakeBackend((1536, 864)), system="Windows")
        shot = Shot(b"", 1280, 720, 1920, 1080, left=1920, top=0)
        self.assertEqual(executor.mouse_point((1920 + 960, 540), shot), (1920 + 768, 432))
        left_of_main = Shot(b"", 1280, 720, 1920, 1080, left=-1920, top=-200)
        self.assertEqual(executor.mouse_point((-1920 + 960, -200 + 540), left_of_main), (-1920 + 768, -200 + 432))

    def test_other_systems_never_scale(self):
        # pyautogui's screen there is every monitor together (Linux) or the main display in points (Mac):
        # a different size says nothing about scaling.
        for system in ("Linux", "Darwin"):
            with self.subTest(system=system):
                executor = Executor(backend=FakeBackend((3840, 1080)), system=system)
                self.assertEqual(executor.mouse_point((960, 540), SHOT), (960, 540))

    def test_unknown_monitor_size_is_left_alone(self):
        executor = Executor(backend=FakeBackend((1536, 864)), system="Windows")
        for shot in (Shot(b"", 1, 1, 0, 1080), Shot(b"", 1, 1, 1920, 0), Shot(b"", 1, 1, 0, 0)):
            with self.subTest(shot=shot):
                self.assertEqual(executor.mouse_point((10, 20), shot), (10, 20))

    def test_the_corners_are_moved_in_so_the_fail_safe_does_not_fire(self):
        # pyautogui stops when the pointer is on a corner pixel: "Show desktop" at the bottom right
        # would end the session as if the user had pulled the emergency stop.
        executor = Executor(backend=FakeBackend((1920, 1080)), system="Windows")
        for corner, moved in (((0, 0), (1, 1)), ((1919, 0), (1918, 1)), ((0, 1079), (1, 1078)), ((1919, 1079), (1918, 1078))):
            with self.subTest(corner=corner):
                self.assertEqual(executor.mouse_point(corner, SHOT), moved)

    def test_edges_away_from_the_corners_are_left_alone(self):
        executor = Executor(backend=FakeBackend((1920, 1080)), system="Windows")
        for point in ((0, 1), (1, 0), (1919, 1), (1918, 1079), (0, 540), (960, 0)):
            with self.subTest(point=point):
                self.assertEqual(executor.mouse_point(point, SHOT), point)


class CornerTests(unittest.TestCase):
    def test_only_exact_corners_move(self):
        self.assertEqual(executor_module._off_the_corners(0, 0, 100, 50), (1, 1))
        self.assertEqual(executor_module._off_the_corners(99, 49, 100, 50), (98, 48))
        self.assertEqual(executor_module._off_the_corners(99, 0, 100, 50), (98, 1))
        self.assertEqual(executor_module._off_the_corners(0, 49, 100, 50), (1, 48))
        self.assertEqual(executor_module._off_the_corners(50, 0, 100, 50), (50, 0))
        self.assertEqual(executor_module._off_the_corners(100, 50, 100, 50), (100, 50))  # off this screen: not a corner of it


class PointAtTests(unittest.TestCase):
    def test_moves_without_clicking(self):
        backend = FakeBackend()
        Executor(backend=backend).point_at((10, 20))
        self.assertEqual(backend.calls, [("moveTo", 10, 20, 0.25)])

    def test_dry_run_does_not_move(self):
        backend = FakeBackend()
        Executor(dry_run=True, backend=backend).point_at((10, 20))
        self.assertEqual(backend.calls, [])


class RunTests(unittest.TestCase):
    def test_clicks(self):
        for action, call in (("click", "click"), ("double_click", "doubleClick"), ("right_click", "rightClick")):
            with self.subTest(action=action):
                backend = FakeBackend()
                decision = Decision(action, x=500.0, y=250.0)
                self.assertEqual(Executor(backend=backend).run(decision, (960, 270)), decision.describe())
                self.assertEqual(backend.calls, [(call, 960, 270)])

    def test_clicks_need_a_position(self):
        for action in ("click", "double_click", "right_click"):
            with self.subTest(action=action):
                backend = FakeBackend()
                with self.assertRaisesRegex(ValueError, f"{action} needs a position"):
                    Executor(backend=backend).run(Decision(action, x=1.0, y=2.0), None)
                self.assertEqual(backend.calls, [])

    def test_plain_text_is_typed_a_letter_at_a_time(self):
        backend = FakeBackend()
        clipboard = FakeClipboard(backend.calls)
        executor = Executor(backend=backend, clipboard=clipboard, key_reader=frozenset)
        self.assertEqual(executor.run(Decision("type", text="hi you"), None), "type 'hi you'")
        self.assertEqual(backend.calls, [("write", letter, None) for letter in "hi you"])

    def test_press_one_key(self):
        backend = FakeBackend()
        self.assertEqual(Executor(backend=backend).run(Decision("press", keys=("enter",)), None), "press enter")
        self.assertEqual(backend.calls, [("press", "enter")])

    def test_hotkey(self):
        backend = FakeBackend()
        self.assertEqual(Executor(backend=backend).run(Decision("hotkey", keys=("ctrl", "shift", "s")), None), "press ctrl+shift+s")
        self.assertEqual(backend.calls, [("hotkey", "ctrl", "shift", "s")])

    def test_scroll_at_a_position(self):
        backend = FakeBackend()
        done = Executor(backend=backend).run(Decision("scroll", x=1.0, y=2.0, amount=-3), (100, 200))
        self.assertEqual(done, "scroll down 3")
        self.assertEqual(backend.calls, [("moveTo", 100, 200, 0.1), ("scroll", -3 * WHEEL_NOTCH)])
        self.assertIs(type(backend.calls[-1][1]), int)  # the system scroll calls take whole notches

    def test_scroll_where_the_pointer_is(self):
        backend = FakeBackend()
        self.assertEqual(Executor(backend=backend).run(Decision("scroll", amount=5), None), "scroll up 5")
        self.assertEqual(backend.calls, [("scroll", 5 * WHEEL_NOTCH)])
        self.assertIs(type(backend.calls[-1][1]), int)

    def test_scroll_amounts_become_wheel_units_on_windows(self):
        # Windows counts a wheel notch as 120 units: one notch per step of the model's amount.
        backend = FakeBackend()
        with patch.object(executor_module, "WHEEL_NOTCH", 120):
            Executor(backend=backend).run(Decision("scroll", amount=-2), None)
        self.assertEqual(backend.calls, [("scroll", -240)])

    def test_wheel_notch_for_this_system(self):
        self.assertEqual(WHEEL_NOTCH, 120 if platform.system() == "Windows" else 1)

    def test_wheel_notch_on_each_system(self):
        for system, notch in (("Windows", 120), ("Darwin", 1), ("Linux", 1)):
            with self.subTest(system=system):
                # A separate copy of the module, loaded as if on that system; the real one is left alone.
                spec = importlib.util.spec_from_file_location(f"executor_on_{system}", executor_module.__file__)
                module = importlib.util.module_from_spec(spec)
                with patch("platform.system", return_value=system):
                    spec.loader.exec_module(module)
                self.assertEqual(module.WHEEL_NOTCH, notch)

    def test_quiet_steps_are_not_for_the_hands(self):
        for decision in (Decision("wait", seconds=1.0), Decision("answer", say="Hi."), Decision("done")):
            with self.subTest(action=decision.action):
                backend = FakeBackend()
                with self.assertRaisesRegex(ValueError, f"{decision.action} is not something the hands do"):
                    Executor(backend=backend).run(decision, None)
                self.assertEqual(backend.calls, [])


class DryRunTests(unittest.TestCase):
    STEPS = (
        (Decision("click", x=500.0, y=250.0), (960, 270)),
        (Decision("double_click", x=500.0, y=250.0), (960, 270)),
        (Decision("right_click", x=500.0, y=250.0), (960, 270)),
        (Decision("type", text="hello"), None),
        (Decision("type", text="café"), None),
        (Decision("press", keys=("enter",)), None),
        (Decision("hotkey", keys=("ctrl", "s")), None),
        (Decision("scroll", amount=-3), (5, 5)),
    )

    def test_says_what_it_would_do_and_does_nothing(self):
        for decision, point in self.STEPS:
            with self.subTest(action=decision.action):
                backend = FakeBackend()
                clipboard = FakeClipboard(backend.calls)
                done = Executor(dry_run=True, backend=backend, clipboard=clipboard).run(decision, point)
                self.assertEqual(done, f"(dry run, nothing done) {decision.describe()}")
                self.assertEqual(backend.calls, [])

    def test_never_loads_pyautogui(self):
        with patch.dict(sys.modules, {"pyautogui": None}):  # importing it now would fail
            executor = Executor(dry_run=True)
            executor.point_at((1, 2))
            self.assertEqual(executor.run(Decision("click", x=1.0, y=2.0), (1, 2)), "(dry run, nothing done) click at (1, 2)")


class Keys:
    """The modifier keys held down, read in turn (the last answer repeats)."""

    def __init__(self, *answers):
        self.answers = list(answers) or [frozenset()]
        self.reads = 0

    def __call__(self):
        self.reads += 1
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


class TypingTests(unittest.TestCase):
    def make(self, system="Windows", keys=None, **options):
        self.backend = FakeBackend()
        self.typed = []
        return Executor(
            backend=self.backend, clipboard=FakeClipboard(self.backend.calls), key_reader=keys or Keys(),
            unicode_typer=lambda text: self.backend.calls.append(("unicode", text)), system=system, **options,
        )

    def test_windows_types_other_languages_as_characters_never_through_the_clipboard(self):
        executor = self.make()
        self.assertEqual(executor.run(Decision("type", text="నమస్తే, 5 pm"), None), "type 'నమస్తే, 5 pm'")
        self.assertEqual(self.backend.calls, [("unicode", "నమస్తే")] + [("write", letter, None) for letter in ", 5 pm"])

    def test_windows_runs_of_each_kind_keep_their_order(self):
        executor = self.make()
        executor.run(Decision("type", text="a€b😀c"), None)
        self.assertEqual(
            self.backend.calls,
            [("write", "a", None), ("unicode", "€"), ("write", "b", None), ("unicode", "😀"), ("write", "c", None)],
        )

    def test_mac_and_linux_paste_the_whole_text_once_and_leave_the_clipboard(self):
        # The old clipboard is never put back: a busy app reading it late would paste that instead
        # (often a password just copied) in place of the approved text.
        for system, keys in (("Darwin", ("command", "v")), ("Linux", ("ctrl", "v"))):
            with self.subTest(system=system):
                executor = self.make(system=system)
                executor.run(Decision("type", text="café au lait"), None)
                self.assertEqual(self.backend.calls, [("copy", "café au lait"), ("hotkey", *keys)])

    def test_plain_text_is_typed_on_every_system(self):
        for system in ("Windows", "Darwin", "Linux"):
            with self.subTest(system=system):
                executor = self.make(system=system)
                executor.run(Decision("type", text="ok"), None)
                self.assertEqual(self.backend.calls, [("write", "o", None), ("write", "k", None)])

    def test_pyperclip_is_the_default_clipboard(self):
        backend = FakeBackend()
        pyperclip = ModuleType("pyperclip")
        pyperclip.copy = lambda text: backend.calls.append(("copy", text))
        with patch.dict(sys.modules, {"pyperclip": pyperclip}):
            Executor(backend=backend, key_reader=frozenset, system="Linux").run(Decision("type", text="café"), None)
        self.assertEqual(backend.calls, [("copy", "café"), ("hotkey", "ctrl", "v")])

    def test_the_stop_key_stops_typing_at_the_next_letter(self):
        stop = threading.Event()
        executor = self.make()

        def write(text, interval=None):
            self.backend.calls.append(("write", text, interval))
            if text == "c":
                stop.set()

        self.backend.write = write
        with self.assertRaisesRegex(Interrupted, "stopped"):
            executor.run(Decision("type", text="abcdef"), None, stop)
        self.assertEqual(self.backend.calls, [("write", "a", None), ("write", "b", None), ("write", "c", None)])

    def test_a_key_held_down_stops_typing(self):
        # Ctrl held while typing would turn the letters into shortcuts.
        executor = self.make(keys=Keys(frozenset(), frozenset(), frozenset({"ctrl", "alt"})))
        with self.assertRaisesRegex(Interrupted, "alt\\+ctrl held down"):
            executor.run(Decision("type", text="hello"), None)
        self.assertEqual(self.backend.calls, [("write", "h", None), ("write", "e", None)])

    def test_nothing_is_typed_or_pasted_with_a_key_already_held(self):
        for system in ("Windows", "Linux"):
            with self.subTest(system=system):
                executor = self.make(system=system, keys=Keys(frozenset({"shift"})))
                with self.assertRaises(Interrupted):
                    executor.run(Decision("type", text="café"), None)
                self.assertEqual(self.backend.calls, [])

    def test_keys_that_cannot_be_read_do_not_stop_typing(self):
        executor = self.make(keys=Keys(None))
        executor.run(Decision("type", text="ok"), None)
        self.assertEqual(len(self.backend.calls), 2)

    def test_another_window_coming_forward_stops_the_typing(self):
        # Approved for Notepad: what is left of the text never reaches the window that came forward.
        titles = Keys("Untitled - Notepad", "Untitled - Notepad", "Database.kdbx - KeePass")
        executor = self.make(title_reader=titles)
        with self.assertRaisesRegex(Interrupted, "the window in front changed"):
            executor.run(Decision("type", text="see you"), None, window="Untitled - Notepad")
        self.assertEqual(self.backend.calls, [("write", "s", None), ("write", "e", None)])
        self.assertEqual(titles.reads, 3)  # before every letter

    def test_the_window_is_checked_before_characters_and_pastes_too(self):
        for system, text in (("Windows", "నమస్తే"), ("Linux", "café")):
            with self.subTest(system=system):
                executor = self.make(system=system, title_reader=lambda: "Database.kdbx - KeePass")
                with self.assertRaisesRegex(Interrupted, "the window in front changed"):
                    executor.run(Decision("type", text=text), None, window="Untitled - Notepad")
                self.assertEqual(self.backend.calls, [])

    def test_the_same_window_types_everything(self):
        executor = self.make(title_reader=lambda: "Untitled - Notepad")
        executor.run(Decision("type", text="a€b"), None, window="Untitled - Notepad")
        self.assertEqual(self.backend.calls, [("write", "a", None), ("unicode", "€"), ("write", "b", None)])

    def test_a_title_that_becomes_readable_is_a_change(self):
        unknown = self.make(title_reader=lambda: None)
        unknown.run(Decision("type", text="ok"), None, window=None)
        self.assertEqual(len(self.backend.calls), 2)
        named = self.make(title_reader=lambda: "Command Prompt")
        with self.assertRaises(Interrupted):
            named.run(Decision("type", text="ok"), None, window=None)

    def test_with_no_window_given_none_is_read(self):
        def never():
            raise AssertionError("the window was read")

        executor = self.make(title_reader=never)
        executor.run(Decision("type", text="ok"), None)
        self.assertEqual(len(self.backend.calls), 2)


DOWN = -32768  # GetAsyncKeyState's answer, a signed short, for a key that is down


class FakeKeyState:
    def __init__(self, *codes):
        self.codes = set(codes)

    def GetAsyncKeyState(self, code):
        return DOWN if code in self.codes else 0


class HotkeyModifierTests(unittest.TestCase):
    """The hotkeys' Alt is the left Alt: the right one is AltGr on many keyboards, reported as Ctrl and Alt."""

    def held(self, *codes):
        with patch.object(executor_module, "_user32", return_value=FakeKeyState(*codes)):
            return executor_module.hotkey_modifiers()

    def test_the_left_ctrl_and_alt(self):
        self.assertEqual(self.held(0x11, 0xA2, 0x12, 0xA4), frozenset({"ctrl", "alt"}))

    def test_altgr_is_not_the_hotkeys_ctrl_and_alt(self):
        # AltGr: Windows adds a left Ctrl of its own to the right Alt.
        self.assertEqual(self.held(0x11, 0xA2, 0x12, 0xA5), frozenset({"ctrl"}))
        self.assertEqual(self.held(0x11, 0xA2, 0x12, 0xA4, 0xA5), frozenset({"ctrl"}))  # both Alts: AltGr is in it

    def test_the_other_modifiers(self):
        self.assertEqual(self.held(0x10, 0x12, 0xA4), frozenset({"shift", "alt"}))
        self.assertEqual(self.held(0x5C), frozenset({"win"}))
        self.assertEqual(self.held(), frozenset())

    def test_none_where_the_keyboard_cannot_be_read(self):
        with patch.object(executor_module, "_user32", side_effect=OSError("no user32 here")):
            self.assertIsNone(executor_module.hotkey_modifiers())


class FakeWindows:
    """user32 and dwmapi as _windows_layout uses them, over a list of windows, the top one first.

    Each window is a dict: title, rect, pid, and the flags hidden, minimized,
    cloaked, no_frame (the DWM cannot give its bounds), broken (reading it fails).
    """

    def __init__(self, windows, front):
        self.windows = {index + 1: window for index, window in enumerate(windows)}
        self.front = front

    # user32
    def GetForegroundWindow(self):
        return self.front

    def EnumWindows(self, callback, _):
        for handle in self.windows:
            if not callback(handle, None):
                break
        return True

    def IsWindowVisible(self, handle):
        return not self.windows[handle].get("hidden")

    def IsIconic(self, handle):
        return bool(self.windows[handle].get("minimized"))

    def GetWindowRect(self, handle, rect):
        self._fill(rect, self.windows[handle]["rect"])
        return True

    def GetWindowTextLengthW(self, handle):
        if self.windows[handle].get("broken"):
            raise OSError("access denied")
        return len(self.windows[handle].get("title", ""))

    def GetWindowTextW(self, handle, buffer, size):
        buffer.value = self.windows[handle].get("title", "")[: size - 1]
        return len(buffer.value)

    def GetWindowThreadProcessId(self, handle, process):
        process._obj.value = self.windows[handle].get("pid", 0)
        return 1

    # dwmapi
    def DwmGetWindowAttribute(self, handle, attribute, target, size):
        window = self.windows[handle]
        if attribute == executor_module.DWMWA_CLOAKED:
            target._obj.value = 1 if window.get("cloaked") else 0
            return 0
        if window.get("no_frame"):
            return -2147024809  # E_INVALIDARG
        self._fill(target, window["rect"])
        return 0

    @staticmethod
    def _fill(pointer, box):
        rect = pointer._obj if hasattr(pointer, "_obj") else pointer.contents
        rect.left, rect.top, rect.right, rect.bottom = box


class WindowsLayoutTests(unittest.TestCase):
    def layout(self, windows, front=1):
        fake = FakeWindows(windows, front)
        dwmapi = SimpleNamespace(DwmGetWindowAttribute=lambda *args: fake.DwmGetWindowAttribute(*args))  # takes argtypes
        with patch.object(executor_module, "_user32", return_value=fake), \
             patch.object(ctypes, "WinDLL", return_value=dwmapi, create=True), \
             patch.object(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE, create=True):
            return executor_module._windows_layout()

    def test_the_windows_on_the_screen_top_first(self):
        windows = [
            {"title": "Untitled - Notepad", "rect": (0, 0, 800, 600), "pid": 100},
            {"title": "Hidden", "rect": (0, 0, 10, 10), "pid": 1, "hidden": True},
            {"title": "Minimized", "rect": (0, 0, 10, 10), "pid": 1, "minimized": True},
            {"title": "On another desktop", "rect": (0, 0, 10, 10), "pid": 1, "cloaked": True},
            {"title": "Database.kdbx - KeePass", "rect": (700, 0, 1920, 1080), "pid": 200, "no_frame": True},
            {"title": "No size", "rect": (5, 5, 5, 50), "pid": 1},
            {"title": "", "rect": (10, 10, 60, 30), "pid": 200},
            {"title": "Program Manager", "rect": (0, 0, 1920, 1080), "pid": 0},
        ]
        self.assertEqual(self.layout(windows), [
            Window("Untitled - Notepad", (0, 0, 800, 600), 100, True),
            Window("Database.kdbx - KeePass", (700, 0, 1920, 1080), 200, False),
            Window(None, (10, 10, 60, 30), 200, False),
            Window("Program Manager", (0, 0, 1920, 1080), None, False),
        ])

    def test_no_window_in_front(self):
        self.assertEqual(self.layout([{"title": "Notepad", "rect": (0, 0, 9, 9), "pid": 7}], front=None),
                         [Window("Notepad", (0, 0, 9, 9), 7, False)])

    def test_a_window_that_cannot_be_read_fails_the_whole_layout(self):
        # Leaving it out could leave a blocked window uncovered: teach mode then uses no screenshot.
        windows = [{"title": "Notepad", "rect": (0, 0, 9, 9), "pid": 7}, {"title": "KeePass", "rect": (0, 0, 9, 9), "broken": True},
                   {"title": "Calculator", "rect": (0, 0, 9, 9), "pid": 8}]
        with self.assertRaisesRegex(OSError, "access denied"):
            self.layout(windows)
        with patch.object(executor_module, "_windows_layout", side_effect=OSError("access denied")), \
             patch.object(executor_module.platform, "system", return_value="Windows"):
            self.assertIsNone(executor_module.window_layout())


class FakeProperty:
    def __init__(self, value):
        self.value = value


class FakeXWindow:
    """An X window as _x11_layout reads it; ``fail`` is raised when it is looked at."""

    def __init__(self, wid, name=None, box=(0, 0, 10, 10), pid=None, mapped=True, fail=None):
        self.id, self.name, self.box, self.pid, self.mapped, self.fail = wid, name, box, pid, mapped, fail

    def get_attributes(self):
        if self.fail is not None:
            raise self.fail
        from Xlib import X

        return SimpleNamespace(map_state=X.IsViewable if self.mapped else X.IsUnmapped)

    def get_geometry(self):
        return SimpleNamespace(width=self.box[2] - self.box[0], height=self.box[3] - self.box[1])

    def get_full_property(self, atom, kind):
        if atom == "_NET_WM_NAME" and self.name:
            return FakeProperty(self.name.encode())
        if atom == "_NET_WM_PID" and self.pid:
            return FakeProperty([self.pid])
        return None


class FakeXDisplay:
    """An X display: ``windows`` bottom first, as X lists them; ``stacking`` as a window manager's list or not."""

    def __init__(self, windows, active=None, stacking=True):
        self.windows = {window.id: window for window in windows}
        self.order = [window.id for window in windows]
        self.active, self.stacking = active, stacking
        self.closed = False
        display = self

        class Root:
            def get_full_property(self, atom, kind):
                if atom == "_NET_CLIENT_LIST_STACKING" and display.stacking:
                    return FakeProperty(display.order)
                if atom == "_NET_ACTIVE_WINDOW" and display.active is not None:
                    return FakeProperty([display.active])
                return None

            def query_tree(self):
                return SimpleNamespace(children=[display.windows[wid] for wid in display.order])

            def translate_coords(self, window, x, y):
                return SimpleNamespace(x=window.box[0], y=window.box[1])

        self.root = Root()

    def screen(self):
        return SimpleNamespace(root=self.root)

    def intern_atom(self, name):
        return name

    def create_resource_object(self, kind, wid):
        return self.windows[wid]

    def close(self):
        self.closed = True


def closed_window_error():
    from Xlib import error

    return error.BadWindow.__new__(error.BadWindow)  # what X says about a window closed in the meantime


@unittest.skipUnless(importlib.util.find_spec("Xlib"), "needs python-xlib")
class X11LayoutTests(unittest.TestCase):
    def layout(self, display):
        with patch("Xlib.display.Display", return_value=display):
            return executor_module._x11_layout()

    def desktop(self, **options):
        return FakeXDisplay([
            FakeXWindow(1, None, (0, 0, 1920, 1080)),  # the desktop, at the bottom
            FakeXWindow(2, "Database.kdbx - KeePass", (1000, 0, 1600, 900), pid=77),
            FakeXWindow(3, "Hidden away", (0, 0, 50, 50), mapped=False),
            FakeXWindow(4, "Notes", (0, 0, 1000, 900), pid=42),
        ], active=4, **options)

    def test_the_windows_top_first_with_their_programs_and_the_one_in_front(self):
        display = self.desktop()
        self.assertEqual(self.layout(display), [
            Window("Notes", (0, 0, 1000, 900), 42, True),
            Window("Database.kdbx - KeePass", (1000, 0, 1600, 900), 77, False),
            Window(None, (0, 0, 1920, 1080), None, False),
        ])
        self.assertTrue(display.closed)

    def test_without_a_window_manager_the_window_tree_is_read(self):
        self.assertEqual([window.title for window in self.layout(self.desktop(stacking=False))],
                         ["Notes", "Database.kdbx - KeePass", None])

    def test_no_window_in_front(self):
        display = self.desktop()
        display.active = None
        self.assertEqual([window.front for window in self.layout(display)], [False, False, False])

    def test_a_window_closed_in_the_meantime_is_left_out(self):
        display = self.desktop()
        display.windows[2].fail = closed_window_error()
        self.assertEqual([window.title for window in self.layout(display)], ["Notes", None])

    def test_any_other_trouble_fails_the_whole_layout(self):
        display = self.desktop()
        display.windows[2].fail = RuntimeError("the server stopped answering")
        with self.assertRaises(RuntimeError):
            self.layout(display)
        self.assertTrue(display.closed)
        with patch("Xlib.display.Display", return_value=display), \
             patch.object(executor_module.platform, "system", return_value="Linux"):
            self.assertIsNone(executor_module.window_layout())


class UnicodeInputTests(unittest.TestCase):
    def test_a_press_and_a_release_for_each_utf16_unit(self):
        events = executor_module.unicode_inputs("é😀")
        self.assertEqual(len(events), 6)  # é is one unit, the emoji two (a surrogate pair)
        units = [0x00E9, 0xD83D, 0xDE00]
        for index, event in enumerate(events):
            with self.subTest(index=index):
                self.assertEqual(event.type, executor_module.INPUT_KEYBOARD)
                self.assertEqual(event.union.ki.wVk, 0)
                self.assertEqual(event.union.ki.wScan, units[index // 2])
                flags = executor_module.KEYEVENTF_UNICODE | (executor_module.KEYEVENTF_KEYUP if index % 2 else 0)
                self.assertEqual(event.union.ki.dwFlags, flags)

    @unittest.skipUnless(ctypes.sizeof(ctypes.c_void_p) == 8, "the layout checked is 64-bit Windows'")
    def test_the_structure_is_the_size_windows_expects(self):
        # SendInput refuses anything but sizeof(INPUT), 40 bytes on 64-bit Windows.
        self.assertEqual(ctypes.sizeof(executor_module._Input), 40)
        self.assertEqual(ctypes.sizeof(executor_module._KeyboardInput), 24)


class ReleaseTests(unittest.TestCase):
    def make(self, keys):
        self.clock = [0.0]
        self.slept = []

        def sleep(seconds):
            self.slept.append(seconds)
            self.clock[0] += seconds

        return Executor(backend=FakeBackend(), key_reader=keys, sleep=sleep, clock=lambda: self.clock[0])

    def test_nothing_held_goes_at_once(self):
        executor = self.make(Keys(frozenset()))
        self.assertTrue(executor.wait_for_release())
        self.assertEqual(self.slept, [])

    def test_waits_while_the_yes_hotkey_is_still_held(self):
        held = frozenset({"ctrl", "alt"})
        keys = Keys(held, held, held, frozenset())
        executor = self.make(keys)
        self.assertTrue(executor.wait_for_release(timeout=2.0))
        self.assertEqual(self.slept, [0.05, 0.05, 0.05])

    def test_gives_up_when_held_past_the_timeout(self):
        executor = self.make(Keys(frozenset({"shift"})))
        self.assertFalse(executor.wait_for_release(timeout=1.0))
        self.assertGreaterEqual(self.clock[0], 1.0)
        self.assertLess(self.clock[0], 1.1)

    def test_keys_that_cannot_be_read_get_a_short_pause(self):
        executor = self.make(Keys(None))
        self.assertTrue(executor.wait_for_release())
        self.assertEqual(self.slept, [0.5])


class QuestionTests(unittest.TestCase):
    def test_window_under_a_point_and_pointer(self):
        backend = FakeBackend()
        backend.position = lambda: (12.0, 34.0)
        seen = []
        executor = Executor(backend=backend, window_reader=lambda point: seen.append(point) or "Calculator")
        self.assertEqual(executor.window_at((5, 6)), "Calculator")
        self.assertEqual(seen, [(5, 6)])
        self.assertEqual(executor.pointer(), (12, 34))

    def test_the_real_readers_say_none_when_they_cannot_tell(self):
        with patch.object(executor_module.platform, "system", return_value="Darwin"):
            self.assertIsNone(executor_module.window_at((1, 2)))
        with patch.object(executor_module, "_x11_window_at", side_effect=RuntimeError("no display")), \
             patch.object(executor_module.platform, "system", return_value="Linux"):
            self.assertIsNone(executor_module.window_at((1, 2)))
        with patch.object(executor_module, "_x11_held", side_effect=RuntimeError("no display")), \
             patch.object(executor_module.platform, "system", return_value="Linux"):
            self.assertIsNone(executor_module.held_modifiers())


class BackendTests(unittest.TestCase):
    def test_pyautogui_is_loaded_when_first_needed_with_its_fail_safe_on(self):
        pyautogui = ModuleType("pyautogui")
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0
        with patch.dict(sys.modules, {"pyautogui": pyautogui}):
            executor = Executor()
            self.assertIs(executor.backend, pyautogui)
        self.assertIs(pyautogui.FAILSAFE, True)
        self.assertEqual(pyautogui.PAUSE, 0.05)
        self.assertIs(executor.backend, pyautogui)  # kept: no second import

    def test_given_backend_is_used(self):
        backend = FakeBackend()
        self.assertIs(Executor(backend=backend).backend, backend)

    def test_window_title_comes_from_the_title_reader(self):
        self.assertEqual(Executor(backend=FakeBackend(), title_reader=lambda: "Untitled - Notepad").foreground_title(), "Untitled - Notepad")
        self.assertIsNone(Executor(backend=FakeBackend(), title_reader=lambda: None).foreground_title())


if __name__ == "__main__":
    unittest.main()
