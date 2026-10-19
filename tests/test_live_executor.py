"""Tests for nexus.live.executor: the hands, driven through a fake pyautogui."""

import importlib.util
import platform
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import nexus.live.executor as executor_module
from nexus.live.executor import WHEEL_NOTCH, Executor
from nexus.live.model import Decision
from nexus.live.screen import Shot

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
        self.assertEqual(Executor(backend=backend).mouse_point((960, 540), SHOT), (960, 540))
        self.assertEqual(backend.calls, [("size",)])

    def test_scaled_display(self):
        executor = Executor(backend=FakeBackend((1536, 864)))  # Windows at 125% without DPI awareness
        self.assertEqual(executor.mouse_point((960, 540), SHOT), (768, 432))
        self.assertEqual(executor.mouse_point((1919, 1079), SHOT), (1535, 863))

    def test_width_and_height_scale_separately(self):
        executor = Executor(backend=FakeBackend((1000, 500)))
        self.assertEqual(executor.mouse_point((100, 100), Shot(b"", 100, 100, 2000, 2000)), (50, 25))
        self.assertEqual(executor.mouse_point((3, 3), Shot(b"", 100, 100, 2000, 2000)), (1, 0))

    def test_unknown_monitor_size_is_left_alone(self):
        executor = Executor(backend=FakeBackend((1536, 864)))
        for shot in (Shot(b"", 1, 1, 0, 1080), Shot(b"", 1, 1, 1920, 0), Shot(b"", 1, 1, 0, 0)):
            with self.subTest(shot=shot):
                self.assertEqual(executor.mouse_point((10, 20), shot), (10, 20))


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

    def test_plain_text_is_typed(self):
        backend = FakeBackend()
        clipboard = FakeClipboard(backend.calls)
        done = Executor(backend=backend, clipboard=clipboard).run(Decision("type", text="hello world\n"), None)
        self.assertEqual(done, "type 'hello world\\n'")
        self.assertEqual(backend.calls, [("write", "hello world\n", 0.02)])

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


class PasteTests(unittest.TestCase):
    def setUp(self):
        self.slept = []
        sleep_patch = patch.object(executor_module, "time", SimpleNamespace(sleep=self.slept.append))
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def run_typing(self, backend, clipboard, system="Windows", text="café"):
        with patch.object(executor_module.platform, "system", return_value=system):
            return Executor(backend=backend, clipboard=clipboard).run(Decision("type", text=text), None)

    def test_other_languages_are_pasted_and_the_clipboard_put_back(self):
        backend = FakeBackend()
        done = self.run_typing(backend, FakeClipboard(backend.calls), text="నమస్తే café")
        self.assertEqual(done, "type 'నమస్తే café'")
        self.assertEqual(backend.calls, [("copy", "నమస్తే café"), ("hotkey", "ctrl", "v"), ("copy", "what the user copied")])
        self.assertEqual(self.slept, [0.15])

    def test_mac_pastes_with_command_v(self):
        backend = FakeBackend()
        self.run_typing(backend, FakeClipboard(backend.calls), system="Darwin")
        self.assertEqual(backend.calls[1], ("hotkey", "command", "v"))

    def test_linux_pastes_with_ctrl_v(self):
        backend = FakeBackend()
        self.run_typing(backend, FakeClipboard(backend.calls), system="Linux")
        self.assertEqual(backend.calls[1], ("hotkey", "ctrl", "v"))

    def test_empty_clipboard_is_put_back(self):
        backend = FakeBackend()
        self.run_typing(backend, FakeClipboard(backend.calls, content=""))
        self.assertEqual(backend.calls, [("copy", "café"), ("hotkey", "ctrl", "v"), ("copy", "")])

    def test_unreadable_clipboard_is_not_put_back(self):
        backend = FakeBackend()
        self.run_typing(backend, FakeClipboard(backend.calls, fail_paste=True))
        self.assertEqual(backend.calls, [("copy", "café"), ("hotkey", "ctrl", "v")])

    def test_clipboard_is_put_back_when_pasting_fails(self):
        backend = FakeBackend(fail_on="hotkey")
        with self.assertRaisesRegex(RuntimeError, "hotkey failed"):
            self.run_typing(backend, FakeClipboard(backend.calls))
        self.assertEqual(backend.calls, [("copy", "café"), ("hotkey", "ctrl", "v"), ("copy", "what the user copied")])

    def test_pyperclip_is_the_default_clipboard(self):
        backend = FakeBackend()
        pyperclip = ModuleType("pyperclip")
        pyperclip.paste = lambda: "earlier"
        pyperclip.copy = lambda text: backend.calls.append(("copy", text))
        with patch.dict(sys.modules, {"pyperclip": pyperclip}):
            self.run_typing(backend, None)
        self.assertEqual(backend.calls, [("copy", "café"), ("hotkey", "ctrl", "v"), ("copy", "earlier")])


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
