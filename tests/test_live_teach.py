"""Tests for nexus.live.teach: turning what you do into NEXUS Vision training examples."""

import json
import tempfile
import types
import unittest
from pathlib import Path

from nexus.live.data import read_examples
from nexus.live.screen import Shot
from nexus.live.settings import LiveSettings
from nexus.live.teach import (
    DOUBLE_CLICK_SECONDS,
    INTERVAL,
    OEM_KEYS,
    SCROLL_PAUSE,
    SETTLE,
    SETTLE_TYPED,
    TYPING_PAUSE,
    WM_MOUSEWHEEL,
    Step,
    TeachRecorder,
    TeachSession,
    WheelNotches,
    describe,
    pynput_key_name,
    save_session,
    wheel_filter,
)


class Clock:
    def __init__(self):
        self.now = 128.0  # a whole binary number, so the boundaries below are exact

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


def shot(label: str, width=1920, height=1080, left=0, top=0) -> Shot:
    return Shot(label.encode(), 1280, 720, width, height, left, top)


class Harness:
    """A recorder with a fake screen, clock and window title; ``snap`` takes a screenshot labelled ``label``."""

    def __init__(self, title="Notepad", under=None, **settings):
        self.clock = Clock()
        self.title = title
        self.under = under  # the window under the pointer, when a test sets one
        self.label = "s0"
        self.screen = {"width": 1920, "height": 1080, "left": 0, "top": 0}
        self.recorder = TeachRecorder(
            LiveSettings(**settings), lambda: shot(self.label, **self.screen), lambda: self.title, clock=self.clock,
            window_reader=lambda point: self.under if self.under is not None else self.title,
        )

    def snap(self, label, wait=0.3):
        self.label = label
        self.recorder.snap()
        self.clock.tick(wait)

    def steps(self):
        return self.recorder.session.steps

    def actions(self):
        return [step.target["action"] for step in self.steps()]

    def images(self):
        return [step.shot.image if step.shot is not None else None for step in self.steps()]


class ConstantTests(unittest.TestCase):
    def test_timings(self):
        self.assertEqual((INTERVAL, SETTLE, SETTLE_TYPED), (0.1, 0.15, 0.05))


class ClickTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.h.recorder.start("open the file menu")

    def test_a_click_is_recorded_on_the_screen_from_before_it(self):
        self.h.snap("before")
        self.h.snap("same-moment", wait=0)  # taken as the click happened: not from before it
        self.h.recorder.on_click(960, 540, "left", True)
        step = self.h.steps()[0]
        self.assertEqual(step.target, {"thought": "", "say": "", "action": "click", "x": 500, "y": 500})
        self.assertEqual(step.shot.image, b"before")

    def test_a_screenshot_is_stamped_when_its_grab_starts(self):
        # Taking a screenshot takes time; the moment it shows is the start of the grab, not its end.
        clock = Clock()

        def slow_capture():
            clock.tick(0.12)
            return shot("menu closed")

        recorder = TeachRecorder(LiveSettings(), slow_capture, lambda: "Notepad", clock=clock)
        recorder.start("x")
        recorder.snap()  # starts at 128.0, ends at 128.12
        clock.now = 128.06
        recorder.on_click(10, 10, "left", True)
        self.assertEqual(recorder.session.steps[0].shot.image, b"menu closed")

    def test_positions_are_0_to_1000_of_the_monitor(self):
        self.h.snap("s")
        for x, y, expected in ((0, 0, (0, 0)), (1919, 1079, (999, 999)), (192, 108, (100, 100))):
            self.h.clock.tick(1)
            self.h.recorder.on_click(x, y, "right", True)
            self.assertEqual((self.h.steps()[-1].target["x"], self.h.steps()[-1].target["y"]), expected)
        self.assertEqual(self.h.actions(), ["right_click"] * 3)

    def test_clicks_off_the_main_monitor_are_left_out(self):
        self.h.snap("s")
        for x, y in ((1920, 10), (10, 1080), (-1, 10), (10, -1)):
            self.h.recorder.on_click(x, y, "left", True)
        self.assertEqual(self.h.steps(), [])

    def test_a_monitor_that_does_not_start_at_zero(self):
        self.h.screen.update(left=-1920, top=0)
        self.h.snap("s")
        self.h.recorder.on_click(-960, 540, "left", True)
        self.assertEqual((self.h.steps()[0].target["x"], self.h.steps()[0].target["y"]), (500, 500))

    def test_releases_and_other_buttons_are_ignored(self):
        self.h.snap("s")
        self.h.recorder.on_click(100, 100, "left", False)
        self.h.recorder.on_click(100, 100, "middle", True)
        self.assertEqual(self.h.steps(), [])

    def test_two_quick_clicks_in_one_place_are_a_double_click(self):
        self.h.snap("s")
        self.h.clock.now = 128.0 + 0.5
        self.h.recorder.on_click(100, 100, "left", True)
        self.h.clock.now = 128.0 + 0.5 + DOUBLE_CLICK_SECONDS
        self.h.recorder.on_click(106, 94, "left", True)
        self.assertEqual(self.h.actions(), ["double_click"])
        self.assertEqual(self.h.images(), [b"s"])

    def test_slow_or_distant_second_clicks_stay_separate(self):
        self.h.snap("s")
        self.h.recorder.on_click(100, 100, "left", True)
        self.h.clock.tick(DOUBLE_CLICK_SECONDS + 0.01)
        self.h.recorder.on_click(100, 100, "left", True)
        self.h.clock.tick(0.1)
        self.h.recorder.on_click(107, 100, "left", True)
        self.h.clock.tick(0.1)
        self.h.recorder.on_click(107, 107, "left", True)
        self.assertEqual(self.h.actions(), ["click"] * 4)

    def test_no_screenshot_yet_means_no_step(self):
        self.h.recorder.on_click(10, 10, "left", True)
        self.assertEqual(self.h.steps(), [])

    def test_a_click_with_keys_held_is_history_only(self):
        # Ctrl+click adds to a selection: saved as a plain click it would teach the wrong thing.
        self.h.snap("s")
        self.h.recorder.on_press("ctrl")
        self.h.recorder.on_click(960, 540, "left", True)
        step = self.h.steps()[0]
        self.assertIsNone(step.shot)
        self.assertEqual(step.described, "ctrl+click at (500, 500)")
        self.assertIn("keys held down", step.why)

    def test_a_held_click_is_not_merged_into_a_double_click(self):
        self.h.snap("s")
        self.h.recorder.on_click(100, 100, "left", True)
        self.h.recorder.on_press("shift")
        self.h.recorder.on_click(100, 100, "left", True)
        self.assertEqual(self.h.actions(), ["click", "click"])


class TooSoonTests(unittest.TestCase):
    """A screenshot taken before the last step had time to show does not go with the next step."""

    def setUp(self):
        self.h = Harness()
        self.h.recorder.start("save the file")

    def test_a_menu_item_clicked_before_any_screenshot_shows_the_menu(self):
        self.h.snap("no menu", wait=0.05)
        self.h.recorder.on_click(10, 10, "left", True)  # opens the menu at 128.05
        self.h.snap("menu drawing", wait=0.2)  # 128.05: no later than the click
        self.h.recorder.on_click(20, 40, "left", True)  # 128.25: Save
        self.assertEqual(self.h.images(), [b"no menu", None])
        self.assertIn("too soon", self.h.steps()[1].why)

    def test_a_screenshot_once_the_step_has_shown_is_used(self):
        self.h.snap("no menu", wait=0.05)
        self.h.recorder.on_click(10, 10, "left", True)  # 128.05
        self.h.clock.tick(SETTLE)
        self.h.snap("menu open", wait=0.1)  # exactly SETTLE after the click
        self.h.recorder.on_click(20, 40, "left", True)
        self.assertEqual(self.h.images(), [b"no menu", b"menu open"])

    def test_typing_shows_sooner_than_a_click(self):
        self.h.snap("empty field", wait=0.5)
        for character in "ok":
            self.h.recorder.on_press(character)  # both at 128.5
        self.h.clock.now = 128.5 + SETTLE_TYPED
        self.h.snap("ok typed", wait=0.01)  # exactly SETTLE_TYPED after the last key
        self.h.recorder.on_press("enter")
        self.assertEqual(self.h.images(), [b"empty field", b"ok typed"])

    def test_a_screenshot_just_before_the_typing_shows_is_too_soon(self):
        self.h.snap("empty field", wait=0.5)
        self.h.recorder.on_press("o")  # 128.5
        self.h.clock.now = 128.5 + SETTLE_TYPED - 0.01
        self.h.snap("maybe typed", wait=0.02)
        self.h.recorder.on_press("enter")
        self.assertEqual(self.h.images(), [b"empty field", None])


class TypingTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.h.recorder.start("write a note")
        self.h.snap("field-focused")

    def type(self, text):
        for character in text:
            self.h.recorder.on_press(character)
            self.h.clock.tick(0.05)

    def test_keys_in_a_row_become_one_type_step_with_the_screen_before_the_first(self):
        self.type("Hi")
        self.h.snap("half-typed", wait=0.05)
        self.h.recorder.on_press("space")
        self.type("you")
        self.h.recorder.on_press("enter")
        self.assertEqual(self.h.actions(), ["type", "press"])
        self.assertEqual(self.h.steps()[0].target["text"], "Hi you")
        self.assertEqual(self.h.steps()[0].shot.image, b"field-focused")
        self.assertEqual(self.h.steps()[1].target["keys"], ["enter"])

    def test_backspace_fixes_the_typing_instead_of_being_a_step(self):
        self.type("helo")
        self.h.recorder.on_press("backspace")
        self.type("lo")
        self.h.recorder.on_press("tab")
        self.assertEqual(self.h.steps()[0].target["text"], "hello")
        self.assertEqual(self.h.actions(), ["type", "press"])

    def test_backspace_with_nothing_typed_is_a_key(self):
        self.h.recorder.on_press("backspace")
        self.assertEqual(self.h.steps()[0].target["keys"], ["backspace"])

    def test_a_pause_ends_the_typing(self):
        self.type("one")
        self.h.clock.tick(TYPING_PAUSE)
        self.h.snap("later")
        self.assertEqual(self.h.actions(), ["type"])
        self.type("two")
        self.h.recorder.on_click(5, 5, "left", True)
        self.assertEqual(self.h.actions(), ["type", "type", "click"])

    def test_the_pause_boundary_is_exact(self):
        self.h.clock.now = 256.0
        self.h.recorder.on_press("a")
        self.h.clock.now = 256.0 + TYPING_PAUSE - 0.5
        self.h.snap("almost")
        self.assertEqual(self.h.actions(), [])
        self.h.clock.now = 256.0 + TYPING_PAUSE
        self.h.snap("exactly")
        self.assertEqual(self.h.actions(), ["type"])

    def test_a_shorter_pause_does_not(self):
        self.type("one")
        self.h.clock.tick(TYPING_PAUSE - 0.2)
        self.h.snap("still typing")
        self.assertEqual(self.h.actions(), [])

    def test_shift_is_typing_and_ctrl_is_a_hotkey(self):
        self.h.recorder.on_press("shift")
        self.type("A")
        self.h.recorder.on_release("shift")
        self.h.recorder.on_press("ctrl")
        self.h.recorder.on_press("s")
        self.h.recorder.on_release("ctrl")
        self.h.recorder.on_press("ctrl")
        self.h.recorder.on_press("shift")
        self.h.recorder.on_press("t")
        self.assertEqual(self.h.actions(), ["type", "hotkey", "hotkey"])
        self.assertEqual(self.h.steps()[0].target["text"], "A")
        self.assertEqual(self.h.steps()[1].target["keys"], ["ctrl", "s"])
        self.assertEqual(self.h.steps()[2].target["keys"], ["ctrl", "shift", "t"])

    def test_shift_with_a_named_key_is_a_hotkey(self):
        self.h.recorder.on_press("shift")
        self.h.recorder.on_press("tab")
        self.h.recorder.on_press("enter")
        self.h.recorder.on_press("space")
        session = self.h.recorder.stop()
        self.assertEqual([step.target.get("keys") or step.target.get("text") for step in session.steps],
                         [["shift", "tab"], ["shift", "enter"], " "])

    def test_nexus_live_hotkeys_are_never_recorded(self):
        for keys in (("ctrl", "alt", "q"), ("ctrl", "alt", "space"), ("ctrl", "alt", "y"), ("ctrl", "alt", "n")):
            for key in keys:
                self.h.recorder.on_press(key)
            for key in keys:
                self.h.recorder.on_release(key)
        self.assertEqual(self.h.steps(), [])
        self.h.recorder.on_press("ctrl")
        self.h.recorder.on_press("alt")
        self.h.recorder.on_press("t")
        self.assertEqual(self.h.steps()[0].target["keys"], ["ctrl", "alt", "t"])


class KeysNexusNeverPressesTests(unittest.TestCase):
    """A key NEXUS Live cannot press, once recorded, made the whole session fail to save."""

    def setUp(self):
        self.h = Harness()
        self.h.recorder.start("write a note")
        self.h.snap("s")

    def test_keys_it_cannot_press_are_not_steps(self):
        for key in ("caps_lock", "print_screen", "media_volume_up", "num_lock", "menu", "pause", "f13"):
            self.h.recorder.on_press(key)
        self.h.recorder.on_press("ctrl")
        self.h.recorder.on_press("f13")
        self.assertEqual(self.h.steps(), [])

    def test_caps_lock_does_not_break_a_run_of_typing(self):
        self.h.recorder.on_press("h")
        self.h.recorder.on_press("caps_lock")
        self.h.recorder.on_press("I")
        self.assertEqual([step.target for step in self.h.recorder.stop().steps],
                         [{"thought": "", "say": "", "action": "type", "text": "hI"}])

    def test_altgr_characters_are_typing(self):
        # AltGr is Ctrl+Alt on Windows: English (India) AltGr+4 is the rupee sign, German AltGr+Q is @.
        for held, made in ((("ctrl", "alt"), "₹"), (("ctrl", "alt"), "@"), (("ctrl", "alt"), "["), (("altgr",), "€")):
            for key in held:
                self.h.recorder.on_press(key)
            self.h.recorder.on_press(made, made)
            for key in held:
                self.h.recorder.on_release(key)
        self.assertEqual([step.target["text"] for step in self.h.recorder.stop().steps], ["₹@[€"])

    def test_altgr_typing_is_saved(self):
        h = self.h
        h.recorder.on_press("ctrl")
        h.recorder.on_press("alt")
        h.recorder.on_press("4", "₹")
        h.recorder.on_release("alt")
        h.recorder.on_release("ctrl")
        h.recorder.on_press("5")
        h.recorder.on_press("0")
        session = h.recorder.stop()
        self.assertEqual([step.target for step in session.steps][0]["text"], "₹50")
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(len(save_session(session, Path(folder))), 1)

    def test_a_shortcut_with_no_character_is_still_a_shortcut(self):
        self.h.recorder.on_press("ctrl")
        self.h.recorder.on_press("alt")
        self.h.recorder.on_press("t", None)
        self.assertEqual(self.h.steps()[0].target["keys"], ["ctrl", "alt", "t"])

    def test_the_windows_key_is_never_recorded(self):
        # Win then R opened the Run box; saved as "press win" it taught a step that is always refused.
        self.h.recorder.on_press("win")
        self.h.recorder.on_press("r", "r")
        self.h.recorder.on_release("win")
        self.h.recorder.on_press("win")
        self.h.recorder.on_release("win")
        self.assertEqual(self.h.steps(), [])
        self.h.recorder.on_press("a", "a")  # once it is up, keys are recorded again
        self.assertEqual([step.target["text"] for step in self.h.recorder.stop().steps], ["a"])


class PrivacyTests(unittest.TestCase):
    def test_nothing_is_recorded_in_blocked_windows(self):
        h = Harness(title="Bitwarden - Vault")
        h.recorder.start("log in")
        h.snap("s")
        h.recorder.on_click(10, 10, "left", True)
        for character in "secret":
            h.recorder.on_press(character)
        h.recorder.on_press("enter")
        h.recorder.on_scroll(10, 10, 0, -2)
        self.assertEqual(h.steps(), [])

    def test_nothing_is_recorded_in_a_window_whose_title_cannot_be_read(self):
        for title in (None, ""):
            with self.subTest(title=title):
                h = Harness(title=title)
                h.recorder.start("log in")
                h.snap("s")
                h.recorder.on_click(10, 10, "left", True)
                for character in "hunter2":
                    h.recorder.on_press(character)
                h.recorder.on_press("enter")
                self.assertEqual(h.steps(), [])

    def test_nothing_is_recorded_in_the_whole_title_windows(self):
        h = Harness(title="Run")
        h.recorder.start("open something")
        h.snap("s")
        h.recorder.on_press("c", "c")
        h.recorder.on_press("enter")
        self.assertEqual(h.steps(), [])

    def test_nothing_is_recorded_in_its_own_console(self):
        # VS Code's own terminal: not on the blocked list, but teach mode was started from it.
        h = Harness(title="nexus - Visual Studio Code")
        h.recorder.own_titles.add("nexus - Visual Studio Code")
        h.recorder.start("x")
        h.snap("s")
        h.recorder.on_press("a", "a")
        self.assertEqual(h.steps(), [])

    def test_a_click_that_lands_in_a_blocked_window_beside_is_not_recorded(self):
        h = Harness(title="Untitled - Notepad", under="Database.kdbx - KeePass")
        h.recorder.start("x")
        h.snap("s")
        h.recorder.on_click(10, 10, "left", True)
        h.recorder.on_scroll(10, 10, 0, -2)
        self.assertEqual(h.steps(), [])

    def test_typing_carried_into_a_blocked_window_is_dropped(self):
        h = Harness()
        h.recorder.start("note")
        h.snap("s")
        for character in "abc":
            h.recorder.on_press(character)
        h.title = "Windows Security"
        h.recorder.on_press("d")
        h.title = "Notepad"
        h.recorder.on_press("enter")
        self.assertEqual(h.actions(), ["press"])

    def test_a_screenshot_taken_with_a_blocked_window_in_front_goes_with_no_step(self):
        h = Harness()
        h.recorder.start("note")
        h.title = "KeePass"
        h.snap("KEEPASS WITH THE BANK PASSWORD SHOWN", wait=0.3)
        h.title = "Untitled - Notepad"
        h.recorder.on_click(10, 10, "left", True)
        self.assertEqual(h.images(), [None])
        self.assertIn("not recorded was in front", h.steps()[0].why)

    def test_a_screenshot_during_which_the_window_changed_goes_with_no_step(self):
        titles = iter(["KeePass", "Untitled - Notepad"])
        h = Harness()
        h.recorder.start("note")
        h.recorder.title_reader = lambda: next(titles)
        h.snap("switching", wait=0.3)
        h.recorder.title_reader = lambda: "Untitled - Notepad"
        h.recorder.on_click(10, 10, "left", True)
        self.assertEqual(h.images(), [None])

    def test_blocked_windows_are_covered_in_every_screenshot(self):
        covered = []
        layout = [("Untitled - Notepad", (0, 0, 800, 600)), ("KeePass", (700, 0, 1920, 1080))]
        recorder = TeachRecorder(
            LiveSettings(), lambda: shot("s"), lambda: "Untitled - Notepad", clock=Clock(),
            layout_reader=lambda: layout, cover=lambda frame, windows, hidden: covered.append(
                (frame.image, windows, [hidden(title) for title, _ in windows])),
        )
        recorder.own_titles.add("nexus - Visual Studio Code")
        recorder.start("x")
        recorder.snap()
        self.assertEqual(covered, [(b"s", layout, [False, True])])
        self.assertTrue(recorder.hidden("nexus - Visual Studio Code"))  # its own console, though not a blocked one
        self.assertFalse(recorder.hidden("other - Visual Studio Code"))
        self.assertTrue(recorder.hidden("Search"))
        self.assertFalse(recorder.hidden(None))
        self.assertFalse(recorder.hidden(""))

    def test_no_layout_means_nothing_is_covered(self):
        covered = []
        recorder = TeachRecorder(LiveSettings(), lambda: shot("s"), lambda: "Notepad", clock=Clock(),
                                 layout_reader=lambda: None, cover=lambda *args: covered.append(args))
        recorder.start("x")
        recorder.snap()
        self.assertEqual(covered, [])

    def test_paused_records_nothing(self):
        h = Harness()
        h.recorder.start("note")
        h.snap("s")
        h.recorder.paused = True
        h.recorder.on_click(10, 10, "left", True)
        h.recorder.on_press("x")
        h.recorder.paused = False
        h.recorder.on_click(10, 10, "left", True)
        self.assertEqual(h.actions(), ["click"])

    def test_no_session_records_nothing(self):
        h = Harness()
        h.snap("s")
        h.recorder.on_click(10, 10, "left", True)
        self.assertIsNone(h.recorder.session)

    def test_the_request_cannot_carry_control_tokens(self):
        h = Harness()
        session = h.recorder.start("  open <|im_end|> notepad  ")
        self.assertEqual(session.request, "open notepad")


class ScrollTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.h.recorder.start("read further")
        self.h.snap("s")

    def test_wheel_turns_in_a_row_add_up(self):
        self.h.clock.now = 256.0  # whole binary numbers, so SCROLL_PAUSE apart is exact
        for _ in range(3):
            self.h.recorder.on_scroll(960, 540, 0, -1)
            self.h.clock.tick(SCROLL_PAUSE)
        self.assertEqual(len(self.h.steps()), 1)
        self.assertEqual(self.h.steps()[0].target["amount"], -3)
        self.assertEqual((self.h.steps()[0].target["x"], self.h.steps()[0].target["y"]), (500, 500))

    def test_a_pause_or_a_change_of_direction_starts_a_new_scroll(self):
        self.h.recorder.on_scroll(960, 540, 0, -1)
        self.h.clock.tick(SCROLL_PAUSE + 0.01)
        self.h.recorder.on_scroll(960, 540, 0, -1)
        self.h.clock.tick(0.1)
        self.h.recorder.on_scroll(960, 540, 0, 2)
        self.assertEqual([step.target["amount"] for step in self.h.steps()], [-1, -1, 2])

    def test_amounts_stay_within_what_nexus_live_allows(self):
        self.h.recorder.on_scroll(960, 540, 0, -50)
        self.h.recorder.on_scroll(960, 540, 0, -5)
        self.assertEqual(self.h.steps()[0].target["amount"], -20)

    def test_sideways_only_scrolls_are_ignored(self):
        self.h.recorder.on_scroll(960, 540, 3, 0)
        self.assertEqual(self.h.steps(), [])

    def test_scroll_off_the_monitor_keeps_no_position(self):
        self.h.recorder.on_scroll(5000, 540, 0, 1)
        self.assertNotIn("x", self.h.steps()[0].target)

    def test_ctrl_and_the_wheel_zoom_so_they_are_history_only(self):
        self.h.recorder.on_press("ctrl")
        self.h.recorder.on_scroll(960, 540, 0, 1)
        self.assertIsNone(self.h.steps()[0].shot)
        self.assertEqual(self.h.steps()[0].described, "ctrl+scroll up 1")


class WheelTests(unittest.TestCase):
    def test_a_touchpad_s_small_deltas_add_up_to_notches(self):
        # 12 deltas of 20, a two-notch swipe; pynput alone made "up" nothing and "down" 12 notches.
        wheel = WheelNotches()
        self.assertEqual(sum(wheel.add(20) for _ in range(12)), 2)
        wheel = WheelNotches()
        self.assertEqual(sum(wheel.add(-20) for _ in range(12)), -2)

    def test_a_mouse_wheel_is_a_notch_a_turn(self):
        wheel = WheelNotches()
        self.assertEqual([wheel.add(120), wheel.add(-120), wheel.add(240)], [1, -1, 2])

    def test_turning_back_starts_again(self):
        wheel = WheelNotches()
        self.assertEqual(wheel.add(100), 0)
        self.assertEqual(wheel.add(-30), 0)  # the 100 up is not taken off: it was the other way
        self.assertEqual(wheel.add(-90), -1)
        self.assertEqual(wheel.add(0), 0)

    def test_the_filter_scrolls_by_notches_and_keeps_pynput_out(self):
        scrolled = []
        recorder = types.SimpleNamespace(on_scroll=lambda x, y, dx, dy: scrolled.append((x, y, dx, dy)))
        event_filter = wheel_filter(recorder)

        def wheel(delta):
            data = types.SimpleNamespace(mouseData=(delta & 0xFFFF) << 16, pt=types.SimpleNamespace(x=5, y=6))
            return event_filter(WM_MOUSEWHEEL, data)

        results = [wheel(-40) for _ in range(6)]
        self.assertEqual(results, [False] * 6)
        self.assertEqual(scrolled, [(5, 6, 0, -1), (5, 6, 0, -1)])
        self.assertTrue(event_filter(0x0201, types.SimpleNamespace()))  # a click goes on to pynput


class ReviewTests(unittest.TestCase):
    def session(self):
        session = TeachSession("note")
        for index, target in enumerate(
            [{"action": "type", "text": "first"}, {"action": "click", "x": 1, "y": 1}, {"action": "type", "text": "second"}]
        ):
            session.steps.append(Step(target, shot(str(index)), "Notepad", float(index)))
        return session

    def review(self, answer):
        from nexus.live.teach import _review_typing

        shown = []
        return _review_typing(self.session(), lambda prompt: answer, shown.append), shown

    def test_everything_typed_is_shown(self):
        drop, shown = self.review("")
        self.assertEqual(drop, [])
        self.assertEqual(shown, ["You typed:", "  1. 'first'", "  2. 'second'"])

    def test_numbers_pick_the_typing_steps_to_drop(self):
        self.assertEqual(self.review("2")[0], [2])
        self.assertEqual(self.review("1, 2")[0], [0, 2])

    def test_numbers_out_of_range_or_not_numbers_are_ignored(self):
        self.assertEqual(self.review("0 3 x -1")[0], [])

    def test_no_typing_asks_nothing(self):
        from nexus.live.teach import _review_typing

        asked = []
        self.assertEqual(_review_typing(TeachSession("x"), lambda prompt: asked.append(prompt) or "", lambda line: None), [])
        self.assertEqual(asked, [])

    def test_typing_kept_only_as_history_is_reviewed_too(self):
        session = TeachSession("note")
        session.steps.append(Step({"action": "type", "text": "hunter2"}, None, "Notepad", 0.0, why="too soon"))
        self.assertEqual(session.typed(), [(0, "hunter2")])


class StopAndSaveTests(unittest.TestCase):
    def record(self):
        h = Harness()
        h.recorder.start("  rename   the file ")
        h.snap("desktop")
        h.recorder.on_click(100, 100, "right", True)
        h.clock.tick(1)
        h.snap("menu open")
        h.recorder.on_click(150, 200, "left", True)
        h.clock.tick(1)
        h.snap("name selected")
        for character in "report":
            h.recorder.on_press(character)
            h.clock.tick(0.05)
        h.snap("typed")
        h.recorder.on_press("enter")
        h.clock.tick(1)
        h.snap("renamed")
        return h

    def test_stop_adds_done_on_the_last_screen(self):
        h = self.record()
        session = h.recorder.stop()
        self.assertEqual([step.target["action"] for step in session.steps], ["right_click", "click", "type", "press", "done"])
        self.assertEqual([step.shot.image for step in session.steps], [b"desktop", b"menu open", b"name selected", b"typed", b"renamed"])
        self.assertEqual(session.steps[-1].target["say"], "Done.")
        self.assertEqual(session.request, "rename the file")
        self.assertIsNone(h.recorder.session)

    def test_stop_with_nothing_done_adds_no_done(self):
        h = Harness()
        h.recorder.start("nothing")
        h.snap("s")
        self.assertEqual(h.recorder.stop().steps, [])

    def test_stop_in_a_blocked_window_adds_no_done(self):
        h = self.record()
        h.title = "PayPal - Checkout"
        self.assertNotIn("done", [step.target["action"] for step in h.recorder.stop().steps])

    def test_stop_too_soon_after_the_last_step_adds_no_done(self):
        h = self.record()
        h.recorder.on_press("f5")  # after the last screenshot, with none since
        self.assertEqual(h.recorder.stop().steps[-1].target["action"], "press")

    def test_saved_examples_carry_the_steps_so_far(self):
        h = self.record()
        session = h.recorder.stop()
        with tempfile.TemporaryDirectory() as folder:
            examples = save_session(session, Path(folder))
            self.assertEqual(len(examples), 5)
            self.assertEqual(examples[0].steps, [])
            self.assertEqual(examples[3].steps, [describe(step.target) for step in session.steps[:3]])
            self.assertEqual(examples[3].steps[2], "type 'report'")
            self.assertTrue(all(example.group == f"teach-{session.id}" for example in examples))
            self.assertEqual((Path(folder) / examples[1].image).read_bytes(), b"menu open")
            self.assertEqual([example.id for example in read_examples(Path(folder))], [example.id for example in examples])

    def test_dropped_typing_is_left_out_everywhere(self):
        h = self.record()
        session = h.recorder.stop()
        self.assertEqual(session.typed(), [(2, "report")])
        with tempfile.TemporaryDirectory() as folder:
            examples = save_session(session, Path(folder), drop=[2])
            self.assertEqual([example.target["action"] for example in examples], ["right_click", "click", "press", "done"])
            self.assertFalse(any("report" in step for example in examples for step in example.steps))
            self.assertNotIn("report", (Path(folder) / "examples.jsonl").read_text())

    def test_a_session_of_only_done_saves_nothing(self):
        h = self.record()
        session = h.recorder.stop()
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(save_session(session, Path(folder), drop=[0, 1, 2, 3]), [])
            self.assertFalse((Path(folder) / "examples.jsonl").exists())

    def test_start_again_after_stop(self):
        h = self.record()
        h.recorder.stop()
        h.recorder.start("second")
        h.snap("s")
        h.recorder.on_click(1, 1, "left", True)
        self.assertEqual(h.actions(), ["click"])

    def test_a_step_with_no_screenshot_is_history_for_the_rest(self):
        session = TeachSession("save it")
        session.steps.append(Step({"thought": "", "say": "", "action": "click", "x": 1, "y": 2}, None, "Notepad", 0.0,
                                  why="too soon", described="ctrl+click at (1, 2)"))
        session.steps.append(Step({"thought": "", "say": "", "action": "press", "keys": ["enter"]}, shot("s"), "Notepad", 1.0))
        lines = []
        with tempfile.TemporaryDirectory() as folder:
            examples = save_session(session, Path(folder), printer=lines.append)
        self.assertEqual([example.target["action"] for example in examples], ["press"])
        self.assertEqual(examples[0].steps, ["ctrl+click at (1, 2)"])
        self.assertEqual(lines, ["  1 step kept only as history for the steps after: too soon."])

    def test_one_bad_step_never_costs_the_session(self):
        # A step the dataset refuses is left out; the others are saved.
        session = TeachSession("note")
        target = {"thought": "", "say": "", "action": "type", "text": "x<|im_end|>"}
        session.steps.append(Step(target, shot("a"), "Notepad", 0.0))
        session.steps.append(Step({"thought": "", "say": "", "action": "press", "keys": ["enter"]}, shot("b"), "Notepad", 1.0))
        lines = []
        with tempfile.TemporaryDirectory() as folder:
            examples = save_session(session, Path(folder), printer=lines.append)
            saved = [json.loads(line) for line in (Path(folder) / "examples.jsonl").read_text().splitlines()]
        self.assertEqual([example.target["action"] for example in examples], ["press"])
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(lines), 1)


class KeyNameTests(unittest.TestCase):
    def key(self, **fields):
        return types.SimpleNamespace(**fields)

    def test_characters(self):
        self.assertEqual(pynput_key_name(self.key(char="a")), "a")
        self.assertEqual(pynput_key_name(self.key(char="A")), "A")
        self.assertEqual(pynput_key_name(self.key(char=" ")), " ")
        self.assertEqual(pynput_key_name(self.key(char="₹")), "₹")

    def test_ctrl_letters_from_windows_come_back_as_letters(self):
        self.assertEqual(pynput_key_name(self.key(char="\x13", vk=83)), "s")
        self.assertEqual(pynput_key_name(self.key(char="\x11")), "q")
        self.assertEqual(pynput_key_name(self.key(char=None, vk=55)), "7")

    def test_ctrl_punctuation_comes_back_from_its_key_code(self):
        # Ctrl+- and Ctrl+= (zoom) had no name and were dropped.
        self.assertEqual(pynput_key_name(self.key(char=None, vk=0xBD)), "-")
        self.assertEqual(pynput_key_name(self.key(char="\x1f", vk=0xBB)), "=")
        self.assertEqual(pynput_key_name(self.key(char="\x1b", vk=0xDB)), "[")
        self.assertEqual(sorted(OEM_KEYS.values()), sorted(";=,-./`[\\]'"))

    def test_named_keys(self):
        cases = {"ctrl_l": "ctrl", "alt_gr": "altgr", "alt_r": "alt", "shift_r": "shift", "cmd": "win", "cmd_r": "win",
                 "page_up": "pageup", "esc": "esc", "enter": "enter", "f5": "f5", "space": "space", "caps_lock": "caps_lock"}
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(pynput_key_name(self.key(char=None, name=name)), expected)

    def test_unknown(self):
        self.assertIsNone(pynput_key_name(self.key(char=None)))
        self.assertIsNone(pynput_key_name(self.key(char=None, vk=0x05)))


if __name__ == "__main__":
    unittest.main()
