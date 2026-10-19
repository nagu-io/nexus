"""Tests for nexus.live.teach: turning what you do into NEXUS Vision training examples."""

import tempfile
import types
import unittest
from pathlib import Path

from nexus.live.data import read_examples
from nexus.live.screen import Shot
from nexus.live.settings import LiveSettings
from nexus.live.teach import (
    BEFORE,
    DOUBLE_CLICK_SECONDS,
    SCROLL_PAUSE,
    TYPING_PAUSE,
    TeachRecorder,
    describe,
    pynput_key_name,
    save_session,
)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


def shot(label: str, width=1920, height=1080, left=0, top=0) -> Shot:
    return Shot(label.encode(), 1280, 720, width, height, left, top)


class Harness:
    """A recorder with a fake screen, clock and window title; ``snap`` takes a screenshot labelled ``label``."""

    def __init__(self, title="Notepad", **settings):
        self.clock = Clock()
        self.title = title
        self.label = "s0"
        self.screen = {"width": 1920, "height": 1080, "left": 0, "top": 0}
        self.recorder = TeachRecorder(
            LiveSettings(**settings), lambda: shot(self.label, **self.screen), lambda: self.title, clock=self.clock
        )

    def snap(self, label, wait=0.3):
        self.label = label
        self.recorder.snap()
        self.clock.tick(wait)

    def steps(self):
        return self.recorder.session.steps

    def actions(self):
        return [step.target["action"] for step in self.steps()]


class ClickTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.h.recorder.start("open the file menu")

    def test_a_click_is_recorded_on_the_screen_from_before_it(self):
        self.h.snap("before")
        self.h.snap("after-click-was-already-on-screen", wait=BEFORE / 3)  # too close to the click to be the "before"
        self.h.recorder.on_click(960, 540, "left", True)
        step = self.h.steps()[0]
        self.assertEqual(step.target, {"thought": "", "say": "", "action": "click", "x": 500, "y": 500})
        self.assertEqual(step.shot.image, b"before")

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
        self.h.clock.now = 128.0  # whole binary numbers, so the boundary below is exact
        self.h.recorder.on_click(100, 100, "left", True)
        self.h.clock.now = 128.0 + DOUBLE_CLICK_SECONDS
        self.h.recorder.on_click(106, 94, "left", True)
        self.assertEqual(self.h.actions(), ["double_click"])

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
        self.h.clock.now = 128.0
        self.h.recorder.on_press("a")
        self.h.clock.now = 128.0 + TYPING_PAUSE - 0.5
        self.h.snap("almost")
        self.assertEqual(self.h.actions(), [])
        self.h.clock.now = 128.0 + TYPING_PAUSE
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


class ScrollTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.h.recorder.start("read further")
        self.h.snap("s")

    def test_wheel_turns_in_a_row_add_up(self):
        self.h.clock.now = 128.0  # whole binary numbers, so SCROLL_PAUSE apart is exact
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


class ReviewTests(unittest.TestCase):
    def session(self):
        from nexus.live.teach import Step, TeachSession

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
        from nexus.live.teach import TeachSession, _review_typing

        asked = []
        self.assertEqual(_review_typing(TeachSession("x"), lambda prompt: asked.append(prompt) or "", lambda line: None), [])
        self.assertEqual(asked, [])


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
        h.recorder.on_press("enter")
        h.clock.tick(1)
        h.snap("renamed")
        return h

    def test_stop_adds_done_on_the_last_screen(self):
        h = self.record()
        session = h.recorder.stop()
        self.assertEqual([step.target["action"] for step in session.steps], ["right_click", "click", "type", "press", "done"])
        self.assertEqual(session.steps[-1].shot.image, b"renamed")
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


class KeyNameTests(unittest.TestCase):
    def key(self, **fields):
        return types.SimpleNamespace(**fields)

    def test_characters(self):
        self.assertEqual(pynput_key_name(self.key(char="a")), "a")
        self.assertEqual(pynput_key_name(self.key(char="A")), "A")
        self.assertEqual(pynput_key_name(self.key(char=" ")), " ")

    def test_ctrl_letters_from_windows_come_back_as_letters(self):
        self.assertEqual(pynput_key_name(self.key(char="\x13", vk=83)), "s")
        self.assertEqual(pynput_key_name(self.key(char="\x11")), "q")
        self.assertEqual(pynput_key_name(self.key(char=None, vk=55)), "7")

    def test_named_keys(self):
        cases = {"ctrl_l": "ctrl", "alt_gr": "alt", "shift_r": "shift", "cmd": "win", "page_up": "pageup", "esc": "esc",
                 "enter": "enter", "f5": "f5", "space": "space"}
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(pynput_key_name(self.key(char=None, name=name)), expected)

    def test_unknown(self):
        self.assertIsNone(pynput_key_name(self.key(char=None)))


if __name__ == "__main__":
    unittest.main()
