"""Tests for nexus.live.policy: what NEXUS Live may do on its own, what it asks about, and what it refuses."""

import dataclasses
import unittest

from nexus.live.model import Decision, parse_decision
from nexus.live.policy import ALLOW, CONFIRM, REFUSE, SYSTEM_COMBOS, SYSTEM_KEYS, Policy, Verdict, lands_at_a_point, own_hotkeys
from nexus.live.settings import DEFAULT_BLOCKED_TITLES, DEFAULT_BLOCKED_WINDOWS, LiveSettings

NOTEPAD = "Untitled - Notepad"

CLICK = Decision("click", x=10.0, y=20.0)
DOUBLE_CLICK = Decision("double_click", x=10.0, y=20.0)
RIGHT_CLICK = Decision("right_click", x=10.0, y=20.0)
SCROLL = Decision("scroll", amount=-3)
SCROLL_AT = Decision("scroll", x=10.0, y=20.0, amount=3)
TYPE = Decision("type", text="hello")
PRESS = Decision("press", keys=("enter",))
HOTKEY = Decision("hotkey", keys=("ctrl", "s"))
WAIT = Decision("wait", seconds=1.0)
ANSWER = Decision("answer", say="It is noon.")
DONE = Decision("done", say="Done.")

POINTER_STEPS = (CLICK, DOUBLE_CLICK, RIGHT_CLICK, SCROLL, SCROLL_AT)
KEYBOARD_STEPS = (TYPE, PRESS, HOTKEY)
ACTING_STEPS = POINTER_STEPS + KEYBOARD_STEPS
QUIET_STEPS = (WAIT, ANSWER, DONE)

WINDOWS_KEY = "I don't use the Windows key: it opens the Run box and system menus."
SYSTEM_COMBO = "I don't press system key combinations like that one."
UNKNOWN_WINDOW = "I can't tell which window is in front."
UNKNOWN_WINDOW_ALLOWLIST = "I can't tell which window is in front, so I can't check it is one I'm allowed to use."
UNKNOWN_POINT = "I can't tell which window is under that point."
UNKNOWN_POINT_ALLOWLIST = "I can't tell which window is under that point, so I can't check it is one I'm allowed to use."
OWN_HOTKEY = "That is one of my own hotkeys, so I don't press it."
SAME = object()


def check(decision, title=NOTEPAD, point=SAME, **settings):
    """The verdict, with the window under a click or scroll the same as the one in front unless ``point`` says otherwise."""
    return Policy(LiveSettings(**settings)).check(decision, title, title if point is SAME else point)


def keys(*names):
    return Decision("hotkey", keys=names)


class ConstantTests(unittest.TestCase):
    def test_outcomes(self):
        self.assertEqual((ALLOW, CONFIRM, REFUSE), ("allow", "confirm", "refuse"))

    def test_system_keys(self):
        self.assertEqual(SYSTEM_KEYS, frozenset({"win"}))
        self.assertEqual(
            SYSTEM_COMBOS,
            (frozenset({"ctrl", "alt", "delete"}), frozenset({"ctrl", "shift", "esc"}), frozenset({"ctrl", "esc"})),
        )

    def test_steps_that_land_at_a_point(self):
        self.assertEqual([lands_at_a_point(step) for step in POINTER_STEPS], [True] * 5)
        self.assertEqual([lands_at_a_point(step) for step in KEYBOARD_STEPS + QUIET_STEPS], [False] * 6)

    def test_verdict(self):
        self.assertEqual(Verdict(ALLOW), Verdict("allow", ""))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            Verdict(ALLOW).outcome = REFUSE  # type: ignore[misc]


class QuietStepTests(unittest.TestCase):
    def test_quiet_steps_are_allowed_anywhere(self):
        for decision in QUIET_STEPS:
            for title in ("Online Banking - Chrome", None, "Calculator", NOTEPAD, ""):
                with self.subTest(action=decision.action, title=title):
                    self.assertEqual(check(decision, title, allowed_windows=("notepad",)), Verdict(ALLOW))
                    self.assertEqual(check(decision, title, confirm="risky"), Verdict(ALLOW))

    def test_acting_steps_are_not_quiet(self):
        for decision in ACTING_STEPS:
            with self.subTest(action=decision.action):
                self.assertEqual(check(decision, "Online Banking - Chrome").outcome, REFUSE)


class SystemKeyTests(unittest.TestCase):
    def test_windows_key_alone_or_with_others(self):
        for decision in (Decision("press", keys=("win",)), keys("win", "r"), keys("ctrl", "win", "d"), keys("r", "win")):
            with self.subTest(keys=decision.keys):
                self.assertEqual(check(decision), Verdict(REFUSE, WINDOWS_KEY))
                self.assertEqual(check(decision, confirm="risky"), Verdict(REFUSE, WINDOWS_KEY))

    def test_windows_key_is_refused_before_the_window_is_looked_at(self):
        self.assertEqual(check(keys("win", "r"), None), Verdict(REFUSE, WINDOWS_KEY))
        self.assertEqual(check(keys("win", "r"), "main.py - Notepad", allowed_windows=("notepad",)), Verdict(REFUSE, WINDOWS_KEY))

    def test_every_name_for_the_windows_key(self):
        for name in ("win", "Windows", "super", "META", "cmd", "command"):
            with self.subTest(name=name):
                decision = parse_decision({"action": "hotkey", "keys": [name, "r"]})
                self.assertEqual(check(decision), Verdict(REFUSE, WINDOWS_KEY))

    def test_system_combinations_in_any_order(self):
        for names in (
            ("ctrl", "alt", "delete"),
            ("delete", "alt", "ctrl"),
            ("ctrl", "shift", "esc"),
            ("esc", "ctrl", "shift"),
            ("ctrl", "alt", "shift", "delete"),
            ("ctrl", "shift", "esc", "a"),
        ):
            with self.subTest(keys=names):
                self.assertEqual(check(keys(*names)), Verdict(REFUSE, SYSTEM_COMBO))
                self.assertEqual(check(keys(*names), None), Verdict(REFUSE, SYSTEM_COMBO))

    def test_system_combinations_by_other_names(self):
        for names in (["Control", "Alt", "Del"], ["ctl", "shift", "escape"]):
            with self.subTest(keys=names):
                decision = parse_decision({"action": "hotkey", "keys": names})
                self.assertEqual(check(decision), Verdict(REFUSE, SYSTEM_COMBO))

    def test_ctrl_esc_opens_start_like_the_windows_key(self):
        for names in (("ctrl", "esc"), ("esc", "ctrl"), ("ctrl", "alt", "esc"), ("ctrl", "esc", "a")):
            with self.subTest(keys=names):
                self.assertEqual(check(keys(*names)), Verdict(REFUSE, SYSTEM_COMBO))
                self.assertEqual(check(keys(*names), confirm="risky"), Verdict(REFUSE, SYSTEM_COMBO))

    def test_part_of_a_system_combination_is_only_asked_about(self):
        for names in (
            ("ctrl", "alt"),
            ("alt", "delete"),
            ("ctrl", "delete"),
            ("ctrl", "shift"),
            ("shift", "esc"),
            ("alt", "esc"),
            ("ctrl", "shift", "delete"),
            ("alt", "shift", "esc"),
        ):
            with self.subTest(keys=names):
                self.assertEqual(check(keys(*names)), Verdict(CONFIRM))
        self.assertEqual(check(Decision("press", keys=("esc",))), Verdict(CONFIRM))


class UnknownWindowTests(unittest.TestCase):
    def test_unknown_window_is_asked_about(self):
        for decision in ACTING_STEPS:
            with self.subTest(action=decision.action):
                self.assertEqual(check(decision, None), Verdict(CONFIRM, UNKNOWN_WINDOW))

    def test_unknown_window_is_asked_about_even_for_risky_mode_clicks(self):
        for decision in POINTER_STEPS:
            with self.subTest(action=decision.action):
                self.assertEqual(check(decision, None, confirm="risky"), Verdict(CONFIRM, UNKNOWN_WINDOW))

    def test_unknown_window_is_refused_when_only_some_windows_are_allowed(self):
        for decision in ACTING_STEPS:
            with self.subTest(action=decision.action):
                self.assertEqual(check(decision, None, allowed_windows=("notepad",)), Verdict(REFUSE, UNKNOWN_WINDOW_ALLOWLIST))


class BlockedWindowTests(unittest.TestCase):
    def test_blocked_windows_are_refused_for_every_acting_step(self):
        for decision in ACTING_STEPS:
            for confirm in ("all", "risky"):
                with self.subTest(action=decision.action, confirm=confirm):
                    self.assertEqual(
                        check(decision, "Chase Bank - Statements", confirm=confirm),
                        Verdict(REFUSE, "The window in front looks like 'bank', which I never act in. Please do this part yourself."),
                    )

    def test_blocked_beats_allowed(self):
        verdict = check(CLICK, "Online Banking - Google Chrome", allowed_windows=("chrome",))
        self.assertEqual(verdict.outcome, REFUSE)
        self.assertIn("'banking'", verdict.reason)

    def test_similar_words_are_not_blocked(self):
        self.assertEqual(check(TYPE, "Bankruptcy notes - Word"), Verdict(CONFIRM))
        self.assertEqual(check(CLICK, "Riverbank Cafe - Maps", confirm="risky"), Verdict(ALLOW))

    def test_the_settings_list_is_the_one_used(self):
        settings = {"blocked_windows": DEFAULT_BLOCKED_WINDOWS + ("slack",)}
        self.assertEqual(check(CLICK, "general - Slack", **settings).outcome, REFUSE)
        self.assertEqual(check(CLICK, "My Bank", blocked_windows=()), Verdict(CONFIRM))

    def test_terminals_and_password_managers(self):
        for title in ("Windows PowerShell", "Administrator: Command Prompt", "Bitwarden", "User Account Control"):
            with self.subTest(title=title):
                self.assertEqual(check(TYPE, title).outcome, REFUSE)


class AllowedWindowTests(unittest.TestCase):
    ALLOWED = {"allowed_windows": ("notepad", "visual studio code")}

    def test_allowed_windows(self):
        self.assertEqual(check(TYPE, "Untitled - Notepad", **self.ALLOWED), Verdict(CONFIRM))
        self.assertEqual(check(TYPE, "main.py - Visual Studio Code", **self.ALLOWED), Verdict(CONFIRM))
        self.assertEqual(check(CLICK, "Untitled - Notepad", confirm="risky", **self.ALLOWED), Verdict(ALLOW))

    def test_other_windows_are_refused(self):
        self.assertEqual(check(CLICK, "Calculator", **self.ALLOWED), Verdict(REFUSE, "'Calculator' is not one of the windows I'm allowed to use."))
        self.assertEqual(check(CLICK, "Calculator", confirm="risky", **self.ALLOWED).outcome, REFUSE)

    def test_allowed_names_are_whole_words(self):
        self.assertEqual(check(CLICK, "Untitled - Notepad", allowed_windows=("note",)).outcome, REFUSE)

    def test_a_window_with_no_title_is_not_on_the_list(self):
        self.assertEqual(check(CLICK, "", **self.ALLOWED), Verdict(REFUSE, "'' is not one of the windows I'm allowed to use."))

    def test_without_a_list_any_window_that_is_not_blocked_will_do(self):
        self.assertEqual(check(CLICK, "Calculator"), Verdict(CONFIRM))


class ConfirmModeTests(unittest.TestCase):
    def test_all_asks_before_every_acting_step(self):
        for decision in ACTING_STEPS:
            with self.subTest(action=decision.action):
                self.assertEqual(check(decision, confirm="all"), Verdict(CONFIRM))

    def test_risky_lets_clicks_and_scrolling_go_ahead(self):
        for decision in POINTER_STEPS:
            with self.subTest(action=decision.action):
                self.assertEqual(check(decision, confirm="risky"), Verdict(ALLOW))

    def test_risky_still_asks_before_typing_and_keys(self):
        for decision in KEYBOARD_STEPS:
            with self.subTest(action=decision.action):
                self.assertEqual(check(decision, confirm="risky"), Verdict(CONFIRM))

    def test_default_is_to_ask_about_everything(self):
        self.assertEqual(Policy(LiveSettings()).check(CLICK, NOTEPAD, NOTEPAD), Verdict(CONFIRM))
        self.assertEqual(Policy(LiveSettings()).check(TYPE, NOTEPAD), Verdict(CONFIRM))


class PointWindowTests(unittest.TestCase):
    """A click or scroll lands in the window under its point, which is checked as well as the one in front."""

    def test_a_blocked_window_under_the_point_is_refused(self):
        for decision in POINTER_STEPS:
            for confirm in ("all", "risky"):
                with self.subTest(action=decision.action, confirm=confirm):
                    self.assertEqual(
                        check(decision, NOTEPAD, "KeePass - Database.kdbx", confirm=confirm),
                        Verdict(REFUSE, "The window under that point looks like 'keepass', which I never act in. Please do this part yourself."),
                    )

    def test_an_unknown_window_under_the_point_is_asked_about_even_in_risky_mode(self):
        for decision in POINTER_STEPS:
            with self.subTest(action=decision.action):
                self.assertEqual(check(decision, NOTEPAD, None, confirm="risky"), Verdict(CONFIRM, UNKNOWN_POINT))
                self.assertEqual(check(decision, NOTEPAD, None), Verdict(CONFIRM, UNKNOWN_POINT))

    def test_an_unknown_window_under_the_point_is_refused_with_an_allowed_list(self):
        self.assertEqual(check(CLICK, NOTEPAD, None, allowed_windows=("notepad",)), Verdict(REFUSE, UNKNOWN_POINT_ALLOWLIST))

    def test_the_window_under_the_point_must_be_allowed_too(self):
        self.assertEqual(
            check(CLICK, NOTEPAD, "Calculator", confirm="risky", allowed_windows=("notepad",)),
            Verdict(REFUSE, "'Calculator' is not one of the windows I'm allowed to use."),
        )

    def test_another_ordinary_window_under_the_point_is_fine(self):
        self.assertEqual(check(CLICK, NOTEPAD, "Calculator", confirm="risky"), Verdict(ALLOW))
        self.assertEqual(check(CLICK, NOTEPAD, "Calculator"), Verdict(CONFIRM))

    def test_the_window_in_front_is_judged_first(self):
        verdict = check(CLICK, "Online Banking - Chrome", "KeePass")
        self.assertIn("The window in front looks like 'banking'", verdict.reason)
        # A refusal for the point beats a question about the window in front.
        self.assertEqual(check(CLICK, None, "KeePass").outcome, REFUSE)

    def test_keys_and_typing_go_to_the_window_in_front_only(self):
        for decision in KEYBOARD_STEPS:
            with self.subTest(action=decision.action):
                self.assertEqual(check(decision, NOTEPAD, "KeePass"), Verdict(CONFIRM))
                self.assertEqual(check(decision, NOTEPAD, None), Verdict(CONFIRM))


class WholeTitleTests(unittest.TestCase):
    def test_the_run_box_start_and_search_are_refused(self):
        for title in ("Run", "Start", "Search", "Cortana", "Task View", "Task Switching", "Quick settings", "Services"):
            for decision in (TYPE, CLICK, PRESS):
                with self.subTest(title=title, action=decision.action):
                    verdict = check(decision, title)
                    self.assertEqual(verdict.outcome, REFUSE)
                    self.assertIn("which I never act in", verdict.reason)

    def test_case_and_spacing_do_not_matter(self):
        for title in ("RUN", " run ", "Task  view", "SEARCH"):
            with self.subTest(title=title):
                self.assertEqual(check(TYPE, title).outcome, REFUSE)

    def test_titles_with_those_words_in_them_are_fine(self):
        for title in ("How to run Python - Google Chrome", "Start here.txt - Notepad", "Search results - Explorer", "Runner"):
            with self.subTest(title=title):
                self.assertEqual(check(TYPE, title), Verdict(CONFIRM))

    def test_the_settings_list_is_the_one_used(self):
        self.assertEqual(check(TYPE, "Run", blocked_titles=()), Verdict(CONFIRM))
        self.assertEqual(check(TYPE, "Notes", blocked_titles=DEFAULT_BLOCKED_TITLES + ("notes",)).outcome, REFUSE)


class OwnHotkeyTests(unittest.TestCase):
    def test_the_default_hotkeys(self):
        self.assertEqual(
            own_hotkeys(LiveSettings()),
            (frozenset({"ctrl", "alt", "space"}), frozenset({"ctrl", "alt", "q"}), frozenset({"ctrl", "alt", "y"}), frozenset({"ctrl", "alt", "n"})),
        )

    def test_the_model_cannot_press_them(self):
        for names in (("ctrl", "alt", "y"), ("alt", "ctrl", "n"), ("ctrl", "alt", "q"), ("ctrl", "alt", "space"), ("ctrl", "alt", "shift", "y")):
            for confirm in ("all", "risky"):
                with self.subTest(keys=names, confirm=confirm):
                    self.assertEqual(check(keys(*names), confirm=confirm), Verdict(REFUSE, OWN_HOTKEY))

    def test_other_combinations_with_those_keys_are_only_asked_about(self):
        for names in (("ctrl", "y"), ("alt", "q"), ("ctrl", "shift", "n"), ("ctrl", "space")):
            with self.subTest(keys=names):
                self.assertEqual(check(keys(*names)), Verdict(CONFIRM))
        self.assertEqual(check(Decision("press", keys=("y",))), Verdict(CONFIRM))

    def test_hotkeys_from_the_settings(self):
        settings = {"hotkey_yes": "<ctrl>+<shift>+y", "hotkey_talk": "<cmd>+<f9>"}
        self.assertEqual(check(keys("ctrl", "shift", "y"), **settings), Verdict(REFUSE, OWN_HOTKEY))
        self.assertEqual(check(keys("ctrl", "alt", "y"), **settings), Verdict(CONFIRM))
        self.assertEqual(own_hotkeys(LiveSettings(**settings))[0], frozenset({"win", "f9"}))


if __name__ == "__main__":
    unittest.main()
