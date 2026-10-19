"""Tests for nexus.live.model: reading the vision model's answer, the shared prompts, and the local model clients."""

import base64
import hashlib
import json
import math
import unittest

import httpx

from nexus.live.model import (
    ACTION_ALIASES,
    ACTIONS,
    DECISION_SCHEMA,
    KEY_ALIASES,
    KEY_NAMES,
    MAX_SAY,
    MAX_SCROLL,
    MAX_TEXT,
    MAX_WAIT,
    OBSERVE_PROMPT,
    OBSERVE_QUESTION,
    OBSERVE_SCHEMA,
    POINTER_ACTIONS,
    PROMPT_VERSION,
    QUIET_ACTIONS,
    Decision,
    DecisionError,
    LlamaCppVision,
    ModelUnavailable,
    OllamaVision,
    VisionModel,
    chat_messages,
    parse_decision,
    request_text,
    system_prompt,
    target_json,
)
from nexus.live.screen import Shot

SHOT = Shot(image=b"\x89PNG\r\n\x1a\nnot really a picture", width=100, height=50, screen_width=200, screen_height=100)


def parse(action, **fields):
    return parse_decision({"action": action, **fields})


class LimitTests(unittest.TestCase):
    def test_limits(self):
        self.assertEqual((MAX_TEXT, MAX_SAY, MAX_SCROLL, MAX_WAIT), (500, 400, 20, 5.0))

    def test_action_names(self):
        self.assertEqual(
            ACTIONS, ("click", "double_click", "right_click", "type", "press", "hotkey", "scroll", "wait", "answer", "done")
        )
        self.assertEqual(POINTER_ACTIONS, ("click", "double_click", "right_click"))
        self.assertEqual(QUIET_ACTIONS, ("wait", "answer", "done"))

    def test_error_types(self):
        self.assertTrue(issubclass(DecisionError, ValueError))
        self.assertTrue(issubclass(ModelUnavailable, RuntimeError))


class JsonExtractionTests(unittest.TestCase):
    def test_bare_object(self):
        self.assertEqual(parse_decision('{"action": "done", "say": "All set."}'), Decision("done", say="All set."))

    def test_object_in_prose(self):
        raw = 'Sure! The next step is {"thought": "menu", "say": "", "action": "click", "x": 10, "y": 20}. Hope that helps.'
        self.assertEqual(parse_decision(raw), Decision("click", thought="menu", x=10.0, y=20.0))

    def test_object_in_a_code_fence(self):
        raw = '```json\n{"action": "press", "keys": ["enter"]}\n```'
        self.assertEqual(parse_decision(raw), Decision("press", keys=("enter",)))

    def test_braces_inside_strings_are_kept(self):
        self.assertEqual(parse_decision('{"action": "type", "text": "a {b} c"}').text, "a {b} c")

    def test_object_inside_an_array_is_read(self):
        self.assertEqual(parse_decision('[{"action": "done"}]'), Decision("done"))

    def test_two_objects_are_not_valid_json(self):
        with self.assertRaisesRegex(DecisionError, "not valid JSON"):
            parse_decision('first {"action": "done"} then {"action": "wait"}')

    def test_broken_json(self):
        for raw in ('{"action": "done",}', "{action: done}", "{'action': 'done'}"):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(DecisionError, "not valid JSON"):
                    parse_decision(raw)

    def test_no_object(self):
        for raw in ("", "done", "[1, 2]", '"done"', "null", "42", "true", "action: done }", "} {"):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(DecisionError, "no JSON object"):
                    parse_decision(raw)

    def test_values_that_are_not_objects(self):
        for value in ([], [{"action": "done"}], None, 42, 3.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(DecisionError, "not a JSON object"):
                    parse_decision(value)  # type: ignore[arg-type]


ALIAS_TABLE = {
    "left_click": "click",
    "leftclick": "click",
    "tap": "click",
    "doubleclick": "double_click",
    "double-click": "double_click",
    "rightclick": "right_click",
    "right-click": "right_click",
    "write": "type",
    "input": "type",
    "type_text": "type",
    "key": "press",
    "keypress": "press",
    "press_key": "press",
    "shortcut": "hotkey",
    "key_combination": "hotkey",
    "sleep": "wait",
    "reply": "answer",
    "respond": "answer",
    "speak": "answer",
    "finish": "done",
    "finished": "done",
    "stop": "done",
}

# Enough for each action to be read.
PAYLOADS = {
    "click": {"x": 10, "y": 20},
    "double_click": {"x": 10, "y": 20},
    "right_click": {"x": 10, "y": 20},
    "type": {"text": "hi"},
    "press": {"keys": ["enter"]},
    "hotkey": {"keys": ["ctrl", "s"]},
    "scroll": {"amount": 3},
    "wait": {"seconds": 2},
    "answer": {"say": "Yes."},
    "done": {},
}


class ActionTests(unittest.TestCase):
    def test_every_action(self):
        for action in ACTIONS:
            with self.subTest(action=action):
                self.assertEqual(parse(action, **PAYLOADS[action]).action, action)

    def test_missing_action(self):
        for data in ({}, {"action": None}, {"action": ""}, {"action": "   "}, {"action": 3}, {"action": ["click"]}):
            with self.subTest(data=data):
                with self.assertRaisesRegex(DecisionError, "the action is missing"):
                    parse_decision(data)

    def test_action_ignores_case_and_spaces(self):
        self.assertEqual(parse_decision({"action": "  DONE  "}), Decision("done"))
        self.assertEqual(parse_decision({"action": "Left_Click", "x": 1, "y": 2}), Decision("click", x=1.0, y=2.0))

    def test_alias_table(self):
        self.assertEqual(ACTION_ALIASES, ALIAS_TABLE)

    def test_every_alias(self):
        for alias, action in ALIAS_TABLE.items():
            with self.subTest(alias=alias):
                self.assertEqual(parse(alias, **PAYLOADS[action]).action, action)

    def test_unknown_actions(self):
        for name in ("drag", "move", "launch", "open_app", "run", "shell", "click_and_hold", "key_down"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(DecisionError, f"unknown action '{name}'"):
                    parse(name, x=1, y=2, keys=["a"], text="a")


class SayAndThoughtTests(unittest.TestCase):
    def test_trimmed(self):
        self.assertEqual(parse("done", say="  Done.  ", thought="  finished  "), Decision("done", say="Done.", thought="finished"))

    def test_missing_or_none_is_empty(self):
        self.assertEqual(parse("done"), Decision("done", say="", thought=""))
        self.assertEqual(parse("done", say=None, thought=None), Decision("done", say="", thought=""))

    def test_say_limit(self):
        self.assertEqual(parse("done", say="a" * 400).say, "a" * 400)
        with self.assertRaisesRegex(DecisionError, "say is longer than 400 characters"):
            parse("done", say="a" * 401)

    def test_say_limit_counts_after_trimming(self):
        self.assertEqual(parse("done", say="  " + "a" * 400 + "  ").say, "a" * 400)

    def test_thought_limit(self):
        self.assertEqual(parse("done", thought="b" * 400).thought, "b" * 400)
        with self.assertRaisesRegex(DecisionError, "thought is longer than 400 characters"):
            parse("done", thought="b" * 401)

    def test_must_be_text(self):
        for value in (5, ["hi"], {"a": 1}, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(DecisionError, "say is not text"):
                    parse("done", say=value)
                with self.assertRaisesRegex(DecisionError, "thought is not text"):
                    parse("done", thought=value)


class PointerTests(unittest.TestCase):
    def test_each_pointer_action_keeps_its_position(self):
        for action in POINTER_ACTIONS:
            with self.subTest(action=action):
                decision = parse(action, x=10, y=20, say="Opening it.", thought="the icon")
                self.assertEqual(decision, Decision(action, say="Opening it.", thought="the icon", x=10.0, y=20.0))
                self.assertIsInstance(decision.x, float)
                self.assertIsInstance(decision.y, float)

    def test_numbers_given_as_text(self):
        self.assertEqual(parse("click", x="12.5", y=" 7 "), Decision("click", x=12.5, y=7.0))

    def test_other_names_for_the_position(self):
        for name in ("point_2d", "point", "coordinate", "coordinates", "position"):
            with self.subTest(name=name):
                self.assertEqual(parse("click", **{name: [10, 20]}), Decision("click", x=10.0, y=20.0))

    def test_tuple_position(self):
        self.assertEqual(parse("click", point=(3, 4)), Decision("click", x=3.0, y=4.0))

    def test_x_and_y_come_first(self):
        self.assertEqual(parse("click", x=1, y=2, point_2d=[3, 4]), Decision("click", x=1.0, y=2.0))

    def test_point_2d_comes_before_point(self):
        self.assertEqual(parse("click", point=[5, 6], point_2d=[3, 4]), Decision("click", x=3.0, y=4.0))

    def test_half_an_x_and_y_falls_back_to_another_name(self):
        self.assertEqual(parse("click", x=1, point_2d=[5, 6]), Decision("click", x=5.0, y=6.0))

    def test_unusable_x_and_y_are_not_replaced(self):
        with self.assertRaisesRegex(DecisionError, "x is not a number"):
            parse("click", x="left", y=2, point_2d=[3, 4])

    def test_position_missing(self):
        for fields in ({}, {"x": 1}, {"y": 1}, {"point": [1, 2, 3]}, {"point": [1]}, {"point": "1,2"}, {"position": None}):
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(DecisionError, r"the position \(x and y\) is missing"):
                    parse("click", **fields)

    def test_true_and_false_are_not_numbers(self):
        with self.assertRaisesRegex(DecisionError, "x is not a number"):
            parse("click", x=True, y=5)
        with self.assertRaisesRegex(DecisionError, "y is not a number"):
            parse("click", x=5, y=False)
        with self.assertRaisesRegex(DecisionError, "x is not a number"):
            parse("click", point=[True, 1])

    def test_not_finite(self):
        for value in (math.nan, math.inf, -math.inf, "nan", "inf", "-Infinity", "NaN"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(DecisionError, "x is not a number"):
                    parse("click", x=value, y=5)
                with self.assertRaisesRegex(DecisionError, "y is not a number"):
                    parse("click", x=5, y=value)

    def test_not_finite_in_the_raw_answer(self):
        for raw in ('{"action": "click", "x": NaN, "y": 1}', '{"action": "click", "x": 1, "y": Infinity}'):
            with self.subTest(raw=raw):
                with self.assertRaises(DecisionError):
                    parse_decision(raw)

    def test_not_numbers(self):
        for value in (None, "", "12px", [1], {"v": 1}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(DecisionError, "x is not a number"):
                    parse("click", x=value, y=5)

    def test_off_screen_positions_are_left_to_to_screen(self):
        self.assertEqual(parse("click", x=-5, y=5000), Decision("click", x=-5.0, y=5000.0))

    def test_huge_whole_number_is_a_decision_error(self):
        # JSON allows any number of digits. A whole number too big for a float is a bad answer
        # (DecisionError), never an OverflowError that would get past the agent's retry.
        huge = "1" + "0" * 400
        with self.assertRaises(DecisionError):
            parse_decision('{"action": "click", "x": ' + huge + ', "y": 5}')


class TypeTests(unittest.TestCase):
    def test_text_is_kept_exactly(self):
        self.assertEqual(parse("type", text="  Hello, world!  "), Decision("type", text="  Hello, world!  "))

    def test_text_limit(self):
        self.assertEqual(parse("type", text="a" * 500).text, "a" * 500)
        with self.assertRaisesRegex(DecisionError, "the text to type is longer than 500 characters"):
            parse("type", text="a" * 501)

    def test_missing_or_empty_text(self):
        for fields in ({}, {"text": ""}, {"text": None}, {"text": 5}, {"text": ["a"]}):
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(DecisionError, "there is no text to type"):
                    parse("type", **fields)

    def test_a_space_is_text(self):
        self.assertEqual(parse("type", text=" ").text, " ")

    def test_newline_and_tab_are_allowed(self):
        self.assertEqual(parse("type", text="line one\nline two\tend").text, "line one\nline two\tend")
        self.assertEqual(parse("type", text="\n").text, "\n")
        self.assertEqual(parse("type", text="\t").text, "\t")

    def test_other_control_characters_are_refused(self):
        for character in ("\x00", "\x07", "\x08", "\x0b", "\x0c", "\r", "\x1b", "\x1f"):
            with self.subTest(character=character):
                with self.assertRaisesRegex(DecisionError, "control characters"):
                    parse("type", text=f"abc{character}def")

    def test_printable_ascii_is_allowed(self):
        printable = "".join(chr(code) for code in range(32, 127))
        self.assertEqual(parse("type", text=printable).text, printable)

    def test_other_languages_are_allowed(self):
        self.assertEqual(parse("type", text="café 日本語 నమస్తే").text, "café 日本語 నమస్తే")

    def test_delete_and_c1_control_characters_are_refused(self):
        # DEL (0x7f) and the C1 controls (0x80 to 0x9f, such as 0x9b, the one-byte terminal escape
        # CSI) are never typed or pasted.
        for character in ("\x7f", "\x9b"):
            with self.assertRaisesRegex(DecisionError, "control characters"):
                parse("type", text=f"abc{character}def")


KEY_ALIAS_TABLE = {
    "control": "ctrl",
    "ctl": "ctrl",
    "return": "enter",
    "escape": "esc",
    "del": "delete",
    "spacebar": "space",
    "windows": "win",
    "super": "win",
    "meta": "win",
    "cmd": "win",
    "command": "win",
    "pgup": "pageup",
    "pgdn": "pagedown",
    "page_up": "pageup",
    "page_down": "pagedown",
    "arrowup": "up",
    "arrowdown": "down",
    "arrowleft": "left",
    "arrowright": "right",
}

EVERY_KEY = (
    list("abcdefghijklmnopqrstuvwxyz0123456789")
    + [f"f{number}" for number in range(1, 13)]
    + ["enter", "tab", "esc", "backspace", "delete", "insert", "space"]
    + ["up", "down", "left", "right", "home", "end", "pageup", "pagedown"]
    + ["ctrl", "alt", "shift", "win"]
    + ["-", "=", "[", "]", ";", "'", ",", ".", "/", "\\", "`"]
)


class KeyTests(unittest.TestCase):
    def test_press_one_key(self):
        self.assertEqual(parse("press", keys=["enter"], say="Sending.", thought="submit"), Decision("press", "Sending.", "submit", keys=("enter",)))

    def test_key_field(self):
        self.assertEqual(parse("press", key="Tab"), Decision("press", keys=("tab",)))

    def test_keys_field_comes_before_key(self):
        self.assertEqual(parse("press", keys=["esc"], key="enter"), Decision("press", keys=("esc",)))

    def test_key_names_ignore_case_and_spaces(self):
        self.assertEqual(parse("press", keys=["  ENTER "]), Decision("press", keys=("enter",)))

    def test_press_with_several_keys_is_a_hotkey(self):
        self.assertEqual(parse("press", keys=["ctrl", "s"], say="Saving."), Decision("hotkey", say="Saving.", keys=("ctrl", "s")))

    def test_keys_joined_with_plus(self):
        self.assertEqual(parse("press", keys="Ctrl + Shift + S"), Decision("hotkey", keys=("ctrl", "shift", "s")))
        self.assertEqual(parse("hotkey", keys="ctrl+s"), Decision("hotkey", keys=("ctrl", "s")))
        self.assertEqual(parse("hotkey", keys="s"), Decision("hotkey", keys=("s",)))
        self.assertEqual(parse("press", keys="ctrl+"), Decision("press", keys=("ctrl",)))

    def test_press_four_keys_but_not_five(self):
        self.assertEqual(parse("press", keys=["ctrl", "alt", "shift", "s"]), Decision("hotkey", keys=("ctrl", "alt", "shift", "s")))
        with self.assertRaisesRegex(DecisionError, "at most 4 keys at once"):
            parse("press", keys=["ctrl", "alt", "shift", "s", "a"])

    def test_hotkey_four_keys_but_not_five(self):
        self.assertEqual(parse("hotkey", keys=["ctrl", "alt", "shift", "s"]), Decision("hotkey", keys=("ctrl", "alt", "shift", "s")))
        with self.assertRaisesRegex(DecisionError, "at most 4 keys at once"):
            parse("hotkey", keys="ctrl+alt+shift+s+a")

    def test_hotkey_with_one_key_stays_a_hotkey(self):
        self.assertEqual(parse("hotkey", keys=["f5"]), Decision("hotkey", keys=("f5",)))

    def test_keys_missing(self):
        for fields in ({}, {"keys": []}, {"keys": None}, {"key": None}, {"keys": 5}, {"keys": "+"}, {"keys": {"key": "a"}}):
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(DecisionError, "keys are missing"):
                    parse("press", **fields)

    def test_not_a_key(self):
        for keys in ("", [""], ["  "], [5], [None], [["ctrl"]]):
            with self.subTest(keys=keys):
                with self.assertRaisesRegex(DecisionError, "not a key"):
                    parse("hotkey", keys=keys)

    def test_keys_it_cannot_press(self):
        for key in ("f13", "f0", "+", "capslock", "printscreen", "numlock", "fn", "winleft", "lwin", "start", "ctrl+s", "é", "enter key"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(DecisionError, "not a key NEXUS Live can press"):
                    parse("press", keys=[key])

    def test_every_key_name(self):
        self.assertEqual(KEY_NAMES, frozenset(EVERY_KEY))
        for key in EVERY_KEY:
            with self.subTest(key=key):
                self.assertEqual(parse("press", keys=[key]).keys, (key,))

    def test_key_alias_table(self):
        self.assertEqual(KEY_ALIASES, KEY_ALIAS_TABLE)

    def test_every_key_alias(self):
        for alias, key in KEY_ALIAS_TABLE.items():
            with self.subTest(alias=alias):
                self.assertEqual(parse("press", keys=[alias.upper()]).keys, (key,))

    def test_the_windows_key_is_read(self):
        # Reading it is fine: the policy is what refuses it (see test_live_policy).
        self.assertEqual(parse("hotkey", keys=["Super", "R"]), Decision("hotkey", keys=("win", "r")))


class ScrollTests(unittest.TestCase):
    def test_amounts_inside_the_limits(self):
        for amount in (1, -1, 7, 20, -20):
            with self.subTest(amount=amount):
                decision = parse("scroll", amount=amount)
                self.assertEqual(decision, Decision("scroll", amount=amount))
                self.assertIs(type(decision.amount), int)

    def test_amounts_beyond_the_limits(self):
        for amount in (21, -21, 100):
            with self.subTest(amount=amount):
                with self.assertRaisesRegex(DecisionError, "amount must be a whole number from -20 to 20, not 0"):
                    parse("scroll", amount=amount)

    def test_zero(self):
        for amount in (0, 0.0, -0.0, "0"):
            with self.subTest(amount=amount):
                with self.assertRaisesRegex(DecisionError, "amount must be a whole number"):
                    parse("scroll", amount=amount)

    def test_fractions(self):
        for amount in (2.5, -0.5, 0.999, 19.5):
            with self.subTest(amount=amount):
                with self.assertRaisesRegex(DecisionError, "amount must be a whole number"):
                    parse("scroll", amount=amount)

    def test_whole_floats_and_text(self):
        for amount, expected in ((3.0, 3), ("-4", -4), (" 5 ", 5)):
            with self.subTest(amount=amount):
                decision = parse("scroll", amount=amount)
                self.assertEqual(decision.amount, expected)
                self.assertIs(type(decision.amount), int)

    def test_amount_missing_or_not_a_number(self):
        for fields in ({}, {"amount": None}, {"amount": True}, {"amount": "a lot"}, {"amount": math.nan}, {"amount": math.inf}):
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(DecisionError, "amount is not a number"):
                    parse("scroll", **fields)

    def test_no_position(self):
        decision = parse("scroll", amount=-3)
        self.assertIsNone(decision.x)
        self.assertIsNone(decision.y)

    def test_position(self):
        self.assertEqual(parse("scroll", amount=3, x=10, y=20, say="Down a bit."), Decision("scroll", say="Down a bit.", x=10.0, y=20.0, amount=3))

    def test_position_by_other_names(self):
        for name in ("point_2d", "point", "coordinate"):
            with self.subTest(name=name):
                self.assertEqual(parse("scroll", amount=3, **{name: [10, 20]}), Decision("scroll", x=10.0, y=20.0, amount=3))

    def test_half_a_position_is_refused(self):
        for fields in ({"x": 5}, {"y": 5}, {"point_2d": [5]}):
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(DecisionError, r"the position \(x and y\) is missing"):
                    parse("scroll", amount=3, **fields)

    def test_position_named_coordinates_or_position_is_kept(self):
        # A scroll takes its position under every name _point reads, so it lands where the model
        # meant, not wherever the pointer happens to be.
        for name in ("coordinates", "position"):
            self.assertEqual(parse("scroll", amount=3, **{name: [10, 20]}), Decision("scroll", x=10.0, y=20.0, amount=3))


class WaitTests(unittest.TestCase):
    def test_one_second_by_default(self):
        self.assertEqual(parse("wait"), Decision("wait", seconds=1.0))

    def test_inside_the_limits(self):
        for seconds, expected in ((5, 5.0), (5.0, 5.0), (0.001, 0.001), (2.5, 2.5), ("3", 3.0)):
            with self.subTest(seconds=seconds):
                self.assertEqual(parse("wait", seconds=seconds), Decision("wait", seconds=expected))

    def test_beyond_the_limits(self):
        for seconds in (5.01, 6, 0, -0.0, -1):
            with self.subTest(seconds=seconds):
                with self.assertRaisesRegex(DecisionError, "seconds must be more than 0 and at most 5"):
                    parse("wait", seconds=seconds)

    def test_not_numbers(self):
        for seconds in (None, True, "soon", math.nan, math.inf):
            with self.subTest(seconds=seconds):
                with self.assertRaisesRegex(DecisionError, "seconds is not a number"):
                    parse("wait", seconds=seconds)


class AnswerAndDoneTests(unittest.TestCase):
    def test_answer_needs_something_to_say(self):
        for fields in ({}, {"say": ""}, {"say": "   "}, {"say": None}):
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(DecisionError, "an answer needs something to say"):
                    parse("answer", **fields)

    def test_answer(self):
        self.assertEqual(parse("answer", say="It is 5 pm.", thought="clock"), Decision("answer", say="It is 5 pm.", thought="clock"))

    def test_done_with_or_without_words(self):
        self.assertEqual(parse("done"), Decision("done"))
        self.assertEqual(parse("done", say="Saved."), Decision("done", say="Saved."))

    def test_quiet_steps_drop_everything_else(self):
        self.assertEqual(parse("done", x=5, y=5, text="hi", keys=["win"], amount=3), Decision("done"))


class DescribeTests(unittest.TestCase):
    def test_pointer_steps(self):
        self.assertEqual(Decision("click", x=500.0, y=250.0).describe(), "click at (500, 250)")
        self.assertEqual(Decision("double_click", x=12.5, y=7.0, thought="the file").describe(), "double click at (12.5, 7): the file")
        self.assertEqual(Decision("right_click", x=0.0, y=1000.0).describe(), "right click at (0, 1000)")

    def test_typing_is_shown_up_to_60_characters(self):
        self.assertEqual(Decision("type", text="hello").describe(), "type 'hello'")
        self.assertEqual(Decision("type", text="a" * 60).describe(), "type '" + "a" * 60 + "'")
        self.assertEqual(Decision("type", text="a" * 61).describe(), "type '" + "a" * 57 + "...'")

    def test_keys(self):
        self.assertEqual(Decision("press", keys=("enter",)).describe(), "press enter")
        self.assertEqual(Decision("hotkey", keys=("ctrl", "shift", "s")).describe(), "press ctrl+shift+s")

    def test_scroll(self):
        self.assertEqual(Decision("scroll", amount=3).describe(), "scroll up 3")
        self.assertEqual(Decision("scroll", amount=1).describe(), "scroll up 1")
        self.assertEqual(Decision("scroll", amount=-1).describe(), "scroll down 1")
        self.assertEqual(Decision("scroll", amount=-20).describe(), "scroll down 20")
        # Never parsed, but the words the user is asked about must not say "up" for it.
        self.assertEqual(Decision("scroll", amount=0).describe(), "scroll down 0")

    def test_wait_and_quiet_steps(self):
        self.assertEqual(Decision("wait", seconds=2.5).describe(), "wait 2.5 s")
        self.assertEqual(Decision("wait", seconds=1.0).describe(), "wait 1 s")
        self.assertEqual(Decision("answer", say="Hi.").describe(), "answer")
        self.assertEqual(Decision("done").describe(), "done")


class PromptTests(unittest.TestCase):
    def test_request_alone(self):
        self.assertEqual(request_text("  Open Notepad  ", [], []), "Request: Open Notepad\nWhat is the next step?")

    def test_request_with_conversation_and_steps(self):
        text = request_text("Save it", ["click at (1, 2)", "type 'x'"], [("user", "open notepad"), ("you", "Done.")])
        self.assertEqual(
            text,
            "Request: Save it\n"
            "Earlier in this conversation:\n"
            "- user: open notepad\n"
            "- you: Done.\n"
            "Steps already taken for this request:\n"
            "1. click at (1, 2)\n"
            "2. type 'x'\n"
            "What is the next step?",
        )

    def test_request_with_steps_only(self):
        self.assertEqual(
            request_text("Save it", ["press ctrl+s"], []),
            "Request: Save it\nSteps already taken for this request:\n1. press ctrl+s\nWhat is the next step?",
        )

    def test_request_with_conversation_only(self):
        self.assertEqual(
            request_text("And now?", [], [("user", "hi")]),
            "Request: And now?\nEarlier in this conversation:\n- user: hi\nWhat is the next step?",
        )

    def test_chat_messages(self):
        image = object()
        messages = chat_messages("the rules", "the request", image)
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": [{"type": "text", "text": "the rules"}]},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "the request"}]},
            ],
        )
        self.assertIs(messages[1]["content"][0]["image"], image)

    def test_target_json_puts_thought_say_and_action_first(self):
        decision = {"x": 1, "action": "click", "y": 2, "say": "", "thought": "the button"}
        self.assertEqual(target_json(decision), '{"thought": "the button", "say": "", "action": "click", "x": 1, "y": 2}')
        self.assertEqual(list(decision), ["x", "action", "y", "say", "thought"])  # the input is left alone

    def test_target_json_with_missing_keys(self):
        self.assertEqual(target_json({"say": "ok", "action": "done"}), '{"say": "ok", "action": "done"}')
        self.assertEqual(target_json({"keys": ["enter"], "action": "press"}), '{"action": "press", "keys": ["enter"]}')

    def test_target_json_keeps_other_languages_readable(self):
        self.assertEqual(target_json({"action": "type", "text": "café 日本"}), '{"action": "type", "text": "café 日本"}')

    def test_system_prompt_positions(self):
        normalized = system_prompt("normalized")
        pixels = system_prompt("pixels")
        self.assertIn("x and y run from 0 to 1000 across the screenshot: x from the left edge, y from the top edge.", normalized)
        self.assertNotIn("pixels of the screenshot", normalized)
        self.assertIn("x and y are pixels of the screenshot, from its top left corner.", pixels)
        self.assertNotIn("0 to 1000", pixels)
        self.assertEqual(system_prompt("anything else"), normalized)

    def test_system_prompt_rules(self):
        for mode in ("normalized", "pixels"):
            with self.subTest(mode=mode):
                prompt = system_prompt(mode)
                self.assertTrue(prompt.startswith("You operate the user's computer for them, one step at a time"))
                self.assertIn("Never type passwords, one-time codes, card or bank details, and never confirm a payment.", prompt)
                self.assertIn("The Windows key is not available", prompt)
                self.assertIn("wait: seconds (at most 5)", prompt)
                self.assertTrue(prompt.endswith("Reply with JSON only."))

    def test_prompt_version_changes_with_the_prompts(self):
        # A fine-tuned NEXUS Vision records the PROMPT_VERSION it was trained on, and local.py refuses
        # one trained on another version. If this fails, the prompts or the answer format changed:
        # bump PROMPT_VERSION in nexus/live/model.py, then put the new version and fingerprint here.
        parts = [
            system_prompt("normalized"),
            system_prompt("pixels"),
            OBSERVE_PROMPT,
            OBSERVE_QUESTION,
            json.dumps(DECISION_SCHEMA, sort_keys=True),
            json.dumps(OBSERVE_SCHEMA, sort_keys=True),
        ]
        fingerprint = hashlib.sha256("\n\n".join(parts).encode("utf-8")).hexdigest()
        self.assertEqual((PROMPT_VERSION, fingerprint), (1, "f5817dd402644de9975a45495bfe73accea88617539f3b91ab50202ad4505624"))

    def test_schemas(self):
        self.assertEqual(DECISION_SCHEMA["properties"]["action"]["enum"], list(ACTIONS))
        self.assertEqual(DECISION_SCHEMA["required"], ["thought", "say", "action"])
        self.assertEqual(DECISION_SCHEMA["properties"]["amount"], {"type": "integer"})
        self.assertEqual(DECISION_SCHEMA["properties"]["keys"], {"type": "array", "items": {"type": "string"}})
        self.assertEqual(OBSERVE_SCHEMA, {"type": "object", "properties": {"say": {"type": "string"}}, "required": ["say"]})
        self.assertEqual(OBSERVE_QUESTION, "Anything that needs my attention?")
        self.assertIn("return an empty string", OBSERVE_PROMPT)


class ScriptedModel(VisionModel):
    """A VisionModel whose raw answers are written in advance."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def _chat(self, system, text, image, schema):
        self.calls.append((system, text, image, schema))
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class VisionModelTests(unittest.TestCase):
    def test_base_class_has_no_model(self):
        self.assertEqual(VisionModel.name, "vision model")
        with self.assertRaises(NotImplementedError):
            VisionModel().decide("goal", [], [], SHOT, "normalized")

    def test_ask_sends_the_shared_prompts(self):
        model = ScriptedModel("raw answer")
        conversation = [("user", "hi"), ("you", "hello")]
        self.assertEqual(model.ask("Open Notepad", ["press esc"], conversation, b"png", "pixels"), "raw answer")
        self.assertEqual(
            model.calls,
            [(system_prompt("pixels"), request_text("Open Notepad", ["press esc"], conversation), b"png", DECISION_SCHEMA)],
        )

    def test_decide_reads_the_answer_for_the_screenshot(self):
        model = ScriptedModel('{"thought": "", "say": "", "action": "click", "x": 10, "y": 20}')
        self.assertEqual(model.decide("Open it", [], [], SHOT, "normalized"), Decision("click", x=10.0, y=20.0))
        system, _, image, schema = model.calls[0]
        self.assertEqual(system, system_prompt("normalized"))
        self.assertIs(image, SHOT.image)
        self.assertIs(schema, DECISION_SCHEMA)

    def test_decide_passes_on_unreadable_answers(self):
        with self.assertRaisesRegex(DecisionError, "no JSON object"):
            ScriptedModel("I would click the button.").decide("Open it", [], [], SHOT, "normalized")

    def test_observe_says_what_needs_attention(self):
        model = ScriptedModel('{"say": "  The build failed.  "}')
        self.assertEqual(model.observe(SHOT), "The build failed.")
        self.assertEqual(model.calls, [(OBSERVE_PROMPT, OBSERVE_QUESTION, SHOT.image, OBSERVE_SCHEMA)])

    def test_observe_can_have_its_own_chat(self):
        class Watcher(ScriptedModel):
            def _observe_chat(self, system, text, image, schema):
                self.observed = (system, text, image, schema)
                return '{"say": "Low battery."}'

        model = Watcher()  # no _chat answers are written, so observe must not use _chat
        self.assertEqual(model.observe(SHOT), "Low battery.")
        self.assertEqual(model.observed, (OBSERVE_PROMPT, OBSERVE_QUESTION, SHOT.image, OBSERVE_SCHEMA))
        self.assertEqual(model.calls, [])

    def test_observe_up_to_the_limit(self):
        self.assertEqual(ScriptedModel(json.dumps({"say": "a" * 400})).observe(SHOT), "a" * 400)

    def test_observe_is_quiet_about_anything_unusable(self):
        for raw in ('{"say": ""}', "{}", "nothing to report", '{"say": 5}', json.dumps({"say": "a" * 401}), "[1, 2]", "{broken"):
            with self.subTest(raw=raw):
                self.assertEqual(ScriptedModel(raw).observe(SHOT), "")

    def test_observe_passes_on_an_unavailable_model(self):
        with self.assertRaises(ModelUnavailable):
            ScriptedModel(ModelUnavailable("Ollama is not running")).observe(SHOT)


class Server:
    """A fake model server for httpx.MockTransport: it records each request and answers with ``respond``."""

    def __init__(self, respond):
        self.respond = respond
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.respond(request)

    @property
    def transport(self):
        return httpx.MockTransport(self)

    def payload(self, index=0):
        return json.loads(self.requests[index].content)


def raising(error_type, message="boom"):
    def respond(request):
        raise error_type(message, request=request)

    return respond


def ollama_answer(content, status=200):
    return lambda request: httpx.Response(status, json={"message": {"role": "assistant", "content": content}})


def llama_answer(content, status=200):
    return lambda request: httpx.Response(status, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


CLICK_JSON = '{"thought": "", "say": "", "action": "click", "x": 10, "y": 20}'
CONVERSATION = [("user", "hi"), ("you", "hello")]


def timeouts(seconds):
    return {"connect": seconds, "read": seconds, "write": seconds, "pool": seconds}


class OllamaVisionTests(unittest.TestCase):
    def make(self, respond, **kwargs):
        server = Server(respond)
        vision = OllamaVision("http://127.0.0.1:11434/", "qwen3-vl:4b", transport=server.transport, **kwargs)
        return vision, server

    def test_names(self):
        vision, _ = self.make(ollama_answer("{}"))
        self.assertEqual((vision.name, vision.model, vision.base_url, vision.timeout), ("qwen3-vl:4b", "qwen3-vl:4b", "http://127.0.0.1:11434", 240.0))

    def test_decide_sends_the_screenshot_and_the_request(self):
        vision, server = self.make(ollama_answer(CLICK_JSON))
        decision = vision.decide("Open the menu", ["press esc"], CONVERSATION, SHOT, "normalized")
        self.assertEqual(decision, Decision("click", x=10.0, y=20.0))
        self.assertEqual(len(server.requests), 1)
        request = server.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), "http://127.0.0.1:11434/api/chat")
        self.assertEqual(request.extensions["timeout"], timeouts(240.0))
        payload = server.payload()
        self.assertEqual(payload["model"], "qwen3-vl:4b")
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["format"], DECISION_SCHEMA)
        self.assertEqual(payload["keep_alive"], "15m")
        self.assertEqual(payload["options"], {"temperature": 0})
        image = base64.b64encode(SHOT.image).decode("ascii")
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": system_prompt("normalized")},
                {"role": "user", "content": request_text("Open the menu", ["press esc"], CONVERSATION), "images": [image]},
            ],
        )
        self.assertEqual(base64.b64decode(payload["messages"][1]["images"][0]), SHOT.image)

    def test_timeout_setting(self):
        vision, server = self.make(ollama_answer(CLICK_JSON), timeout=12.5)
        vision.decide("Open it", [], [], SHOT, "pixels")
        self.assertEqual(server.requests[0].extensions["timeout"], timeouts(12.5))
        self.assertEqual(server.payload()["messages"][0]["content"], system_prompt("pixels"))

    def test_observe(self):
        vision, server = self.make(ollama_answer('{"say": "A new message from Ravi."}'))
        self.assertEqual(vision.observe(SHOT), "A new message from Ravi.")
        payload = server.payload()
        self.assertEqual(payload["format"], OBSERVE_SCHEMA)
        self.assertEqual(payload["options"], {"temperature": 0})
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": OBSERVE_PROMPT},
                {"role": "user", "content": OBSERVE_QUESTION, "images": [base64.b64encode(SHOT.image).decode("ascii")]},
            ],
        )

    def test_not_running(self):
        vision, _ = self.make(raising(httpx.ConnectError, "connection refused"))
        with self.assertRaises(ModelUnavailable) as caught:
            vision.decide("Open it", [], [], SHOT, "normalized")
        self.assertEqual(
            str(caught.exception),
            "Ollama is not running at http://127.0.0.1:11434. Start it (open the Ollama app, or run: ollama serve).",
        )
        self.assertTrue(caught.exception.__suppress_context__)

    def test_too_slow(self):
        vision, _ = self.make(raising(httpx.ReadTimeout, "timed out"), timeout=12.5)
        with self.assertRaises(ModelUnavailable) as caught:
            vision.decide("Open it", [], [], SHOT, "normalized")
        self.assertTrue(str(caught.exception).startswith("qwen3-vl:4b did not answer within 12.5 s. A smaller model"))

    def test_other_transport_errors(self):
        for error_type in (httpx.ReadError, httpx.RemoteProtocolError):
            with self.subTest(error_type=error_type):
                vision, _ = self.make(raising(error_type))
                with self.assertRaises(ModelUnavailable) as caught:
                    vision.decide("Open it", [], [], SHOT, "normalized")
                self.assertEqual(str(caught.exception), f"could not reach Ollama: {error_type.__name__}")

    def test_model_not_downloaded(self):
        vision, _ = self.make(lambda request: httpx.Response(404, json={"error": "model not found"}))
        with self.assertRaises(ModelUnavailable) as caught:
            vision.decide("Open it", [], [], SHOT, "normalized")
        self.assertEqual(str(caught.exception), "qwen3-vl:4b is not downloaded. Run: ollama pull qwen3-vl:4b")

    def test_error_statuses(self):
        for status in (400, 500, 503):
            with self.subTest(status=status):
                vision, _ = self.make(lambda request, status=status: httpx.Response(status, text="something went wrong"))
                with self.assertRaises(ModelUnavailable) as caught:
                    vision.decide("Open it", [], [], SHOT, "normalized")
                self.assertEqual(str(caught.exception), f"Ollama answered {status}: something went wrong")

    def test_error_body_is_cut_to_200_characters(self):
        vision, _ = self.make(lambda request: httpx.Response(500, text="a" * 200 + "b" * 100))
        with self.assertRaises(ModelUnavailable) as caught:
            vision.decide("Open it", [], [], SHOT, "normalized")
        self.assertEqual(str(caught.exception), "Ollama answered 500: " + "a" * 200)

    def test_status_below_400_is_read(self):
        vision, _ = self.make(ollama_answer(CLICK_JSON, status=399))
        self.assertEqual(vision.decide("Open it", [], [], SHOT, "normalized"), Decision("click", x=10.0, y=20.0))

    def test_answers_without_a_message(self):
        for body in (b"not json", b"{}", b'{"message": {}}', b'{"message": "text"}', b"[1, 2]", b'{"message": null}'):
            with self.subTest(body=body):
                vision, _ = self.make(lambda request, body=body: httpx.Response(200, content=body))
                with self.assertRaises(ModelUnavailable) as caught:
                    vision.decide("Open it", [], [], SHOT, "normalized")
                self.assertEqual(str(caught.exception), "Ollama's answer had no message in it")

    def test_unreadable_content_is_a_decision_error(self):
        vision, _ = self.make(ollama_answer("I think you should click Save."))
        with self.assertRaises(DecisionError):
            vision.decide("Open it", [], [], SHOT, "normalized")

    def test_installed_models(self):
        models = {"models": [{"name": "qwen3-vl:4b"}, {"name": ""}, {"model": "no-name"}, {"name": "llava:7b"}]}
        vision, server = self.make(lambda request: httpx.Response(200, json=models))
        self.assertEqual(vision.installed_models(), ["qwen3-vl:4b", "llava:7b"])
        request = server.requests[0]
        self.assertEqual((request.method, str(request.url)), ("GET", "http://127.0.0.1:11434/api/tags"))
        self.assertEqual(request.extensions["timeout"], timeouts(5))

    def test_installed_models_when_ollama_cannot_be_asked(self):
        for respond in (
            lambda request: httpx.Response(500, text="oops"),
            lambda request: httpx.Response(404, text="no such page"),
            raising(httpx.ConnectError),
            raising(httpx.ReadTimeout),
            lambda request: httpx.Response(200, json={}),
        ):
            with self.subTest(respond=respond):
                vision, _ = self.make(respond)
                self.assertEqual(vision.installed_models(), [])

    def test_installed_models_when_something_else_answers(self):
        # A 200 answer that is not JSON (another program on Ollama's port) means no models,
        # not a crash of `nexus live --check`.
        vision, _ = self.make(lambda request: httpx.Response(200, text="<html>not Ollama</html>"))
        self.assertEqual(vision.installed_models(), [])


class LlamaCppVisionTests(unittest.TestCase):
    def make(self, respond, **kwargs):
        server = Server(respond)
        vision = LlamaCppVision("http://127.0.0.1:8080/", transport=server.transport, **kwargs)
        return vision, server

    def test_names(self):
        vision, _ = self.make(llama_answer("{}"))
        self.assertEqual((vision.name, vision.base_url, vision.timeout), ("llama-server at http://127.0.0.1:8080", "http://127.0.0.1:8080", 240.0))

    def test_decide_sends_the_screenshot_and_the_request(self):
        vision, server = self.make(llama_answer(CLICK_JSON))
        decision = vision.decide("Open the menu", ["press esc"], CONVERSATION, SHOT, "pixels")
        self.assertEqual(decision, Decision("click", x=10.0, y=20.0))
        self.assertEqual(len(server.requests), 1)
        request = server.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(request.extensions["timeout"], timeouts(240.0))
        payload = server.payload()
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 400)
        self.assertEqual(payload["response_format"], {"type": "json_object", "schema": DECISION_SCHEMA})
        data_uri = "data:image/png;base64," + base64.b64encode(SHOT.image).decode("ascii")
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": system_prompt("pixels")},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": request_text("Open the menu", ["press esc"], CONVERSATION)},
                    ],
                },
            ],
        )

    def test_timeout_setting(self):
        vision, server = self.make(llama_answer(CLICK_JSON), timeout=12.5)
        vision.decide("Open it", [], [], SHOT, "normalized")
        self.assertEqual(server.requests[0].extensions["timeout"], timeouts(12.5))

    def test_observe(self):
        vision, server = self.make(llama_answer('{"say": "Battery at 5 percent."}'))
        self.assertEqual(vision.observe(SHOT), "Battery at 5 percent.")
        payload = server.payload()
        self.assertEqual(payload["response_format"], {"type": "json_object", "schema": OBSERVE_SCHEMA})
        self.assertEqual(payload["messages"][0], {"role": "system", "content": OBSERVE_PROMPT})
        self.assertEqual(payload["messages"][1]["content"][1], {"type": "text", "text": OBSERVE_QUESTION})

    def test_not_running(self):
        vision, _ = self.make(raising(httpx.ConnectError, "connection refused"))
        with self.assertRaises(ModelUnavailable) as caught:
            vision.decide("Open it", [], [], SHOT, "normalized")
        self.assertEqual(
            str(caught.exception),
            "llama-server is not running at http://127.0.0.1:8080. Start it with your model and its --mmproj file.",
        )

    def test_too_slow(self):
        vision, _ = self.make(raising(httpx.ReadTimeout, "timed out"), timeout=12.5)
        with self.assertRaises(ModelUnavailable) as caught:
            vision.decide("Open it", [], [], SHOT, "normalized")
        self.assertEqual(str(caught.exception), "llama-server did not answer within 12.5 s.")

    def test_other_transport_errors(self):
        vision, _ = self.make(raising(httpx.ReadError))
        with self.assertRaises(ModelUnavailable) as caught:
            vision.decide("Open it", [], [], SHOT, "normalized")
        self.assertEqual(str(caught.exception), "could not reach llama-server: ReadError")

    def test_error_statuses(self):
        for status in (400, 404, 500):
            with self.subTest(status=status):
                vision, _ = self.make(lambda request, status=status: httpx.Response(status, text="no slot available"))
                with self.assertRaises(ModelUnavailable) as caught:
                    vision.decide("Open it", [], [], SHOT, "normalized")
                self.assertEqual(str(caught.exception), f"llama-server answered {status}: no slot available")

    def test_error_body_is_cut_to_200_characters(self):
        vision, _ = self.make(lambda request: httpx.Response(503, text="a" * 200 + "b" * 100))
        with self.assertRaises(ModelUnavailable) as caught:
            vision.decide("Open it", [], [], SHOT, "normalized")
        self.assertEqual(str(caught.exception), "llama-server answered 503: " + "a" * 200)

    def test_status_below_400_is_read(self):
        vision, _ = self.make(llama_answer(CLICK_JSON, status=399))
        self.assertEqual(vision.decide("Open it", [], [], SHOT, "normalized"), Decision("click", x=10.0, y=20.0))

    def test_first_choice_is_used(self):
        choices = {"choices": [{"message": {"content": CLICK_JSON}}, {"message": {"content": '{"action": "done"}'}}]}
        vision, _ = self.make(lambda request: httpx.Response(200, json=choices))
        self.assertEqual(vision.decide("Open it", [], [], SHOT, "normalized"), Decision("click", x=10.0, y=20.0))

    def test_answers_without_a_message(self):
        for body in (b"not json", b"{}", b'{"choices": []}', b'{"choices": [{}]}', b'{"choices": null}', b'{"choices": [{"message": "x"}]}'):
            with self.subTest(body=body):
                vision, _ = self.make(lambda request, body=body: httpx.Response(200, content=body))
                with self.assertRaises(ModelUnavailable) as caught:
                    vision.decide("Open it", [], [], SHOT, "normalized")
                self.assertEqual(str(caught.exception), "llama-server's answer had no message in it")


if __name__ == "__main__":
    unittest.main()
