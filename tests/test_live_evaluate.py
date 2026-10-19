"""Tests for nexus.live.evaluate: grading a model's answers against the recorded steps."""

import json
import tempfile
import unittest
from pathlib import Path

from nexus.live.data import Example, split
from nexus.live.evaluate import NEAR, eval_fraction_of, evaluate, grade, held_out, report_table, summarize
from nexus.live.model import VisionModel


def example(target, box=None, **changes):
    values = {"id": "e", "source": "os-atlas", "group": "g", "image": "images/a.png", "request": "Click OK", "target": target, "box": box}
    values.update(changes)
    return Example(**values)


def answer(**decision):
    return json.dumps({"thought": "", "say": "", **decision})


BOX = [400.0, 400.0, 600.0, 500.0]


class GradeClickTests(unittest.TestCase):
    def setUp(self):
        self.example = example({"action": "click", "x": 500, "y": 450}, box=BOX)

    def test_inside_the_box_is_right_edges_included(self):
        for x, y in ((500, 450), (400, 400), (600, 500), (401, 499)):
            with self.subTest(x=x, y=y):
                result = grade(self.example, answer(action="click", x=x, y=y))
                self.assertTrue(result.correct)
                self.assertTrue(result.in_box)

    def test_outside_the_box_is_wrong_even_when_close(self):
        for x, y in ((399.9, 450), (600.1, 450), (500, 399), (500, 501)):
            with self.subTest(x=x, y=y):
                result = grade(self.example, answer(action="click", x=x, y=y))
                self.assertFalse(result.correct)
                self.assertFalse(result.in_box)

    def test_the_wrong_kind_of_click_is_wrong_but_its_landing_is_kept(self):
        result = grade(self.example, answer(action="double_click", x=500, y=450))
        self.assertFalse(result.correct)
        self.assertTrue(result.in_box)
        self.assertEqual(result.answered, "double_click")

    def test_distance_is_measured(self):
        self.assertEqual(grade(self.example, answer(action="click", x=530, y=490)).distance, 50.0)

    def test_without_a_box_near_enough_is_right(self):
        teach = example({"action": "click", "x": 500, "y": 450}, box=None, source="teach")
        self.assertTrue(grade(teach, answer(action="click", x=500 + NEAR, y=450)).correct)
        self.assertFalse(grade(teach, answer(action="click", x=500 + NEAR + 0.5, y=450)).correct)
        self.assertIsNone(grade(teach, answer(action="click", x=500, y=450)).in_box)

    def test_a_non_pointer_answer_to_a_click(self):
        result = grade(self.example, answer(action="done", say="Done."))
        self.assertFalse(result.correct)
        self.assertIsNone(result.distance)


class GradeOtherActionsTests(unittest.TestCase):
    def test_typing_must_match_apart_from_surrounding_space(self):
        typed = example({"action": "type", "text": "hello world"})
        self.assertTrue(grade(typed, answer(action="type", text=" hello world ")).correct)
        self.assertFalse(grade(typed, answer(action="type", text="hello World")).correct)
        self.assertFalse(grade(typed, answer(action="press", keys=["enter"])).correct)

    def test_keys_must_be_the_same_keys_press_or_hotkey(self):
        save = example({"action": "hotkey", "keys": ["ctrl", "s"]})
        self.assertTrue(grade(save, answer(action="hotkey", keys=["s", "ctrl"])).correct)
        self.assertTrue(grade(save, answer(action="press", keys=["ctrl", "s"])).correct)
        self.assertFalse(grade(save, answer(action="hotkey", keys=["ctrl", "shift", "s"])).correct)
        enter = example({"action": "press", "keys": ["enter"]})
        self.assertTrue(grade(enter, answer(action="press", keys=["return"])).correct)

    def test_scrolling_must_go_the_same_way(self):
        down = example({"action": "scroll", "amount": -3})
        self.assertTrue(grade(down, answer(action="scroll", amount=-10)).correct)
        self.assertFalse(grade(down, answer(action="scroll", amount=1)).correct)
        up = example({"action": "scroll", "amount": 2})
        self.assertTrue(grade(up, answer(action="scroll", amount=1)).correct)

    def test_done_answer_and_wait_must_be_the_same_action(self):
        done = example({"action": "done", "say": "Done."})
        self.assertTrue(grade(done, answer(action="done", say="All finished.")).correct)
        self.assertFalse(grade(done, answer(action="click", x=1, y=1)).correct)

    def test_an_answer_nexus_live_cannot_act_on_is_invalid(self):
        done = example({"action": "done", "say": "Done."})
        for raw in ("", "I would click the button", '{"action": "fly"}', answer(action="click")):
            with self.subTest(raw=raw):
                result = grade(done, raw)
                self.assertEqual(result.answered, "invalid")
                self.assertFalse(result.correct)
                self.assertTrue(result.error)


class SummaryTests(unittest.TestCase):
    def test_shares_and_click_figures(self):
        clicked = example({"action": "click", "x": 500, "y": 450}, box=BOX)
        grades = [
            grade(clicked, answer(action="click", x=500, y=450), seconds=1.0),
            grade(clicked, answer(action="click", x=900, y=900), seconds=2.0),
            grade(example({"action": "done", "say": "x"}), "nonsense", seconds=3.0),
            grade(example({"action": "type", "text": "a"}), answer(action="type", text="a"), seconds=2.0),
        ]
        summary = summarize(grades)
        self.assertEqual(summary["examples"], 4)
        self.assertEqual(summary["share_right"], 0.5)
        self.assertEqual(summary["share_invalid"], 0.25)
        self.assertEqual(summary["clicks_in_box"], 0.5)
        self.assertEqual(summary["by_action"]["click"], {"examples": 2, "right": 1, "share_right": 0.5})
        self.assertEqual(grades[1].distance, 602.1)  # from (500, 450) to (900, 900)
        self.assertEqual(summary["median_click_distance"], 301.1)  # the middle of 0 and 602.1
        self.assertEqual(summary["seconds_per_step"], 2.0)

    def test_empty(self):
        self.assertEqual(summarize([]), {"examples": 0})

    def test_table(self):
        table = report_table({"NEXUS Vision": {"share_right": 0.75, "share_invalid": 0.0, "clicks_in_box": None,
                                                "median_click_distance": 12.5, "seconds_per_step": 1.2},
                              "Base model": {"share_right": 0.5}})
        self.assertIn("| Steps right | 75.0% | 50.0% |", table)
        self.assertIn("| Clicks inside the box | n/a | n/a |", table)
        self.assertIn("| Median click distance (of 1000) | 12.5 | n/a |", table)


class FakeModel(VisionModel):
    def __init__(self, replies):
        self.replies = list(replies)
        self.asked = []

    def _chat(self, system, text, image, schema):
        self.asked.append((system, text, image))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class EvaluateTests(unittest.TestCase):
    def test_puts_each_example_to_the_model_as_nexus_live_would(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "images").mkdir()
            (Path(folder) / "images" / "a.png").write_bytes(b"PNG")
            items = [
                (Path(folder), example({"action": "click", "x": 500, "y": 450}, box=BOX, steps=["click at (1, 2)"])),
                (Path(folder), example({"action": "done", "say": "x"})),
            ]
            model = FakeModel([answer(action="click", x=450, y=420), RuntimeError("out of memory")])
            lines = []
            summary, grades = evaluate(model, items, printer=lines.append)
        self.assertEqual(model.asked[0][2], b"PNG")
        self.assertIn("1. click at (1, 2)", model.asked[0][1])
        self.assertTrue(grades[0].correct)
        self.assertIn("out of memory", grades[1].error)
        self.assertEqual(summary["share_right"], 0.5)
        self.assertEqual(lines[-1], "  2 of 2: 1 right")

    def test_wrong_answers_are_shown_so_a_log_says_why(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "images").mkdir()
            (Path(folder) / "images" / "a.png").write_bytes(b"PNG")
            items = [(Path(folder), example({"action": "done", "say": "x"}, id=f"e{index}")) for index in range(5)]
            replies = ["I think you should click it" + "!" * 400] * 4 + [answer(action="done", say="ok")]
            lines = []
            summary, grades = evaluate(FakeModel(replies), items, printer=lines.append)
        shown = [line for line in lines if line.startswith("    e")]
        self.assertEqual(len(shown), 3)  # the first three, not every one
        self.assertIn("e0: the answer has no JSON object in it | said: 'I think you should click it", shown[0])
        self.assertEqual(len(grades[0].raw), 300)  # what was said is kept, cut short
        self.assertEqual(grades[4].raw, answer(action="done", say="ok"))

    def test_what_was_generated_is_kept_and_shown(self):
        # An answer that is only the end-of-turn marker decodes to "": the token count tells it from a failure.
        class Generating(FakeModel):
            def _chat(self, system, text, image, schema):
                self.last_generation = {"new_tokens": 1, "raw": "<|im_end|>"}
                return super()._chat(system, text, image, schema)

        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "images").mkdir()
            (Path(folder) / "images" / "a.png").write_bytes(b"PNG")
            items = [(Path(folder), example({"action": "done", "say": "x"}))]
            lines = []
            summary, grades = evaluate(Generating([""]), items, printer=lines.append)
        self.assertEqual((grades[0].tokens, grades[0].raw_tokens), (1, "<|im_end|>"))
        self.assertIn("| 1 new tokens: '<|im_end|>'", lines[0])

    def test_a_failed_question_keeps_no_generation_from_before(self):
        class Failing(FakeModel):
            last_generation = {"new_tokens": 9, "raw": "old"}

        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "images").mkdir()
            (Path(folder) / "images" / "a.png").write_bytes(b"PNG")
            items = [(Path(folder), example({"action": "done", "say": "x"}))]
            _, grades = evaluate(Failing([RuntimeError("out of memory")]), items, printer=lambda line: None)
        self.assertIsNone(grades[0].tokens)

    def test_the_model_is_loaded_before_the_clock_starts(self):
        # seconds_per_step was a mean that included loading the model, which hid how fast answers failed.
        order = []

        class Loading(FakeModel):
            def load(self):
                order.append("load")

            def _chat(self, system, text, image, schema):
                order.append("ask")
                return super()._chat(system, text, image, schema)

        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "images").mkdir()
            (Path(folder) / "images" / "a.png").write_bytes(b"PNG")
            items = [(Path(folder), example({"action": "done", "say": "x"}, id=str(index))) for index in range(2)]
            evaluate(Loading([answer(action="done", say="y")] * 2), items, printer=lambda line: None)
        self.assertEqual(order, ["load", "ask", "ask"])

    def test_a_load_that_fails_is_graded_with_each_question(self):
        class Broken(FakeModel):
            def load(self):
                raise RuntimeError("no model here")

        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "images").mkdir()
            (Path(folder) / "images" / "a.png").write_bytes(b"PNG")
            items = [(Path(folder), example({"action": "done", "say": "x"}))]
            _, grades = evaluate(Broken([RuntimeError("no model here")]), items, printer=lambda line: None)
        self.assertIn("no model here", grades[0].error)

    def test_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "images").mkdir()
            (Path(folder) / "images" / "a.png").write_bytes(b"PNG")
            items = [(Path(folder), example({"action": "done", "say": "x"}, id=str(index))) for index in range(5)]
            summary, grades = evaluate(FakeModel([answer(action="done", say="y")] * 2), items, limit=2, printer=lambda line: None)
        self.assertEqual(len(grades), 2)

    def test_held_out_matches_the_training_split(self):
        items = [(Path("."), example({"action": "done", "say": "x"}, id=str(index), group=f"g{index}")) for index in range(100)]
        _, expected = split([item for _, item in items], 0.2)
        self.assertEqual([item.id for _, item in held_out(items, 0.2)], [item.id for item in expected])

    def test_eval_fraction_comes_from_the_training_record(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(eval_fraction_of(Path(folder)), 0.1)
            (Path(folder) / "training.json").write_text(json.dumps({"options": {"eval_fraction": 0.25}}))
            self.assertEqual(eval_fraction_of(Path(folder)), 0.25)
            (Path(folder) / "training.json").write_text("garbage")
            self.assertEqual(eval_fraction_of(Path(folder), default=0.3), 0.3)


if __name__ == "__main__":
    unittest.main()
