"""Tests for nexus.live.data: the training examples and the OS-Atlas converter."""

import importlib.util
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

from nexus.live.data import (
    EXAMPLES_FILE,
    MIN_SIDE,
    AtlasElement,
    Example,
    atlas_elements,
    atlas_examples,
    choose,
    clean_box,
    folder_reader,
    read_examples,
    request_for,
    split,
    to_messages,
    write_examples,
)
from nexus.live.model import PROMPT_VERSION, request_text, system_prompt, target_json

HAS_PIL = importlib.util.find_spec("PIL") is not None


def click(x=500, y=500, **extra):
    return {"thought": "", "say": "", "action": "click", "x": x, "y": y, **extra}


def example(**changes):
    values = {
        "id": "e1",
        "source": "teach",
        "group": "g1",
        "image": "images/a.png",
        "request": "Open the menu",
        "target": click(),
    }
    values.update(changes)
    return Example(**values)


def png(width=200, height=100, color=(10, 120, 200)):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


class ExampleCheckTests(unittest.TestCase):
    def test_a_good_example_passes(self):
        example().check()

    def test_blank_request_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no request"):
            example(request="   ").check()

    def test_a_step_nexus_live_cannot_take_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown action"):
            example(target={"action": "fly"}).check()

    def test_clicks_must_be_on_the_screenshot(self):
        example(target=click(0, 0)).check()
        example(target=click(1000, 1000)).check()
        for x, y in ((1000.5, 10), (10, 1001), (-1, 10), (10, -0.1)):
            with self.subTest(x=x, y=y), self.assertRaisesRegex(ValueError, "off the screenshot"):
                example(target=click(x, y)).check()

    def test_pixel_examples_are_not_held_to_the_0_to_1000_range(self):
        example(target=click(1500, 900), coordinates="pixels").check()

    def test_box_must_be_inside_and_the_right_way_round(self):
        example(box=[0, 0, 1000, 1000]).check()
        for box in ([10, 10, 10, 20], [10, 10, 20, 10], [20, 10, 10, 30], [-1, 0, 10, 10], [0, 0, 1001, 10], [0, 0, 10, 1000.1]):
            with self.subTest(box=box), self.assertRaisesRegex(ValueError, "box"):
                example(box=box).check()

    def test_json_round_trip(self):
        original = example(steps=["click at (1, 2)"], conversation=[["user", "hi"]], box=[1.5, 2, 3, 4])
        self.assertEqual(Example.from_json(original.to_json()), original)

    def test_new_examples_carry_the_current_prompt_version(self):
        self.assertEqual(example().prompt_version, PROMPT_VERSION)

    def test_text_that_reads_as_a_control_token_is_refused_anywhere(self):
        # The tokenizer turns "<|im_end|>" into the real end-of-turn marker: in a request it forges a turn,
        # in an answer it cuts the part the model learns from short.
        token = "<|im_end|>"
        for changes in (
            {"request": f"open {token} notepad"},
            {"steps": [f"click at (1, 2): {token}"]},
            {"conversation": [["user", f"hi {token}"]]},
            {"target": {"thought": "", "say": "", "action": "type", "text": f"x{token}"}},
            {"target": click(thought=f"the {token} button")},
            {"target": {"thought": "", "say": f"done {token}", "action": "done"}},
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "control tokens"):
                example(**changes).check()

    def test_brackets_that_are_not_a_token_are_fine(self):
        example(request="compare a <| b |> c and x < y").check()


class WriteReadTests(unittest.TestCase):
    def test_write_then_read(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(write_examples(folder, [example(id="a"), example(id="b")]), 2)
            write_examples(folder, [example(id="c")], append=True)
            self.assertEqual([item.id for item in read_examples(Path(folder))], ["a", "b", "c"])
            self.assertEqual([item.id for item in read_examples(Path(folder) / EXAMPLES_FILE)], ["a", "b", "c"])

    def test_one_bad_example_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            write_examples(folder, [example(id="kept")])
            with self.assertRaises(ValueError):
                write_examples(folder, [example(id="good"), example(id="bad", request="")], append=True)
            self.assertEqual([item.id for item in read_examples(Path(folder))], ["kept"])


class MessagesTests(unittest.TestCase):
    def test_conversation_is_what_nexus_live_sends_plus_the_answer(self):
        item = example(steps=["click at (5, 6)"], conversation=[["user", "hello"], ["assistant", "hi"]])
        messages = to_messages(item, "IMAGE")
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
        self.assertEqual(messages[0]["content"][0]["text"], system_prompt("normalized"))
        self.assertEqual(messages[1]["content"][0], {"type": "image", "image": "IMAGE"})
        self.assertEqual(
            messages[1]["content"][1]["text"],
            request_text("Open the menu", ["click at (5, 6)"], [("user", "hello"), ("assistant", "hi")]),
        )
        self.assertEqual(messages[2]["content"][0]["text"], target_json(item.target))

    def test_without_the_answer(self):
        self.assertEqual([message["role"] for message in to_messages(example(), "IMAGE", with_answer=False)], ["system", "user"])

    def test_pixel_examples_get_the_pixel_prompt(self):
        messages = to_messages(example(coordinates="pixels", target=click(10, 10)), "IMAGE")
        self.assertEqual(messages[0]["content"][0]["text"], system_prompt("pixels"))


class SplitTests(unittest.TestCase):
    def test_groups_never_straddle_the_split_and_the_split_is_stable(self):
        items = [example(id=f"{group}-{index}", group=f"group-{group}") for group in range(200) for index in range(3)]
        train, held = split(items, 0.25)
        self.assertEqual(len(train) + len(held), len(items))
        self.assertFalse({item.group for item in train} & {item.group for item in held})
        self.assertGreater(len(held), 0)
        self.assertGreater(len(train), len(held))
        again_train, again_held = split(list(reversed(items)), 0.25)
        self.assertEqual({item.id for item in held}, {item.id for item in again_held})

    def test_fraction_zero_holds_nothing_out(self):
        items = [example(id=str(index), group=str(index)) for index in range(50)]
        self.assertEqual(split(items, 0.0), (items, []))

    def test_the_boundary_is_exclusive(self):
        import hashlib

        # A group whose hash starts with a zero byte sits exactly on the 0.0 boundary: it is trained on.
        group = next(f"g{index}" for index in range(100000) if hashlib.sha256(f"g{index}".encode()).digest()[0] == 0)
        train, held = split([example(group=group)], 0.0)
        self.assertEqual((len(train), len(held)), (1, 0))
        train, held = split([example(group=group)], 1 / 256)
        self.assertEqual((len(train), len(held)), (0, 1))


class CleanBoxTests(unittest.TestCase):
    def test_a_box_inside_is_kept_as_it_is(self):
        self.assertEqual(clean_box([0.1, 0.2, 0.3, 0.4]), (0.1, 0.2, 0.3, 0.4))

    def test_a_box_running_a_little_past_the_edge_is_clipped(self):
        # OS-Atlas has boxes like this one, from its macOS file.
        box = clean_box([0.8555555555555555, 0.4588888888888889, 1.0909722222222222, 0.7144444444444444])
        self.assertEqual(box[2], 1.0)
        self.assertAlmostEqual(box[0], 0.8555555555555555)

    def test_every_edge_is_clipped_to_the_screenshot(self):
        self.assertEqual(clean_box([-0.05, 0.2, 0.15, 0.3]), (0.0, 0.2, 0.15, 0.3))
        self.assertEqual(clean_box([0.1, -0.1, 0.2, 0.1]), (0.1, 0.0, 0.2, 0.1))
        self.assertEqual(clean_box([0.9, 0.2, 1.05, 0.3]), (0.9, 0.2, 1.0, 0.3))
        self.assertEqual(clean_box([0.1, 0.95, 0.2, 1.02]), (0.1, 0.95, 0.2, 1.0))

    def test_a_box_mostly_off_the_screenshot_is_dropped(self):
        self.assertIsNone(clean_box([0.9, 0.1, 1.3, 0.2]))  # 25% visible
        self.assertIsNone(clean_box([0.9, 0.1, 1.105, 0.2]))  # 49% visible
        self.assertIsNotNone(clean_box([0.9, 0.1, 1.1, 0.2]))  # exactly half visible
        self.assertIsNotNone(clean_box([0.1, -0.1, 0.2, 0.1]))  # half above the top edge

    def test_never_read_as_0_to_1000(self):
        # The old converter divided anything over 1 by 1000; a 1.09 edge became 0.00109.
        self.assertIsNone(clean_box([100, 200, 300, 400]))
        self.assertIsNone(clean_box([0.5, 0.5, 2.5, 0.6]))
        self.assertIsNotNone(clean_box([0.5, 0.5, 1.0, 0.6]))

    def test_unusable_boxes(self):
        for raw in (None, [0.1, 0.2, 0.3], ["a", 0, 1, 1], [0.3, 0.1, 0.2, 0.4], [0.1, 0.4, 0.3, 0.2], [0.1, 0.1, 0.1, 0.2],
                    [float("nan"), 0, 0.5, 0.5], [0, 0, float("inf"), 0.5]):
            with self.subTest(raw=raw):
                self.assertIsNone(clean_box(raw))

    def test_tiny_boxes_are_dropped(self):
        self.assertIsNone(clean_box([0.5, 0.5, 0.5 + MIN_SIDE / 2, 0.6]))
        self.assertIsNone(clean_box([0.5, 0.5, 0.6, 0.5 + MIN_SIDE / 2]))
        self.assertIsNotNone(clean_box([0.5, 0.5, 0.5 + MIN_SIDE * 1.01, 0.5 + MIN_SIDE * 1.01]))


class AtlasElementsTests(unittest.TestCase):
    def test_grouped_desktop_layout(self):
        annotations = [
            {
                "img_filename": "a_sub1.png",
                "elements": [
                    {"instruction": "Help and feedback", "bbox": [0.742, 0.797, 0.966, 0.842]},
                    {"instruction": "Copilot", "bbox": [0.182, 0.0, 0.382, 0.049], "data_type": "AXButton"},
                ],
            },
            {"img_filename": "b.png", "elements": [{"instruction": "Clear the Streets!", "bbox": [0.37, 0.5, 0.44, 0.56]}]},
        ]
        by_image = atlas_elements(annotations)
        self.assertEqual(sorted(by_image), ["a_sub1.png", "b.png"])
        self.assertEqual([element.instruction for element in by_image["a_sub1.png"]], ["Help and feedback", "Copilot"])
        self.assertEqual(by_image["a_sub1.png"][1].box, (0.182, 0.0, 0.382, 0.049))

    def test_flat_layout(self):
        annotations = [
            {"img_filename": "m.png", "instruction": "Settings", "bbox": [0.1, 0.1, 0.2, 0.2]},
            {"img_filename": "m.png", "instruction": "Back", "bbox": [0.3, 0.1, 0.4, 0.2]},
        ]
        self.assertEqual([element.instruction for element in atlas_elements(annotations)["m.png"]], ["Settings", "Back"])

    def test_wrapped_in_a_dict(self):
        annotations = {"data": [{"img_filename": "m.png", "instruction": "Settings", "bbox": [0.1, 0.1, 0.2, 0.2]}]}
        self.assertIn("m.png", atlas_elements(annotations))

    def test_an_instruction_naming_two_elements_is_ambiguous_and_left_out(self):
        annotations = [
            {
                "img_filename": "a.png",
                "elements": [
                    {"instruction": "Read More", "bbox": [0.1, 0.1, 0.2, 0.2]},
                    {"instruction": "read  more", "bbox": [0.5, 0.5, 0.6, 0.6]},
                    {"instruction": "Home", "bbox": [0.3, 0.3, 0.4, 0.4]},
                ],
            },
            {"img_filename": "b.png", "elements": [{"instruction": "Read More", "bbox": [0.1, 0.1, 0.2, 0.2]}]},
        ]
        by_image = atlas_elements(annotations)
        self.assertEqual([element.instruction for element in by_image["a.png"]], ["Home"])
        self.assertEqual([element.instruction for element in by_image["b.png"]], ["Read More"])

    def test_a_duplicate_with_a_bad_box_still_makes_the_other_ambiguous(self):
        annotations = [
            {
                "img_filename": "a.png",
                "elements": [
                    {"instruction": "OK", "bbox": [0.1, 0.1, 0.2, 0.2]},
                    {"instruction": "OK", "bbox": [5, 5, 6, 6]},
                ],
            }
        ]
        self.assertEqual(atlas_elements(annotations), {})

    def test_junk_is_skipped(self):
        annotations = [
            "not a dict",
            {"img_filename": "", "instruction": "x", "bbox": [0.1, 0.1, 0.2, 0.2]},
            {"img_filename": "a.png", "elements": "nope"},
            {"img_filename": "b.png", "elements": [None, {"instruction": "", "bbox": [0.1, 0.1, 0.2, 0.2]}]},
            {"img_filename": "c.png", "elements": [{"instruction": "y" * 201, "bbox": [0.1, 0.1, 0.2, 0.2]}]},
            {"img_filename": "d.png", "elements": [{"instruction": "Fine", "bbox": [0.1, 0.1, 0.2, 0.2]}]},
        ]
        self.assertEqual(sorted(atlas_elements(annotations)), ["d.png"])
        self.assertEqual(atlas_elements("not a list"), {})

    def test_instructions_up_to_200_characters_are_kept(self):
        annotations = [{"img_filename": "a.png", "elements": [{"instruction": "z" * 200, "bbox": [0.1, 0.1, 0.2, 0.2]}]}]
        self.assertEqual(len(atlas_elements(annotations)["a.png"][0].instruction), 200)

    def test_instructions_that_are_page_code_are_left_out(self):
        # Seen in OS-Atlas's Windows annotations: a CSS class where a label should be.
        instructions = ["Class: octicon arrow-symbol-mktg", "id=submit-button", "classname: btn", "the glyphicon search",
                        "Open <|im_start|> menu", "Classic view", "Identity settings", "Sign in"]
        annotations = [{"img_filename": f"{index}.png", "elements": [{"instruction": text, "bbox": [0.1, 0.1, 0.2, 0.2]}]}
                       for index, text in enumerate(instructions)]
        kept = [elements[0].instruction for elements in atlas_elements(annotations).values()]
        self.assertEqual(sorted(kept), ["Classic view", "Identity settings", "Sign in"])

    def test_whitespace_in_instructions_is_tidied(self):
        annotations = [{"img_filename": "a.png", "elements": [{"instruction": "  New \n window ", "bbox": [0.1, 0.1, 0.2, 0.2]}]}]
        self.assertEqual(atlas_elements(annotations)["a.png"][0].instruction, "New window")


def elements(count, prefix="e"):
    return [AtlasElement(f"{prefix}{index}", (0.1, 0.1, 0.2, 0.2)) for index in range(count)]


class ChooseTests(unittest.TestCase):
    def test_at_most_per_image_from_each_and_limit_in_all(self):
        by_image = {f"img{index}.png": elements(5) for index in range(10)}
        chosen = choose(by_image, limit=7, per_image=3, seed=1)
        self.assertEqual(sum(len(picked) for _, picked in chosen), 7)
        self.assertTrue(all(1 <= len(picked) <= 3 for _, picked in chosen))
        self.assertEqual(len(chosen), 3)

    def test_no_limit_takes_per_image_from_every_screenshot(self):
        by_image = {f"img{index}.png": elements(5) for index in range(4)}
        chosen = choose(by_image, limit=None, per_image=2)
        self.assertEqual(len(chosen), 4)
        self.assertTrue(all(len(picked) == 2 for _, picked in chosen))

    def test_same_seed_same_choice_whatever_the_input_order(self):
        by_image = {f"img{index}.png": elements(4) for index in range(20)}
        reordered = dict(reversed(list(by_image.items())))
        self.assertEqual(choose(by_image, 10, 2, seed=3), choose(reordered, 10, 2, seed=3))
        self.assertNotEqual(choose(by_image, 10, 2, seed=3), choose(by_image, 10, 2, seed=4))

    def test_screenshots_are_picked_at_random_not_in_name_order(self):
        by_image = {f"img{index:03d}.png": elements(1) for index in range(100)}
        picked = [name for name, _ in choose(by_image, limit=10, per_image=1, seed=7)]
        self.assertEqual(len(picked), 10)
        self.assertNotEqual(picked, sorted(by_image)[:10])
        self.assertNotEqual(sorted(picked), sorted(by_image)[:10])

    def test_limit_zero_chooses_nothing(self):
        self.assertEqual(choose({"a.png": elements(3)}, limit=0), [])


class RequestForTests(unittest.TestCase):
    def test_labels_become_click_requests_with_the_label_kept_exactly(self):
        self.assertEqual(request_for("Help and feedback"), 'Click "Help and feedback"')
        self.assertEqual(request_for("  Copilot "), 'Click "Copilot"')

    def test_instructions_that_already_say_what_to_do_are_kept(self):
        self.assertEqual(request_for("open the settings"), "Open the settings")
        self.assertEqual(request_for("go to the home page"), "Go to the home page")
        self.assertEqual(request_for("Close"), "Close")

    def test_templates_vary_with_the_random_source(self):
        made = {request_for("Copilot", random.Random(seed)) for seed in range(40)}
        self.assertGreater(len(made), 1)
        self.assertTrue(all('"Copilot"' in request for request in made))

    def test_empty(self):
        self.assertEqual(request_for("   "), "")


@unittest.skipUnless(HAS_PIL, "needs Pillow")
class AtlasExamplesTests(unittest.TestCase):
    def test_click_in_the_middle_of_each_box_with_the_box_kept_for_grading(self):
        chosen = [("dir/shot.png", [AtlasElement("Copilot", (0.2, 0.0, 0.4, 0.05)), AtlasElement("open help", (0.7, 0.8, 0.9, 0.84))])]
        with tempfile.TemporaryDirectory() as folder:
            made = atlas_examples(chosen, lambda name: png(), folder)
            self.assertEqual(len(made), 2)
            first, second = made
            self.assertEqual(first.target, {"thought": "", "say": "", "action": "click", "x": 300, "y": 25})
            self.assertEqual(first.box, [200.0, 0.0, 400.0, 50.0])
            self.assertEqual(first.group, second.group)
            self.assertEqual(second.request, "Open help")
            self.assertTrue((Path(folder) / first.image).is_file())
            self.assertEqual(first.source, "os-atlas")

    def test_screenshots_are_shrunk_to_the_width_nexus_live_uses(self):
        from PIL import Image

        chosen = [("big.png", [AtlasElement("Copilot", (0.2, 0.2, 0.4, 0.4))])]
        with tempfile.TemporaryDirectory() as folder:
            made = atlas_examples(chosen, lambda name: png(2560, 1440), folder, max_width=1280)
            with Image.open(Path(folder) / made[0].image) as picture:
                self.assertEqual(picture.size, (1280, 720))

    def test_missing_and_broken_screenshots_are_skipped(self):
        chosen = [("gone.png", elements(1)), ("broken.png", elements(1)), ("fine.png", elements(1))]
        reader = {"broken.png": b"not a png", "fine.png": png()}.get
        with tempfile.TemporaryDirectory() as folder:
            made = atlas_examples(chosen, reader, folder)
            self.assertEqual(len(made), 1)
            self.assertFalse(list((Path(folder) / "images").glob("*.part")))

    def test_done_share_adds_a_stop_example_after_the_click(self):
        chosen = [(f"s{index}.png", elements(2)) for index in range(30)]
        with tempfile.TemporaryDirectory() as folder:
            made = atlas_examples(chosen, lambda name: png(), folder, done_share=0.5)
            done = [item for item in made if item.target["action"] == "done"]
            clicks = [item for item in made if item.target["action"] == "click"]
            self.assertEqual(len(clicks), 60)
            self.assertTrue(10 < len(done) < 50)
            for item in done:
                clicked = next(other for other in clicks if other.id == item.id[: -len("-done")])
                self.assertEqual(item.steps, [f"click at ({clicked.target['x']:g}, {clicked.target['y']:g})"])
                self.assertEqual(item.request, clicked.request)
                self.assertIsNone(item.box)

    def test_no_done_examples_by_default(self):
        with tempfile.TemporaryDirectory() as folder:
            made = atlas_examples([("a.png", elements(3))], lambda name: png(), folder)
            self.assertTrue(all(item.target["action"] == "click" for item in made))

    def test_folder_reader_finds_images_by_file_name_when_the_path_differs(self):
        with tempfile.TemporaryDirectory() as folder:
            nested = Path(folder) / "windows_images" / "deeper"
            nested.mkdir(parents=True)
            (nested / "shot.png").write_bytes(b"data")
            read = folder_reader(Path(folder))
            self.assertEqual(read("shot.png"), b"data")
            self.assertEqual(read("windows_images/deeper/shot.png"), b"data")
            self.assertIsNone(read("other.png"))

    def test_from_os_atlas_end_to_end(self):
        from nexus.live.data import from_os_atlas

        annotations = [{"img_filename": f"s{index}.png", "elements": [{"instruction": f"Button {index}", "bbox": [0.1, 0.1, 0.3, 0.2]}]} for index in range(5)]
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source"
            source.mkdir()
            for index in range(5):
                (source / f"s{index}.png").write_bytes(png())
            (Path(folder) / "ann.json").write_text(json.dumps(annotations))
            made = from_os_atlas(Path(folder) / "ann.json", source, Path(folder) / "out", limit=3)
            self.assertEqual(len(made), 3)
            self.assertEqual({item.target["x"] for item in made}, {200})


if __name__ == "__main__":
    unittest.main()
