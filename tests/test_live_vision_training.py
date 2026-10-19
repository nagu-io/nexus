"""NEXUS Vision's training, loading and evaluation code, run for real on a tiny Qwen3-VL on the CPU.

Skipped unless torch, transformers and peft are installed (pip install -e .[vision]).
The tiny model's answers are noise; these tests check that every step of the
plumbing works: the loss falls on the answer only, the adapter and model card
are written, NEXUS Live's local backend loads them, and evaluation runs.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from live_tiny_vision import build_tiny_qwen3_vl, vision_stack_available

from nexus.live.data import IMAGES_DIR, Example, write_examples
from nexus.live.model import PROMPT_VERSION, target_json

STACK = vision_stack_available()


def make_dataset(folder: Path, count: int = 10) -> list[Example]:
    from PIL import Image, ImageDraw

    (folder / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    examples = []
    for index in range(count):
        picture = Image.new("RGB", (320, 180), (240, 240, 240))
        ImageDraw.Draw(picture).rectangle([20 + index * 10, 40, 80 + index * 10, 70], fill=(30, 90, 200))
        name = f"{IMAGES_DIR}/s{index}.png"
        picture.save(folder / name)
        box = [round((20 + index * 10) / 320 * 1000, 1), 222.2, round((80 + index * 10) / 320 * 1000, 1), 388.9]
        examples.append(
            Example(
                id=f"click{index}", source="test", group=f"g{index}", image=name, request='Click "Blue"', box=box,
                target={"thought": "", "say": "", "action": "click", "x": round((box[0] + box[2]) / 2), "y": 306},
            )
        )
        if index % 3 == 0:
            examples.append(
                Example(
                    id=f"type{index}", source="test", group=f"g{index}", image=name, request="Type hello",
                    target={"thought": "", "say": "", "action": "type", "text": "hello"}, steps=["click at (100, 300)"],
                )
            )
    write_examples(folder, examples)
    return examples


@unittest.skipUnless(STACK, "needs the vision extra: torch, transformers, peft")
class VisionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.base = build_tiny_qwen3_vl(cls.root / "tiny")
        cls.data = cls.root / "data"
        cls.examples = make_dataset(cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def options(self, out, **changes):
        from nexus.live.finetune import TrainOptions

        values = dict(data=[self.data], out=out, base=str(self.base), max_steps=2, batch_size=2, grad_accum=1,
                      eval_fraction=0.3, workers=0, log_steps=1, save_steps=1000)
        values.update(changes)
        return TrainOptions(**values)

    def test_the_loss_is_on_the_answer_only(self):
        from transformers import AutoProcessor

        from nexus.live.finetune import IGNORE_INDEX, AnswerCollator, load_items

        processor = AutoProcessor.from_pretrained(self.base)
        processor.tokenizer.padding_side = "right"
        items, _ = load_items([self.data])
        batch = AnswerCollator(processor, 1280)(items[:3])
        self.assertIn("pixel_values", batch)
        lengths = batch["attention_mask"].sum(1).tolist()
        self.assertNotEqual(len(set(lengths)), 1, "the batch should need padding")
        for row in range(3):
            labels = batch["labels"][row]
            kept = labels[labels != IGNORE_INDEX].tolist()
            self.assertEqual(processor.tokenizer.decode(kept), target_json(items[row][1].target) + "<|im_end|>")
            padded = batch["attention_mask"][row] == 0
            self.assertTrue((labels[padded] == IGNORE_INDEX).all())

    def test_train_writes_what_the_local_backend_loads(self):
        from nexus.live.evaluate import evaluate, held_out
        from nexus.live.finetune import load_items, train
        from nexus.live.local import LocalVision, read_card

        out = self.root / "trained"
        result = train(self.options(out), printer=lambda line: None)
        self.assertEqual(result.steps, 2)
        self.assertGreater(result.train_examples, 0)
        self.assertGreater(result.eval_examples, 0)
        self.assertIsNotNone(result.eval_loss)
        for name in ("adapter_config.json", "adapter_model.safetensors", "nexus_live.json", "training.json"):
            self.assertTrue((out / name).exists(), name)
        card = read_card(out)
        self.assertEqual((card.prompt_version, card.coordinates, card.base_model, card.screen_width), (PROMPT_VERSION, "normalized", str(self.base), 1280))
        adapter = json.loads((out / "adapter_config.json").read_text())
        self.assertTrue(all("language_model" in name or "|" in name for name in [adapter["target_modules"]]))
        summary = json.loads((out / "training.json").read_text())
        self.assertEqual(summary["options"]["eval_fraction"], 0.3)

        vision = LocalVision(out)
        reply = vision.ask('Click "Blue"', [], [], (self.data / "images" / "s0.png").read_bytes(), "normalized")
        self.assertIsInstance(reply, str)
        # What was generated is kept with the special tokens left in, so "nothing but <|im_end|>" shows.
        self.assertGreaterEqual(vision.last_generation["new_tokens"], 1)
        self.assertIsInstance(vision.last_generation["raw"], str)
        items, _ = load_items([self.data])
        summary, grades = evaluate(vision, held_out(items, 0.3), limit=2, printer=lambda line: None)
        self.assertEqual(summary["examples"], 2)
        self.assertTrue(all(grade.tokens is not None and grade.tokens >= 1 for grade in grades))

        # Watch mode asks the base model (the adapter switched off); requests use NEXUS Vision.
        from nexus.live.screen import Shot

        switched_off = []
        original = vision._model.disable_adapter
        vision._model.disable_adapter = lambda: (switched_off.append(True), original())[1]
        image = (self.data / "images" / "s2.png").read_bytes()
        self.assertIsInstance(vision.observe(Shot(image, 320, 180, 320, 180)), str)
        self.assertEqual(switched_off, [True])
        vision.ask("Type hello", [], [], image, "normalized")
        self.assertEqual(switched_off, [True])
        vision.close()

    def test_the_vision_encoder_is_left_alone_unless_asked(self):
        from nexus.live.finetune import add_adapter, load_base, pick_precision

        precision = pick_precision("off")
        model, _ = load_base(str(self.base), precision)
        tuned = add_adapter(model, self.options(self.root / "x"), precision)
        names = [name for name, parameter in tuned.named_parameters() if parameter.requires_grad]
        self.assertTrue(names)
        self.assertTrue(all("language_model" in name for name in names))
        model, _ = load_base(str(self.base), precision)
        tuned = add_adapter(model, self.options(self.root / "y", tune_vision=True), precision)
        names = [name for name, parameter in tuned.named_parameters() if parameter.requires_grad]
        self.assertTrue(any("visual" in name for name in names))

    def test_merge_makes_a_standalone_model_the_local_backend_loads(self):
        from nexus.live.finetune import merge, train
        from nexus.live.local import LocalVision

        adapter = self.root / "for-merge"
        train(self.options(adapter, max_steps=1), printer=lambda line: None)
        merged = merge(adapter, self.root / "merged", printer=lambda line: None)
        self.assertTrue((merged / "config.json").exists())
        self.assertFalse((merged / "adapter_config.json").exists())
        self.assertTrue((merged / "nexus_live.json").exists())
        vision = LocalVision(merged)
        self.assertIsInstance(vision.ask("Type hello", [], [], (self.data / "images" / "s1.png").read_bytes(), "normalized"), str)
        vision.close()

    def test_unusable_examples_are_counted_and_skipped(self):
        from nexus.live.finetune import load_items

        folder = self.root / "mixed"
        make_dataset(folder, count=2)
        lines = (folder / "examples.jsonl").read_text().splitlines()
        # Recorded under older prompts: still used when its step passes today's checks (the prompts
        # around it are rebuilt at training time), counted invalid when it does not.
        old = json.loads(lines[0]) | {"id": "old", "prompt_version": PROMPT_VERSION - 1}
        typing = {"thought": "", "say": "", "action": "type", "text": "hi\nthere"}
        self.assertEqual(json.loads(lines[1])["target"]["action"], "type")
        old_broken = json.loads(lines[1]) | {"id": "old-broken", "prompt_version": PROMPT_VERSION - 1, "target": typing}
        newer = json.loads(lines[0]) | {"id": "newer", "prompt_version": PROMPT_VERSION + 1}
        broken = json.loads(lines[0]) | {"id": "broken", "request": ""}
        missing = json.loads(lines[0]) | {"id": "missing", "image": "images/gone.png"}
        with open(folder / "examples.jsonl", "a") as handle:
            for item in (old, old_broken, newer, broken, missing):
                handle.write(json.dumps(item) + "\n")
        items, skipped = load_items([folder])
        self.assertEqual([example.id for _, example in items][-1], "old")
        self.assertEqual(len(items), len(lines) + 1)
        self.assertEqual(skipped, {"newer prompt version": 1, "invalid": 2, "image missing": 1})

    def test_a_loss_that_becomes_nan_stops_the_run(self):
        # Trainer used to log the running average in place of a NaN loss, so an overflowed run
        # "finished normally" with loss 0.0 and its adapter was written and uploaded.
        import functools

        from transformers import Qwen3VLForConditionalGeneration

        from nexus.live.finetune import TrainingDataError, train

        original = Qwen3VLForConditionalGeneration.forward

        @functools.wraps(original)
        def overflowing(model, *args, **kwargs):
            output = original(model, *args, **kwargs)
            if output.loss is not None:
                output.loss = output.loss * float("nan")  # what an overflow in float16 gives
            return output

        for eval_fraction in (0.3, 0.0):  # with no held-out examples, the training loss alone has to catch it
            with self.subTest(eval_fraction=eval_fraction):
                out = self.root / f"overflowed-{eval_fraction}"
                logged = []
                with mock.patch.object(Qwen3VLForConditionalGeneration, "forward", overflowing):
                    with self.assertRaisesRegex(TrainingDataError, "became NaN"):
                        train(self.options(out, max_steps=4, log_steps=2, eval_fraction=eval_fraction), printer=logged.append)
                self.assertFalse((out / "adapter_model.safetensors").exists())
                self.assertIn("step 1 of 4: loss nan", logged)  # the loss as it is, not a running average of 0.0

    def test_a_held_out_loss_that_is_nan_stops_the_run_too(self):
        from transformers import Trainer

        from nexus.live.finetune import TrainingDataError, train

        out = self.root / "nan-held-out"
        with mock.patch.object(Trainer, "evaluate", return_value={"eval_loss": float("nan")}):
            with self.assertRaisesRegex(TrainingDataError, "became NaN"):
                train(self.options(out, max_steps=1), printer=lambda line: None)
        self.assertFalse((out / "adapter_model.safetensors").exists())

    def test_the_trained_model_is_freed_when_train_returns(self):
        # Mixed precision (every graphics card run) leaves the model in a reference cycle: without a
        # collection it stayed in memory while NEXUS Vision was evaluated next to it.
        import gc
        import weakref

        from nexus.live import finetune

        alive = []
        real_add_adapter = finetune.add_adapter

        def spy(model, options, precision):
            tuned = real_add_adapter(model, options, precision)
            alive.append(weakref.ref(tuned))
            return tuned

        mixed = finetune.Precision("cpu", "bfloat16", False)
        gc.disable()  # so the result does not depend on when Python happens to collect
        try:
            with mock.patch.object(finetune, "add_adapter", spy), \
                 mock.patch.object(finetune, "pick_precision", return_value=mixed):
                finetune.train(self.options(self.root / "freed", max_steps=1), printer=lambda line: None)
            self.assertIsNone(alive[0]())
        finally:
            gc.enable()

    def test_no_data_is_a_clear_error(self):
        from nexus.live.finetune import TrainingDataError, train

        empty = self.root / "empty"
        write_examples(empty, [])
        with self.assertRaisesRegex(TrainingDataError, "No usable examples"):
            train(self.options(self.root / "none", data=[empty]), printer=lambda line: None)

    def test_build_runs_end_to_end_without_the_network(self):
        from nexus.live.build import parser, run

        out = self.root / "built"
        args = parser().parse_args(
            ["--limit", "0", "--teach", str(self.data), "--work", str(self.root / "work"), "--out", str(out),
             "--base", str(self.base), "--max-steps", "1", "--batch-size", "2", "--grad-accum", "1",
             "--workers", "0", "--eval-limit", "2", "--push", ""]
        )
        lines = []
        freed = []
        from nexus.live import finetune

        real_free = finetune.free_memory
        with mock.patch.object(finetune, "free_memory", side_effect=lambda: (freed.append(len(lines)), real_free())):
            report = run(args, printer=lines.append)
        # Once after training, and before each model is evaluated: nothing earlier stays on the graphics card.
        self.assertEqual(len(freed), 3)
        self.assertEqual(sorted(report["results"]), ["Base model", "NEXUS Vision"])
        readme = (out / "README.md").read_text()
        self.assertIn("base_model: Qwen/Qwen3-VL-2B-Instruct", readme)
        self.assertIn("| Steps right |", readme)
        self.assertEqual(sorted(json.loads((out / "eval.json").read_text())), ["Base model", "NEXUS Vision"])
        details = json.loads((out / "eval_details.json").read_text())
        self.assertEqual(len(details["NEXUS Vision"]), report["results"]["NEXUS Vision"]["examples"])
        self.assertIn("raw", details["NEXUS Vision"][0])
        self.assertIn("torch ", lines[0])  # the run's log starts with the versions and the card
        self.assertFalse(any("Uploaded" in line for line in lines))


@unittest.skipUnless(STACK, "needs the vision extra: torch, transformers, peft")
class BuildArgumentTests(unittest.TestCase):
    def test_no_data_at_all_stops_with_a_message(self):
        from nexus.live.build import parser, run

        with tempfile.TemporaryDirectory() as folder:
            args = parser().parse_args(["--limit", "0", "--work", folder, "--push", ""])
            with self.assertRaises(SystemExit):
                run(args, printer=lambda line: None)


if __name__ == "__main__":
    unittest.main()
