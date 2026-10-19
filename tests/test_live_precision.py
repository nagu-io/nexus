"""Tests for how NEXUS Vision picks its number format on a given graphics card, and recovers from an overflow."""

import importlib.util
import json
import math
import os
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from nexus.live import app, finetune, local
from nexus.live.data import Example, write_examples
from nexus.live.finetune import NonFiniteLoss, Precision, TrainingDataError, TrainOptions, loss_is_bad, pick_precision
from nexus.live.hardware import Hardware
from nexus.live.settings import LiveSettings

GB = 1024**3


def fake_torch(cuda=True, capability=(7, 5), memory_gb=15.0, bf16_reported=True):
    """A stand-in for torch with just the CUDA questions the code asks."""
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float16, torch.float32 = "bfloat16", "float16", "float32"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        get_device_capability=lambda index=0: capability,
        get_device_properties=lambda index=0: types.SimpleNamespace(total_memory=memory_gb * GB),
        # Recent torch says yes on a T4 too, counting slow emulation: the code must not trust it.
        is_bf16_supported=lambda including_emulation=True: bf16_reported,
    )
    return torch


class PickPrecisionTests(unittest.TestCase):
    def pick(self, four_bit="auto", compute="auto", **card):
        with mock.patch.dict(sys.modules, {"torch": fake_torch(**card)}):
            return pick_precision(four_bit, compute)

    def test_a_free_t4_trains_in_float16_on_a_4_bit_base(self):
        # A T4 is compute capability 7.5 with 15 GB; torch reports bfloat16 "supported" by emulation.
        self.assertEqual(self.pick(capability=(7, 5), memory_gb=15), Precision("cuda", "float16", True))

    def test_a_laptop_rtx_3050_uses_bfloat16_on_a_4_bit_base(self):
        self.assertEqual(self.pick(capability=(8, 6), memory_gb=4), Precision("cuda", "bfloat16", True))

    def test_a_24_gb_card_trains_the_full_base_in_bfloat16(self):
        self.assertEqual(self.pick(capability=(8, 6), memory_gb=24), Precision("cuda", "bfloat16", False))

    def test_the_12_gb_boundary(self):
        self.assertTrue(self.pick(capability=(8, 9), memory_gb=11.9).four_bit)
        self.assertFalse(self.pick(capability=(8, 9), memory_gb=12).four_bit)

    def test_capability_8_is_the_first_with_bfloat16(self):
        self.assertEqual(self.pick(capability=(8, 0), memory_gb=40).dtype_name, "bfloat16")
        self.assertEqual(self.pick(capability=(7, 9), memory_gb=40).dtype_name, "float16")

    def test_four_bit_can_be_forced_either_way(self):
        self.assertTrue(self.pick(four_bit="on", capability=(8, 6), memory_gb=24).four_bit)
        self.assertFalse(self.pick(four_bit="off", capability=(8, 6), memory_gb=4).four_bit)

    def test_compute_can_be_chosen(self):
        self.assertEqual(self.pick(compute="fp32").dtype_name, "float32")
        self.assertEqual(self.pick(compute="fp16", capability=(8, 6)).dtype_name, "float16")

    def test_bfloat16_on_a_card_without_it_is_refused(self):
        with self.assertRaisesRegex(TrainingDataError, "no bfloat16"):
            self.pick(compute="bf16", capability=(7, 5))

    def test_no_graphics_card(self):
        self.assertEqual(self.pick(cuda=False), Precision("cpu", "float32", False))
        with self.assertRaisesRegex(TrainingDataError, "NVIDIA"):
            self.pick(four_bit="on", cuda=False)


class LossGuardTests(unittest.TestCase):
    def test_bad_losses(self):
        self.assertTrue(loss_is_bad({"loss": math.nan}))
        self.assertTrue(loss_is_bad({"loss": math.inf}))
        self.assertTrue(loss_is_bad({"loss": -math.inf}))
        self.assertFalse(loss_is_bad({"loss": 2.5}))
        self.assertFalse(loss_is_bad({"grad_norm": math.nan}))  # a skipped float16 step is normal; only the loss counts
        self.assertFalse(loss_is_bad(None))

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "needs transformers")
    def test_the_callback_stops_training_at_the_first_bad_loss(self):
        guard = finetune._guard()
        control = types.SimpleNamespace(should_training_stop=False)
        guard.on_log(None, None, control, logs={"loss": 1.2})
        self.assertFalse(control.should_training_stop)
        self.assertFalse(guard.tripped)
        guard.on_log(None, None, control, logs={"loss": math.nan})
        self.assertTrue(control.should_training_stop)
        self.assertTrue(guard.tripped)


class FakeSaved:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, folder):
        self.saved_to = Path(folder)


class FallbackTests(unittest.TestCase):
    """train() restarts in float32 when float16 overflows, and only then."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        data = self.root / "data"
        (data / "images").mkdir(parents=True)
        examples = []
        for index in range(30):
            (data / "images" / f"{index}.png").write_bytes(b"png")
            examples.append(Example(id=str(index), source="t", group=f"g{index}", image=f"images/{index}.png",
                                    request="Click OK", target={"action": "click", "x": 1, "y": 2}))
        write_examples(data, examples)
        self.options = TrainOptions(data=[data], out=self.root / "out", eval_fraction=0.2)
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def fit(self, overflow_in):
        def fit(options, precision, train_items, eval_items, out, printer):
            self.calls.append(precision.dtype_name)
            (out / "checkpoints").mkdir(exist_ok=True)
            if precision.dtype_name in overflow_in:
                raise NonFiniteLoss()
            return FakeSaved(), FakeSaved(), types.SimpleNamespace(global_step=7, training_loss=0.5), 0.6
        return fit

    def run_train(self, precision, overflow_in, options=None):
        lines = []
        with mock.patch.object(finetune, "pick_precision", return_value=precision), \
             mock.patch.object(finetune, "_fit", side_effect=self.fit(overflow_in)):
            result = finetune.train(options or self.options, printer=lines.append)
        return result, lines

    def test_float16_overflow_starts_again_in_float32(self):
        result, lines = self.run_train(Precision("cuda", "float16", True), overflow_in={"float16"})
        self.assertEqual(self.calls, ["float16", "float32"])
        self.assertIn("float32", result.precision)
        self.assertTrue(any("overflowed in float16" in line for line in lines))
        summary = json.loads((self.root / "out" / "training.json").read_text())
        self.assertIn("float32", summary["precision"])
        self.assertTrue((self.root / "out" / "nexus_live.json").exists())

    def test_the_first_run_checkpoints_are_cleared_before_the_second(self):
        cleared = []
        original = finetune.shutil.rmtree

        def rmtree(path, ignore_errors=False):
            cleared.append(Path(path).name)
            original(path, ignore_errors=ignore_errors)

        with mock.patch.object(finetune.shutil, "rmtree", side_effect=rmtree):
            self.run_train(Precision("cuda", "float16", True), overflow_in={"float16"})
        self.assertEqual(cleared, ["checkpoints"])

    def test_no_overflow_trains_once(self):
        result, lines = self.run_train(Precision("cuda", "float16", True), overflow_in=set())
        self.assertEqual(self.calls, ["float16"])
        self.assertEqual(result.steps, 7)

    def test_an_overflow_in_float32_too_is_a_clear_error(self):
        with self.assertRaisesRegex(TrainingDataError, "even in float32"):
            self.run_train(Precision("cuda", "float16", True), overflow_in={"float16", "float32"})

    def test_no_fallback_from_bfloat16(self):
        with self.assertRaisesRegex(TrainingDataError, "became NaN"):
            self.run_train(Precision("cuda", "bfloat16", True), overflow_in={"bfloat16"})
        self.assertEqual(self.calls, ["bfloat16"])

    def test_no_fallback_when_float16_was_asked_for(self):
        with self.assertRaisesRegex(TrainingDataError, "became NaN"):
            self.run_train(Precision("cuda", "float16", True), overflow_in={"float16"},
                           options=replace(self.options, compute="fp16"))
        self.assertEqual(self.calls, ["float16"])


class OneGpuTests(unittest.TestCase):
    def test_first_card_unless_one_is_chosen(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            finetune.use_one_gpu()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            finetune.use_one_gpu()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")


class RuntimeDtypeTests(unittest.TestCase):
    def test_bfloat16_on_ampere_and_later_float16_before(self):
        self.assertEqual(local._compute_dtype(fake_torch(capability=(8, 6))), "bfloat16")
        self.assertEqual(local._compute_dtype(fake_torch(capability=(8, 0))), "bfloat16")
        self.assertEqual(local._compute_dtype(fake_torch(capability=(7, 5))), "float16")
        self.assertEqual(local._compute_dtype(fake_torch(cuda=False)), "float16")


class CudaTorchCheckTests(unittest.TestCase):
    def lines(self, gpu, cuda):
        settings = replace(LiveSettings(), model_dir="/nonexistent/nexus_vision")
        hardware = Hardware(system="Windows", ram_gb=16, gpu_name=gpu, vram_gb=4 if gpu else None, apple_silicon=False)
        with mock.patch.object(app, "_torch_sees_cuda", return_value=cuda), \
             mock.patch.object(app.importlib.util, "find_spec", return_value=object()):
            return [line for line in app._local_lines(settings, hardware) if line[1] == "Graphics card"]

    def test_a_cpu_only_torch_next_to_an_nvidia_card_is_flagged(self):
        lines = self.lines("NVIDIA GeForce RTX 3050 Laptop GPU", cuda=False)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0][0], "warn")
        self.assertIn("pytorch.org", lines[0][2])

    def test_a_cuda_torch_is_fine(self):
        self.assertEqual(self.lines("NVIDIA GeForce RTX 3050 Laptop GPU", cuda=True), [])

    def test_no_nvidia_card_no_warning(self):
        self.assertEqual(self.lines(None, cuda=False), [])


if __name__ == "__main__":
    unittest.main()
