"""Tests for nexus.live.hardware: which local vision model fits this computer."""

import unittest

from nexus.live.hardware import MODELS, Hardware, recommend_model

SMALL, MEDIUM, LARGE = "qwen3-vl:2b", "qwen3-vl:4b", "qwen3-vl:8b"


def computer(ram=None, vram=None, apple=False, gpu=None):
    return Hardware(system="Test OS 1", ram_gb=ram, gpu_name=gpu, vram_gb=vram, apple_silicon=apple)


def pick(**hardware):
    return recommend_model(computer(**hardware))[0]


class ModelListTests(unittest.TestCase):
    def test_smallest_first(self):
        self.assertEqual([tag for tag, _ in MODELS], [SMALL, MEDIUM, LARGE])

    def test_the_advice_states_the_same_limits_as_recommend_model(self):
        self.assertEqual(
            MODELS,
            (
                (SMALL, "1.9 GB download; runs on almost any laptop, slowly without a graphics card"),
                (MEDIUM, "3.3 GB download; needs a 4 GB graphics card or 16 GB of memory"),
                (LARGE, "6.1 GB download; needs an 8 GB graphics card or an Apple Silicon Mac with 16 GB"),
            ),
        )

    def test_the_reason_comes_with_the_model(self):
        for hardware, index in ((computer(), 0), (computer(vram=4), 1), (computer(vram=8), 2)):
            with self.subTest(index=index):
                self.assertEqual(recommend_model(hardware), MODELS[index])


class RecommendTests(unittest.TestCase):
    def test_graphics_card_memory(self):
        cases = ((24, LARGE), (8, LARGE), (7.99, MEDIUM), (6, MEDIUM), (4, MEDIUM), (3.99, SMALL), (2, SMALL))
        for vram, expected in cases:
            with self.subTest(vram=vram):
                self.assertEqual(pick(vram=vram), expected)

    def test_main_memory_without_a_graphics_card(self):
        cases = ((64, MEDIUM), (16, MEDIUM), (15.99, SMALL), (8, SMALL))
        for ram, expected in cases:
            with self.subTest(ram=ram):
                self.assertEqual(pick(ram=ram), expected)

    def test_apple_silicon_shares_its_memory_with_the_graphics(self):
        cases = ((32, LARGE), (16, LARGE), (15.99, SMALL), (8, SMALL))
        for ram, expected in cases:
            with self.subTest(ram=ram):
                self.assertEqual(pick(ram=ram, apple=True), expected)

    def test_lots_of_memory_does_not_make_a_small_card_large(self):
        self.assertEqual(pick(ram=128, vram=6), MEDIUM)
        self.assertEqual(pick(ram=128, vram=2), MEDIUM)

    def test_large_card_with_little_memory(self):
        self.assertEqual(pick(ram=8, vram=12), LARGE)
        self.assertEqual(pick(ram=None, vram=8), LARGE)

    def test_unknown_or_zero_sizes(self):
        self.assertEqual(pick(), SMALL)
        self.assertEqual(pick(ram=0.0, vram=0.0), SMALL)
        self.assertEqual(pick(apple=True), SMALL)


class DescribeTests(unittest.TestCase):
    def test_graphics_card(self):
        self.assertEqual(computer(ram=31.6, vram=12.0, gpu="RTX 3060").describe(), "Test OS 1, 32 GB memory, RTX 3060, 12 GB")

    def test_graphics_card_of_unknown_size(self):
        self.assertEqual(computer(ram=16, gpu="RTX 3060").describe(), "Test OS 1, 16 GB memory, RTX 3060")

    def test_apple_silicon(self):
        self.assertEqual(computer(ram=16, apple=True).describe(), "Test OS 1, 16 GB memory, Apple Silicon (memory shared with the graphics)")

    def test_no_graphics_card_and_unknown_memory(self):
        self.assertEqual(computer().describe(), "Test OS 1, memory unknown, no NVIDIA graphics card found")
        self.assertEqual(computer(ram=0.0).describe(), "Test OS 1, memory unknown, no NVIDIA graphics card found")


if __name__ == "__main__":
    unittest.main()
