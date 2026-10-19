"""Tests for nexus.live.screen: turning the model's positions into pixels, shrinking screenshots, and change detection."""

import importlib.util
import io
import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import nexus.live.screen as screen_module
from nexus.live.screen import NORMALIZED_SPAN, OutsideScreen, Shot, shrink, thumbnail_difference, to_screen

HAS_PILLOW = importlib.util.find_spec("PIL") is not None

# The model saw a 1280 by 720 copy of a 1920 by 1080 monitor.
SHOT = Shot(image=b"", width=1280, height=720, screen_width=1920, screen_height=1080)
# The model saw a 1280 by 720 copy of a 2560 by 1440 monitor that starts at (100, 50) on the desk.
OFFSET_SHOT = Shot(image=b"", width=1280, height=720, screen_width=2560, screen_height=1440, left=100, top=50)


class NormalizedTests(unittest.TestCase):
    def test_span(self):
        self.assertEqual(NORMALIZED_SPAN, 1000.0)

    def test_corners_and_middle(self):
        self.assertEqual(to_screen(0, 0, SHOT, "normalized"), (0, 0))
        self.assertEqual(to_screen(500, 500, SHOT, "normalized"), (960, 540))
        self.assertEqual(to_screen(333, 250, SHOT, "normalized"), (639, 270))

    def test_far_edge_is_on_screen_and_clamped_to_the_last_pixel(self):
        self.assertEqual(to_screen(1000, 1000, SHOT, "normalized"), (1919, 1079))
        self.assertEqual(to_screen(1000, 0, SHOT, "normalized"), (1919, 0))
        self.assertEqual(to_screen(0, 1000, SHOT, "normalized"), (0, 1079))
        self.assertEqual(to_screen(999.9, 999.9, SHOT, "normalized"), (1919, 1079))

    def test_just_outside_each_edge(self):
        for x, y in ((-0.001, 500), (1000.001, 500), (500, -0.001), (500, 1000.001), (-1, -1), (1001, 0)):
            with self.subTest(x=x, y=y):
                with self.assertRaises(OutsideScreen):
                    to_screen(x, y, SHOT, "normalized")

    def test_span_does_not_depend_on_the_screenshot_size(self):
        with self.assertRaises(OutsideScreen):
            to_screen(1280, 0, SHOT, "normalized")

    def test_monitor_offset_is_added(self):
        self.assertEqual(to_screen(0, 0, OFFSET_SHOT, "normalized"), (100, 50))
        self.assertEqual(to_screen(1000, 1000, OFFSET_SHOT, "normalized"), (100 + 2559, 50 + 1439))

    def test_monitor_left_of_the_main_one(self):
        shot = Shot(image=b"", width=1280, height=720, screen_width=1920, screen_height=1080, left=-1920, top=-200)
        self.assertEqual(to_screen(0, 0, shot, "normalized"), (-1920, -200))
        self.assertEqual(to_screen(1000, 1000, shot, "normalized"), (-1, 879))

    def test_message(self):
        with self.assertRaises(OutsideScreen) as caught:
            to_screen(1000.5, 10, SHOT, "normalized")
        self.assertEqual(str(caught.exception), "(1000.5, 10) is outside the screenshot (0 to 1000, 0 to 1000)")


class PixelTests(unittest.TestCase):
    def test_corners_and_middle(self):
        self.assertEqual(to_screen(0, 0, OFFSET_SHOT, "pixels"), (100, 50))
        self.assertEqual(to_screen(640, 360, OFFSET_SHOT, "pixels"), (100 + 1280, 50 + 720))

    def test_far_edge_is_on_screen_and_clamped_to_the_last_pixel(self):
        self.assertEqual(to_screen(1280, 720, OFFSET_SHOT, "pixels"), (100 + 2559, 50 + 1439))

    def test_x_runs_over_the_width_and_y_over_the_height(self):
        self.assertEqual(to_screen(721, 0, OFFSET_SHOT, "pixels"), (100 + 1442, 50))
        with self.assertRaises(OutsideScreen):
            to_screen(0, 721, OFFSET_SHOT, "pixels")

    def test_just_outside_each_edge(self):
        for x, y in ((-0.01, 0), (1280.01, 0), (0, -0.01), (0, 720.01), (1281, 360), (640, 721)):
            with self.subTest(x=x, y=y):
                with self.assertRaises(OutsideScreen):
                    to_screen(x, y, OFFSET_SHOT, "pixels")

    def test_unshrunk_screenshot(self):
        shot = Shot(image=b"", width=100, height=100, screen_width=100, screen_height=100)
        self.assertEqual(to_screen(0.9, 99.9, shot, "pixels"), (0, 99))
        self.assertEqual(to_screen(100, 100, shot, "pixels"), (99, 99))

    def test_message(self):
        with self.assertRaises(OutsideScreen) as caught:
            to_screen(1280.5, 0, OFFSET_SHOT, "pixels")
        self.assertEqual(str(caught.exception), "(1280.5, 0) is outside the screenshot (0 to 1280, 0 to 720)")


class InputTests(unittest.TestCase):
    def test_outside_screen_is_a_value_error(self):
        self.assertTrue(issubclass(OutsideScreen, ValueError))

    def test_not_numbers(self):
        for x, y in ((None, 5.0), (5.0, None), ("5", 5.0), (5.0, "5"), ([1], 5.0)):
            with self.subTest(x=x, y=y):
                with self.assertRaises(OutsideScreen):
                    to_screen(x, y, SHOT, "normalized")

    def test_not_finite(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                for mode in ("normalized", "pixels"):
                    with self.assertRaises(OutsideScreen):
                        to_screen(value, 5.0, SHOT, mode)
                    with self.assertRaises(OutsideScreen):
                        to_screen(5.0, value, SHOT, mode)

    def test_unknown_mode(self):
        for mode in ("percent", "Normalized", ""):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError) as caught:
                    to_screen(1, 1, SHOT, mode)
                self.assertNotIsInstance(caught.exception, OutsideScreen)
                self.assertIn("unknown coordinate mode", str(caught.exception))

    def test_whole_numbers_work_too(self):
        self.assertEqual(to_screen(500, 500, SHOT, "normalized"), to_screen(500.0, 500.0, SHOT, "normalized"))


class DpiTests(unittest.TestCase):
    def fake_ctypes(self, fail_new=False, fail_old=False):
        calls = []

        def new_call(level):
            calls.append(("SetProcessDpiAwareness", level))
            if fail_new:
                raise OSError("not on this Windows")

        def old_call():
            calls.append(("SetProcessDPIAware",))
            if fail_old:
                raise OSError("not on this Windows either")

        windll = SimpleNamespace(shcore=SimpleNamespace(SetProcessDpiAwareness=new_call), user32=SimpleNamespace(SetProcessDPIAware=old_call))
        return SimpleNamespace(windll=windll), calls

    def test_only_on_windows(self):
        fake, calls = self.fake_ctypes()
        with patch.object(screen_module, "ctypes", fake), patch.object(screen_module.platform, "system", return_value="Linux"):
            screen_module._make_dpi_aware()
        self.assertEqual(calls, [])

    def test_per_monitor_awareness_on_windows(self):
        fake, calls = self.fake_ctypes()
        with patch.object(screen_module, "ctypes", fake), patch.object(screen_module.platform, "system", return_value="Windows"):
            screen_module._make_dpi_aware()
        self.assertEqual(calls, [("SetProcessDpiAwareness", 2)])

    def test_older_windows_falls_back(self):
        fake, calls = self.fake_ctypes(fail_new=True)
        with patch.object(screen_module, "ctypes", fake), patch.object(screen_module.platform, "system", return_value="Windows"):
            screen_module._make_dpi_aware()
        self.assertEqual(calls, [("SetProcessDpiAwareness", 2), ("SetProcessDPIAware",)])

    def test_failure_is_not_an_error(self):
        fake, calls = self.fake_ctypes(fail_new=True, fail_old=True)
        with patch.object(screen_module, "ctypes", fake), patch.object(screen_module.platform, "system", return_value="Windows"):
            screen_module._make_dpi_aware()
        self.assertEqual(len(calls), 2)


def png(size, colour=0, mode="L", pixels=None):
    """A PNG of ``size`` in one colour, with optional {(x, y): colour} pixels painted on it."""
    from PIL import Image

    image = Image.new(mode, size, colour)
    for position, value in (pixels or {}).items():
        image.putpixel(position, value)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def opened(data):
    from PIL import Image

    return Image.open(io.BytesIO(data))


@unittest.skipUnless(HAS_PILLOW, "needs Pillow")
class ShrinkTests(unittest.TestCase):
    def image(self, size):
        from PIL import Image

        return Image.new("RGB", size, (30, 60, 90))

    def test_wide_screen_is_shrunk_keeping_its_shape(self):
        shot = shrink(self.image((2560, 1440)), 1280, 2560, 1440, left=-2560, top=10)
        self.assertEqual(shot, Shot(shot.image, 1280, 720, 2560, 1440, -2560, 10))
        self.assertTrue(shot.image.startswith(b"\x89PNG\r\n\x1a\n"))
        with opened(shot.image) as picture:
            self.assertEqual(picture.size, (1280, 720))

    def test_exactly_the_limit_is_left_alone(self):
        shot = shrink(self.image((1280, 720)), 1280, 1280, 720)
        self.assertEqual((shot.width, shot.height), (1280, 720))

    def test_one_pixel_over_the_limit_is_shrunk(self):
        shot = shrink(self.image((1281, 720)), 1280, 1281, 720)
        self.assertEqual((shot.width, shot.height), (1280, 719))
        with opened(shot.image) as picture:
            self.assertEqual(picture.size, (1280, 719))

    def test_smaller_screens_are_not_enlarged(self):
        shot = shrink(self.image((800, 600)), 1280, 800, 600)
        self.assertEqual((shot.width, shot.height, shot.left, shot.top), (800, 600, 0, 0))

    def test_height_is_rounded(self):
        shot = shrink(self.image((3, 4)), 2, 3, 4)
        self.assertEqual((shot.width, shot.height), (2, 3))

    def test_height_is_at_least_one_pixel(self):
        shot = shrink(self.image((5000, 1)), 1280, 5000, 1)
        self.assertEqual((shot.width, shot.height), (1280, 1))

    def test_unoptimized_png_shows_the_same_picture(self):
        from PIL import Image

        image = Image.new("RGB", (64, 32), (0, 0, 0))
        image.putpixel((3, 4), (255, 128, 0))
        quick, small = shrink(image, 1280, 64, 32, optimize=False), shrink(image, 1280, 64, 32)
        with opened(quick.image) as first, opened(small.image) as second:
            self.assertEqual(first.tobytes(), second.tobytes())
            self.assertEqual(first.getpixel((3, 4)), (255, 128, 0))


class ThumbnailDifferenceTests(unittest.TestCase):
    def test_no_earlier_screenshot_is_all_changed_and_needs_no_pillow(self):
        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            self.assertEqual(thumbnail_difference(None, b"anything"), 1.0)

    @unittest.skipUnless(HAS_PILLOW, "needs Pillow")
    def test_same_picture(self):
        picture = png((64, 36), 100)
        self.assertEqual(thumbnail_difference(picture, picture), 0.0)

    @unittest.skipUnless(HAS_PILLOW, "needs Pillow")
    def test_black_to_white(self):
        self.assertEqual(thumbnail_difference(png((64, 36), 0), png((64, 36), 255)), 1.0)

    @unittest.skipUnless(HAS_PILLOW, "needs Pillow")
    def test_half_the_screen(self):
        left_half = {(x, y): 255 for x in range(32) for y in range(36)}
        self.assertEqual(thumbnail_difference(png((64, 36), 0), png((64, 36), 0, pixels=left_half)), 0.5)

    @unittest.skipUnless(HAS_PILLOW, "needs Pillow")
    def test_a_change_of_24_grey_levels_does_not_count_but_25_does(self):
        self.assertEqual(thumbnail_difference(png((64, 36), 100), png((64, 36), 124)), 0.0)
        self.assertEqual(thumbnail_difference(png((64, 36), 100), png((64, 36), 125)), 1.0)
        self.assertEqual(thumbnail_difference(png((64, 36), 124), png((64, 36), 100)), 0.0)
        self.assertEqual(thumbnail_difference(png((64, 36), 125), png((64, 36), 100)), 1.0)

    @unittest.skipUnless(HAS_PILLOW, "needs Pillow")
    def test_screenshots_of_other_sizes_are_compared_small(self):
        self.assertEqual(thumbnail_difference(png((640, 360), 80), png((64, 36), 80)), 0.0)

    @unittest.skipUnless(HAS_PILLOW, "needs Pillow")
    def test_colour_is_compared_as_grey(self):
        red, black = png((64, 36), (255, 0, 0), mode="RGB"), png((64, 36), (0, 0, 0), mode="RGB")
        dark_blue = png((64, 36), (0, 0, 30), mode="RGB")
        self.assertEqual(thumbnail_difference(black, red), 1.0)
        self.assertEqual(thumbnail_difference(black, dark_blue), 0.0)

    @unittest.skipUnless(HAS_PILLOW, "needs Pillow")
    def test_custom_thumbnail_size(self):
        self.assertEqual(thumbnail_difference(png((2, 1), 0), png((2, 1), 0, pixels={(1, 0): 255}), size=(2, 1)), 0.5)


if __name__ == "__main__":
    unittest.main()
