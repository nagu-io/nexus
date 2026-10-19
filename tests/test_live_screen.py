"""Tests for nexus.live.screen: turning the model's positions into pixels, shrinking screenshots, and change detection."""

import dataclasses
import importlib.util
import io
import math
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import nexus.live.screen as screen_module
from nexus.live.screen import (
    NORMALIZED_SPAN,
    Frame,
    OutsideScreen,
    Shot,
    Window,
    changed_near,
    cover_windows,
    grab,
    primary_monitor,
    share_of,
    shrink,
    thumbnail_difference,
    to_screen,
)

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

    def test_shot_cannot_be_changed(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            SHOT.width = 1  # type: ignore[misc]

    def test_not_numbers(self):
        for x, y in ((None, 5.0), (5.0, None), ("5", 5.0), (5.0, "5"), ([1], 5.0)):
            with self.subTest(x=x, y=y):
                with self.assertRaises(OutsideScreen) as caught:
                    to_screen(x, y, SHOT, "normalized")
                self.assertEqual(str(caught.exception), f"not a position: {x!r}, {y!r}")

    def test_not_finite(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                for mode in ("normalized", "pixels"):
                    with self.assertRaises(OutsideScreen) as caught:
                        to_screen(value, 5.0, SHOT, mode)
                    self.assertEqual(str(caught.exception), f"not a position: {value!r}, 5.0")
                    with self.assertRaises(OutsideScreen) as caught:
                        to_screen(5.0, value, SHOT, mode)
                    self.assertEqual(str(caught.exception), f"not a position: 5.0, {value!r}")

    def test_not_a_position_is_found_before_the_mode_is_read(self):
        with self.assertRaises(OutsideScreen):
            to_screen(math.nan, 5.0, SHOT, "percent")

    def test_unknown_mode(self):
        for mode in ("percent", "Normalized", ""):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError) as caught:
                    to_screen(1, 1, SHOT, mode)
                self.assertNotIsInstance(caught.exception, OutsideScreen)
                self.assertEqual(str(caught.exception), f"unknown coordinate mode {mode!r}")

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


def patterned(width, height):
    """An RGB picture with enough detail that PNG optimization makes a difference."""
    from PIL import Image

    image = Image.new("RGB", (width, height))
    image.putdata([((x * 7) % 256, (y * 5) % 256, ((x ^ y) * 3) % 256) for y in range(height) for x in range(width)])
    return image


class FakeMss:
    """The mss package: monitors[0] is every monitor together, monitors[1] the main one."""

    def __init__(self, main, pixel=(30, 20, 10, 255), pattern=False):
        self.monitors = [{"left": -1920, "top": -30, "width": 1920 + main["width"], "height": 1080}, main]
        self.pixel = bytes(pixel)  # blue, green, red, unused: the order mss gives
        self.pattern = pattern
        self.grabbed = []

    def module(self):
        module = ModuleType("mss")
        module.mss = lambda: self
        return module

    def __enter__(self):
        return self

    def __exit__(self, *error):
        return False

    def grab(self, monitor):
        self.grabbed.append(monitor)
        width, height = monitor["width"], monitor["height"]
        if self.pattern:
            data = bytes(value for y in range(height) for x in range(width) for value in ((x * 7) % 256, (y * 5) % 256, ((x ^ y) * 3) % 256, 255))
        else:
            data = self.pixel * (width * height)
        return SimpleNamespace(size=(width, height), bgra=data)


@unittest.skipUnless(HAS_PILLOW, "needs Pillow")
class CaptureTests(unittest.TestCase):
    def capture(self, fake, *args, **kwargs):
        with patch.dict(sys.modules, {"mss": fake.module()}), patch.object(screen_module.platform, "system", return_value="Linux"):
            return screen_module.capture(*args, **kwargs)

    def test_main_monitor_is_captured_and_shrunk(self):
        main = {"left": 100, "top": 50, "width": 40, "height": 20}
        fake = FakeMss(main)
        shot = self.capture(fake, 20)
        self.assertEqual(fake.grabbed, [main])
        self.assertEqual((shot.width, shot.height, shot.screen_width, shot.screen_height, shot.left, shot.top), (20, 10, 40, 20, 100, 50))
        with opened(shot.image) as picture:
            self.assertEqual(picture.mode, "RGB")
            self.assertEqual(picture.getpixel((5, 5)), (10, 20, 30))

    def test_shrunk_to_1280_wide_by_default(self):
        self.assertEqual(self.capture(FakeMss({"left": 0, "top": 0, "width": 1281, "height": 2})).width, 1280)
        self.assertEqual(self.capture(FakeMss({"left": 0, "top": 0, "width": 1279, "height": 2})).width, 1279)

    def test_optimized_by_default(self):
        main = {"left": 0, "top": 0, "width": 320, "height": 180}
        optimized = self.capture(FakeMss(main, pattern=True))
        quick = self.capture(FakeMss(main, pattern=True), optimize=False)
        self.assertLess(len(optimized.image), len(quick.image))

    def test_the_main_monitor_is_captured_even_when_listed_second(self):
        # A laptop with an external screen to its right, which the system lists first.
        laptop = {"left": 0, "top": 0, "width": 40, "height": 20}
        external = {"left": 40, "top": 0, "width": 60, "height": 30}
        fake = FakeMss(laptop)
        fake.monitors = [{"left": 0, "top": 0, "width": 100, "height": 30}, external, laptop]
        shot = self.capture(fake, 100)
        self.assertEqual(fake.grabbed, [laptop])
        self.assertEqual((shot.screen_width, shot.left), (40, 0))


class PrimaryMonitorTests(unittest.TestCase):
    ALL = {"left": -1920, "top": 0, "width": 3840, "height": 1080}

    def test_the_one_marked_primary(self):
        first = {"left": -1920, "top": 0, "width": 1920, "height": 1080}
        main = {"left": 0, "top": 0, "width": 1920, "height": 1080, "is_primary": True}
        self.assertIs(primary_monitor([self.ALL, first, main]), main)
        self.assertIs(primary_monitor([self.ALL, {"left": 0, "top": 0, "width": 1, "height": 1}, main]), main)

    def test_without_marks_the_one_at_the_origin(self):
        # Older mss marks none; Windows always puts the main monitor at (0, 0).
        external = {"left": 1920, "top": 0, "width": 2560, "height": 1440}
        laptop = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        self.assertIs(primary_monitor([self.ALL, external, laptop]), laptop)

    def test_otherwise_the_first(self):
        first = {"left": 10, "top": 0, "width": 800, "height": 600}
        second = {"left": 810, "top": 0, "width": 800, "height": 600}
        self.assertIs(primary_monitor([self.ALL, first, second]), first)


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

    def test_png_is_optimized_unless_asked_not_to(self):
        image = patterned(320, 180)
        self.assertLess(len(shrink(image, 1280, 320, 180).image), len(shrink(image, 1280, 320, 180, optimize=False).image))

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
    def test_top_half_of_the_screen(self):
        top_half = {(x, y): 255 for x in range(64) for y in range(18)}
        self.assertEqual(thumbnail_difference(png((64, 36), 0), png((64, 36), 0, pixels=top_half)), 0.5)

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


@unittest.skipUnless(HAS_PILLOW, "needs Pillow")
class ChangedNearTests(unittest.TestCase):
    """Whether what is under a click changed between the model's look and the click."""

    def test_same_picture(self):
        picture = png((200, 100), 90)
        self.assertEqual(changed_near(picture, picture, (0.5, 0.5)), 0.0)
        self.assertEqual(changed_near(picture, picture, None), 0.0)

    def test_a_change_under_the_point(self):
        middle = {(x, y): 255 for x in range(80, 120) for y in range(35, 65)}
        before, after = png((200, 100), 0), png((200, 100), 0, pixels=middle)
        self.assertGreater(changed_near(before, after, (0.5, 0.5)), 0.9)

    def test_a_change_elsewhere_does_not_count(self):
        # A video playing in a corner, say.
        corner = {(x, y): 255 for x in range(0, 40) for y in range(0, 30)}
        before, after = png((200, 100), 0), png((200, 100), 0, pixels=corner)
        self.assertEqual(changed_near(before, after, (0.75, 0.75)), 0.0)
        self.assertGreater(changed_near(before, after, None), 0.0)

    def test_the_box_is_the_span_around_the_point(self):
        # span 0.2 of a 200 by 100 picture: 40 by 20 pixels around (100, 50).
        inside = {(x, y): 255 for x in range(80, 120) for y in range(40, 60)}
        outside = {(x, y): 255 for x in list(range(0, 79)) + list(range(121, 200)) for y in range(100)}
        before = png((200, 100), 0)
        self.assertEqual(changed_near(before, png((200, 100), 0, pixels=inside), (0.5, 0.5), span=0.2), 1.0)
        self.assertEqual(changed_near(before, png((200, 100), 0, pixels=outside), (0.5, 0.5), span=0.2), 0.0)

    def test_points_at_the_edges(self):
        before, after = png((200, 100), 0), png((200, 100), 255)
        for around in ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
            with self.subTest(around=around):
                self.assertEqual(changed_near(before, after, around), 1.0)

    def test_screenshots_of_different_sizes_are_all_changed(self):
        self.assertEqual(changed_near(png((200, 100), 0), png((100, 100), 0), (0.5, 0.5)), 1.0)

    def test_24_grey_levels_do_not_count_but_25_do(self):
        self.assertEqual(changed_near(png((50, 50), 100), png((50, 50), 124), (0.5, 0.5)), 0.0)
        self.assertEqual(changed_near(png((50, 50), 100), png((50, 50), 125), (0.5, 0.5)), 1.0)


class ShareOfTests(unittest.TestCase):
    def test_normalized(self):
        self.assertEqual(share_of(500, 250, SHOT, "normalized"), (0.5, 0.25))
        self.assertEqual(share_of(0, 1000, SHOT, "normalized"), (0.0, 1.0))

    def test_pixels(self):
        shot = Shot(b"", 1280, 720, 1920, 1080)
        self.assertEqual(share_of(640, 180, shot, "pixels"), (0.5, 0.25))


@unittest.skipUnless(HAS_PILLOW, "needs Pillow")
class FrameTests(unittest.TestCase):
    def frame(self, size=(100, 50), screen=(200, 100), colour=(200, 200, 200), left=0, top=0):
        from PIL import Image

        return Frame(Image.new("RGB", size, colour), screen[0], screen[1], left, top)

    def test_a_frame_is_a_png_only_when_asked(self):
        frame = self.frame()
        self.assertIsNone(frame._png)
        self.assertEqual((frame.width, frame.height, frame.screen_width, frame.screen_height), (100, 50, 200, 100))
        with opened(frame.image) as picture:
            self.assertEqual(picture.size, (100, 50))
        self.assertIs(frame.image, frame.image)  # made once

    def test_grab_shrinks_like_capture(self):
        main = {"left": 0, "top": 0, "width": 40, "height": 20}
        fake = FakeMss(main)
        with patch.dict(sys.modules, {"mss": fake.module()}), patch.object(screen_module.platform, "system", return_value="Linux"):
            frame = grab(20)
        self.assertEqual((frame.width, frame.height, frame.screen_width, frame.left), (20, 10, 40, 0))
        self.assertEqual(frame.picture.getpixel((5, 5)), (10, 20, 30))


NOTEPAD, KEEPASS, CHROME = 101, 202, 303  # the programs the windows belong to


def notepad(box, front=True):
    return Window("Untitled - Notepad", box, NOTEPAD, front)


def keepass(box, title="KeePass"):
    return Window(title, box, KEEPASS)


def keepass_hidden(title):
    return title == "KeePass"


@unittest.skipUnless(HAS_PILLOW, "needs Pillow")
class CoverWindowsTests(unittest.TestCase):
    """Teach mode's screenshots never show a blocked window, wherever it is on the screen."""

    def frame(self):
        from PIL import Image

        # A 200 by 100 screen seen at half size.
        return Frame(Image.new("RGB", (100, 50), (200, 200, 200)), 200, 100)

    def black(self, frame, x, y):
        return frame.picture.getpixel((x, y)) == (0, 0, 0)

    def test_a_blocked_window_beside_the_one_in_use_is_blacked_out(self):
        frame = self.frame()
        before = frame.image  # a PNG made before covering is not the one saved
        layout = [notepad((0, 0, 100, 100)), keepass((100, 0, 200, 100))]
        self.assertTrue(cover_windows(frame, layout, keepass_hidden))
        self.assertTrue(self.black(frame, 75, 25))
        self.assertFalse(self.black(frame, 25, 25))
        self.assertNotEqual(frame.image, before)
        with opened(frame.image) as picture:
            self.assertEqual(picture.convert("RGB").getpixel((75, 25)), (0, 0, 0))

    def test_the_window_in_front_uncovers_only_the_part_of_it_that_it_covers(self):
        # KeePass lies behind Notepad, which covers its left half.
        frame = self.frame()
        layout = [notepad((0, 0, 120, 100)), keepass((60, 0, 200, 100))]
        cover_windows(frame, layout, keepass_hidden)
        self.assertFalse(self.black(frame, 55, 25))  # Notepad, over KeePass
        self.assertTrue(self.black(frame, 65, 25))  # KeePass, where it shows
        self.assertTrue(self.black(frame, 99, 49))

    def test_a_blocked_window_on_top_covers_what_is_under_it(self):
        frame = self.frame()
        layout = [keepass((40, 20, 80, 60)), notepad((0, 0, 200, 100))]
        cover_windows(frame, layout, keepass_hidden)
        self.assertTrue(self.black(frame, 30, 20))
        self.assertFalse(self.black(frame, 10, 10))

    def test_edges_are_covered_generously_and_uncovered_strictly(self):
        # Half-pixel edges round outward for a hidden window, inward for the one in front over it.
        frame = self.frame()
        layout = [Window("Notes", (0, 0, 101, 100), NOTEPAD, True), keepass((99, 0, 200, 100))]
        cover_windows(frame, layout, keepass_hidden)
        self.assertTrue(self.black(frame, 50, 10))  # 100 to 101 on the screen: half Notes, half KeePass
        self.assertFalse(self.black(frame, 49, 10))

    def test_a_pixel_a_hidden_window_only_partly_covers_is_covered(self):
        # At half size, pixel 50 shows screen pixels 100 and 101; KeePass starts at 101.
        frame = self.frame()
        cover_windows(frame, [keepass((101, 0, 200, 100))], lambda title: True)
        self.assertTrue(self.black(frame, 50, 10))
        self.assertFalse(self.black(frame, 49, 10))

    def test_nothing_hidden_leaves_the_picture_alone(self):
        frame = self.frame()
        original = frame.picture
        self.assertFalse(cover_windows(frame, [notepad((0, 0, 200, 100))], lambda title: False))
        self.assertIs(frame.picture, original)

    def test_windows_off_this_monitor_or_with_no_size_do_nothing(self):
        frame = self.frame()
        layout = [keepass((-500, 0, -100, 100)), keepass((50, 50, 50, 90))]
        self.assertFalse(cover_windows(frame, layout, lambda title: True))

    def test_a_monitor_that_does_not_start_at_zero(self):
        from PIL import Image

        frame = Frame(Image.new("RGB", (100, 50), (200, 200, 200)), 200, 100, left=-200, top=0)
        cover_windows(frame, [keepass((-100, 0, 0, 100))], lambda title: True)
        self.assertTrue(self.black(frame, 75, 25))
        self.assertFalse(self.black(frame, 25, 25))

    def test_a_see_through_window_on_top_never_uncovers_a_blocked_one(self):
        # A graphics card's full-screen overlay, a screen recorder's frame, any untitled window on top:
        # visible top-level windows that show what is under them.
        for overlay in (Window("NVIDIA GeForce Overlay", (0, 0, 200, 100), 404), Window(None, (0, 0, 200, 100), None)):
            with self.subTest(overlay=overlay.title):
                frame = self.frame()
                layout = [overlay, notepad((0, 0, 100, 100)), keepass((100, 0, 200, 100))]
                self.assertTrue(cover_windows(frame, layout, keepass_hidden))
                self.assertTrue(self.black(frame, 75, 25))
                self.assertFalse(self.black(frame, 25, 25))

    def test_a_window_not_in_front_never_hides_a_blocked_one(self):
        frame = self.frame()
        layout = [Window("Calculator", (100, 0, 150, 100), 505), notepad((0, 0, 100, 100)), keepass((100, 0, 200, 100))]
        cover_windows(frame, layout, keepass_hidden)
        self.assertTrue(self.black(frame, 60, 25))  # under the calculator, KeePass is still covered

    def test_a_blocked_program_s_other_windows_are_covered_titled_or_not(self):
        # KeePass's tooltip, drop-down or dialog over Notepad: its own window, no blocked title of its own.
        frame = self.frame()
        layout = [keepass((80, 20, 140, 60), title=None), keepass((20, 60, 60, 100), title="Edit Entry"),
                  notepad((0, 0, 100, 100)), keepass((100, 0, 200, 100))]
        cover_windows(frame, layout, keepass_hidden)
        self.assertTrue(self.black(frame, 45, 20))  # the tooltip, over Notepad
        self.assertTrue(self.black(frame, 20, 40))  # the dialog, over Notepad
        self.assertFalse(self.black(frame, 25, 15))  # Notepad itself

    def test_a_program_s_window_in_front_stays_when_another_of_its_windows_is_blocked(self):
        # One browser program, two windows: the bank's behind the one being worked in.
        frame = self.frame()
        layout = [Window("Inbox - Gmail - Google Chrome", (0, 0, 120, 100), CHROME, True),
                  Window("Online Banking - Google Chrome", (60, 0, 200, 100), CHROME)]
        cover_windows(frame, layout, lambda title: "Banking" in (title or ""))
        self.assertFalse(self.black(frame, 25, 25))
        self.assertFalse(self.black(frame, 55, 25))  # the window in front, over the bank's
        self.assertTrue(self.black(frame, 65, 25))

    def test_windows_whose_program_is_unknown_are_not_grouped(self):
        frame = self.frame()
        layout = [Window("Untitled - Notepad", (0, 0, 100, 100), None, True), Window(None, (100, 0, 150, 100), None),
                  Window("KeePass", (150, 0, 200, 100), None)]
        cover_windows(frame, layout, keepass_hidden)
        self.assertFalse(self.black(frame, 60, 25))  # an untitled window of no known program, beside KeePass
        self.assertTrue(self.black(frame, 90, 25))


if __name__ == "__main__":
    unittest.main()
