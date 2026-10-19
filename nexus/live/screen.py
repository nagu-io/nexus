"""Looking at the screen: a screenshot for the model, and its positions back on the screen.

The model sees a shrunken copy of the main monitor. It answers with a position
on that copy; ``to_screen`` turns it back into a position on the monitor.
"""

from __future__ import annotations

import ctypes
import io
import math
import platform
from dataclasses import dataclass

# 0 to 1000 across the screenshot, as Qwen3-VL gives positions.
NORMALIZED_SPAN = 1000.0


@dataclass(frozen=True)
class Shot:
    image: bytes  # PNG of what the model is shown
    width: int  # its size
    height: int
    screen_width: int  # the monitor it was taken from
    screen_height: int
    left: int = 0  # where that monitor starts, on a desk of several
    top: int = 0


class OutsideScreen(ValueError):
    """The model pointed outside the screenshot."""


def _make_dpi_aware() -> None:
    # Without this, Windows with display scaling (125%, 150%) reports a
    # smaller screen than the screenshot, and clicks land in the wrong place.
    if platform.system() != "Windows":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]  # per-monitor aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
        except Exception:
            pass


def capture(max_width: int = 1280, optimize: bool = True) -> Shot:
    """A screenshot of the main monitor, shrunk to at most ``max_width`` pixels wide.

    ``optimize=False`` saves the PNG quickly at a slightly bigger size, for
    teach mode, which takes one every few tenths of a second.
    """
    import mss  # type: ignore
    from PIL import Image  # type: ignore

    _make_dpi_aware()
    with mss.mss() as grabber:
        monitor = grabber.monitors[1]
        raw = grabber.grab(monitor)
        image = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    return shrink(image, max_width, monitor["width"], monitor["height"], monitor["left"], monitor["top"], optimize)


def shrink(
    image, max_width: int, screen_width: int, screen_height: int, left: int = 0, top: int = 0, optimize: bool = True
) -> Shot:
    """The Shot for an image of a screen, shrunk to ``max_width`` keeping its shape."""
    from PIL import Image  # type: ignore

    width, height = image.size
    if width > max_width:
        height = max(1, round(height * max_width / width))
        width = max_width
        image = image.resize((width, height), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=optimize)
    return Shot(buffer.getvalue(), width, height, screen_width, screen_height, left, top)


def to_screen(x: float, y: float, shot: Shot, mode: str) -> tuple[int, int]:
    """A position the model gave on the screenshot, as a pixel on the monitor.

    Raises OutsideScreen for a position that is not on the screenshot: a model
    that points off the edge has misread it, and clicking the nearest edge
    instead could press something it never meant.
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))) or not (math.isfinite(x) and math.isfinite(y)):
        raise OutsideScreen(f"not a position: {x!r}, {y!r}")
    if mode == "normalized":
        span_x = span_y = NORMALIZED_SPAN
    elif mode == "pixels":
        span_x, span_y = float(shot.width), float(shot.height)
    else:
        raise ValueError(f"unknown coordinate mode {mode!r}")
    if not (0 <= x <= span_x and 0 <= y <= span_y):
        raise OutsideScreen(f"({x:g}, {y:g}) is outside the screenshot (0 to {span_x:g}, 0 to {span_y:g})")
    screen_x = min(shot.screen_width - 1, int(x / span_x * shot.screen_width))
    screen_y = min(shot.screen_height - 1, int(y / span_y * shot.screen_height))
    return shot.left + screen_x, shot.top + screen_y


def thumbnail_difference(before: bytes | None, after: bytes, size: tuple[int, int] = (64, 36)) -> float:
    """How much of the screen changed between two screenshots, from 0 (nothing) to 1 (all of it).

    Compared on small grey copies, so a moving cursor or a clock does not count
    for much. With no earlier screenshot, everything has changed.
    """
    if before is None:
        return 1.0
    from PIL import Image  # type: ignore

    def grey(data: bytes) -> list[int]:
        with Image.open(io.BytesIO(data)) as picture:
            return list(picture.convert("L").resize(size).getdata())

    first, second = grey(before), grey(after)
    changed = sum(1 for a, b in zip(first, second) if abs(a - b) > 24)
    return changed / len(first)
