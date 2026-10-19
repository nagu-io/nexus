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


@dataclass(frozen=True)
class Window:
    """A window on the screen, as teach mode's masking sees it."""

    title: str | None
    box: tuple[int, int, int, int]  # left, top, right, bottom, on the screen
    process: int | None = None  # the program it belongs to, where that can be read
    front: bool = False  # the window in front


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


def primary_monitor(monitors: list[dict]) -> dict:
    """The main monitor from mss's list (the first entry is all of them together).

    The one marked primary (mss 10.2 and later), else the one at (0, 0), which
    on Windows is always the primary, else the first. The order mss lists them
    in is the system's, and the first is not always the main one.
    """
    screens = monitors[1:]
    for monitor in screens:
        if monitor.get("is_primary"):
            return monitor
    for monitor in screens:
        if monitor.get("left") == 0 and monitor.get("top") == 0:
            return monitor
    return screens[0]


class Frame:
    """A screenshot kept as a picture, made into a PNG only when something needs the PNG.

    Teach mode takes several a second and keeps only the few that go with a
    step: encoding every one would cost more than taking it. It has the same
    fields as a Shot.
    """

    def __init__(self, picture, screen_width: int, screen_height: int, left: int = 0, top: int = 0):
        self.picture = picture
        self.width, self.height = picture.size
        self.screen_width, self.screen_height = screen_width, screen_height
        self.left, self.top = left, top
        self._png: bytes | None = None

    @property
    def image(self) -> bytes:
        if self._png is None:
            buffer = io.BytesIO()
            self.picture.save(buffer, format="PNG")
            self._png = buffer.getvalue()
        return self._png


def _grab():
    """The main monitor as a picture, and the monitor it was taken from."""
    import mss  # type: ignore
    from PIL import Image  # type: ignore

    _make_dpi_aware()
    with mss.mss() as grabber:
        monitor = primary_monitor(grabber.monitors)
        raw = grabber.grab(monitor)
        image = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    return image, monitor


def capture(max_width: int = 1280, optimize: bool = True) -> Shot:
    """A screenshot of the main monitor, shrunk to at most ``max_width`` pixels wide.

    ``optimize=False`` saves the PNG quickly at a slightly bigger size.
    """
    image, monitor = _grab()
    return shrink(image, max_width, monitor["width"], monitor["height"], monitor["left"], monitor["top"], optimize)


def grab(max_width: int = 1280) -> Frame:
    """A screenshot of the main monitor, shrunk like ``capture``'s, but not yet a PNG."""
    image, monitor = _grab()
    return Frame(_shrunk(image, max_width), monitor["width"], monitor["height"], monitor["left"], monitor["top"])


def _shrunk(image, max_width: int):
    from PIL import Image  # type: ignore

    width, height = image.size
    if width > max_width:
        height = max(1, round(height * max_width / width))
        width = max_width
        image = image.resize((width, height), Image.LANCZOS)
    return image


def shrink(
    image, max_width: int, screen_width: int, screen_height: int, left: int = 0, top: int = 0, optimize: bool = True
) -> Shot:
    """The Shot for an image of a screen, shrunk to ``max_width`` keeping its shape."""
    image = _shrunk(image, max_width)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=optimize)
    return Shot(buffer.getvalue(), image.size[0], image.size[1], screen_width, screen_height, left, top)


def cover_windows(frame: Frame, layout: list[Window], hidden) -> bool:
    """Black out every part of ``frame`` where a window that ``hidden(title)`` picks can be seen.

    ``layout`` is the visible windows, the top one first. A hidden window is
    covered whole, and so is every window of the same program but the one in
    front (its pop-ups, tooltips and dialogs, titled or not). Only the window in
    front, laid over a hidden one, uncovers what it hides: any other window may
    be see-through (an overlay, a screen recorder's frame), so it is never
    trusted to hide anything. True if anything was covered.
    """
    from PIL import Image, ImageDraw  # type: ignore

    hidden_programs = {window.process for window in layout if window.process is not None and hidden(window.title)}
    scale_x, scale_y = frame.width / max(1, frame.screen_width), frame.height / max(1, frame.screen_height)
    mask = Image.new("1", (frame.width, frame.height), 0)
    draw = ImageDraw.Draw(mask)
    for window in reversed(layout):  # the bottom window first, each laid over the ones below it
        left, top, right, bottom = window.box
        x0, y0 = (left - frame.left) * scale_x, (top - frame.top) * scale_y
        x1, y1 = (right - frame.left) * scale_x, (bottom - frame.top) * scale_y
        if hidden(window.title) or (window.process in hidden_programs and not window.front):
            box = (math.floor(x0), math.floor(y0), math.ceil(x1) - 1, math.ceil(y1) - 1)  # every pixel it touches
            fill = 1
        elif window.front:
            box = (math.ceil(x0), math.ceil(y0), math.floor(x1) - 1, math.floor(y1) - 1)  # only pixels it covers whole
            fill = 0
        else:
            continue
        if box[2] >= box[0] and box[3] >= box[1]:
            draw.rectangle(box, fill=fill)
    if mask.getbbox() is None:
        return False
    frame.picture = frame.picture.copy()
    frame.picture.paste((0, 0, 0), None, mask)
    frame._png = None
    return True


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
            return list(picture.convert("L").resize(size).tobytes())  # one byte per pixel

    first, second = grey(before), grey(after)
    changed = sum(1 for a, b in zip(first, second) if abs(a - b) > 24)
    return changed / len(first)


def changed_near(before: bytes, after: bytes, around: tuple[float, float] | None, span: float = 0.16) -> float:
    """How much changed between two screenshots of the same screen, from 0 to 1, near a point.

    ``around`` is the point as shares of the width and height (0 to 1); the
    part compared is a box ``span`` of the screen across, centred on it. With
    no point the whole screen is compared. What is under a click is what
    matters: a video playing elsewhere on the screen does not count.
    """
    from PIL import Image  # type: ignore

    with Image.open(io.BytesIO(before)) as first, Image.open(io.BytesIO(after)) as second:
        if first.size != second.size:
            return 1.0
        first_grey, second_grey = first.convert("L"), second.convert("L")
    if around is not None:
        width, height = first_grey.size
        x, y = around[0] * width, around[1] * height
        half_width, half_height = max(1.0, span * width / 2), max(1.0, span * height / 2)
        box = (
            int(max(0, x - half_width)),
            int(max(0, y - half_height)),
            int(min(width, x + half_width)),
            int(min(height, y + half_height)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            return 0.0
        first_grey, second_grey = first_grey.crop(box), second_grey.crop(box)
    size = (32, 32) if around is not None else (64, 36)
    first_pixels = first_grey.resize(size).tobytes()
    second_pixels = second_grey.resize(size).tobytes()
    return sum(1 for a, b in zip(first_pixels, second_pixels) if abs(a - b) > 24) / len(first_pixels)


def share_of(x: float, y: float, shot: Shot, mode: str) -> tuple[float, float]:
    """A position the model gave, as shares (0 to 1) of the screenshot's width and height."""
    if mode == "normalized":
        return x / NORMALIZED_SPAN, y / NORMALIZED_SPAN
    return x / max(1, shot.width), y / max(1, shot.height)
