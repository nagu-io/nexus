"""Watch mode: NEXUS Live keeps an eye on the screen and speaks up when something needs you.

It only looks and talks. It is never given the hands (no Executor), so watch
mode cannot click or type, whatever the model says. It looks closer only when
enough of the screen has changed, so a still screen costs nothing.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Protocol

from nexus.live.model import ModelUnavailable
from nexus.live.screen import Shot, thumbnail_difference


class Observer(Protocol):
    def observe(self, shot: Shot) -> str: ...


class ScreenWatcher:
    def __init__(
        self,
        observer: Observer,
        speaker: Any,
        capture: Callable[[], Shot],
        interval: float = 10.0,
        change: float = 0.06,
        difference: Callable[[bytes | None, bytes], float] = thumbnail_difference,
    ):
        self.observer = observer
        self.speaker = speaker
        self.capture = capture
        self.interval = interval
        self.change = change
        self.difference = difference
        self.paused = threading.Event()  # set while a request is being handled
        self._stop = threading.Event()
        self._last_image: bytes | None = None
        self._last_said = ""

    def look_once(self) -> str:
        """One look: says and returns what needs attention, or returns an empty string."""
        shot = self.capture()
        changed = self.difference(self._last_image, shot.image)
        self._last_image = shot.image
        if changed < self.change:
            return ""
        try:
            said = self.observer.observe(shot).strip()
        except ModelUnavailable:
            return ""
        if not said or said == self._last_said:
            return ""
        self._last_said = said
        self.speaker.say(said)
        return said

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            if not self.paused.is_set():
                self.look_once()

    def stop(self) -> None:
        self._stop.set()
