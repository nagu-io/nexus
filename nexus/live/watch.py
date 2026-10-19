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
        self._looking = threading.Lock()  # held for the whole of each look
        self._last_image: bytes | None = None
        self._last_said = ""
        self._last_problem = ""

    def look_once(self) -> str:
        """One look: says and returns what needs attention, or returns an empty string."""
        shot = self.capture()
        changed = self.difference(self._last_image, shot.image)
        self._last_image = shot.image
        if changed < self.change:
            return ""
        try:
            said = self.observer.observe(shot).strip()
        except ModelUnavailable as error:
            self._problem(str(error))
            return ""
        if not said or said == self._last_said:
            return ""
        self._last_said = said
        self.speaker.say(said)
        return said

    def _problem(self, message: str) -> None:
        """Say why watch mode could not look, once for each new reason; it keeps trying."""
        if message != self._last_problem:
            self._last_problem = message
            self.speaker.say(f"Watch mode couldn't look at the screen ({message}). I'll keep trying.")

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            with self._looking:
                if self.paused.is_set():
                    continue
                try:
                    self.look_once()
                except Exception as error:  # one failed look (out of memory, a screenshot error) never ends watching
                    self._problem(f"{type(error).__name__}: {error}")

    def pause(self) -> None:
        """Stop looking, and wait for a look under way to finish.

        A request is about to use the model: it must not take its screenshot
        while a look is still holding the model, then wait for it and act on
        a screen that has moved on.
        """
        self.paused.set()
        with self._looking:
            pass

    def resume(self) -> None:
        self.paused.clear()

    def stop(self) -> None:
        self._stop.set()
