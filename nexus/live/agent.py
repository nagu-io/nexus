"""The loop: look at the screen, pick a step, check it, ask, act, and look again."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from nexus.live.executor import Interrupted
from nexus.live.model import Decision, DecisionError, ModelUnavailable
from nexus.live.policy import CONFIRM, REFUSE, Policy, Verdict, lands_at_a_point
from nexus.live.screen import OutsideScreen, Shot, changed_near, share_of, to_screen
from nexus.live.settings import LiveSettings

# Outcomes of one request.
FINISHED = "finished"  # the model said it was done, or answered
REFUSED = "refused"  # the policy would not let a step run
DECLINED = "declined"  # the user said no (or did not answer)
STOPPED = "stopped"  # the stop hotkey, a key held down, or the pointer thrown into a corner
GAVE_UP = "gave_up"  # the model could not be used, kept giving answers that could not be read, or something failed
OUT_OF_STEPS = "out_of_steps"
NO_WINDOW = "no_window"  # NEXUS Live's own window stayed in front, so there was nothing to work in

# How much may change between the model's look and the step, before the step is
# dropped and the screen looked at again: near a click (what is under it), or
# anywhere, for keys and typing.
CHANGED_NEAR_POINT = 0.3
CHANGED_ANYWHERE = 0.25
WINDOW_POLL = 0.25  # seconds between looks at which window is in front, while waiting for one
WINDOW_WAITS = 3  # times one request waits for a window before giving up


class Vision(Protocol):
    def decide(self, goal: str, steps: list[str], conversation: list[tuple[str, str]], shot: Shot, mode: str) -> Decision: ...


class Confirmer(Protocol):
    def ask(self, description: str) -> bool: ...


@dataclass
class Result:
    outcome: str
    said: str = ""
    steps: list[str] = field(default_factory=list)


def _fail_safe(error: BaseException) -> bool:
    # pyautogui's FailSafeException, without importing pyautogui to name it.
    return type(error).__name__ == "FailSafeException"


class LiveAgent:
    def __init__(
        self,
        settings: LiveSettings,
        vision: Vision,
        executor: Any,
        speaker: Any,
        confirmer: Confirmer,
        capture: Callable[[], Shot],
        sleep: Callable[[float], None] = time.sleep,
        difference: Callable[[bytes, bytes, tuple[float, float] | None], float] = changed_near,
    ):
        self.settings = settings
        self.vision = vision
        self.executor = executor
        self.speaker = speaker
        self.confirmer = confirmer
        self.capture = capture
        self.sleep = sleep
        self.difference = difference
        self.policy = Policy(settings)
        self.stop_event = threading.Event()
        self.conversation: list[tuple[str, str]] = []
        # Titles NEXUS Live's own console window has had. A request typed there
        # starts with that window in front: NEXUS waits for the user to pick the
        # window to work in, and never types or clicks into its own console.
        self.own_windows: set[str] = set()

    def stop(self) -> None:
        """Stop the request under way, before its next step (and typing, at the next character)."""
        self.stop_event.set()
        self.speaker.stop()

    def new_request(self) -> None:
        """A request is starting: a stop pressed from now on cancels it."""
        self.stop_event.clear()

    def note_own_window(self, title: str | None) -> None:
        if title:
            self.own_windows.add(title)

    def _remember(self, goal: str, said: str) -> None:
        self.conversation.append(("user", goal))
        if said:
            self.conversation.append(("you", said))
        self.conversation = self.conversation[-2 * self.settings.history_turns :]

    def _decide(self, goal: str, steps: list[str], shot: Shot) -> Decision | None:
        """The model's next step; one more try when its answer cannot be read."""
        for attempt in range(2):
            try:
                return self.vision.decide(goal, steps, list(self.conversation), shot, self.settings.coordinates)
            except DecisionError as error:
                if attempt == 1:
                    self.speaker.say(f"I couldn't read the model's answer ({error}), so I stopped.")
                    return None
                steps = steps + [f"(your last answer could not be used: {error}; answer again)"]
        return None

    def handle(self, goal: str, fresh: bool = True) -> Result:
        """Work on one request until it is done, refused, declined, stopped, or out of steps.

        ``fresh=False`` keeps a stop pressed since ``new_request`` (while the
        request was being spoken, say): such a request is not carried out. Any
        failure ends the request, never NEXUS Live: the next one can still come.
        """
        steps: list[str] = []
        try:
            return self._handle(goal.strip(), fresh, steps)
        except Exception as error:
            if _fail_safe(error):
                message, outcome = "Stopped: the pointer went into a corner of the screen.", STOPPED
            else:
                message, outcome = f"Something went wrong ({type(error).__name__}: {error}), so I stopped.", GAVE_UP
            self.speaker.say(message)
            return Result(outcome, said=message, steps=steps)

    def _handle(self, goal: str, fresh: bool, steps: list[str]) -> Result:
        if fresh:
            self.stop_event.clear()
        if not goal:
            return Result(FINISHED)
        taken = waits = 0
        while taken < self.settings.max_steps:
            if self.stop_event.is_set():
                return Result(STOPPED, steps=steps)
            shot = self.capture()
            title = self.executor.foreground_title()
            if title in self.own_windows:
                waits += 1
                if waits <= WINDOW_WAITS and self._wait_for_a_window():
                    continue
                if self.stop_event.is_set():
                    return Result(STOPPED, steps=steps)
                message = "No other window came to the front, so I stopped. Click the window to use, then ask again."
                self.speaker.say(message)
                self._remember(goal, message)
                return Result(NO_WINDOW, said=message, steps=steps)
            taken += 1
            try:
                decision = self._decide(goal, steps, shot)
            except ModelUnavailable as error:
                self.speaker.say(str(error))
                return Result(GAVE_UP, steps=steps)
            if decision is None:
                return Result(GAVE_UP, steps=steps)
            if self.stop_event.is_set():
                return Result(STOPPED, steps=steps)

            if decision.action in ("answer", "done"):
                self.speaker.say(decision.say)
                self._remember(goal, decision.say)
                return Result(FINISHED, said=decision.say, steps=steps)
            if decision.action == "wait":
                if decision.say:
                    self.speaker.say(decision.say)
                self.sleep(decision.seconds)
                steps.append(decision.describe())
                continue

            point = target = None
            if lands_at_a_point(decision):
                if decision.x is None:  # a scroll with no position scrolls where the pointer is
                    point = self.executor.pointer()
                else:
                    try:
                        point = self.executor.mouse_point(to_screen(decision.x, decision.y, shot, self.settings.coordinates), shot)
                    except OutsideScreen as error:
                        steps.append(f"(that position was off the screen: {error}; point at something on it)")
                        continue
                target = self.executor.window_at(point)

            if target is not None and target in self.own_windows:
                verdict = Verdict(REFUSE, "That point is in my own window, which I never click or type in.")
            else:
                verdict = self.policy.check(decision, title, target)
            if verdict.outcome == REFUSE:
                self.speaker.say(verdict.reason)
                self._remember(goal, verdict.reason)
                return Result(REFUSED, said=verdict.reason, steps=steps)

            if verdict.outcome == CONFIRM:
                if point is not None and decision.x is not None:
                    self.executor.point_at(point)
                # What the model says about its step stays unsaid while the user decides: text on
                # the screen can steer it, and the question is built only from what NEXUS checked.
                approved = self.confirmer.ask(self._question(decision, title, target, verdict))
                if self.stop_event.is_set():
                    return Result(STOPPED, steps=steps)
                if not approved:
                    self._remember(goal, "The user said no.")
                    return Result(DECLINED, steps=steps)
            elif decision.say:
                self.speaker.say(decision.say)  # nothing to approve: what it is doing, for the user to hear

            if not self._keys_released():
                if self.stop_event.is_set():
                    return Result(STOPPED, steps=steps)
                message = "Keys stayed held down, so I didn't do it: with them, it would not be the step you agreed to."
                self.speaker.say(message)
                self._remember(goal, message)
                return Result(STOPPED, said=message, steps=steps)
            if self.stop_event.is_set():
                return Result(STOPPED, steps=steps)

            skipped = self._moved_on(decision, title, point, target, shot)
            if skipped:
                steps.append(f"(not done: {skipped} before '{decision.describe()}')")
                continue

            try:
                done = self.executor.run(decision, point, self.stop_event)
            except Interrupted as error:
                if self.stop_event.is_set():
                    return Result(STOPPED, steps=steps + [f"(stopped part-way: {decision.describe()})"])
                message = f"I stopped typing part-way: {error}. Some of it may be typed already."
                self.speaker.say(message)
                self._remember(goal, message)
                return Result(STOPPED, said=message, steps=steps + [f"(stopped part-way: {decision.describe()})"])
            steps.append(done)
            self.sleep(self.settings.settle_seconds)
        message = f"I stopped after {self.settings.max_steps} steps. Ask again to carry on."
        self.speaker.say(message)
        self._remember(goal, message)
        return Result(OUT_OF_STEPS, said=message, steps=steps)

    def _question(self, decision: Decision, title: str | None, target: str | None, verdict: Verdict) -> str:
        """The step as the user is asked about it, from facts NEXUS read itself: what, where, and in which window."""
        window = target if lands_at_a_point(decision) else title
        question = decision.confirmation() + (f" in {window!r}" if window else "")
        if verdict.reason:
            question += f" ({verdict.reason.rstrip('.')})"
        return question

    def _wait_for_a_window(self) -> bool:
        """Wait for the user to bring the window to work in to the front: True once one is there."""
        self.speaker.say("Click the window you want me to work in, and I'll start there.")
        for _ in range(max(1, int(self.settings.confirm_timeout / WINDOW_POLL))):
            if self.stop_event.is_set():
                return False
            self.sleep(WINDOW_POLL)
            if self.executor.foreground_title() not in self.own_windows:
                self.sleep(self.settings.settle_seconds)  # let it come all the way forward
                return True
        return False

    def _keys_released(self) -> bool:
        """Whether the modifier keys are up, waiting a little for them (the yes hotkey is still held, say)."""
        if self.executor.wait_for_release(2.0):
            return True
        self.speaker.say("Let go of the keys and I'll go on.")
        return self.executor.wait_for_release(10.0)

    def _moved_on(self, decision: Decision, title: str | None, point: Any, target: str | None, shot: Shot) -> str:
        """Why the step can no longer be trusted to land where the model meant it to, or "".

        The model chose it on a screenshot that is now seconds old. A different
        window in front, a different window under the point, or a changed screen
        under it (a dialog that closed, a list that scrolled) means looking again.
        """
        if self.executor.foreground_title() != title:
            return "the window in front changed"
        if point is not None and self.executor.window_at(point) != target:
            return "the window under that point changed"
        now = self.capture()
        if now.image == shot.image:
            return ""
        if lands_at_a_point(decision) and decision.x is not None:
            around: tuple[float, float] | None = share_of(decision.x, decision.y, shot, self.settings.coordinates)
            limit = CHANGED_NEAR_POINT
        else:
            around, limit = None, CHANGED_ANYWHERE
        try:
            changed = self.difference(shot.image, now.image, around)
        except Exception:  # screenshots that cannot be compared say nothing either way
            return ""
        return "the screen changed" if changed > limit else ""
