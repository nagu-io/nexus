"""The loop: look at the screen, pick a step, check it, ask, act, and look again."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from nexus.live.model import POINTER_ACTIONS, Decision, DecisionError, ModelUnavailable
from nexus.live.policy import CONFIRM, REFUSE, Policy
from nexus.live.screen import OutsideScreen, Shot, to_screen
from nexus.live.settings import LiveSettings

# Outcomes of one request.
FINISHED = "finished"  # the model said it was done, or answered
REFUSED = "refused"  # the policy would not let a step run
DECLINED = "declined"  # the user said no (or did not answer)
STOPPED = "stopped"  # the stop hotkey
GAVE_UP = "gave_up"  # the model could not be used, or kept giving answers that could not be read
OUT_OF_STEPS = "out_of_steps"


class Vision(Protocol):
    def decide(self, goal: str, steps: list[str], conversation: list[tuple[str, str]], shot: Shot, mode: str) -> Decision: ...


class Confirmer(Protocol):
    def ask(self, description: str) -> bool: ...


@dataclass
class Result:
    outcome: str
    said: str = ""
    steps: list[str] = field(default_factory=list)


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
    ):
        self.settings = settings
        self.vision = vision
        self.executor = executor
        self.speaker = speaker
        self.confirmer = confirmer
        self.capture = capture
        self.sleep = sleep
        self.policy = Policy(settings)
        self.stop_event = threading.Event()
        self.conversation: list[tuple[str, str]] = []

    def stop(self) -> None:
        """Stop the request under way, before its next step."""
        self.stop_event.set()
        self.speaker.stop()

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

    def handle(self, goal: str) -> Result:
        """Work on one request until it is done, refused, declined, stopped, or out of steps."""
        goal = goal.strip()
        self.stop_event.clear()
        steps: list[str] = []
        if not goal:
            return Result(FINISHED)
        for _ in range(self.settings.max_steps):
            if self.stop_event.is_set():
                return Result(STOPPED, steps=steps)
            shot = self.capture()
            title = self.executor.foreground_title()
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
            if decision.say:
                self.speaker.say(decision.say)
            if decision.action == "wait":
                self.sleep(decision.seconds)
                steps.append(decision.describe())
                continue

            verdict = self.policy.check(decision, title)
            if verdict.outcome == REFUSE:
                self.speaker.say(verdict.reason)
                self._remember(goal, verdict.reason)
                return Result(REFUSED, said=verdict.reason, steps=steps)

            point = None
            if decision.action in POINTER_ACTIONS or (decision.action == "scroll" and decision.x is not None):
                try:
                    point = self.executor.mouse_point(to_screen(decision.x, decision.y, shot, self.settings.coordinates), shot)
                except OutsideScreen as error:
                    steps.append(f"(that position was off the screen: {error}; point at something on it)")
                    continue

            if verdict.outcome == CONFIRM:
                if point is not None:
                    self.executor.point_at(point)
                approved = self.confirmer.ask(decision.describe())
                if self.stop_event.is_set():
                    return Result(STOPPED, steps=steps)
                if not approved:
                    self._remember(goal, "The user said no.")
                    return Result(DECLINED, steps=steps)

            # The step was checked against the window in front when the screenshot
            # was taken. If another one has come forward since (a pop-up, the user
            # switching), the step could land in it: look again instead.
            if self.executor.foreground_title() != title:
                steps.append(f"(not done: the window in front changed before '{decision.describe()}')")
                continue

            done = self.executor.run(decision, point)
            steps.append(done)
            self.sleep(self.settings.settle_seconds)
        message = f"I stopped after {self.settings.max_steps} steps. Ask again to carry on."
        self.speaker.say(message)
        self._remember(goal, message)
        return Result(OUT_OF_STEPS, said=message, steps=steps)
