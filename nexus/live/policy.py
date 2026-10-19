"""What NEXUS Live may do on its own, what needs the user's go-ahead, and what it never does."""

from __future__ import annotations

from dataclasses import dataclass

from nexus.live.model import POINTER_ACTIONS, QUIET_ACTIONS, Decision
from nexus.live.settings import LiveSettings, window_matches

ALLOW = "allow"
CONFIRM = "confirm"
REFUSE = "refuse"

# Keys that open system menus or run things: the Windows key (Run box, Start
# search, settings), and the secure attention and Task Manager combinations.
SYSTEM_KEYS = frozenset({"win"})
SYSTEM_COMBOS = (
    frozenset({"ctrl", "alt", "delete"}),
    frozenset({"ctrl", "shift", "esc"}),
)


@dataclass(frozen=True)
class Verdict:
    outcome: str
    reason: str = ""


class Policy:
    def __init__(self, settings: LiveSettings):
        self.settings = settings

    def check(self, decision: Decision, window_title: str | None) -> Verdict:
        """Whether ``decision`` may run in the window titled ``window_title`` (None: unknown)."""
        if decision.action in QUIET_ACTIONS:
            return Verdict(ALLOW)

        keys = set(decision.keys)
        if keys & SYSTEM_KEYS:
            return Verdict(REFUSE, "I don't use the Windows key: it opens the Run box and system menus.")
        if any(combo <= keys for combo in SYSTEM_COMBOS):
            return Verdict(REFUSE, "I don't press system key combinations like that one.")

        if window_title is None:
            if self.settings.allowed_windows:
                return Verdict(REFUSE, "I can't tell which window is in front, so I can't check it is one I'm allowed to use.")
            return Verdict(CONFIRM, "I can't tell which window is in front.")

        blocked = window_matches(window_title, self.settings.blocked_windows)
        if blocked:
            return Verdict(
                REFUSE,
                f"The window in front looks like {blocked!r}, which I never act in. Please do this part yourself.",
            )
        if self.settings.allowed_windows and not window_matches(window_title, self.settings.allowed_windows):
            return Verdict(REFUSE, f"{window_title!r} is not one of the windows I'm allowed to use.")

        if self.settings.confirm == "risky" and (decision.action in POINTER_ACTIONS or decision.action == "scroll"):
            return Verdict(ALLOW)
        return Verdict(CONFIRM)
