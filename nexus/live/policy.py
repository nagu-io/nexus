"""What NEXUS Live may do on its own, what needs the user's go-ahead, and what it never does."""

from __future__ import annotations

from dataclasses import dataclass

from nexus.live.model import POINTER_ACTIONS, QUIET_ACTIONS, Decision
from nexus.live.settings import LiveSettings, blocked_by, window_matches
from nexus.live.voice import hotkey_keys

ALLOW = "allow"
CONFIRM = "confirm"
REFUSE = "refuse"

# Keys that open system menus or run things: the Windows key (Run box, Start
# search, settings), Ctrl+Esc (Start, like the Windows key), and the secure
# attention and Task Manager combinations.
SYSTEM_KEYS = frozenset({"win"})
SYSTEM_COMBOS = (
    frozenset({"ctrl", "alt", "delete"}),
    frozenset({"ctrl", "shift", "esc"}),
    frozenset({"ctrl", "esc"}),
)


@dataclass(frozen=True)
class Verdict:
    outcome: str
    reason: str = ""


def lands_at_a_point(decision: Decision) -> bool:
    """Whether the step goes to whatever window is under a point (a click, or a scroll) rather than the one in front."""
    return decision.action in POINTER_ACTIONS or decision.action == "scroll"


def own_hotkeys(settings: LiveSettings) -> tuple[frozenset[str], ...]:
    """NEXUS Live's own hotkeys (talk, stop, yes, no) as sets of key names."""
    combos = []
    for hotkey in (settings.hotkey_talk, settings.hotkey_stop, settings.hotkey_yes, settings.hotkey_no):
        modifiers, last = hotkey_keys(hotkey)
        if last:
            combos.append(modifiers | {last})
    return tuple(combos)


class Policy:
    def __init__(self, settings: LiveSettings):
        self.settings = settings
        self.own_hotkeys = own_hotkeys(settings)

    def check(self, decision: Decision, window_title: str | None, point_title: str | None = None) -> Verdict:
        """Whether ``decision`` may run, with ``window_title`` in front (None: unknown).

        A click or a scroll lands in the window under its point, which need not
        be the one in front (a window beside it, a second window of a blocked
        app): ``point_title`` is the title of that window (None: unknown), and it
        is held to the same rules.
        """
        if decision.action in QUIET_ACTIONS:
            return Verdict(ALLOW)

        refusal = self.key_refusal(decision)
        if refusal:
            return Verdict(REFUSE, refusal)

        verdicts = [self._judge(window_title, "in front")]
        if lands_at_a_point(decision):
            verdicts.append(self._judge(point_title, "under that point"))
        for verdict in verdicts:
            if verdict is not None and verdict.outcome == REFUSE:
                return verdict
        for verdict in verdicts:
            if verdict is not None:
                return verdict

        if self.settings.confirm == "risky" and lands_at_a_point(decision):
            return Verdict(ALLOW)
        return Verdict(CONFIRM)

    def key_refusal(self, decision: Decision) -> str:
        """Why NEXUS Live never presses ``decision``'s keys, in any window; "" when it may."""
        keys = set(decision.keys)
        if keys & SYSTEM_KEYS:
            return "I don't use the Windows key: it opens the Run box and system menus."
        if any(combo <= keys for combo in SYSTEM_COMBOS):
            return "I don't press system key combinations like that one."
        if any(combo <= keys for combo in self.own_hotkeys):
            # It would answer my own question, stop me, or start recording whatever is said nearby.
            return "That is one of my own hotkeys, so I don't press it."
        return ""

    def _judge(self, title: str | None, where: str) -> Verdict | None:
        """A refusal, or a question, for a window that rules a step out or cannot be checked; None when it is fine."""
        if title is None:
            if self.settings.allowed_windows:
                return Verdict(REFUSE, f"I can't tell which window is {where}, so I can't check it is one I'm allowed to use.")
            return Verdict(CONFIRM, f"I can't tell which window is {where}.")
        blocked = blocked_by(title, self.settings)
        if blocked:
            return Verdict(
                REFUSE,
                f"The window {where} looks like {blocked!r}, which I never act in. Please do this part yourself.",
            )
        if self.settings.allowed_windows and not window_matches(title, self.settings.allowed_windows):
            return Verdict(REFUSE, f"{title!r} is not one of the windows I'm allowed to use.")
        return None
