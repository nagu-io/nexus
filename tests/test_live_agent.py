"""Tests for nexus.live.agent: the loop that looks, decides, checks, asks, acts and looks again."""

import unittest
from types import SimpleNamespace

from nexus.live.agent import DECLINED, FINISHED, GAVE_UP, NO_WINDOW, OUT_OF_STEPS, REFUSED, STOPPED, LiveAgent, Result
from nexus.live.executor import Executor, Interrupted
from nexus.live.model import Decision, DecisionError, ModelUnavailable, VisionModel
from nexus.live.screen import Shot
from nexus.live.settings import LiveSettings

# The model sees a 1000 by 500 copy of a 2000 by 1000 monitor.
SHOT = Shot(image=b"screenshot", width=1000, height=500, screen_width=2000, screen_height=1000)
NOTEPAD = "Untitled - Notepad"
BANK = "Online Banking - Google Chrome"
# The middle of the screenshot: (1000, 500) on the monitor in normalized mode, then (1001, 501) for the mouse.
CLICK = Decision("click", x=500.0, y=500.0)
DONE = Decision("done", say="All done.")
BANK_REFUSAL = "The window in front looks like 'banking', which I never act in. Please do this part yourself."


class FakeVision:
    """Hands out the answers it was given: a Decision, an exception to raise, or a function to call first."""

    def __init__(self, rig, answers):
        self.rig = rig
        self.answers = list(answers)
        self.calls = []

    def decide(self, goal, steps, conversation, shot, mode):
        self.calls.append(SimpleNamespace(goal=goal, steps=list(steps), conversation=list(conversation), shot=shot, mode=mode))
        self.rig.log.append("decide")
        answer = self.answers.pop(0)
        if callable(answer):
            answer = answer()
        if isinstance(answer, BaseException):
            raise answer
        return answer


def in_turn(values):
    """A list read in turn, the last one repeating (one value on its own is a list of one)."""
    return list(values) if isinstance(values, list) else [values]


class FakeExecutor:
    """The hands, recorded.

    ``titles`` is the window in front and ``targets`` the window under a point:
    each one title, or a list read in turn (the last one repeats). ``releases``
    are the answers to "are the modifier keys up?", and ``failure`` is raised
    by run(). ``windows`` keeps the window each run() was told the step is for.
    """

    def __init__(self, rig, titles, targets, releases=True, failure=None):
        self.rig = rig
        self.titles = in_turn(titles)
        self.targets = in_turn(targets)
        self.releases = in_turn(releases)
        self.failure = failure
        self.runs = []
        self.windows = []

    @staticmethod
    def _next(values):
        return values.pop(0) if len(values) > 1 else values[0]

    def foreground_title(self):
        self.rig.log.append("title")
        return self._next(self.titles)

    def window_at(self, point):
        self.rig.log.append(("window_at", point))
        return self._next(self.targets)

    def pointer(self):
        self.rig.log.append("pointer")
        return (7, 8)

    def mouse_point(self, point, shot):
        self.rig.log.append(("mouse_point", point, shot))
        return (point[0] + 1, point[1] + 1)  # changed, so a test can tell the agent used it

    def point_at(self, point):
        self.rig.log.append(("point_at", point))

    def wait_for_release(self, timeout):
        self.rig.log.append(("release", timeout))
        return self._next(self.releases)

    def run(self, decision, point, stop=None, window="(not given)"):
        self.rig.log.append(("run", decision, point))
        self.runs.append((decision, point))
        self.windows.append(window)
        if self.failure is not None:
            failure = self.failure() if callable(self.failure) else self.failure
            raise failure
        return f"did {decision.describe()}"


class FakeSpeaker:
    def __init__(self, rig):
        self.rig = rig
        self.said = []
        self.stops = 0

    def say(self, text):
        self.rig.log.append(("say", text))
        self.said.append(text)

    def stop(self):
        self.stops += 1


class FakeConfirmer:
    """Answers with the given yes (True) and no (False) answers, or calls a function for the answer."""

    def __init__(self, rig, answers):
        self.rig = rig
        self.answers = list(answers)
        self.asked = []

    def ask(self, description):
        self.rig.log.append(("ask", description))
        self.asked.append(description)
        answer = self.answers.pop(0)
        return answer() if callable(answer) else answer


class Rig:
    """A LiveAgent wired to fakes that write everything they are asked to do into one log.

    ``targets`` (the window under a click) defaults to the window in front;
    ``shots`` are the screenshots handed out in turn (the last repeats), and
    ``changed`` how much the screen differs when two are compared (a list is
    read in turn, the last one repeating).
    """

    def __init__(self, *answers, titles=NOTEPAD, targets="same", confirms=(), releases=True, failure=None,
                 shots=None, changed=0.0, **settings):
        self.log = []
        self.settings = LiveSettings(**settings)
        self.vision = FakeVision(self, answers)
        self.executor = FakeExecutor(self, titles, titles if targets == "same" else targets, releases, failure)
        self.speaker = FakeSpeaker(self)
        self.confirmer = FakeConfirmer(self, confirms)
        self.slept = []
        self.on_sleep = None
        self.shots = in_turn(shots if shots is not None else SHOT)
        self.changed = in_turn(changed)
        self.compared = []
        self.agent = LiveAgent(
            self.settings, self.vision, self.executor, self.speaker, self.confirmer, capture=self.capture, sleep=self.sleep,
            difference=self.difference,
        )

    def capture(self):
        self.log.append("capture")
        return self.shots.pop(0) if len(self.shots) > 1 else self.shots[0]

    def difference(self, before, after, around):
        self.compared.append((before, after, around))
        return self.changed.pop(0) if len(self.changed) > 1 else self.changed[0]

    def sleep(self, seconds):
        self.log.append(("sleep", seconds))
        self.slept.append(seconds)
        if self.on_sleep:
            self.on_sleep()

    @property
    def captures(self):
        return self.log.count("capture")

    def events(self, kind):
        return [event for event in self.log if isinstance(event, tuple) and event[0] == kind]

    def then_stop(self, answer):
        """An answer that presses the stop hotkey before it is given."""

        def give():
            self.agent.stop()
            return answer

        return give


class ScriptedModel(VisionModel):
    """A real VisionModel (prompts and parsing included) whose raw answers are written in advance."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.texts = []

    def _chat(self, system, text, image, schema):
        self.texts.append(text)
        return self.answers.pop(0)


class ResultTests(unittest.TestCase):
    def test_outcome_names(self):
        self.assertEqual(
            (FINISHED, REFUSED, DECLINED, STOPPED, GAVE_UP, OUT_OF_STEPS, NO_WINDOW),
            ("finished", "refused", "declined", "stopped", "gave_up", "out_of_steps", "no_window"),
        )

    def test_defaults(self):
        first, second = Result(FINISHED), Result(FINISHED)
        self.assertEqual((first.said, first.steps), ("", []))
        self.assertIsNot(first.steps, second.steps)


class FinishTests(unittest.TestCase):
    def test_empty_request_does_nothing(self):
        rig = Rig()
        self.assertEqual(rig.agent.handle("   "), Result(FINISHED))
        self.assertEqual(rig.log, [])
        self.assertEqual(rig.agent.conversation, [])

    def test_answer_is_spoken_and_remembered(self):
        rig = Rig(Decision("answer", say="It is five o'clock."), titles=BANK)
        result = rig.agent.handle("  what time is it?  ")
        self.assertEqual(result, Result(FINISHED, said="It is five o'clock.", steps=[]))
        self.assertEqual(rig.vision.calls[0].goal, "what time is it?")
        self.assertEqual(rig.speaker.said, ["It is five o'clock."])
        self.assertEqual(rig.agent.conversation, [("user", "what time is it?"), ("you", "It is five o'clock.")])
        self.assertEqual((rig.executor.runs, rig.confirmer.asked), ([], []))

    def test_done_with_nothing_to_say_remembers_only_the_request(self):
        rig = Rig(Decision("done"))
        self.assertEqual(rig.agent.handle("close it"), Result(FINISHED, said="", steps=[]))
        self.assertEqual(rig.agent.conversation, [("user", "close it")])

    def test_mode_and_screenshot_go_to_the_model(self):
        rig = Rig(DONE, coordinates="pixels")
        rig.agent.handle("look")
        call = rig.vision.calls[0]
        self.assertEqual((call.mode, call.steps, call.conversation), ("pixels", [], []))
        self.assertIs(call.shot, SHOT)


QUESTION = "click at (500, 500) in 'Untitled - Notepad'"


class ActingTests(unittest.TestCase):
    def test_a_click_is_shown_asked_about_rechecked_then_done(self):
        click = Decision("click", x=500.0, y=500.0, say="Clicking Save.", thought="the Save button")
        rig = Rig(click, DONE, confirms=(True,))
        result = rig.agent.handle("save the file")
        self.assertEqual(result, Result(FINISHED, said="All done.", steps=["did click at (500, 500): the Save button"]))
        self.assertEqual(
            rig.log,
            [
                "capture",
                "title",
                "decide",
                ("mouse_point", (1000, 500), SHOT),
                ("window_at", (1001, 501)),
                ("point_at", (1001, 501)),
                ("ask", QUESTION),
                ("release", 2.0),
                "title",
                ("window_at", (1001, 501)),
                "capture",
                ("run", click, (1001, 501)),
                ("sleep", 0.8),
                "capture",
                "title",
                "decide",
                ("say", "All done."),
            ],
        )
        self.assertEqual(rig.vision.calls[1].steps, ["did click at (500, 500): the Save button"])
        self.assertEqual(rig.agent.conversation, [("user", "save the file"), ("you", "All done.")])

    def test_each_step_is_carried_out_for_the_window_it_was_approved_in(self):
        # Typing stops when another window comes forward part-way (the executor checks it before each letter).
        typing = Decision("type", text="see you at the cafe")
        rig = Rig(typing, CLICK, DONE, confirms=(True, True))
        rig.agent.handle("reply and click")
        self.assertEqual(rig.executor.windows, [NOTEPAD, NOTEPAD])
        rig = Rig(typing, DONE, titles=None, confirms=(True,))
        rig.agent.handle("reply")
        self.assertEqual(rig.executor.windows, [None])  # unknown then, so any known window later is a change

    def test_settle_pause_is_the_setting(self):
        rig = Rig(CLICK, DONE, confirms=(True,), settle_seconds=0.25)
        rig.agent.handle("click")
        self.assertEqual(rig.slept, [0.25])

    def test_risky_mode_clicks_without_asking(self):
        rig = Rig(CLICK, DONE, confirm="risky")
        result = rig.agent.handle("click the middle")
        self.assertEqual(result.steps, ["did click at (500, 500)"])
        self.assertEqual(rig.confirmer.asked, [])
        self.assertEqual(rig.events("point_at"), [])
        self.assertEqual(rig.executor.runs, [(CLICK, (1001, 501))])
        # The window in front and the one under the click are both checked again right before it.
        before_click = rig.log[: rig.log.index(("run", CLICK, (1001, 501)))]
        self.assertEqual(before_click.count("title"), 2)
        self.assertEqual(len([event for event in before_click if event[0] == "window_at"]), 2)

    def test_what_the_model_says_on_the_way_is_not_said_when_nothing_is_asked(self):
        # Text on the screen can steer the model's words; heard just before a question, they would
        # speak for the step being asked about. Only its final answer is said.
        click = Decision("click", x=500.0, y=500.0, say="This only closes the pop-up, it is safe.")
        typing = Decision("type", text="CONFIRM 50000")
        rig = Rig(click, typing, DONE, confirm="risky", confirms=(True,))
        rig.agent.handle("close the pop-up")
        self.assertEqual(rig.executor.runs, [(click, (1001, 501)), (typing, None)])
        self.assertEqual(rig.speaker.said, ["All done."])
        self.assertEqual(rig.confirmer.asked, ["type 'CONFIRM 50000' (13 characters) in 'Untitled - Notepad'"])

    def test_risky_mode_still_asks_before_typing(self):
        typing = Decision("type", text="hello")
        rig = Rig(typing, DONE, confirm="risky", confirms=(True,))
        rig.agent.handle("say hello")
        self.assertEqual(rig.confirmer.asked, ["type 'hello' (5 characters) in 'Untitled - Notepad'"])
        self.assertEqual(rig.events("point_at"), [])
        self.assertEqual(rig.events("window_at"), [])
        self.assertEqual(rig.executor.runs, [(typing, None)])

    def test_the_whole_text_is_asked_about(self):
        # Typing approved from its first 57 characters let 443 more, Enter presses included, through.
        text = "Thanks for the update, I will reply properly tomorrow morning. " * 7
        typing = Decision("type", text=text)
        rig = Rig(typing, DONE, confirms=(True,))
        rig.agent.handle("reply")
        self.assertEqual(rig.confirmer.asked, [f"type {text!r} ({len(text)} characters) in 'Untitled - Notepad'"])

    def test_pixel_positions(self):
        click = Decision("click", x=900.0, y=450.0)
        rig = Rig(click, DONE, coordinates="pixels", confirm="risky")
        rig.agent.handle("click near the corner")
        self.assertEqual(rig.events("mouse_point"), [("mouse_point", (1800, 900), SHOT)])
        self.assertEqual(rig.executor.runs, [(click, (1801, 901))])
        self.assertEqual(rig.compared, [])  # the same screenshot both times: nothing to compare

    def test_scroll_without_a_position_scrolls_where_the_pointer_is(self):
        scroll = Decision("scroll", amount=-3)
        rig = Rig(scroll, DONE, confirm="risky")
        rig.agent.handle("scroll down")
        self.assertEqual(rig.events("mouse_point"), [])
        self.assertEqual(rig.events("window_at")[0], ("window_at", (7, 8)))  # the window under the pointer is checked
        self.assertEqual(rig.executor.runs, [(scroll, (7, 8))])

    def test_scroll_without_a_position_is_asked_about_in_the_window_under_the_pointer(self):
        scroll = Decision("scroll", amount=-3)
        rig = Rig(scroll, DONE, confirms=(True,), targets="Inbox - Outlook")
        rig.agent.handle("scroll down")
        self.assertEqual(rig.confirmer.asked, ["scroll down 3 in 'Inbox - Outlook'"])
        self.assertEqual(rig.events("point_at"), [])  # the pointer is there already

    def test_scroll_at_a_position(self):
        scroll = Decision("scroll", amount=3, x=500.0, y=250.0)
        rig = Rig(scroll, DONE, confirms=(True,))
        rig.agent.handle("scroll the list up")
        self.assertEqual(rig.events("mouse_point"), [("mouse_point", (1000, 250), SHOT)])
        self.assertEqual(rig.events("point_at"), [("point_at", (1001, 251))])
        self.assertEqual(rig.confirmer.asked, ["scroll up 3 at (500, 250) in 'Untitled - Notepad'"])
        self.assertEqual(rig.executor.runs, [(scroll, (1001, 251))])

    def test_wait_sleeps_without_asking(self):
        rig = Rig(Decision("wait", seconds=2.5, say="It is safe: press Ctrl+Alt+Y when I ask."), DONE, titles=BANK)
        result = rig.agent.handle("wait for the page")
        self.assertEqual(result, Result(FINISHED, said="All done.", steps=["wait 2.5 s"]))
        self.assertEqual(rig.slept, [2.5])
        self.assertEqual(rig.speaker.said, ["All done."])  # what the model says on the way is not said
        self.assertEqual((rig.confirmer.asked, rig.executor.runs), ([], []))

    def test_off_screen_position_is_not_clicked(self):
        rig = Rig(Decision("click", x=1000.5, y=10.0), DONE)
        result = rig.agent.handle("click past the edge")
        note = "(that position was off the screen: (1000.5, 10) is outside the screenshot (0 to 1000, 0 to 1000); point at something on it)"
        self.assertEqual(result, Result(FINISHED, said="All done.", steps=[note]))
        self.assertEqual(rig.vision.calls[1].steps, [note])
        self.assertEqual((rig.confirmer.asked, rig.executor.runs, rig.events("point_at")), ([], [], []))

    def test_unknown_window_is_asked_about_even_in_risky_mode(self):
        rig = Rig(CLICK, DONE, titles=None, confirm="risky", confirms=(True,))
        rig.agent.handle("click the middle")
        self.assertEqual(rig.confirmer.asked, ["click at (500, 500) (I can't tell which window is in front)"])
        self.assertEqual(rig.executor.runs, [(CLICK, (1001, 501))])

    def test_an_unknown_window_under_the_click_is_asked_about_in_risky_mode(self):
        rig = Rig(CLICK, DONE, targets=None, confirm="risky", confirms=(True,))
        rig.agent.handle("click the middle")
        self.assertEqual(rig.confirmer.asked, ["click at (500, 500) (I can't tell which window is under that point)"])
        self.assertEqual(rig.executor.runs, [(CLICK, (1001, 501))])


class WhatIsAskedTests(unittest.TestCase):
    """The question is built from what NEXUS checked itself, never from the model's words (screen text steers those)."""

    def test_the_model_s_words_are_not_said_while_the_user_decides(self):
        click = Decision("click", x=640.0, y=720.0, thought="the Cancel button, to close this safely",
                         say="This only closes the pop-up, it is safe.")
        rig = Rig(click, DONE, titles="Confirm transfer - Mozilla Firefox", confirms=(True,))
        rig.agent.handle("close the pop-up")
        self.assertEqual(rig.confirmer.asked, ["click at (640, 720) in 'Confirm transfer - Mozilla Firefox'"])
        self.assertNotIn("This only closes the pop-up, it is safe.", rig.speaker.said)
        self.assertEqual(rig.speaker.said, ["All done."])

    def test_the_window_under_the_click_is_named(self):
        rig = Rig(CLICK, DONE, targets="Calculator", confirms=(True,))
        rig.agent.handle("click")
        self.assertEqual(rig.confirmer.asked, ["click at (500, 500) in 'Calculator'"])

    def test_keys_name_the_window_in_front(self):
        rig = Rig(Decision("hotkey", keys=("ctrl", "s")), DONE, confirms=(True,))
        rig.agent.handle("save")
        self.assertEqual(rig.confirmer.asked, ["press ctrl+s in 'Untitled - Notepad'"])

    def test_a_window_with_no_title_is_not_named(self):
        rig = Rig(Decision("press", keys=("enter",)), DONE, titles="", confirms=(True,))
        rig.agent.handle("press enter")
        self.assertEqual(rig.confirmer.asked, ["press enter"])


class RefusalTests(unittest.TestCase):
    def test_blocked_window_is_refused_before_pointing_or_asking(self):
        rig = Rig(CLICK, titles=BANK)
        result = rig.agent.handle("pay the bill")
        self.assertEqual(result, Result(REFUSED, said=BANK_REFUSAL, steps=[]))
        self.assertEqual(rig.speaker.said, [BANK_REFUSAL])
        self.assertEqual(rig.agent.conversation, [("user", "pay the bill"), ("you", BANK_REFUSAL)])
        self.assertEqual((rig.confirmer.asked, rig.executor.runs, rig.events("point_at")), ([], [], []))

    def test_a_blocked_window_under_the_click_is_refused_in_every_mode(self):
        # KeePass beside the window in front: the click would land in KeePass.
        for confirm in ("all", "risky"):
            with self.subTest(confirm=confirm):
                rig = Rig(CLICK, targets="Database.kdbx - KeePass", confirm=confirm)
                result = rig.agent.handle("click it")
                self.assertEqual(result.outcome, REFUSED)
                self.assertEqual(result.said, "The window under that point looks like 'keepass', which I never act in. Please do this part yourself.")
                self.assertEqual((rig.confirmer.asked, rig.executor.runs, rig.events("point_at")), ([], [], []))

    def test_windows_key_is_refused(self):
        rig = Rig(Decision("hotkey", keys=("win", "r")))
        result = rig.agent.handle("open the run box")
        self.assertEqual(result.outcome, REFUSED)
        self.assertEqual(result.said, "I don't use the Windows key: it opens the Run box and system menus.")
        self.assertEqual(rig.executor.runs, [])

    def test_its_own_hotkeys_are_refused(self):
        rig = Rig(Decision("hotkey", keys=("ctrl", "alt", "y")))
        result = rig.agent.handle("approve it")
        self.assertEqual((result.outcome, result.said), (REFUSED, "That is one of my own hotkeys, so I don't press it."))
        self.assertEqual(rig.executor.runs, [])

    def test_unknown_window_is_refused_when_only_some_windows_are_allowed(self):
        rig = Rig(CLICK, titles=None, allowed_windows=("notepad",))
        self.assertEqual(rig.agent.handle("click").outcome, REFUSED)
        self.assertEqual(rig.executor.runs, [])

    def test_steps_before_a_refusal_are_kept(self):
        rig = Rig(Decision("wait", seconds=1.0), CLICK, titles=BANK)
        self.assertEqual(rig.agent.handle("pay").steps, ["wait 1 s"])


class DeclineTests(unittest.TestCase):
    def test_no_from_the_user(self):
        rig = Rig(CLICK, DONE, confirms=(False,))
        result = rig.agent.handle("click the middle")
        self.assertEqual(result, Result(DECLINED, said="", steps=[]))
        self.assertEqual(rig.executor.runs, [])
        self.assertEqual(rig.events("release"), [])
        self.assertEqual(len(rig.vision.calls), 1)
        self.assertEqual(rig.agent.conversation, [("user", "click the middle"), ("you", "The user said no.")])


class StopTests(unittest.TestCase):
    def test_stop_cuts_off_speech(self):
        rig = Rig()
        rig.agent.stop()
        self.assertTrue(rig.agent.stop_event.is_set())
        self.assertEqual(rig.speaker.stops, 1)

    def test_stop_before_a_request_does_not_cancel_it(self):
        rig = Rig(DONE)
        rig.agent.stop()
        self.assertEqual(rig.agent.handle("hello").outcome, FINISHED)

    def test_a_stop_while_the_request_was_spoken_cancels_it(self):
        # Voice mode: the stop key pressed while talking drops what was said, it is never carried out.
        rig = Rig(CLICK, DONE, confirms=(True,))
        rig.agent.new_request()
        rig.agent.stop()
        self.assertEqual(rig.agent.handle("delete the files in my Downloads", fresh=False), Result(STOPPED))
        self.assertEqual((rig.vision.calls, rig.executor.runs), ([], []))

    def test_new_request_forgets_an_earlier_stop(self):
        rig = Rig(DONE)
        rig.agent.stop()
        rig.agent.new_request()
        self.assertEqual(rig.agent.handle("hello", fresh=False).outcome, FINISHED)

    def test_stop_while_the_model_thinks(self):
        rig = Rig()
        rig.vision.answers = [rig.then_stop(Decision("click", x=500.0, y=500.0, say="Clicking."))]
        result = rig.agent.handle("click")
        self.assertEqual(result, Result(STOPPED, steps=[]))
        self.assertEqual((rig.speaker.said, rig.confirmer.asked, rig.executor.runs), ([], [], []))

    def test_stop_while_the_model_thinks_beats_its_answer(self):
        rig = Rig()
        rig.vision.answers = [rig.then_stop(Decision("answer", say="It is noon."))]
        self.assertEqual(rig.agent.handle("time?"), Result(STOPPED, steps=[]))
        self.assertEqual(rig.speaker.said, [])
        self.assertEqual(rig.agent.conversation, [])

    def test_stop_while_waiting_for_the_go_ahead(self):
        rig = Rig(CLICK, DONE)
        rig.confirmer.answers = [rig.then_stop(True)]
        result = rig.agent.handle("click")
        self.assertEqual(result, Result(STOPPED, steps=[]))
        self.assertEqual(rig.executor.runs, [])
        self.assertEqual(rig.agent.conversation, [])

    def test_stop_after_a_step_ends_before_the_next_look(self):
        rig = Rig(CLICK, DONE, confirms=(True,))
        rig.on_sleep = rig.agent.stop
        result = rig.agent.handle("click")
        self.assertEqual(result, Result(STOPPED, steps=["did click at (500, 500)"]))
        self.assertEqual(rig.captures, 2)  # the look, and the check that the screen had not moved on
        self.assertEqual(len(rig.vision.calls), 1)

    def test_stop_while_typing(self):
        typing = Decision("type", text="hello world")
        rig = Rig(typing, DONE, confirms=(True,))

        def stopped_part_way():
            rig.agent.stop_event.set()
            return Interrupted("stopped")

        rig.executor.failure = stopped_part_way
        result = rig.agent.handle("type it")
        self.assertEqual(result, Result(STOPPED, steps=["(stopped part-way: type 'hello world')"]))
        self.assertEqual(len(rig.vision.calls), 1)


class HeldKeyTests(unittest.TestCase):
    """The yes hotkey is still held when it fires: a step taken then would be Ctrl+Alt+letters."""

    def test_the_step_waits_for_the_keys_to_come_up(self):
        rig = Rig(CLICK, DONE, confirms=(True,))
        rig.agent.handle("click")
        self.assertLess(rig.log.index(("release", 2.0)), rig.log.index(("run", CLICK, (1001, 501))))

    def test_slow_hands_are_asked_to_let_go(self):
        rig = Rig(CLICK, DONE, confirms=(True,), releases=[False, True])
        rig.agent.handle("click")
        self.assertEqual(rig.events("release"), [("release", 2.0), ("release", 10.0)])
        self.assertIn("Let go of the keys and I'll go on.", rig.speaker.said)
        self.assertEqual(rig.executor.runs, [(CLICK, (1001, 501))])

    def test_keys_that_stay_down_mean_the_step_is_not_taken(self):
        rig = Rig(CLICK, DONE, confirms=(True,), releases=False)
        result = rig.agent.handle("click")
        message = "Keys stayed held down, so I didn't do it: with them, it would not be the step you agreed to."
        self.assertEqual(result, Result(STOPPED, said=message, steps=[]))
        self.assertEqual(rig.executor.runs, [])

    def test_steps_without_a_question_wait_for_the_keys_too(self):
        rig = Rig(CLICK, DONE, confirm="risky", releases=False)
        self.assertEqual(rig.agent.handle("click").outcome, STOPPED)
        self.assertEqual(rig.executor.runs, [])

    def test_a_key_pressed_while_typing_stops_the_typing(self):
        rig = Rig(Decision("type", text="hello"), DONE, confirms=(True,), failure=Interrupted("ctrl held down"))
        result = rig.agent.handle("type hello")
        self.assertEqual(result.outcome, STOPPED)
        self.assertEqual(result.said, "I stopped typing part-way: ctrl held down. Some of it may be typed already.")
        self.assertEqual(len(rig.vision.calls), 1)


class WindowChangeTests(unittest.TestCase):
    NOTE = "(not done: the window in front changed before 'click at (500, 500)')"

    def test_window_change_while_asking_skips_the_step(self):
        rig = Rig(CLICK, DONE, titles=[NOTEPAD, BANK, BANK], targets=NOTEPAD, confirms=(True,))
        result = rig.agent.handle("click the middle")
        self.assertEqual(result, Result(FINISHED, said="All done.", steps=[self.NOTE]))
        self.assertEqual(rig.executor.runs, [])
        self.assertEqual(rig.slept, [])
        self.assertEqual(rig.vision.calls[1].steps, [self.NOTE])

    def test_window_change_without_asking_skips_the_step(self):
        rig = Rig(CLICK, DONE, titles=[NOTEPAD, "Calculator", "Calculator"], targets=NOTEPAD, confirm="risky")
        result = rig.agent.handle("click the middle")
        self.assertEqual(result.steps, [self.NOTE])
        self.assertEqual((rig.confirmer.asked, rig.executor.runs), ([], []))

    def test_window_that_stays_unknown_is_not_a_change(self):
        rig = Rig(CLICK, DONE, titles=None, confirms=(True,))
        rig.agent.handle("click the middle")
        self.assertEqual(rig.executor.runs, [(CLICK, (1001, 501))])

    def test_a_new_window_under_the_click_skips_it(self):
        # A pop-up opened over the spot while the user decided.
        rig = Rig(CLICK, DONE, targets=[NOTEPAD, "Update available"], confirms=(True,))
        result = rig.agent.handle("click the middle")
        self.assertEqual(result.steps, ["(not done: the window under that point changed before 'click at (500, 500)')"])
        self.assertEqual(rig.executor.runs, [])


class ScreenChangeTests(unittest.TestCase):
    """The model chose its step on a screenshot that is seconds old by the time it is carried out."""

    LATER = Shot(image=b"later", width=1000, height=500, screen_width=2000, screen_height=1000)
    NOTE = "(not done: the screen changed before 'click at (500, 500)')"

    SAID = "I didn't do that: the screen changed after I looked. Looking again."

    def test_a_changed_screen_under_the_click_skips_it(self):
        # Changed since the model looked, then still a moment later: the screen moved on (a dialog closed).
        rig = Rig(CLICK, DONE, confirms=(True,), shots=[SHOT, self.LATER, self.LATER], changed=[0.31, 0.3])
        result = rig.agent.handle("click OK")
        self.assertEqual(result.steps, [self.NOTE])
        self.assertEqual(rig.executor.runs, [])
        self.assertEqual(rig.compared, [(b"screenshot", b"later", (0.5, 0.5)), (b"later", b"later", (0.5, 0.5))])
        self.assertEqual(rig.slept, [0.3])  # one more look, a moment later
        self.assertEqual(rig.speaker.said, [self.SAID, "All done."])

    def test_a_place_that_keeps_changing_is_moving_not_moved_on(self):
        # A video under the click changes at every look: no screenshot is ever new enough, so the
        # click goes ahead once two more looks show it never settles.
        playing = [Shot(image=f"frame {n}".encode(), width=1000, height=500, screen_width=2000, screen_height=1000)
                   for n in range(4)]
        rig = Rig(CLICK, DONE, confirms=(True,), shots=playing, changed=[0.61, 0.58, 0.6])
        result = rig.agent.handle("skip the ad")
        self.assertEqual(result.steps, ["did click at (500, 500)"])
        self.assertEqual(rig.compared, [(b"frame 0", b"frame 1", (0.5, 0.5)), (b"frame 1", b"frame 2", (0.5, 0.5)),
                                        (b"frame 2", b"frame 3", (0.5, 0.5))])
        self.assertEqual(rig.slept, [0.3, 0.3, 0.8])
        self.assertEqual(rig.speaker.said, ["All done."])

    def test_a_change_that_settles_on_the_last_look_skips_it(self):
        rig = Rig(CLICK, DONE, confirms=(True,), shots=[SHOT, self.LATER], changed=[0.31, 0.5, 0.3])
        result = rig.agent.handle("click OK")
        self.assertEqual(result.steps, [self.NOTE])
        self.assertEqual(rig.executor.runs, [])
        self.assertEqual(len(rig.compared), 3)

    def test_why_a_step_was_not_done_is_said_once_a_request(self):
        rig = Rig(CLICK, CLICK, DONE, confirms=(True, True), shots=[SHOT, self.LATER, self.LATER, SHOT, self.LATER, self.LATER],
                  changed=[0.31, 0.0, 0.31, 0.0])
        result = rig.agent.handle("click OK")
        self.assertEqual(result.steps, [self.NOTE, self.NOTE])
        self.assertEqual(rig.speaker.said, [self.SAID, "All done."])
        rig.vision.answers = [CLICK, DONE]
        rig.confirmer.answers = [True]
        rig.shots = [SHOT, self.LATER, self.LATER]
        rig.changed = [0.31, 0.0]
        rig.agent.handle("click OK again")
        self.assertEqual(rig.speaker.said, [self.SAID, "All done.", self.SAID, "All done."])  # said again for the next request

    def test_a_small_change_under_the_click_is_fine(self):
        rig = Rig(CLICK, DONE, confirms=(True,), shots=[SHOT, self.LATER, self.LATER], changed=0.3)
        rig.agent.handle("click OK")
        self.assertEqual(rig.executor.runs, [(CLICK, (1001, 501))])
        self.assertEqual(len(rig.compared), 1)  # no more looks for a screen that did not change

    def test_keys_compare_the_whole_screen(self):
        typing = Decision("type", text="hi")
        rig = Rig(typing, DONE, confirms=(True,), shots=[SHOT, self.LATER, self.LATER], changed=[0.26, 0.25])
        result = rig.agent.handle("type hi")
        self.assertEqual(result.steps, ["(not done: the screen changed before 'type 'hi'')"])
        self.assertEqual(rig.compared, [(b"screenshot", b"later", None), (b"later", b"later", None)])
        rig = Rig(typing, DONE, confirms=(True,), shots=[SHOT, self.LATER, self.LATER], changed=0.25)
        rig.agent.handle("type hi")
        self.assertEqual(rig.executor.runs, [(typing, None)])

    def test_pixel_positions_are_compared_where_they_are(self):
        click = Decision("click", x=250.0, y=125.0)
        rig = Rig(click, DONE, coordinates="pixels", confirms=(True,), shots=[SHOT, self.LATER, self.LATER])
        rig.agent.handle("click")
        self.assertEqual(rig.compared[0][2], (0.25, 0.25))

    def test_screenshots_that_cannot_be_compared_do_not_block_the_step(self):
        rig = Rig(CLICK, DONE, confirms=(True,), shots=[SHOT, self.LATER, self.LATER])
        rig.agent.difference = lambda before, after, around: 1 / 0
        rig.agent.handle("click")
        self.assertEqual(rig.executor.runs, [(CLICK, (1001, 501))])

    def test_a_look_that_fails_while_checking_for_motion_counts_as_a_changed_screen(self):
        rig = Rig(CLICK, DONE, confirms=(True,), shots=[SHOT, self.LATER], changed=0.5)
        looks = []

        def capture():
            looks.append("look")
            if len(looks) == 3:  # the first look for motion
                raise OSError("the screen could not be read")
            return SHOT if len(looks) == 1 else self.LATER

        rig.agent.capture = capture
        result = rig.agent.handle("click")
        self.assertEqual(result, Result(FINISHED, said="All done.", steps=[self.NOTE]))
        self.assertEqual(rig.executor.runs, [])

    def test_stop_while_looking_for_motion(self):
        rig = Rig(CLICK, DONE, confirms=(True,), shots=[SHOT, self.LATER, SHOT, self.LATER], changed=0.6)
        rig.on_sleep = rig.agent.stop
        self.assertEqual(rig.agent.handle("click"), Result(STOPPED, steps=[]))
        self.assertEqual(rig.executor.runs, [])


class OwnWindowTests(unittest.TestCase):
    """A request typed into NEXUS Live's console starts with that console in front."""

    CONSOLE = "MINGW64:/c/Users/nagu/nexus"

    def rig(self, *answers, titles, **settings):
        rig = Rig(*answers, titles=titles, **settings)
        rig.agent.note_own_window(self.CONSOLE)
        return rig

    def test_it_waits_for_the_user_to_pick_a_window(self):
        rig = self.rig(Decision("type", text="hi"), DONE, titles=[self.CONSOLE, self.CONSOLE, self.CONSOLE, NOTEPAD], confirms=(True,))
        result = rig.agent.handle("say hi")
        self.assertEqual(result.outcome, FINISHED)
        self.assertEqual(rig.speaker.said[0], "Click the window you want me to work in, and I'll start there.")
        self.assertEqual(rig.slept[:4], [0.25, 0.25, 0.25, 0.8])  # three looks at the window in front, then a moment to settle
        self.assertEqual(rig.confirmer.asked, ["type 'hi' (2 characters) in 'Untitled - Notepad'"])
        self.assertEqual(len(rig.vision.calls), 2)  # the model never saw the console screenshot

    def test_waiting_does_not_use_up_steps(self):
        rig = self.rig(DONE, titles=[self.CONSOLE, NOTEPAD], max_steps=1)
        self.assertEqual(rig.agent.handle("finish").outcome, FINISHED)

    def test_no_window_picked_in_time(self):
        rig = self.rig(DONE, titles=self.CONSOLE, confirm_timeout=3.0)
        result = rig.agent.handle("do something")
        message = "No other window came to the front, so I stopped. Click the window to use, then ask again."
        self.assertEqual(result, Result(NO_WINDOW, said=message, steps=[]))
        self.assertEqual(rig.slept, [0.25] * 12)
        self.assertEqual(rig.vision.calls, [])

    def test_stop_while_waiting(self):
        rig = self.rig(DONE, titles=self.CONSOLE)
        rig.on_sleep = rig.agent.stop
        self.assertEqual(rig.agent.handle("do something").outcome, STOPPED)

    def test_back_in_its_own_window_too_often(self):
        # The console was in front at every look, however often another window came forward between them.
        rig = self.rig(DONE, titles=[self.CONSOLE, NOTEPAD, self.CONSOLE, NOTEPAD, self.CONSOLE, NOTEPAD, self.CONSOLE])
        result = rig.agent.handle("do something")
        self.assertEqual(result.outcome, NO_WINDOW)
        self.assertEqual(rig.vision.calls, [])
        self.assertEqual(rig.speaker.said.count("Click the window you want me to work in, and I'll start there."), 3)

    def test_a_click_into_its_own_window_is_refused(self):
        rig = self.rig(CLICK, titles=NOTEPAD, targets=self.CONSOLE, confirm="risky")
        result = rig.agent.handle("click")
        self.assertEqual((result.outcome, result.said), (REFUSED, "That point is in my own window, which I never click or type in."))
        self.assertEqual(rig.executor.runs, [])

    def test_only_real_titles_are_noted(self):
        rig = Rig()
        rig.agent.note_own_window(None)
        rig.agent.note_own_window("")
        self.assertEqual(rig.agent.own_windows, set())
        rig.agent.note_own_window("Windows PowerShell")
        rig.agent.note_own_window("Windows PowerShell")
        self.assertEqual(rig.agent.own_windows, {"Windows PowerShell"})


class FailureTests(unittest.TestCase):
    """A failure ends the request, never NEXUS Live: the console or voice loop goes on to the next one."""

    def test_the_model_failing_gives_up_the_request(self):
        rig = Rig(RuntimeError("CUDA out of memory"))
        result = rig.agent.handle("tidy up")
        message = "Something went wrong (RuntimeError: CUDA out of memory), so I stopped."
        self.assertEqual(result, Result(GAVE_UP, said=message, steps=[]))
        self.assertEqual(rig.speaker.said, [message])

    def test_the_hands_failing_gives_up_the_request_keeping_the_steps(self):
        rig = Rig(Decision("wait", seconds=1.0), CLICK, confirm="risky", failure=OSError("no display"))
        result = rig.agent.handle("click")
        self.assertEqual(result, Result(GAVE_UP, said="Something went wrong (OSError: no display), so I stopped.", steps=["wait 1 s"]))

    def test_the_pointer_in_a_corner_is_a_stop(self):
        FailSafeException = type("FailSafeException", (Exception,), {})
        rig = Rig(CLICK, confirm="risky", failure=FailSafeException("PyAutoGUI fail-safe triggered"))
        result = rig.agent.handle("click")
        self.assertEqual(result, Result(STOPPED, said="Stopped: the pointer went into a corner of the screen.", steps=[]))


class ModelTroubleTests(unittest.TestCase):
    def test_unreadable_answer_is_retried_once_on_the_same_screenshot(self):
        rig = Rig(DecisionError("unknown action 'drag'"), DONE)
        result = rig.agent.handle("tidy up")
        self.assertEqual(result, Result(FINISHED, said="All done.", steps=[]))
        self.assertEqual(rig.captures, 1)
        self.assertEqual(
            [call.steps for call in rig.vision.calls],
            [[], ["(your last answer could not be used: unknown action 'drag'; answer again)"]],
        )
        self.assertIs(rig.vision.calls[1].shot, SHOT)

    def test_retry_note_is_not_kept_as_a_step(self):
        rig = Rig(CLICK, DecisionError("bad"), DONE, confirms=(True,))
        result = rig.agent.handle("click then finish")
        self.assertEqual(result.steps, ["did click at (500, 500)"])
        self.assertEqual(
            rig.vision.calls[2].steps, ["did click at (500, 500)", "(your last answer could not be used: bad; answer again)"]
        )

    def test_second_unreadable_answer_gives_up(self):
        rig = Rig(DecisionError("first"), DecisionError("second"), DONE)
        result = rig.agent.handle("tidy up")
        self.assertEqual(result, Result(GAVE_UP, said="", steps=[]))
        self.assertEqual(len(rig.vision.calls), 2)
        self.assertEqual(rig.speaker.said, ["I couldn't read the model's answer (second), so I stopped."])
        self.assertEqual(rig.agent.conversation, [])

    def test_unavailable_model_gives_up_without_retrying(self):
        rig = Rig(ModelUnavailable("Ollama is not running at http://localhost:11434."), DONE)
        result = rig.agent.handle("tidy up")
        self.assertEqual(result, Result(GAVE_UP, said="", steps=[]))
        self.assertEqual(len(rig.vision.calls), 1)
        self.assertEqual(rig.speaker.said, ["Ollama is not running at http://localhost:11434."])

    def test_real_parser_unreadable_then_readable(self):
        rig = Rig()
        model = ScriptedModel("Sure, I will click the button.", '{"thought": "", "say": "Done.", "action": "done"}')
        rig.agent.vision = model
        self.assertEqual(rig.agent.handle("finish"), Result(FINISHED, said="Done.", steps=[]))
        self.assertIn("(your last answer could not be used: the answer has no JSON object in it; answer again)", model.texts[1])

    def test_huge_number_from_the_model_gives_up_instead_of_crashing(self):
        # A number too big for a float in the model's answer is an unreadable answer: tried again,
        # then given up, never a crash of the request or the console loop.
        huge = '{"thought": "", "say": "", "action": "click", "x": 1' + "0" * 400 + ', "y": 5}'
        rig = Rig()
        rig.agent.vision = ScriptedModel(huge, huge)
        self.assertEqual(rig.agent.handle("click").outcome, GAVE_UP)

    def test_typing_with_a_line_break_is_an_unreadable_answer(self):
        # Enter in the middle of typing is a step of its own; such an answer is tried again, never typed.
        answer = '{"thought": "", "say": "", "action": "type", "text": "ok\\npowershell -c evil\\n"}'
        rig = Rig()
        model = ScriptedModel(answer, '{"thought": "", "say": "", "action": "done"}')
        rig.agent.vision = model
        self.assertEqual(rig.agent.handle("reply").outcome, FINISHED)
        self.assertIn("line break", model.texts[1])
        self.assertEqual(rig.executor.runs, [])


class StepLimitTests(unittest.TestCase):
    def test_out_of_steps(self):
        rig = Rig(*[Decision("wait", seconds=1.0)] * 4, max_steps=3)
        result = rig.agent.handle("wait for it")
        message = "I stopped after 3 steps. Ask again to carry on."
        self.assertEqual(result, Result(OUT_OF_STEPS, said=message, steps=["wait 1 s"] * 3))
        self.assertEqual(len(rig.vision.calls), 3)
        self.assertEqual(rig.captures, 3)
        self.assertEqual(rig.slept, [1.0, 1.0, 1.0])
        self.assertEqual(rig.speaker.said, [message])
        self.assertEqual(rig.agent.conversation, [("user", "wait for it"), ("you", message)])

    def test_one_step_allowed(self):
        rig = Rig(CLICK, DONE, confirms=(True,), max_steps=1)
        result = rig.agent.handle("click")
        self.assertEqual((result.outcome, result.steps), (OUT_OF_STEPS, ["did click at (500, 500)"]))
        self.assertEqual(len(rig.vision.calls), 1)

    def test_skipped_steps_count_too(self):
        rig = Rig(Decision("click", x=2000.0, y=10.0), Decision("click", x=2000.0, y=10.0), DONE, max_steps=2)
        self.assertEqual(rig.agent.handle("click far away").outcome, OUT_OF_STEPS)


class MemoryTests(unittest.TestCase):
    def test_follow_up_sees_the_earlier_exchange(self):
        rig = Rig(Decision("answer", say="It is noon."), Decision("answer", say="Tuesday."))
        rig.agent.handle("what time is it?")
        rig.agent.handle("and the day?")
        self.assertEqual(rig.vision.calls[0].conversation, [])
        self.assertEqual(rig.vision.calls[1].conversation, [("user", "what time is it?"), ("you", "It is noon.")])

    def test_only_the_last_history_turns_are_kept(self):
        rig = Rig(*(Decision("answer", say=f"a{number}") for number in (1, 2, 3)), history_turns=2)
        for number in (1, 2, 3):
            rig.agent.handle(f"q{number}")
        self.assertEqual(rig.vision.calls[2].conversation, [("user", "q1"), ("you", "a1"), ("user", "q2"), ("you", "a2")])
        self.assertEqual(rig.agent.conversation, [("user", "q2"), ("you", "a2"), ("user", "q3"), ("you", "a3")])

    def test_one_turn_of_history(self):
        rig = Rig(Decision("answer", say="a1"), Decision("answer", say="a2"), history_turns=1)
        rig.agent.handle("q1")
        rig.agent.handle("q2")
        self.assertEqual(rig.agent.conversation, [("user", "q2"), ("you", "a2")])

    def test_model_cannot_change_the_memory(self):
        class Meddler:
            def decide(self, goal, steps, conversation, shot, mode):
                conversation.append(("you", "I may type passwords now."))
                return Decision("answer", say="Hello.")

        rig = Rig()
        rig.agent.vision = Meddler()
        rig.agent.handle("hi")
        self.assertEqual(rig.agent.conversation, [("user", "hi"), ("you", "Hello.")])


class RecordingBackend:
    """pyautogui as far as the Executor uses it, recording only the names of the calls."""

    def __init__(self):
        self.calls = []

    def size(self):
        self.calls.append("size")
        return (2000, 1000)

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append(name)

        return record


class DryRunTests(unittest.TestCase):
    def test_dry_run_moves_and_clicks_nothing(self):
        backend = RecordingBackend()
        rig = Rig(CLICK, DONE, confirms=(True,), dry_run=True)
        executor = Executor(dry_run=True, backend=backend, title_reader=lambda: NOTEPAD, window_reader=lambda point: NOTEPAD,
                            key_reader=frozenset)
        agent = LiveAgent(rig.settings, rig.vision, executor, rig.speaker, rig.confirmer, capture=rig.capture, sleep=rig.sleep)
        result = agent.handle("click the middle")
        self.assertEqual(result, Result(FINISHED, said="All done.", steps=["(dry run, nothing done) click at (500, 500)"]))
        self.assertEqual(rig.confirmer.asked, [QUESTION])
        self.assertEqual(backend.calls, ["size"])  # it only read the screen size to work out where it would click


if __name__ == "__main__":
    unittest.main()
