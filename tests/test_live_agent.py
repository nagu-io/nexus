"""Tests for nexus.live.agent: the loop that looks, decides, checks, asks, acts and looks again."""

import unittest
from types import SimpleNamespace

from nexus.live.agent import DECLINED, FINISHED, GAVE_UP, OUT_OF_STEPS, REFUSED, STOPPED, LiveAgent, Result
from nexus.live.executor import Executor
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


class FakeExecutor:
    """The hands, recorded. ``titles`` is the window in front: one title, or a list read in turn (the last one repeats)."""

    def __init__(self, rig, titles):
        self.rig = rig
        self.titles = list(titles) if isinstance(titles, list) else [titles]
        self.runs = []

    def foreground_title(self):
        self.rig.log.append("title")
        return self.titles.pop(0) if len(self.titles) > 1 else self.titles[0]

    def mouse_point(self, point, shot):
        self.rig.log.append(("mouse_point", point, shot))
        return (point[0] + 1, point[1] + 1)  # changed, so a test can tell the agent used it

    def point_at(self, point):
        self.rig.log.append(("point_at", point))

    def run(self, decision, point):
        self.rig.log.append(("run", decision, point))
        self.runs.append((decision, point))
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
    """A LiveAgent wired to fakes that write everything they are asked to do into one log."""

    def __init__(self, *answers, titles=NOTEPAD, confirms=(), **settings):
        self.log = []
        self.settings = LiveSettings(**settings)
        self.vision = FakeVision(self, answers)
        self.executor = FakeExecutor(self, titles)
        self.speaker = FakeSpeaker(self)
        self.confirmer = FakeConfirmer(self, confirms)
        self.slept = []
        self.on_sleep = None
        self.agent = LiveAgent(
            self.settings, self.vision, self.executor, self.speaker, self.confirmer, capture=self.capture, sleep=self.sleep
        )

    def capture(self):
        self.log.append("capture")
        return SHOT

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
            (FINISHED, REFUSED, DECLINED, STOPPED, GAVE_UP, OUT_OF_STEPS),
            ("finished", "refused", "declined", "stopped", "gave_up", "out_of_steps"),
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


class ActingTests(unittest.TestCase):
    def test_a_click_is_shown_asked_about_rechecked_then_done(self):
        click = Decision("click", x=500.0, y=500.0, say="Clicking Save.")
        rig = Rig(click, DONE, confirms=(True,))
        result = rig.agent.handle("save the file")
        self.assertEqual(result, Result(FINISHED, said="All done.", steps=["did click at (500, 500)"]))
        self.assertEqual(
            rig.log,
            [
                "capture",
                "title",
                "decide",
                ("say", "Clicking Save."),
                ("mouse_point", (1000, 500), SHOT),
                ("point_at", (1001, 501)),
                ("ask", "click at (500, 500)"),
                "title",
                ("run", click, (1001, 501)),
                ("sleep", 0.8),
                "capture",
                "title",
                "decide",
                ("say", "All done."),
            ],
        )
        self.assertEqual(rig.vision.calls[1].steps, ["did click at (500, 500)"])
        self.assertEqual(rig.agent.conversation, [("user", "save the file"), ("you", "All done.")])

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
        # The window is still checked again right before the click.
        self.assertEqual(rig.log[: rig.log.index(("run", CLICK, (1001, 501)))].count("title"), 2)

    def test_risky_mode_still_asks_before_typing(self):
        typing = Decision("type", text="hello")
        rig = Rig(typing, DONE, confirm="risky", confirms=(True,))
        rig.agent.handle("say hello")
        self.assertEqual(rig.confirmer.asked, ["type 'hello'"])
        self.assertEqual(rig.events("point_at"), [])
        self.assertEqual(rig.executor.runs, [(typing, None)])

    def test_pixel_positions(self):
        click = Decision("click", x=900.0, y=450.0)
        rig = Rig(click, DONE, coordinates="pixels", confirm="risky")
        rig.agent.handle("click near the corner")
        self.assertEqual(rig.events("mouse_point"), [("mouse_point", (1800, 900), SHOT)])
        self.assertEqual(rig.executor.runs, [(click, (1801, 901))])

    def test_scroll_without_a_position_scrolls_where_the_pointer_is(self):
        scroll = Decision("scroll", amount=-3)
        rig = Rig(scroll, DONE, confirm="risky")
        rig.agent.handle("scroll down")
        self.assertEqual(rig.events("mouse_point"), [])
        self.assertEqual(rig.executor.runs, [(scroll, None)])

    def test_scroll_at_a_position(self):
        scroll = Decision("scroll", amount=3, x=500.0, y=250.0)
        rig = Rig(scroll, DONE, confirms=(True,))
        rig.agent.handle("scroll the list up")
        self.assertEqual(rig.events("mouse_point"), [("mouse_point", (1000, 250), SHOT)])
        self.assertEqual(rig.events("point_at"), [("point_at", (1001, 251))])
        self.assertEqual(rig.confirmer.asked, ["scroll up 3"])
        self.assertEqual(rig.executor.runs, [(scroll, (1001, 251))])

    def test_wait_sleeps_without_asking(self):
        rig = Rig(Decision("wait", seconds=2.5), DONE, titles=BANK)
        result = rig.agent.handle("wait for the page")
        self.assertEqual(result, Result(FINISHED, said="All done.", steps=["wait 2.5 s"]))
        self.assertEqual(rig.slept, [2.5])
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
        self.assertEqual(rig.confirmer.asked, ["click at (500, 500)"])
        self.assertEqual(rig.executor.runs, [(CLICK, (1001, 501))])


class RefusalTests(unittest.TestCase):
    def test_blocked_window_is_refused_before_pointing_or_asking(self):
        rig = Rig(CLICK, titles=BANK)
        result = rig.agent.handle("pay the bill")
        self.assertEqual(result, Result(REFUSED, said=BANK_REFUSAL, steps=[]))
        self.assertEqual(rig.speaker.said, [BANK_REFUSAL])
        self.assertEqual(rig.agent.conversation, [("user", "pay the bill"), ("you", BANK_REFUSAL)])
        self.assertEqual((rig.confirmer.asked, rig.executor.runs), ([], []))
        self.assertEqual((rig.events("mouse_point"), rig.events("point_at")), ([], []))

    def test_windows_key_is_refused(self):
        rig = Rig(Decision("hotkey", keys=("win", "r")))
        result = rig.agent.handle("open the run box")
        self.assertEqual(result.outcome, REFUSED)
        self.assertEqual(result.said, "I don't use the Windows key: it opens the Run box and system menus.")
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
        self.assertEqual(rig.captures, 1)
        self.assertEqual(len(rig.vision.calls), 1)


class WindowChangeTests(unittest.TestCase):
    NOTE = "(not done: the window in front changed before 'click at (500, 500)')"

    def test_window_change_while_asking_skips_the_step(self):
        rig = Rig(CLICK, DONE, titles=[NOTEPAD, BANK, BANK], confirms=(True,))
        result = rig.agent.handle("click the middle")
        self.assertEqual(result, Result(FINISHED, said="All done.", steps=[self.NOTE]))
        self.assertEqual(rig.executor.runs, [])
        self.assertEqual(rig.slept, [])
        self.assertEqual(rig.vision.calls[1].steps, [self.NOTE])

    def test_window_change_without_asking_skips_the_step(self):
        rig = Rig(CLICK, DONE, titles=[NOTEPAD, "Calculator", "Calculator"], confirm="risky")
        result = rig.agent.handle("click the middle")
        self.assertEqual(result.steps, [self.NOTE])
        self.assertEqual((rig.confirmer.asked, rig.executor.runs), ([], []))

    def test_window_that_stays_unknown_is_not_a_change(self):
        rig = Rig(CLICK, DONE, titles=None, confirms=(True,))
        rig.agent.handle("click the middle")
        self.assertEqual(rig.executor.runs, [(CLICK, (1001, 501))])


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

    @unittest.expectedFailure
    def test_huge_number_from_the_model_gives_up_instead_of_crashing(self):
        # BUG: parse_decision raises OverflowError (not DecisionError) for a whole number too big
        # for a float, so _decide does not catch it and handle() crashes the request (and the
        # console loop above it) instead of retrying and giving up.
        huge = '{"thought": "", "say": "", "action": "click", "x": 1' + "0" * 400 + ', "y": 5}'
        rig = Rig()
        rig.agent.vision = ScriptedModel(huge, huge)
        self.assertEqual(rig.agent.handle("click").outcome, GAVE_UP)


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
        executor = Executor(dry_run=True, backend=backend, title_reader=lambda: NOTEPAD)
        agent = LiveAgent(rig.settings, rig.vision, executor, rig.speaker, rig.confirmer, capture=rig.capture, sleep=rig.sleep)
        result = agent.handle("click the middle")
        self.assertEqual(result, Result(FINISHED, said="All done.", steps=["(dry run, nothing done) click at (500, 500)"]))
        self.assertEqual(rig.confirmer.asked, ["click at (500, 500)"])
        self.assertEqual(backend.calls, ["size"])  # it only read the screen size to work out where it would click


if __name__ == "__main__":
    unittest.main()
