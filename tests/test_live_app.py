"""Tests for nexus.live.app's LiveSession: the voice and typing loops, the stop key, and watch mode around requests."""

import threading
import unittest
from dataclasses import replace
from unittest.mock import patch

import nexus.live.app as app_module
from nexus.live.agent import FINISHED, Result
from nexus.live.app import LiveSession
from nexus.live.settings import LiveSettings

CONSOLE = "MINGW64:/c/Users/nagu/nexus"


def session(lines=None):
    settings = replace(LiveSettings(), speak=False, backend="ollama", model="test-model")
    live = LiveSession(settings, "test-model", printer=(lines.append if lines is not None else lambda line: None))
    live.handled = []

    def handle(goal, fresh=True):
        live.handled.append((goal, fresh))
        return Result(FINISHED)

    live.agent.handle = handle
    return live


class FakeRecorder:
    """Records one utterance per talk key press; ``during`` runs while recording (the user pressing keys)."""

    def __init__(self, live, during):
        self.live = live
        self.during = list(during)

    def record(self, stop):
        action = self.during.pop(0)
        if action:
            action()
        return [0.1]


class FakeTranscriber:
    def __init__(self, live, texts):
        self.live = live
        self.texts = list(texts)
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        text = self.texts.pop(0)
        return text() if callable(text) else text


class ListenLoopTests(unittest.TestCase):
    def listen(self, live, during, texts):
        """Run the voice loop with fake recording and speech to text, until it hears "goodbye".

        On a thread with a time limit: a loop that waits for a talk key nobody presses fails the
        test instead of hanging it.
        """
        recorder = FakeRecorder(live, during)
        transcriber = FakeTranscriber(live, texts)
        live.talk_requested.set()
        with patch.object(app_module, "Recorder", return_value=recorder), \
             patch.object(app_module, "Transcriber", return_value=transcriber):
            loop = threading.Thread(target=live.listen_loop, daemon=True)
            loop.start()
            loop.join(5)
        self.assertFalse(loop.is_alive(), "the voice loop never reached \"goodbye\"")
        return transcriber

    def test_a_request_is_handled_keeping_a_stop_pressed_since_the_talk_key(self):
        live = session()
        self.listen(live, [live.talk_requested.set, None], ["open notepad", "goodbye"])
        self.assertEqual(live.handled, [("open notepad", False)])

    def test_stop_while_talking_drops_what_was_said(self):
        # The review's case: "delete the files in my Downloads... no wait", then the stop key.
        lines = []
        live = session(lines)

        def stop_then_talk_again():
            live.on_stop()
            live.talk_requested.set()

        transcriber = self.listen(live, [stop_then_talk_again, None], ["goodbye"])
        self.assertEqual(live.handled, [])
        self.assertEqual(transcriber.calls, 1)  # only the second recording was written out
        self.assertIn("NEXUS: stopped.", lines)
        self.assertIn("(dropped what you said)", lines)

    def test_stop_while_it_is_written_out_drops_it_too(self):
        live = session()

        def stop_while_transcribing():
            live.on_stop()
            live.talk_requested.set()
            return "delete the files in my Downloads"

        self.listen(live, [None, None], [stop_while_transcribing, "goodbye"])
        self.assertEqual(live.handled, [])

    def test_a_stop_before_the_talk_key_does_not_drop_the_next_request(self):
        live = session()
        live.on_stop()
        self.listen(live, [live.talk_requested.set, None], ["what time is it", "goodbye"])
        self.assertEqual(live.handled, [("what time is it", False)])


class TypeLoopTests(unittest.TestCase):
    def test_the_console_a_request_is_typed_in_is_noted_as_its_own(self):
        live = session()
        live.executor._title_reader = lambda: CONSOLE
        requests = iter(["open notepad", "  ", "exit"])
        live.type_loop(read=lambda prompt: next(requests))
        self.assertEqual(live.handled, [("open notepad", True)])
        self.assertEqual(live.agent.own_windows, {CONSOLE})

    def test_note_console_at_the_start(self):
        live = session()
        live.executor._title_reader = lambda: "Windows PowerShell"
        live.note_console()
        self.assertEqual(live.agent.own_windows, {"Windows PowerShell"})
        live.executor._title_reader = lambda: None
        live.note_console()
        self.assertEqual(live.agent.own_windows, {"Windows PowerShell"})


class FakeWatcher:
    def __init__(self, log):
        self.log = log

    def pause(self):
        self.log.append("pause")

    def resume(self):
        self.log.append("resume")


class WatchAroundRequestsTests(unittest.TestCase):
    def test_watching_is_paused_for_a_request_and_resumed_after(self):
        live = session()
        log = []
        live.watcher = FakeWatcher(log)

        def handle(goal, fresh=True):
            log.append(("handle", goal, fresh))
            return Result(FINISHED)

        live.agent.handle = handle
        live.handle("open notepad")
        self.assertEqual(log, ["pause", ("handle", "open notepad", True), "resume"])

    def test_resumed_even_when_the_request_fails(self):
        live = session()
        log = []
        live.watcher = FakeWatcher(log)
        live.agent.handle = lambda goal, fresh=True: 1 / 0
        with self.assertRaises(ZeroDivisionError):
            live.handle("x")
        self.assertEqual(log, ["pause", "resume"])


class DesktopCheckTests(unittest.TestCase):
    """`nexus live --check` tries every reader the safety checks use, on the computer itself."""

    def test_everything_readable(self):
        lines = app_module.desktop_lines(
            front=lambda: "Untitled - Notepad",
            layout=lambda: [(None, (0, 0, 5, 5)), ("Untitled - Notepad", (0, 0, 800, 600))],
            under=lambda point: "Untitled - Notepad" if point == (400, 300) else None,
            held=frozenset,
        )
        self.assertEqual([level for level, _, _ in lines], ["ok"] * 4)
        self.assertIn("the middle of 'Untitled - Notepad' reads as 'Untitled - Notepad'", lines[2][2])

    def test_nothing_readable_says_what_it_costs(self):
        lines = app_module.desktop_lines(front=lambda: None, layout=lambda: None, under=lambda point: None, held=lambda: None)
        self.assertEqual([(level, name) for level, name, _ in lines],
                         [("warn", "Window in front"), ("warn", "Windows on the screen"), ("warn", "Keys held down")])
        self.assertIn("ask before every step", lines[0][2])

    def test_a_window_under_a_point_that_cannot_be_told(self):
        lines = app_module.desktop_lines(front=lambda: "x", layout=lambda: [("x", (0, 0, 10, 10))],
                                         under=lambda point: None, held=lambda: frozenset({"ctrl"}))
        self.assertEqual(lines[2][0], "warn")
        self.assertIn("now: ctrl", lines[3][2])


if __name__ == "__main__":
    unittest.main()
