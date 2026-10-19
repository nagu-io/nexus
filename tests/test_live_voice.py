"""Tests for nexus.live.voice: hotkey names, the speaker, and the yes/no hotkey confirmer."""

import queue
import sys
import threading
import unittest
from types import ModuleType
from unittest.mock import patch

import nexus.live.voice as voice_module
from nexus.live.voice import HotkeyConfirmer, Speaker, hotkey_words


class HotkeyWordsTests(unittest.TestCase):
    def test_default_hotkeys(self):
        self.assertEqual(hotkey_words("<ctrl>+<alt>+y"), "Ctrl+Alt+Y")
        self.assertEqual(hotkey_words("<ctrl>+<alt>+n"), "Ctrl+Alt+N")
        self.assertEqual(hotkey_words("<ctrl>+<alt>+q"), "Ctrl+Alt+Q")
        self.assertEqual(hotkey_words("<ctrl>+<alt>+<space>"), "Ctrl+Alt+Space")

    def test_single_characters_are_capitals_and_names_start_with_one(self):
        self.assertEqual(hotkey_words("a"), "A")
        self.assertEqual(hotkey_words("1"), "1")
        self.assertEqual(hotkey_words("<up>"), "Up")
        self.assertEqual(hotkey_words("<f5>"), "F5")
        self.assertEqual(hotkey_words("<f12>"), "F12")
        self.assertEqual(hotkey_words("<cmd>+k"), "Cmd+K")
        self.assertEqual(hotkey_words("<CTRL>+<ALT>+Y"), "Ctrl+Alt+Y")
        self.assertEqual(hotkey_words("<ctrl>+X"), "Ctrl+X")  # only the angle brackets are taken off

    def test_spaces_and_empty_parts_are_dropped(self):
        self.assertEqual(hotkey_words("<ctrl> + <shift> + q"), "Ctrl+Shift+Q")
        self.assertEqual(hotkey_words("<ctrl>++y"), "Ctrl+Y")
        self.assertEqual(hotkey_words(" + y + "), "Y")
        self.assertEqual(hotkey_words(""), "")


class FakeEngine:
    """pyttsx3's engine, recording what it was asked to say."""

    def __init__(self, fail_say=False, fail_stop=False):
        self.fail_say = fail_say
        self.fail_stop = fail_stop
        self.said = []
        self.runs = 0
        self.stops = 0

    def say(self, text):
        if self.fail_say:
            raise RuntimeError("no voice on this computer")
        self.said.append(text)

    def runAndWait(self):
        self.runs += 1

    def stop(self):
        self.stops += 1
        if self.fail_stop:
            raise RuntimeError("cannot stop")


class Factory:
    """Makes the engine (or fails to), counting the calls and the threads they came from."""

    def __init__(self, engine=None, error=None):
        self.engine = engine
        self.error = error
        self.threads = []

    @property
    def calls(self):
        return len(self.threads)

    def __call__(self):
        self.threads.append(threading.current_thread())
        if self.error:
            raise self.error
        return self.engine


def speaker(engine=None, error=None, **options):
    printed = []
    factory = Factory(engine if engine is not None else FakeEngine(), error)
    return Speaker(engine_factory=factory, printer=printed.append, **{"background": False, **options}), factory, printed


class SpeakerTests(unittest.TestCase):
    def test_says_and_prints(self):
        voice, factory, printed = speaker()
        voice.say("  Hello there.  ")
        self.assertEqual(printed, ["NEXUS: Hello there."])
        self.assertEqual(factory.engine.said, ["Hello there."])
        self.assertEqual(factory.engine.runs, 1)

    def test_nothing_to_say(self):
        voice, factory, printed = speaker()
        for text in ("", "   ", None):
            voice.say(text)
        self.assertEqual(printed, [])
        self.assertEqual(factory.calls, 0)

    def test_with_speech_off_it_only_prints(self):
        voice, factory, printed = speaker(enabled=False)
        voice.say("Hi.")
        self.assertEqual(printed, ["NEXUS: Hi."])
        self.assertEqual(factory.calls, 0)

    def test_the_engine_is_made_once(self):
        voice, factory, _ = speaker()
        voice.say("One.")
        voice.say("Two.")
        self.assertEqual(factory.calls, 1)
        self.assertEqual(factory.engine.said, ["One.", "Two."])
        self.assertEqual(factory.engine.runs, 2)

    def test_an_engine_that_cannot_be_made_is_not_tried_again(self):
        voice, factory, printed = speaker(error=RuntimeError("no espeak"))
        voice.say("One.")
        voice.say("Two.")
        self.assertEqual(printed, ["NEXUS: One.", "NEXUS: Two."])
        self.assertEqual(factory.calls, 1)

    def test_an_engine_that_fails_to_speak_is_dropped(self):
        engine = FakeEngine(fail_say=True)
        voice, factory, printed = speaker(engine)
        voice.say("One.")
        voice.say("Two.")
        self.assertEqual(printed, ["NEXUS: One.", "NEXUS: Two."])
        self.assertEqual((engine.runs, factory.calls), (0, 1))

    def test_stop_without_an_engine_makes_none(self):
        voice, factory, _ = speaker()
        voice.stop()
        self.assertEqual(factory.calls, 0)

    def test_stop_cuts_off_the_engine(self):
        voice, factory, _ = speaker()
        voice.say("A long sentence.")
        voice.stop()
        self.assertEqual(factory.engine.stops, 1)

    def test_stop_ignores_engine_errors(self):
        engine = FakeEngine(fail_stop=True)
        voice, _, _ = speaker(engine)
        voice.say("Hi.")
        voice.stop()
        self.assertEqual(engine.stops, 1)

    def test_close_without_a_thread_does_nothing(self):
        voice, factory, _ = speaker()
        voice.close()
        self.assertEqual(factory.calls, 0)


class DefaultEngineTests(unittest.TestCase):
    """The system voice through pyttsx3, with fake pyttsx3 and comtypes modules standing in."""

    def make(self, system, comtypes=None):
        engine = FakeEngine()
        engine.properties = {}
        engine.setProperty = engine.properties.__setitem__
        pyttsx3 = ModuleType("pyttsx3")
        pyttsx3.init = lambda: engine
        modules = {"pyttsx3": pyttsx3, "comtypes": comtypes}
        with patch.dict(sys.modules, modules), patch.object(voice_module.platform, "system", return_value=system):
            return voice_module._default_engine(), engine

    def fake_comtypes(self, fail=False):
        calls = []
        comtypes = ModuleType("comtypes")

        def co_initialize():
            calls.append("CoInitialize")
            if fail:
                raise OSError("COM is not available")

        comtypes.CoInitialize = co_initialize
        return comtypes, calls

    def test_system_voice_at_a_comfortable_rate(self):
        made, engine = self.make("Linux")
        self.assertIs(made, engine)
        self.assertEqual(engine.properties, {"rate": 185})

    def test_windows_sets_up_com_on_the_speech_thread(self):
        comtypes, calls = self.fake_comtypes()
        made, engine = self.make("Windows", comtypes)
        self.assertIs(made, engine)
        self.assertEqual(calls, ["CoInitialize"])

    def test_com_is_left_alone_elsewhere(self):
        comtypes, calls = self.fake_comtypes()
        self.make("Darwin", comtypes)
        self.assertEqual(calls, [])

    def test_com_trouble_does_not_stop_the_voice(self):
        comtypes, _ = self.fake_comtypes(fail=True)
        made, engine = self.make("Windows", comtypes)
        self.assertIs(made, engine)
        made, engine = self.make("Windows", None)  # comtypes not installed
        self.assertIs(made, engine)


class BackgroundSpeakerTests(unittest.TestCase):
    def test_speech_runs_in_order_on_one_daemon_thread(self):
        voice, factory, printed = speaker(background=True)
        voice.say("One.")
        thread = voice._thread
        voice.say("Two.")
        self.assertIs(voice._thread, thread)
        self.assertEqual(printed, ["NEXUS: One.", "NEXUS: Two."])  # printed at once, on the caller's thread
        voice.close()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(factory.engine.said, ["One.", "Two."])
        self.assertEqual((thread.name, thread.daemon), ("nexus-live-speech", True))
        self.assertEqual(factory.threads, [thread])  # the engine is made on the thread that uses it

    def test_stop_drops_what_has_not_been_said(self):
        entered, release = threading.Event(), threading.Event()

        class SlowEngine(FakeEngine):
            def say(self, text):
                super().say(text)
                if text == "One.":
                    entered.set()
                    release.wait(timeout=5)

        engine = SlowEngine()
        voice, _, _ = speaker(engine, background=True)
        voice.say("One.")
        self.assertTrue(entered.wait(timeout=5))
        voice.say("Two.")
        voice.say("Three.")
        voice.stop()
        release.set()
        thread = voice._thread
        voice.close()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(engine.said, ["One."])
        self.assertEqual(engine.stops, 1)


class ListeningSpeaker:
    """Hears the confirmer's question and, if told to, presses the answer hotkeys straight away."""

    def __init__(self):
        self.said = []
        self.on_question = None

    def say(self, text):
        self.said.append(text)
        if self.on_question and text.startswith("I'll "):
            self.on_question()


class RecordingQueue:
    """An answer queue that is always empty, recording how long it was asked to wait."""

    def __init__(self):
        self.timeouts = []

    def empty(self):
        return True

    def get(self, timeout=None):
        self.timeouts.append(timeout)
        raise queue.Empty


class HotkeyConfirmerTests(unittest.TestCase):
    def make(self, timeout=5.0, yes="<ctrl>+<alt>+y", no="<ctrl>+<alt>+n"):
        listener = ListeningSpeaker()
        return HotkeyConfirmer(listener, yes, no, timeout=timeout), listener

    def test_yes(self):
        confirmer, listener = self.make()
        listener.on_question = confirmer.yes
        self.assertIs(confirmer.ask("click at (500, 250)"), True)
        self.assertEqual(listener.said, ["I'll click at (500, 250). Ctrl+Alt+Y to go ahead, Ctrl+Alt+N to skip."])

    def test_no(self):
        confirmer, listener = self.make()
        listener.on_question = confirmer.no
        self.assertIs(confirmer.ask("type 'hello'"), False)
        self.assertEqual(listener.said, ["I'll type 'hello'. Ctrl+Alt+Y to go ahead, Ctrl+Alt+N to skip."])

    def test_hotkeys_from_the_settings(self):
        confirmer, listener = self.make(yes="<cmd>+<shift>+y", no="<f9>")
        listener.on_question = confirmer.yes
        confirmer.ask("press enter")
        self.assertEqual((confirmer.yes_words, confirmer.no_words), ("Cmd+Shift+Y", "F9"))
        self.assertEqual(listener.said, ["I'll press enter. Cmd+Shift+Y to go ahead, F9 to skip."])

    def test_a_yes_pressed_before_the_question_does_not_count(self):
        confirmer, listener = self.make()
        confirmer.yes()
        confirmer.yes()
        listener.on_question = confirmer.no
        self.assertIs(confirmer.ask("press enter"), False)

    def test_a_no_pressed_before_the_question_does_not_count(self):
        confirmer, listener = self.make()
        confirmer.no()
        listener.on_question = confirmer.yes
        self.assertIs(confirmer.ask("press enter"), True)

    def test_the_first_answer_counts_and_the_rest_are_dropped_before_the_next_question(self):
        confirmer, listener = self.make()
        listener.on_question = lambda: (confirmer.yes(), confirmer.no())
        self.assertIs(confirmer.ask("press enter"), True)
        listener.on_question = confirmer.yes
        self.assertIs(confirmer.ask("press tab"), True)

    def test_no_answer_in_time_is_a_no(self):
        confirmer, listener = self.make(timeout=0)
        self.assertIs(confirmer.ask("press enter"), False)
        self.assertEqual(listener.said, ["I'll press enter. Ctrl+Alt+Y to go ahead, Ctrl+Alt+N to skip.", "No answer, so I didn't do it."])

    def test_waits_as_long_as_the_setting(self):
        confirmer, _ = self.make(timeout=12.5)
        confirmer._answers = RecordingQueue()
        self.assertIs(confirmer.ask("press enter"), False)
        self.assertEqual(confirmer._answers.timeouts, [12.5])

    def test_default_wait(self):
        self.assertEqual(HotkeyConfirmer(ListeningSpeaker(), "<ctrl>+<alt>+y", "<ctrl>+<alt>+n").timeout, 30.0)


if __name__ == "__main__":
    unittest.main()
