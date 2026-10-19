"""Tests for nexus.live.watch: watch mode looks at the screen and speaks up only when something new needs the user."""

import importlib.util
import io
import unittest

from nexus.live.model import ModelUnavailable
from nexus.live.screen import Shot, thumbnail_difference
from nexus.live.watch import ScreenWatcher

HAS_PILLOW = importlib.util.find_spec("PIL") is not None


def shot(image):
    return Shot(image=image, width=64, height=36, screen_width=64, screen_height=36)


class FakeObserver:
    """Says the given sentences in turn (or raises the given exception), and remembers what it was shown."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.seen = []

    def observe(self, screenshot):
        self.seen.append(screenshot)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


class Differences:
    """How much changed, for each look in turn, recording what was compared."""

    def __init__(self, *values):
        self.values = list(values)
        self.compared = []

    def __call__(self, before, after):
        self.compared.append((before, after))
        return self.values.pop(0)


class Screens:
    """Hands out the given screenshots in turn."""

    def __init__(self, *images):
        self.shots = [shot(image) for image in images]
        self.count = 0

    def __call__(self):
        self.count += 1
        return self.shots.pop(0)


def watcher(observer, differences, images, change=0.06, interval=10.0):
    speaker = FakeSpeaker()
    screens = Screens(*images)
    return ScreenWatcher(observer, speaker, screens, interval=interval, change=change, difference=differences), speaker, screens


class LookOnceTests(unittest.TestCase):
    def test_defaults(self):
        watch = ScreenWatcher(FakeObserver(), FakeSpeaker(), Screens())
        self.assertEqual((watch.interval, watch.change), (10.0, 0.06))
        self.assertIs(watch.difference, thumbnail_difference)
        self.assertFalse(watch.paused.is_set())

    def test_first_look_speaks_up(self):
        observer = FakeObserver("  The build failed.  ")
        differences = Differences(1.0)
        watch, speaker, _ = watcher(observer, differences, [b"A"])
        self.assertEqual(watch.look_once(), "The build failed.")
        self.assertEqual(speaker.said, ["The build failed."])
        self.assertEqual(differences.compared, [(None, b"A")])
        self.assertEqual(observer.seen, [shot(b"A")])

    def test_small_change_is_not_looked_at(self):
        observer = FakeObserver("Something.")
        watch, speaker, _ = watcher(observer, Differences(0.0599), [b"A"], change=0.06)
        self.assertEqual(watch.look_once(), "")
        self.assertEqual((observer.seen, speaker.said), ([], []))

    def test_change_of_exactly_the_threshold_is_looked_at(self):
        observer = FakeObserver("A download finished.")
        watch, speaker, _ = watcher(observer, Differences(0.06), [b"A"], change=0.06)
        self.assertEqual(watch.look_once(), "A download finished.")
        self.assertEqual(len(observer.seen), 1)

    def test_change_over_the_threshold_is_looked_at(self):
        observer = FakeObserver("A new message.")
        watch, _, _ = watcher(observer, Differences(0.0601), [b"A"], change=0.06)
        self.assertEqual(watch.look_once(), "A new message.")

    def test_each_look_is_compared_with_the_one_before(self):
        differences = Differences(1.0, 0.0, 0.0)
        watch, _, _ = watcher(FakeObserver("Low battery."), differences, [b"A", b"B", b"C"])
        for _ in range(3):
            watch.look_once()
        self.assertEqual(differences.compared, [(None, b"A"), (b"A", b"B"), (b"B", b"C")])

    def test_same_sentence_is_said_once(self):
        observer = FakeObserver("Low battery.", "Low battery.")
        watch, speaker, _ = watcher(observer, Differences(1.0, 1.0), [b"A", b"B"])
        self.assertEqual([watch.look_once(), watch.look_once()], ["Low battery.", ""])
        self.assertEqual(speaker.said, ["Low battery."])

    def test_a_sentence_can_come_back_after_another(self):
        observer = FakeObserver("Low battery.", "A new message.", "Low battery.")
        watch, speaker, _ = watcher(observer, Differences(1.0, 1.0, 1.0), [b"A", b"B", b"C"])
        for _ in range(3):
            watch.look_once()
        self.assertEqual(speaker.said, ["Low battery.", "A new message.", "Low battery."])

    def test_nothing_to_say_is_silent_and_does_not_forget_the_last_sentence(self):
        observer = FakeObserver("Low battery.", "   ", "", "Low battery.")
        watch, speaker, _ = watcher(observer, Differences(1.0, 1.0, 1.0, 1.0), [b"A", b"B", b"C", b"D"])
        self.assertEqual([watch.look_once() for _ in range(4)], ["Low battery.", "", "", ""])
        self.assertEqual(speaker.said, ["Low battery."])

    def test_unavailable_model_is_silent_and_watching_goes_on(self):
        observer = FakeObserver(ModelUnavailable("Ollama is not running"), "The upload finished.")
        watch, speaker, _ = watcher(observer, Differences(1.0, 1.0), [b"A", b"B"])
        self.assertEqual(watch.look_once(), "")
        self.assertEqual(speaker.said, [])
        self.assertEqual(watch.look_once(), "The upload finished.")


class ScriptedStop:
    """Stands in for the stop event: wait() returns the scripted answers in turn and can run a step first."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.timeouts = []

    def wait(self, timeout=None):
        self.timeouts.append(timeout)
        stop, before = self.steps.pop(0)
        if before:
            before()
        return stop


class RunTests(unittest.TestCase):
    def make(self, looks):
        observer = FakeObserver(*[f"Something new {number}." for number in range(looks)])
        images = [str(number).encode() for number in range(looks)]
        return watcher(observer, Differences(*[1.0] * looks), images, interval=7.5)

    def test_looks_after_every_wait_until_stopped(self):
        watch, speaker, screens = self.make(2)
        watch._stop = ScriptedStop((False, None), (False, None), (True, None))
        watch.run()
        self.assertEqual(screens.count, 2)
        self.assertEqual(speaker.said, ["Something new 0.", "Something new 1."])
        self.assertEqual(watch._stop.timeouts, [7.5, 7.5, 7.5])

    def test_paused_while_a_request_is_handled(self):
        watch, _, screens = self.make(0)
        watch.paused.set()
        watch._stop = ScriptedStop((False, None), (False, None), (True, None))
        watch.run()
        self.assertEqual(screens.count, 0)

    def test_pausing_and_carrying_on(self):
        watch, _, screens = self.make(2)
        watch._stop = ScriptedStop(
            (False, None),
            (False, watch.paused.set),
            (False, watch.paused.clear),
            (True, None),
        )
        watch.run()
        self.assertEqual(screens.count, 2)

    def test_stopped_before_it_starts(self):
        watch, _, screens = self.make(0)
        watch.stop()
        watch.run()  # returns at once: the real event is already set, so the 7.5 s wait does not happen
        self.assertEqual(screens.count, 0)


@unittest.skipUnless(HAS_PILLOW, "needs Pillow")
class RealDifferenceTests(unittest.TestCase):
    def png(self, grey):
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("L", (64, 36), grey).save(buffer, format="PNG")
        return buffer.getvalue()

    def test_still_screen_costs_nothing(self):
        dark, light = self.png(0), self.png(255)
        observer = FakeObserver("An error dialog.", "Your build is done.")
        speaker = FakeSpeaker()
        screens = Screens(dark, dark, light)
        watch = ScreenWatcher(observer, speaker, screens)
        self.assertEqual([watch.look_once() for _ in range(3)], ["An error dialog.", "", "Your build is done."])
        self.assertEqual(len(observer.seen), 2)


if __name__ == "__main__":
    unittest.main()
