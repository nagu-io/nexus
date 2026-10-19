"""Ears and voice: push-to-talk recording, speech to text on this computer, and spoken replies.

Speech to text is faster-whisper (the model downloads once, about 150 MB for
"base", then works offline). Replies use the system's own voice through
pyttsx3: Windows' built-in voices, the Mac's, or espeak on Linux.
"""

from __future__ import annotations

import platform
import queue
import threading
import time
from typing import Any, Callable


def hotkey_words(hotkey: str) -> str:
    """'<ctrl>+<alt>+y' as 'Ctrl+Alt+Y', for saying out loud and printing."""
    parts = [part.strip().strip("<>") for part in hotkey.split("+") if part.strip()]
    return "+".join(part.capitalize() if len(part) > 1 else part.upper() for part in parts)


def _default_engine() -> Any:
    if platform.system() == "Windows":
        try:  # the speech engine is a COM object, made on the thread that uses it
            import comtypes  # type: ignore

            comtypes.CoInitialize()
        except Exception:
            pass
    import pyttsx3  # type: ignore

    engine = pyttsx3.init()
    engine.setProperty("rate", 185)
    return engine


class Speaker:
    """Says things out loud, and always prints them too.

    Speech runs on its own thread, so a long sentence never holds up the hotkeys
    or a request, and watch mode can speak from its thread too.
    """

    def __init__(
        self,
        enabled: bool = True,
        engine_factory: Callable[[], Any] = _default_engine,
        printer: Callable[[str], None] = print,
        background: bool = True,
    ):
        self.enabled = enabled
        self._factory = engine_factory
        self._print = printer
        self._background = background
        self._engine: Any = None
        self._failed = False
        self._lines: "queue.Queue[str | None]" = queue.Queue()
        self._thread: threading.Thread | None = None

    def _get_engine(self) -> Any:
        if self._engine is None and not self._failed:
            try:
                self._engine = self._factory()
            except Exception:
                self._failed = True
        return self._engine

    def _speak(self, text: str) -> None:
        engine = self._get_engine()
        if engine is None:
            return
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            self._failed, self._engine = True, None

    def _worker(self) -> None:
        while True:
            text = self._lines.get()
            if text is None:
                return
            self._speak(text)

    def say(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._print(f"NEXUS: {text}")
        if not self.enabled:
            return
        if not self._background:
            self._speak(text)
            return
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, name="nexus-live-speech", daemon=True)
            self._thread.start()
        self._lines.put(text)

    def stop(self) -> None:
        """Drop what has not been said yet, and cut off what is being said."""
        while not self._lines.empty():
            try:
                self._lines.get_nowait()
            except queue.Empty:
                break
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    def close(self) -> None:
        if self._thread is not None:
            self._lines.put(None)


class Recorder:
    """Records from the microphone until told to stop, the user goes quiet, or time runs out."""

    def __init__(self, sample_rate: int = 16000, max_seconds: float = 15.0, silence_seconds: float = 1.2, threshold: float = 0.012):
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.silence_seconds = silence_seconds
        self.threshold = threshold

    def record(self, stop: threading.Event) -> Any:
        import numpy as np  # type: ignore
        import sounddevice as sd  # type: ignore

        chunks: "queue.Queue[Any]" = queue.Queue()

        def callback(data, frames, timing, status):  # noqa: ANN001
            chunks.put(data.copy())

        collected: list[Any] = []
        heard_speech = False
        quiet_since: float | None = None
        started = time.monotonic()
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", callback=callback):
            while not stop.is_set() and time.monotonic() - started < self.max_seconds:
                try:
                    chunk = chunks.get(timeout=0.1)
                except queue.Empty:
                    continue
                collected.append(chunk)
                loud = float(np.sqrt(np.mean(chunk**2))) > self.threshold
                if loud:
                    heard_speech, quiet_since = True, None
                elif heard_speech:
                    quiet_since = quiet_since or time.monotonic()
                    if time.monotonic() - quiet_since >= self.silence_seconds:
                        break
        if not collected:
            return np.zeros(0, dtype="float32")
        return np.concatenate(collected).reshape(-1)


class Transcriber:
    """faster-whisper on this computer: on the graphics card if there is a CUDA one, else the processor."""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel  # type: ignore

            try:
                import ctranslate2  # type: ignore

                on_gpu = ctranslate2.get_cuda_device_count() > 0
            except Exception:
                on_gpu = False
            self._model = WhisperModel(
                self.model_size, device="cuda" if on_gpu else "cpu", compute_type="float16" if on_gpu else "int8"
            )
        return self._model

    def transcribe(self, audio: Any) -> str:
        if audio is None or len(audio) == 0:
            return ""
        segments, _ = self._get_model().transcribe(audio, beam_size=1, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()


class HotkeyConfirmer:
    """Asks before an action; the answer comes from the yes and no hotkeys, so focus never moves.

    Clicking a console to type "y" would bring the console to the front, and the
    action would then land somewhere else; a global hotkey leaves the window
    the action is for in front. No answer within the timeout is a no.
    """

    def __init__(self, speaker: Speaker, yes_hotkey: str, no_hotkey: str, timeout: float = 30.0):
        self.speaker = speaker
        self.yes_words = hotkey_words(yes_hotkey)
        self.no_words = hotkey_words(no_hotkey)
        self.timeout = timeout
        self._answers: "queue.Queue[bool]" = queue.Queue()

    def yes(self) -> None:
        self._answers.put(True)

    def no(self) -> None:
        self._answers.put(False)

    def ask(self, description: str) -> bool:
        while not self._answers.empty():  # a key pressed before the question is no answer to it
            try:
                self._answers.get_nowait()
            except queue.Empty:
                break
        self.speaker.say(f"I'll {description}. {self.yes_words} to go ahead, {self.no_words} to skip.")
        try:
            return self._answers.get(timeout=self.timeout)
        except queue.Empty:
            self.speaker.say("No answer, so I didn't do it.")
            return False
