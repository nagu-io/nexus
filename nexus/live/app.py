"""Running NEXUS Live: putting the parts together, the hotkeys, and the console."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from nexus.live.agent import LiveAgent, Result
from nexus.live.executor import Executor
from nexus.live.hardware import detect, recommend_model
from nexus.live.local import LocalVision, card_problem, read_card
from nexus.live.model import LlamaCppVision, OllamaVision, VisionModel
from nexus.live.screen import capture
from nexus.live.settings import LiveSettings
from nexus.live.voice import HotkeyConfirmer, Recorder, Speaker, Transcriber, hotkey_filter, hotkey_words
from nexus.live.watch import ScreenWatcher

# Python packages Live needs, and what each is for.
PACKAGES = (
    ("mss", "screenshots"),
    ("PIL", "shrinking screenshots (Pillow)"),
    ("pyautogui", "mouse and keyboard"),
    ("pyperclip", "typing text that is not plain English"),
    ("pynput", "the hotkeys"),
    ("pyttsx3", "speaking (it uses the system's own voice)"),
    ("sounddevice", "the microphone"),
    ("numpy", "the microphone"),
    ("faster_whisper", "speech to text"),
)


VISION_PACKAGES = (("torch", "running NEXUS Vision"), ("transformers", "running NEXUS Vision"), ("peft", "loading its adapter"))


def resolve_model(settings: LiveSettings) -> tuple[str, str]:
    """The vision model to use, and why."""
    if settings.backend == "local":
        return f"NEXUS Vision ({settings.model_dir})", "your own fine-tuned model, run in this process"
    if settings.backend == "llamacpp":
        return f"llama-server at {settings.llamacpp_url}", "a GGUF served by llama.cpp"
    if settings.model.lower() != "auto":
        return settings.model, "set in NEXUS_LIVE_MODEL"
    hardware = detect()
    model, why = recommend_model(hardware)
    return model, f"picked for this computer ({hardware.describe()}): {why}"


def make_vision(settings: LiveSettings, model: str) -> VisionModel:
    """The vision model for the configured backend. Nothing is loaded or contacted until it is first asked."""
    if settings.backend == "local":
        return LocalVision(settings.model_dir, settings.coordinates, settings.load_4bit)
    if settings.backend == "llamacpp":
        return LlamaCppVision(settings.llamacpp_url, timeout=settings.request_timeout)
    return OllamaVision(settings.ollama_url, model, timeout=settings.request_timeout)


def apply_model_card(settings: LiveSettings) -> LiveSettings:
    """With the local backend, use the screenshot width NEXUS Vision was trained at, unless one is set."""
    if settings.backend != "local" or os.getenv("NEXUS_LIVE_SCREEN_WIDTH"):
        return settings
    try:
        card = read_card(Path(settings.model_dir).expanduser())
    except (OSError, ValueError):
        return settings
    if card is None or card.screen_width == settings.screen_max_width or not 480 <= card.screen_width <= 3840:
        return settings
    return replace(settings, screen_max_width=card.screen_width)


def _local_lines(settings: LiveSettings, hardware: Any = None) -> list[tuple[str, str, str]]:
    lines: list[tuple[str, str, str]] = []
    folder = Path(settings.model_dir).expanduser()
    problem = card_problem(folder, settings.coordinates)
    if problem:
        lines.append(("error", "NEXUS Vision", problem))
    else:
        detail = ""
        try:
            training = json.loads((folder / "training.json").read_text(encoding="utf-8"))
            detail = f", trained on {training.get('train_examples', '?')} examples"
        except (OSError, ValueError):
            pass
        lines.append(("ok", "NEXUS Vision", f"found in {folder}{detail}"))
    missing = [(name, why) for name, why in VISION_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        names = ", ".join(f"{name} ({why})" for name, why in missing)
        lines.append(("error", "Vision packages", f"missing {names}. Run: pip install -e .[vision]"))
    elif hardware is not None and getattr(hardware, "gpu_name", None) and not _torch_sees_cuda():
        # On Windows, pip's own torch has no CUDA: NEXUS Vision would then run on the processor.
        lines.append(
            (
                "warn",
                "Graphics card",
                f"{hardware.gpu_name} is here, but this torch was built without CUDA, so NEXUS Vision would run on the "
                "processor, slowly. Install the CUDA build of torch from https://pytorch.org/get-started/locally/ "
                "(your system, pip, CUDA).",
            )
        )
    return lines


def _torch_sees_cuda() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _llamacpp_lines(settings: LiveSettings) -> list[tuple[str, str, str]]:
    import httpx

    try:
        healthy = httpx.get(f"{settings.llamacpp_url.rstrip('/')}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        healthy = False
    if healthy:
        return [("ok", "llama-server", f"running at {settings.llamacpp_url}")]
    return [("error", "llama-server", f"not running at {settings.llamacpp_url}. Start it with your GGUF and its --mmproj file.")]


def _ollama_lines(settings: LiveSettings, model: str) -> list[tuple[str, str, str]]:
    vision = OllamaVision(settings.ollama_url, model, timeout=5)
    installed = vision.installed_models()
    online = bool(installed) or _ollama_online(settings.ollama_url)
    if not online:
        return [("error", "Ollama", f"not running at {settings.ollama_url}. Install it from ollama.com and open it.")]
    if model not in installed and f"{model}:latest" not in installed:
        return [("error", "Ollama", f"running, but {model} is not downloaded. Run: ollama pull {model}")]
    return [("ok", "Ollama", f"running, {model} is downloaded")]


def checks(settings: LiveSettings) -> list[tuple[str, str, str]]:
    """(level, name, message) lines for `nexus live --check`: ok, warn or error."""
    lines: list[tuple[str, str, str]] = []
    hardware = detect()
    lines.append(("ok", "This computer", hardware.describe()))
    model, why = resolve_model(settings)
    lines.append(("ok", "Vision model", f"{model} ({why})"))
    if settings.backend == "local":
        lines.extend(_local_lines(settings, hardware))
    elif settings.backend == "llamacpp":
        lines.extend(_llamacpp_lines(settings))
    else:
        lines.extend(_ollama_lines(settings, model))

    missing = [(name, why) for name, why in PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        names = ", ".join(f"{name} ({why})" for name, why in missing)
        lines.append(("error", "Python packages", f"missing {names}. Run: pip install -e .[live]"))
    else:
        lines.append(("ok", "Python packages", "all there"))
    if platform.system() == "Linux" and importlib.util.find_spec("tkinter") is None:
        # pyautogui imports mouseinfo, which ends the program on Linux when tkinter is missing.
        lines.append(("error", "tkinter", "pyautogui needs it on Linux. Run: sudo apt install python3-tk"))

    if importlib.util.find_spec("mss") and importlib.util.find_spec("PIL"):
        try:
            shot = capture(320)
            lines.append(("ok", "Screen", f"can see the main monitor ({shot.screen_width}x{shot.screen_height})"))
        except Exception as error:
            lines.append(("error", "Screen", f"cannot take a screenshot: {type(error).__name__}: {error}"))
    lines.extend(desktop_lines())
    lines.append(
        (
            "ok",
            "Safety",
            f"asks before {'every action' if settings.confirm == 'all' else 'typing and keys'}; "
            f"never acts in windows like: {', '.join(settings.blocked_windows[:6])} and {len(settings.blocked_windows) - 6} more"
            + (f"; only acts in: {', '.join(settings.allowed_windows)}" if settings.allowed_windows else ""),
        )
    )
    return lines


def desktop_lines(
    front: Callable[[], str | None] | None = None,
    layout: Callable[[], list | None] | None = None,
    under: Callable[[tuple[int, int]], str | None] | None = None,
    held: Callable[[], frozenset[str] | None] | None = None,
) -> list[tuple[str, str, str]]:
    """Whether the safety checks can read this desktop: the window in front, the windows on the screen,
    the window under a point, and the keys held down. What cannot be read makes NEXUS Live more careful."""
    from nexus.live import executor

    front, layout, under, held = (front or executor.foreground_title, layout or executor.window_layout,
                                  under or executor.window_at, held or executor.held_modifiers)
    lines: list[tuple[str, str, str]] = []
    title = front()
    if title is None:
        lines.append(("warn", "Window in front", "cannot be read: NEXUS Live will ask before every step, and teach mode records nothing"))
    else:
        lines.append(("ok", "Window in front", f"reads as {title!r}"))
    windows = layout()
    if windows is None:
        lines.append(("warn", "Windows on the screen", "cannot be read: teach mode cannot black out private windows in its screenshots"))
    else:
        lines.append(("ok", "Windows on the screen", f"{len(windows)} can be seen"))
    named = [window for window in windows or [] if window.title]
    if named:
        left, top, right, bottom = named[0].box
        seen = under(((left + right) // 2, (top + bottom) // 2))
        if seen is None:
            lines.append(("warn", "Window under a click", "cannot be told: with confirm=risky, clicks are asked about too"))
        else:
            lines.append(("ok", "Window under a click", f"the middle of {named[0].title!r} reads as {seen!r}"))
    keys = held()
    if keys is None:
        lines.append(("warn", "Keys held down", "cannot be read: after each yes it waits half a second for your hand to come off the keys"))
    else:
        lines.append(("ok", "Keys held down", "can be read" + (f" (now: {'+'.join(sorted(keys))})" if keys else "")))
    return lines


def _ollama_online(base_url: str) -> bool:
    import httpx

    try:
        return httpx.get(f"{base_url.rstrip('/')}/api/version", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


class LiveSession:
    """Everything NEXUS Live needs for one sitting, wired together."""

    def __init__(self, settings: LiveSettings, model: str, printer: Callable[[str], None] = print):
        self.settings = settings
        self.model = model
        self.print = printer
        self.speaker = Speaker(enabled=settings.speak, printer=printer)
        self.vision = make_vision(settings, model)
        self.confirmer = HotkeyConfirmer(self.speaker, settings.hotkey_yes, settings.hotkey_no, settings.confirm_timeout)
        self.executor = Executor(dry_run=settings.dry_run)
        self.agent = LiveAgent(
            settings,
            self.vision,
            self.executor,
            self.speaker,
            self.confirmer,
            # Unoptimized PNGs: the same pixels for the model, made several times faster (each step
            # takes two screenshots: the one the model sees, and one to check the screen before acting).
            capture=lambda: capture(settings.screen_max_width, optimize=False),
        )
        self.talk_requested = threading.Event()
        self.recording_stop = threading.Event()
        self.recording = threading.Event()
        self._hotkeys: Any = None
        self.watcher: ScreenWatcher | None = None

    # Hotkey callbacks: they run on pynput's thread, so they only set flags.
    def on_talk(self) -> None:
        if self.recording.is_set():
            self.recording_stop.set()
        else:
            self.talk_requested.set()

    def on_stop(self) -> None:
        # Also cancels a request still being spoken (listen_loop drops it), and a question waiting for an answer.
        self.recording_stop.set()
        self.agent.stop()
        self.confirmer.cancel()
        self.print("NEXUS: stopped.")

    def note_console(self) -> None:
        """Remember the window in front as NEXUS Live's own console (it is, while a request is typed there)."""
        self.agent.note_own_window(self.executor.foreground_title())

    def start_hotkeys(self) -> None:
        from pynput import keyboard  # type: ignore

        callbacks = {
            self.settings.hotkey_talk: self.on_talk,
            self.settings.hotkey_stop: self.on_stop,
            self.settings.hotkey_yes: self.confirmer.yes,
            self.settings.hotkey_no: self.confirmer.no,
        }
        options: dict[str, Any] = {}
        if platform.system() == "Windows":
            # Keep the hotkeys' last key from the window being worked on, where Ctrl+Alt+letter
            # is a shortcut, or a character on keyboards with AltGr (Ctrl+Alt+Q is @ in German).
            options["win32_event_filter"] = hotkey_filter(callbacks, lambda: self._hotkeys)
        self._hotkeys = keyboard.GlobalHotKeys(callbacks, **options)
        self._hotkeys.start()

    def start_watching(self) -> None:
        self.watcher = ScreenWatcher(
            self.vision,
            self.speaker,
            capture=lambda: capture(min(self.settings.screen_max_width, 1024), optimize=False),
            interval=self.settings.watch_interval,
            change=self.settings.watch_change,
        )
        threading.Thread(target=self.watcher.run, name="nexus-live-watch", daemon=True).start()

    def handle(self, goal: str, fresh: bool = True) -> Result:
        if self.watcher:
            self.watcher.pause()  # and wait for a look under way, so the request's screenshot is not old by the time it acts
        try:
            return self.agent.handle(goal, fresh=fresh)
        finally:
            if self.watcher:
                self.watcher.resume()

    def close(self) -> None:
        if self._hotkeys is not None:
            self._hotkeys.stop()
        if self.watcher:
            self.watcher.stop()
        self.speaker.close()
        if isinstance(self.vision, LocalVision):
            self.vision.close()

    def keys_help(self) -> str:
        s = self.settings
        return (
            f"{hotkey_words(s.hotkey_talk)} talk (press again when done), {hotkey_words(s.hotkey_stop)} stop, "
            f"{hotkey_words(s.hotkey_yes)} yes, {hotkey_words(s.hotkey_no)} no"
        )

    def listen_loop(self) -> None:
        """Voice mode: wait for the talk hotkey, record, transcribe, and handle what was said."""
        recorder = Recorder()
        transcriber = Transcriber(self.settings.whisper_model)
        self.speaker.say("Listening for the talk key.")
        while True:
            self.talk_requested.wait()
            self.talk_requested.clear()
            self.recording_stop.clear()
            self.agent.new_request()  # the stop key, from now on, cancels what is being said
            self.recording.set()
            self.print("(listening... press the talk key again, or stop talking)")
            try:
                audio = recorder.record(self.recording_stop)
            except Exception as error:  # a microphone that fails ends this request, never voice mode
                self.speaker.say(f"I couldn't record that ({type(error).__name__}: {error}).")
                continue
            finally:
                self.recording.clear()
            if self.agent.stop_event.is_set():
                self.print("(dropped what you said)")
                continue
            try:
                text = transcriber.transcribe(audio)
            except Exception as error:
                self.speaker.say(f"I couldn't make out what you said ({type(error).__name__}: {error}).")
                continue
            if self.agent.stop_event.is_set():
                self.print("(dropped what you said)")
                continue
            if not text:
                self.speaker.say("I didn't catch that.")
                continue
            self.print(f"You: {text}")
            if text.strip().lower().rstrip(".!") in {"goodbye", "exit", "quit", "stop listening"}:
                self.speaker.say("Goodbye.")
                return
            self.handle(text, fresh=False)  # a stop pressed since the talk key still counts

    def type_loop(self, read: Callable[[str], str] = input) -> None:
        """Text mode: read requests from the console."""
        while True:
            try:
                goal = read("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if goal.lower() in {"exit", "quit", "q"}:
                return
            if goal:
                self.note_console()  # the request was typed here, so this window is NEXUS Live's own
                self.handle(goal)


def session_settings(
    base: LiveSettings,
    *,
    dry_run: bool,
    model: str | None,
    confirm: str | None,
    backend: str | None = None,
) -> LiveSettings:
    changes: dict[str, Any] = {}
    if dry_run:
        changes["dry_run"] = True
    if backend:
        changes["backend"] = backend
    if model:
        changes["model"] = model
        if not backend and base.backend == "local":  # naming an Ollama model means the Ollama backend
            changes["backend"] = "ollama"
    if confirm:
        changes["confirm"] = confirm
    return apply_model_card(replace(base, **changes) if changes else base)
