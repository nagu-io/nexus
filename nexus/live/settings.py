"""Settings for NEXUS Live, the local screen, voice and action assistant.

Everything NEXUS Live uses runs on this computer: the vision model (NEXUS
Vision, your own fine-tune, by default), speech to text with faster-whisper,
and speech through the system's own voice. Nothing is sent to a cloud model, so
there is no per-token cost.

Each setting can be set in .env (or the environment) with the name shown next
to it below.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Where the vision model runs. All three are on this computer.
#   local: NEXUS Vision (or any Hugging Face vision model folder) loaded in-process with transformers.
#   llamacpp: a GGUF served by llama.cpp's llama-server.
#   ollama: a model in Ollama.
BACKENDS = ("local", "llamacpp", "ollama")

# How the vision model gives screen positions.
#   normalized: 0 to 1000 across the screenshot, whatever its size (Qwen3-VL).
#   pixels: pixels of the screenshot it was sent (Qwen2.5-VL).
COORDINATE_MODES = ("normalized", "pixels")

# What needs the user's go-ahead before it happens.
#   all: every click, key and piece of typing.
#   risky: typing and keys; clicks and scrolling go ahead on their own.
CONFIRM_MODES = ("all", "risky")

# Windows NEXUS Live never acts in, matched as whole words in the title of the
# window in front, whatever case. Typing or clicking here could leak a
# secret, move money, or run a command; ask the user to do it themselves.
DEFAULT_BLOCKED_WINDOWS = (
    "password",
    "passwords",
    "keepass",
    "bitwarden",
    "1password",
    "lastpass",
    "credential manager",
    "windows security",
    "bank",
    "banking",
    "netbanking",
    "paypal",
    "upi",
    "wallet",
    "metamask",
    "powershell",
    "command prompt",
    "cmd.exe",
    "windows terminal",
    "terminal",
    "git bash",
    "mingw64",
    "mingw32",
    "msys2",
    "cygwin",
    "anaconda prompt",
    "miniforge prompt",
    "registry editor",
    "group policy",
    "task scheduler",
    "task manager",
    "user account control",
)

# Windows NEXUS Live never acts in, matched against the whole title: the Run box,
# the Start menu and Windows search run whatever is typed into them, and their
# titles are ordinary words that would block far too much as whole-word matches.
DEFAULT_BLOCKED_TITLES = (
    "run",
    "start",
    "search",
    "cortana",
    "task view",
    "task switching",
    "action center",
    "notification center",
    "quick settings",
    "services",
)


def _env(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else raw.strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_number(name: str, default: float, low: float, high: float, *, integer: bool = False) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, not {raw!r}") from None
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low:g} and {high:g}, not {value:g}")
    return value


def _env_list(name: str) -> tuple[str, ...]:
    raw = os.getenv(name) or ""
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def _choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = _env(name, default).lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}, not {value!r}")
    return value


def _plain(title: str | None) -> str:
    # Consoles show the command line as typed ("Command Prompt - nexus  live"): runs of spaces count as one.
    return " ".join((title or "").split()).lower()


def window_matches(title: str, words: tuple[str, ...]) -> str | None:
    """The first of ``words`` found as a whole word (or phrase) in ``title``, or None."""
    lowered = _plain(title)
    for word in words:
        if word and re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", lowered):
            return word
    return None


def title_is(title: str | None, titles: tuple[str, ...]) -> str | None:
    """The one of ``titles`` that is the whole of ``title`` (whatever case and spacing), or None."""
    lowered = _plain(title)
    for candidate in titles:
        if candidate and lowered == _plain(candidate):
            return candidate
    return None


def blocked_by(title: str | None, settings: "LiveSettings") -> str | None:
    """The blocked word or whole title that ``title`` matches, or None: a window NEXUS Live never acts or records in."""
    return window_matches(title or "", settings.blocked_windows) or title_is(title, settings.blocked_titles)


@dataclass(frozen=True)
class LiveSettings:
    backend: str = "local"  # NEXUS_LIVE_BACKEND: local, llamacpp or ollama
    model_dir: str = "~/.nexus/live/nexus_vision"  # NEXUS_LIVE_MODEL_DIR: the NEXUS Vision folder, for the local backend
    load_4bit: str = "auto"  # NEXUS_LIVE_LOAD_4BIT: auto (on for graphics cards under 8 GB), on, off
    llamacpp_url: str = "http://127.0.0.1:8080"  # NEXUS_LIVE_LLAMACPP_URL
    model: str = "auto"  # NEXUS_LIVE_MODEL: the Ollama model for the ollama backend, or auto to pick for this computer
    ollama_url: str = "http://localhost:11434"  # OLLAMA_BASE_URL
    coordinates: str = "normalized"  # NEXUS_LIVE_COORDINATES
    confirm: str = "all"  # NEXUS_LIVE_CONFIRM
    max_steps: int = 8  # NEXUS_LIVE_MAX_STEPS: actions for one request before it stops and asks
    screen_max_width: int = 1280  # NEXUS_LIVE_SCREEN_WIDTH: screenshots are shrunk to this before the model sees them
    allowed_windows: tuple[str, ...] = ()  # NEXUS_LIVE_ALLOWED_WINDOWS: if set, the only windows it may act in
    blocked_windows: tuple[str, ...] = DEFAULT_BLOCKED_WINDOWS  # plus NEXUS_LIVE_BLOCKED_WINDOWS
    blocked_titles: tuple[str, ...] = DEFAULT_BLOCKED_TITLES  # whole titles: the Run box, Start, search
    hotkey_talk: str = "<ctrl>+<alt>+<space>"  # NEXUS_LIVE_HOTKEY_TALK
    hotkey_stop: str = "<ctrl>+<alt>+q"  # NEXUS_LIVE_HOTKEY_STOP
    hotkey_yes: str = "<ctrl>+<alt>+y"  # NEXUS_LIVE_HOTKEY_YES
    hotkey_no: str = "<ctrl>+<alt>+n"  # NEXUS_LIVE_HOTKEY_NO
    whisper_model: str = "base"  # NEXUS_LIVE_WHISPER_MODEL: tiny, base, small, medium
    speak: bool = True  # NEXUS_LIVE_SPEAK
    dry_run: bool = False  # NEXUS_LIVE_DRY_RUN: say what it would do, and do nothing
    request_timeout: float = 240.0  # NEXUS_LIVE_TIMEOUT: seconds to wait for the model (a CPU is slow)
    confirm_timeout: float = 30.0  # NEXUS_LIVE_CONFIRM_TIMEOUT: no answer by then is a no
    settle_seconds: float = 0.8  # pause after an action so the screen can change before the next look
    watch_interval: float = 10.0  # NEXUS_LIVE_WATCH_INTERVAL: seconds between looks in watch mode
    watch_change: float = 0.06  # NEXUS_LIVE_WATCH_CHANGE: share of the screen that must change before it looks closer
    history_turns: int = 6  # earlier questions and answers it keeps, for follow-ups
    extra_blocked: tuple[str, ...] = field(default=())

    @classmethod
    def from_env(cls) -> "LiveSettings":
        """Read the settings from the environment (and .env, through nexus.config)."""
        from nexus.config import config

        extra = _env_list("NEXUS_LIVE_BLOCKED_WINDOWS")
        return cls(
            backend=_choice("NEXUS_LIVE_BACKEND", "local", BACKENDS),
            model_dir=_env("NEXUS_LIVE_MODEL_DIR", "~/.nexus/live/nexus_vision"),
            load_4bit=_choice("NEXUS_LIVE_LOAD_4BIT", "auto", ("auto", "on", "off")),
            llamacpp_url=_env("NEXUS_LIVE_LLAMACPP_URL", "http://127.0.0.1:8080"),
            model=_env("NEXUS_LIVE_MODEL", "auto"),
            ollama_url=_env("NEXUS_LIVE_OLLAMA_URL", config.ollama_base_url),
            coordinates=_choice("NEXUS_LIVE_COORDINATES", "normalized", COORDINATE_MODES),
            confirm=_choice("NEXUS_LIVE_CONFIRM", "all", CONFIRM_MODES),
            max_steps=int(_env_number("NEXUS_LIVE_MAX_STEPS", 8, 1, 50, integer=True)),
            screen_max_width=int(_env_number("NEXUS_LIVE_SCREEN_WIDTH", 1280, 480, 3840, integer=True)),
            allowed_windows=_env_list("NEXUS_LIVE_ALLOWED_WINDOWS"),
            blocked_windows=DEFAULT_BLOCKED_WINDOWS + extra,
            hotkey_talk=_env("NEXUS_LIVE_HOTKEY_TALK", "<ctrl>+<alt>+<space>"),
            hotkey_stop=_env("NEXUS_LIVE_HOTKEY_STOP", "<ctrl>+<alt>+q"),
            hotkey_yes=_env("NEXUS_LIVE_HOTKEY_YES", "<ctrl>+<alt>+y"),
            hotkey_no=_env("NEXUS_LIVE_HOTKEY_NO", "<ctrl>+<alt>+n"),
            whisper_model=_env("NEXUS_LIVE_WHISPER_MODEL", "base"),
            speak=_env_bool("NEXUS_LIVE_SPEAK", True),
            dry_run=_env_bool("NEXUS_LIVE_DRY_RUN", False),
            request_timeout=_env_number("NEXUS_LIVE_TIMEOUT", 240.0, 5, 3600),
            confirm_timeout=_env_number("NEXUS_LIVE_CONFIRM_TIMEOUT", 30.0, 3, 600),
            watch_interval=_env_number("NEXUS_LIVE_WATCH_INTERVAL", 10.0, 2, 3600),
            watch_change=_env_number("NEXUS_LIVE_WATCH_CHANGE", 0.06, 0.001, 1),
            extra_blocked=extra,
        )
