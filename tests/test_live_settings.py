"""Tests for nexus.live.settings: reading NEXUS Live's settings and matching window titles."""

import dataclasses
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import nexus.config  # noqa: F401  (loads .env once, before any test patches the environment)
from nexus.live.settings import (
    BACKENDS,
    CONFIRM_MODES,
    COORDINATE_MODES,
    DEFAULT_BLOCKED_WINDOWS,
    LiveSettings,
    window_matches,
)

OLLAMA_URL = "http://ollama.test:11434"


def settings_from(**env: str) -> LiveSettings:
    """LiveSettings.from_env() with exactly ``env`` as the NEXUS_LIVE_* variables."""
    with patch.dict(os.environ, env, clear=False):
        for key in [key for key in os.environ if key.startswith("NEXUS_LIVE_") and key not in env]:
            del os.environ[key]
        with patch("nexus.config.config", SimpleNamespace(ollama_base_url=OLLAMA_URL)):
            return LiveSettings.from_env()


class DefaultsTests(unittest.TestCase):
    def test_dataclass_defaults(self):
        settings = LiveSettings()
        self.assertEqual(settings.backend, "local")
        self.assertEqual(settings.model_dir, "~/.nexus/live/nexus_vision")
        self.assertEqual(settings.load_4bit, "auto")
        self.assertEqual(settings.llamacpp_url, "http://127.0.0.1:8080")
        self.assertEqual(settings.model, "auto")
        self.assertEqual(settings.ollama_url, "http://localhost:11434")
        self.assertEqual(settings.coordinates, "normalized")
        self.assertEqual(settings.confirm, "all")
        self.assertEqual(settings.max_steps, 8)
        self.assertEqual(settings.screen_max_width, 1280)
        self.assertEqual(settings.allowed_windows, ())
        self.assertEqual(settings.blocked_windows, DEFAULT_BLOCKED_WINDOWS)
        self.assertEqual(settings.hotkey_talk, "<ctrl>+<alt>+<space>")
        self.assertEqual(settings.hotkey_stop, "<ctrl>+<alt>+q")
        self.assertEqual(settings.hotkey_yes, "<ctrl>+<alt>+y")
        self.assertEqual(settings.hotkey_no, "<ctrl>+<alt>+n")
        self.assertEqual(settings.whisper_model, "base")
        self.assertIs(settings.speak, True)
        self.assertIs(settings.dry_run, False)
        self.assertEqual(settings.request_timeout, 240.0)
        self.assertEqual(settings.confirm_timeout, 30.0)
        self.assertEqual(settings.settle_seconds, 0.8)
        self.assertEqual(settings.watch_interval, 10.0)
        self.assertEqual(settings.watch_change, 0.06)
        self.assertEqual(settings.history_turns, 6)
        self.assertEqual(settings.extra_blocked, ())

    def test_nothing_set_gives_the_defaults(self):
        self.assertEqual(settings_from(), LiveSettings(ollama_url=OLLAMA_URL))

    def test_blank_values_give_the_defaults(self):
        blank = {
            name: "   "
            for name in (
                "NEXUS_LIVE_BACKEND", "NEXUS_LIVE_MODEL_DIR", "NEXUS_LIVE_LOAD_4BIT", "NEXUS_LIVE_LLAMACPP_URL",
                "NEXUS_LIVE_MODEL", "NEXUS_LIVE_OLLAMA_URL", "NEXUS_LIVE_COORDINATES", "NEXUS_LIVE_CONFIRM",
                "NEXUS_LIVE_MAX_STEPS", "NEXUS_LIVE_SCREEN_WIDTH", "NEXUS_LIVE_ALLOWED_WINDOWS",
                "NEXUS_LIVE_BLOCKED_WINDOWS", "NEXUS_LIVE_HOTKEY_TALK", "NEXUS_LIVE_HOTKEY_STOP",
                "NEXUS_LIVE_HOTKEY_YES", "NEXUS_LIVE_HOTKEY_NO", "NEXUS_LIVE_WHISPER_MODEL", "NEXUS_LIVE_SPEAK",
                "NEXUS_LIVE_DRY_RUN", "NEXUS_LIVE_TIMEOUT", "NEXUS_LIVE_CONFIRM_TIMEOUT",
                "NEXUS_LIVE_WATCH_INTERVAL", "NEXUS_LIVE_WATCH_CHANGE",
            )
        }
        self.assertEqual(settings_from(**blank), LiveSettings(ollama_url=OLLAMA_URL))

    def test_ollama_url_comes_from_the_nexus_config_unless_set(self):
        self.assertEqual(settings_from().ollama_url, OLLAMA_URL)
        self.assertEqual(settings_from(NEXUS_LIVE_OLLAMA_URL=" http://gpu-box:11434 ").ollama_url, "http://gpu-box:11434")

    def test_settings_cannot_be_changed_once_made(self):
        settings = LiveSettings()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            settings.confirm = "risky"  # type: ignore[misc]


class ChoiceTests(unittest.TestCase):
    def test_the_choices(self):
        self.assertEqual(BACKENDS, ("local", "llamacpp", "ollama"))
        self.assertEqual(COORDINATE_MODES, ("normalized", "pixels"))
        self.assertEqual(CONFIRM_MODES, ("all", "risky"))

    def test_every_backend(self):
        for value in ("local", "llamacpp", "ollama"):
            with self.subTest(value=value):
                self.assertEqual(settings_from(NEXUS_LIVE_BACKEND=value).backend, value)

    def test_every_coordinate_mode(self):
        for value in ("normalized", "pixels"):
            with self.subTest(value=value):
                self.assertEqual(settings_from(NEXUS_LIVE_COORDINATES=value).coordinates, value)

    def test_every_confirm_mode(self):
        for value in ("all", "risky"):
            with self.subTest(value=value):
                self.assertEqual(settings_from(NEXUS_LIVE_CONFIRM=value).confirm, value)

    def test_every_4bit_choice(self):
        for value in ("auto", "on", "off"):
            with self.subTest(value=value):
                self.assertEqual(settings_from(NEXUS_LIVE_LOAD_4BIT=value).load_4bit, value)

    def test_choices_ignore_case_and_spaces(self):
        settings = settings_from(
            NEXUS_LIVE_BACKEND="  OLLAMA ", NEXUS_LIVE_COORDINATES="Pixels", NEXUS_LIVE_CONFIRM=" RISKY", NEXUS_LIVE_LOAD_4BIT="Off"
        )
        self.assertEqual((settings.backend, settings.coordinates, settings.confirm, settings.load_4bit), ("ollama", "pixels", "risky", "off"))

    def test_unknown_choices_are_refused_naming_the_variable(self):
        cases = {
            "NEXUS_LIVE_BACKEND": "cloud",
            "NEXUS_LIVE_COORDINATES": "percent",
            "NEXUS_LIVE_CONFIRM": "none",
            "NEXUS_LIVE_LOAD_4BIT": "yes",
        }
        for variable, value in cases.items():
            with self.subTest(variable=variable):
                with self.assertRaises(ValueError) as caught:
                    settings_from(**{variable: value})
                self.assertIn(variable, str(caught.exception))
                self.assertIn(repr(value), str(caught.exception))

    def test_unknown_backend_message(self):
        with self.assertRaises(ValueError) as caught:
            settings_from(NEXUS_LIVE_BACKEND="Cloud")
        self.assertEqual(str(caught.exception), "NEXUS_LIVE_BACKEND must be one of local, llamacpp, ollama, not 'cloud'")


# variable, field, lowest allowed, just below it, highest allowed, just above it
NUMBERS = (
    ("NEXUS_LIVE_MAX_STEPS", "max_steps", "1", "0", "50", "51"),
    ("NEXUS_LIVE_SCREEN_WIDTH", "screen_max_width", "480", "479", "3840", "3841"),
    ("NEXUS_LIVE_TIMEOUT", "request_timeout", "5", "4.99", "3600", "3600.01"),
    ("NEXUS_LIVE_CONFIRM_TIMEOUT", "confirm_timeout", "3", "2.99", "600", "600.01"),
    ("NEXUS_LIVE_WATCH_INTERVAL", "watch_interval", "2", "1.99", "3600", "3600.01"),
    ("NEXUS_LIVE_WATCH_CHANGE", "watch_change", "0.001", "0.0009", "1", "1.001"),
)


class NumberTests(unittest.TestCase):
    def test_the_edges_are_allowed(self):
        for variable, name, lowest, _, highest, _ in NUMBERS:
            for value in (lowest, highest):
                with self.subTest(variable=variable, value=value):
                    self.assertEqual(getattr(settings_from(**{variable: value}), name), float(value))

    def test_just_outside_the_edges_is_refused_naming_the_variable(self):
        for variable, _, _, below, _, above in NUMBERS:
            for value in (below, above):
                with self.subTest(variable=variable, value=value):
                    with self.assertRaises(ValueError) as caught:
                        settings_from(**{variable: value})
                    self.assertIn(variable, str(caught.exception))
                    self.assertIn("must be between", str(caught.exception))

    def test_out_of_range_messages(self):
        for env, message in (
            ({"NEXUS_LIVE_MAX_STEPS": "0"}, "NEXUS_LIVE_MAX_STEPS must be between 1 and 50, not 0"),
            ({"NEXUS_LIVE_MAX_STEPS": "51"}, "NEXUS_LIVE_MAX_STEPS must be between 1 and 50, not 51"),
            ({"NEXUS_LIVE_WATCH_CHANGE": "1.001"}, "NEXUS_LIVE_WATCH_CHANGE must be between 0.001 and 1, not 1.001"),
        ):
            with self.subTest(env=env):
                with self.assertRaises(ValueError) as caught:
                    settings_from(**env)
                self.assertEqual(str(caught.exception), message)

    def test_words_are_refused_naming_the_variable(self):
        for variable, *_ in NUMBERS:
            with self.subTest(variable=variable):
                with self.assertRaises(ValueError) as caught:
                    settings_from(**{variable: "soon"})
                self.assertEqual(str(caught.exception), f"{variable} must be a number, not 'soon'")

    def test_whole_number_settings_refuse_fractions(self):
        # Fractions inside the range, so only the whole-number rule can refuse them.
        for variable, value in (("NEXUS_LIVE_MAX_STEPS", "2.5"), ("NEXUS_LIVE_SCREEN_WIDTH", "600.5")):
            with self.subTest(variable=variable):
                with self.assertRaises(ValueError) as caught:
                    settings_from(**{variable: value})
                self.assertEqual(str(caught.exception), f"{variable} must be a number, not {value!r}")

    def test_whole_number_settings_are_ints(self):
        settings = settings_from(NEXUS_LIVE_MAX_STEPS=" 12 ", NEXUS_LIVE_SCREEN_WIDTH="1024")
        self.assertEqual((settings.max_steps, settings.screen_max_width), (12, 1024))
        self.assertIs(type(settings.max_steps), int)
        self.assertIs(type(settings.screen_max_width), int)

    def test_not_a_finite_number_is_refused(self):
        for variable in ("NEXUS_LIVE_TIMEOUT", "NEXUS_LIVE_CONFIRM_TIMEOUT", "NEXUS_LIVE_WATCH_INTERVAL", "NEXUS_LIVE_WATCH_CHANGE"):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(variable=variable, value=value):
                    with self.assertRaises(ValueError) as caught:
                        settings_from(**{variable: value})
                    self.assertIn(variable, str(caught.exception))

    def test_numbers_inside_the_range(self):
        settings = settings_from(
            NEXUS_LIVE_TIMEOUT="90.5", NEXUS_LIVE_CONFIRM_TIMEOUT="12", NEXUS_LIVE_WATCH_INTERVAL="30", NEXUS_LIVE_WATCH_CHANGE="0.25"
        )
        self.assertEqual(
            (settings.request_timeout, settings.confirm_timeout, settings.watch_interval, settings.watch_change),
            (90.5, 12.0, 30.0, 0.25),
        )


class FlagTests(unittest.TestCase):
    def test_words_for_yes(self):
        for value in ("1", "true", "yes", "on", "TRUE", " Yes ", "On"):
            with self.subTest(value=value):
                self.assertIs(settings_from(NEXUS_LIVE_DRY_RUN=value).dry_run, True)

    def test_words_for_no(self):
        for value in ("0", "false", "no", "off", "FALSE"):
            with self.subTest(value=value):
                self.assertIs(settings_from(NEXUS_LIVE_SPEAK=value).speak, False)

    def test_blank_flags_keep_their_defaults(self):
        settings = settings_from(NEXUS_LIVE_SPEAK="  ", NEXUS_LIVE_DRY_RUN="")
        self.assertIs(settings.speak, True)
        self.assertIs(settings.dry_run, False)


class TextSettingTests(unittest.TestCase):
    def test_each_text_setting_is_read_and_trimmed(self):
        settings = settings_from(
            NEXUS_LIVE_MODEL_DIR=" D:/models/nexus_vision ",
            NEXUS_LIVE_LLAMACPP_URL="http://127.0.0.1:9090",
            NEXUS_LIVE_MODEL="qwen3-vl:8b",
            NEXUS_LIVE_HOTKEY_TALK="<ctrl>+<shift>+<space>",
            NEXUS_LIVE_HOTKEY_STOP="<ctrl>+<shift>+q",
            NEXUS_LIVE_HOTKEY_YES="<ctrl>+<shift>+y",
            NEXUS_LIVE_HOTKEY_NO="<ctrl>+<shift>+n",
            NEXUS_LIVE_WHISPER_MODEL=" small ",
        )
        self.assertEqual(settings.model_dir, "D:/models/nexus_vision")
        self.assertEqual(settings.llamacpp_url, "http://127.0.0.1:9090")
        self.assertEqual(settings.model, "qwen3-vl:8b")
        self.assertEqual(settings.hotkey_talk, "<ctrl>+<shift>+<space>")
        self.assertEqual(settings.hotkey_stop, "<ctrl>+<shift>+q")
        self.assertEqual(settings.hotkey_yes, "<ctrl>+<shift>+y")
        self.assertEqual(settings.hotkey_no, "<ctrl>+<shift>+n")
        self.assertEqual(settings.whisper_model, "small")

    def test_allowed_windows_are_a_trimmed_lowercase_list(self):
        settings = settings_from(NEXUS_LIVE_ALLOWED_WINDOWS="  Notepad , ,Visual Studio Code ,  ")
        self.assertEqual(settings.allowed_windows, ("notepad", "visual studio code"))

    def test_extra_blocked_windows_are_added_to_the_defaults(self):
        settings = settings_from(NEXUS_LIVE_BLOCKED_WINDOWS="Slack, Discord ,")
        self.assertEqual(settings.extra_blocked, ("slack", "discord"))
        self.assertEqual(settings.blocked_windows, DEFAULT_BLOCKED_WINDOWS + ("slack", "discord"))
        self.assertEqual(window_matches("general - Slack", settings.blocked_windows), "slack")
        # The defaults cannot be switched off through this variable.
        self.assertEqual(window_matches("My Bank", settings.blocked_windows), "bank")


# A title for every default blocked word, and the word it must be refused for.
BLOCKED_TITLES = (
    ("Password Manager", "password"),
    ("Saved passwords - Google Chrome", "passwords"),
    ("Database.kdbx - KeePass", "keepass"),
    ("Bitwarden", "bitwarden"),
    ("1Password 8 - Vault", "1password"),
    ("LastPass", "lastpass"),
    ("Credential Manager", "credential manager"),
    ("Windows Security", "windows security"),
    ("My Bank - Statements", "bank"),
    ("Online Banking", "banking"),
    ("SBI NetBanking", "netbanking"),
    ("PayPal Checkout", "paypal"),
    ("Pay with UPI", "upi"),
    ("Crypto Wallet", "wallet"),
    ("MetaMask Notification", "metamask"),
    ("Windows PowerShell", "powershell"),
    ("Administrator: Command Prompt", "command prompt"),
    ("C:\\Windows\\system32\\cmd.exe", "cmd.exe"),
    ("Windows Terminal", "windows terminal"),
    ("Terminal - zsh", "terminal"),
    ("Registry Editor", "registry editor"),
    ("Local Group Policy Editor", "group policy"),
    ("Task Scheduler", "task scheduler"),
    ("User Account Control", "user account control"),
)


class DefaultBlockedWindowsTests(unittest.TestCase):
    def test_every_default_word_has_a_title_here(self):
        self.assertEqual(sorted(word for _, word in BLOCKED_TITLES), sorted(DEFAULT_BLOCKED_WINDOWS))

    def test_every_default_word_blocks_its_window(self):
        for title, word in BLOCKED_TITLES:
            with self.subTest(title=title):
                self.assertEqual(window_matches(title, DEFAULT_BLOCKED_WINDOWS), word)

    def test_ordinary_windows_are_not_blocked(self):
        for title in (
            "Untitled - Notepad",
            "Bankruptcy notes - Word",
            "Riverbank Cafe - Google Maps",
            "Groupie - Spotify",
            "Terminally Ill (film) - Wikipedia",
            "Inbox - Outlook",
            "",
        ):
            with self.subTest(title=title):
                self.assertIsNone(window_matches(title, DEFAULT_BLOCKED_WINDOWS))


class WindowMatchesTests(unittest.TestCase):
    def test_whole_words_only(self):
        self.assertEqual(window_matches("My Bank - Chrome", ("bank",)), "bank")
        self.assertIsNone(window_matches("Bankruptcy notes", ("bank",)))
        self.assertIsNone(window_matches("Riverbank Cafe", ("bank",)))

    def test_title_case_does_not_matter(self):
        for title in ("MY BANK", "my bank", "My Bank"):
            with self.subTest(title=title):
                self.assertEqual(window_matches(title, ("bank",)), "bank")

    def test_at_the_start_end_or_alone(self):
        for title in ("bank", "bank statement", "my bank"):
            with self.subTest(title=title):
                self.assertEqual(window_matches(title, ("bank",)), "bank")

    def test_digits_count_as_part_of_a_word(self):
        self.assertIsNone(window_matches("bank2", ("bank",)))
        self.assertIsNone(window_matches("2bank", ("bank",)))
        self.assertIsNone(window_matches("upi2 settings", ("upi",)))
        self.assertIsNone(window_matches("1password", ("password",)))
        self.assertIsNone(window_matches("password1", ("password",)))

    def test_letters_next_to_the_word(self):
        self.assertIsNone(window_matches("xbank", ("bank",)))
        self.assertIsNone(window_matches("bankx", ("bank",)))

    def test_a_word_with_digits(self):
        self.assertEqual(window_matches("1Password 8", ("1password",)), "1password")
        self.assertIsNone(window_matches("x1password", ("1password",)))
        self.assertIsNone(window_matches("21password", ("1password",)))
        self.assertIsNone(window_matches("1passwords", ("1password",)))

    def test_punctuation_ends_a_word(self):
        for title in ("(bank)", "bank:", "bank.", "my-bank", "bank_statement", "bank/2024"):
            with self.subTest(title=title):
                self.assertEqual(window_matches(title, ("bank",)), "bank")

    def test_phrases(self):
        self.assertEqual(window_matches("Administrator: Command Prompt", ("command prompt",)), "command prompt")
        self.assertIsNone(window_matches("Command Prompts explained", ("command prompt",)))
        self.assertIsNone(window_matches("commandprompt", ("command prompt",)))
        self.assertIsNone(window_matches("Command - Prompt", ("command prompt",)))

    def test_special_characters_are_matched_literally(self):
        self.assertEqual(window_matches("cmd.exe - ping", ("cmd.exe",)), "cmd.exe")
        self.assertIsNone(window_matches("cmdxexe", ("cmd.exe",)))
        self.assertEqual(window_matches("Learning C++ - Notes", ("c++",)), "c++")

    def test_the_first_word_in_the_list_wins(self):
        self.assertEqual(window_matches("Bank password reset", ("password", "bank")), "password")
        self.assertEqual(window_matches("Bank password reset", ("bank", "password")), "bank")

    def test_nothing_to_match(self):
        self.assertIsNone(window_matches(None, ("bank",)))  # type: ignore[arg-type]
        # A missing title is empty text, never a placeholder that some word could match.
        self.assertIsNone(window_matches(None, ("none", "null", "xxxx")))  # type: ignore[arg-type]
        self.assertIsNone(window_matches("", ("bank",)))
        self.assertIsNone(window_matches("My Bank", ()))

    def test_empty_words_are_skipped(self):
        self.assertIsNone(window_matches("- -", ("",)))
        self.assertIsNone(window_matches("", ("",)))
        self.assertEqual(window_matches("my bank", ("", "bank")), "bank")


if __name__ == "__main__":
    unittest.main()
