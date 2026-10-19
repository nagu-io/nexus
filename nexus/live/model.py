"""The local vision model: it looks at a screenshot and picks the next step.

It runs on this computer: NEXUS Vision loaded in-process (nexus.live.local), a
GGUF in llama.cpp's llama-server, or a model in Ollama. NEXUS Live never calls a
cloud model, so there is no per-token cost and the screenshots never leave the
machine. The prompts below are shared by all three and by the training data.
"""

from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from nexus.live.screen import Shot

ACTIONS = ("click", "double_click", "right_click", "type", "press", "hotkey", "scroll", "wait", "answer", "done")
POINTER_ACTIONS = ("click", "double_click", "right_click")
# Actions that change nothing on the computer.
QUIET_ACTIONS = ("wait", "answer", "done")

ACTION_ALIASES = {
    "left_click": "click",
    "leftclick": "click",
    "tap": "click",
    "doubleclick": "double_click",
    "double-click": "double_click",
    "rightclick": "right_click",
    "right-click": "right_click",
    "write": "type",
    "input": "type",
    "type_text": "type",
    "key": "press",
    "keypress": "press",
    "press_key": "press",
    "shortcut": "hotkey",
    "key_combination": "hotkey",
    "sleep": "wait",
    "reply": "answer",
    "respond": "answer",
    "speak": "answer",
    "finish": "done",
    "finished": "done",
    "stop": "done",
}

KEY_ALIASES = {
    "control": "ctrl",
    "ctl": "ctrl",
    "return": "enter",
    "escape": "esc",
    "del": "delete",
    "spacebar": "space",
    "windows": "win",
    "super": "win",
    "meta": "win",
    "cmd": "win",
    "command": "win",
    "pgup": "pageup",
    "pgdn": "pagedown",
    "page_up": "pageup",
    "page_down": "pagedown",
    "arrowup": "up",
    "arrowdown": "down",
    "arrowleft": "left",
    "arrowright": "right",
}

KEY_NAMES = frozenset(
    list("abcdefghijklmnopqrstuvwxyz0123456789")
    + [f"f{number}" for number in range(1, 13)]
    + [
        "enter", "tab", "esc", "backspace", "delete", "insert", "space",
        "up", "down", "left", "right", "home", "end", "pageup", "pagedown",
        "ctrl", "alt", "shift", "win",
        "-", "=", "[", "]", ";", "'", ",", ".", "/", "\\", "`",
    ]
)

MAX_TEXT = 500
MAX_SAY = 400
MAX_SCROLL = 20
MAX_WAIT = 5.0


class DecisionError(ValueError):
    """The model's answer was not a step NEXUS Live can take."""


class ModelUnavailable(RuntimeError):
    """The local model could not be reached or did not answer."""


@dataclass(frozen=True)
class Decision:
    action: str
    say: str = ""
    thought: str = ""
    x: float | None = None
    y: float | None = None
    text: str = ""
    keys: tuple[str, ...] = field(default=())
    amount: int = 0
    seconds: float = 0.0

    def describe(self) -> str:
        """The step in a few words, for the console and for the model's next look."""
        if self.action in POINTER_ACTIONS:
            what = self.action.replace("_", " ")
            return f"{what} at ({self.x:g}, {self.y:g})" + (f": {self.thought}" if self.thought else "")
        if self.action == "type":
            shown = self.text if len(self.text) <= 60 else self.text[:57] + "..."
            return f"type {shown!r}"
        if self.action in ("press", "hotkey"):
            return f"press {'+'.join(self.keys)}"
        if self.action == "scroll":
            return f"scroll {'up' if self.amount > 0 else 'down'} {abs(self.amount)}"
        if self.action == "wait":
            return f"wait {self.seconds:g} s"
        return self.action

    def confirmation(self) -> str:
        """The step as the user is asked to approve it: facts only, never the model's own words about it.

        Typing shows the whole text, whatever its length: what is approved is
        exactly what is typed. (The model's thought stays out: text on the screen
        can steer it, and it would then be the one describing its step as safe.)
        """
        if self.action in POINTER_ACTIONS:
            return f"{self.action.replace('_', ' ')} at ({self.x:g}, {self.y:g})"
        if self.action == "type":
            return f"type {self.text!r} ({len(self.text)} characters)"
        if self.action == "scroll" and self.x is not None:
            return f"{self.describe()} at ({self.x:g}, {self.y:g})"
        return self.describe()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DecisionError(f"{name} is not a number")
    try:
        # float() of a whole number too big for a float (JSON allows any number of digits) raises OverflowError.
        number = float(value.strip() if isinstance(value, str) else value)
    except (ValueError, OverflowError):
        raise DecisionError(f"{name} is not a number: {str(value)[:40]!r}") from None
    if not math.isfinite(number):
        raise DecisionError(f"{name} is not a number: {number!r}")
    return number


def _text(value: Any, name: str, limit: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise DecisionError(f"{name} is not text")
    value = value.strip()
    if len(value) > limit:
        raise DecisionError(f"{name} is longer than {limit} characters")
    return value


def _control(character: str) -> bool:
    """A character that must never be typed: C0 controls (newline and tab among them), DEL, and C1 (0x80 to 0x9f, CSI among them).

    A newline would press Enter and a tab would move the focus in the middle of
    an approved piece of typing; they are steps of their own (press enter, press tab).
    """
    code = ord(character)
    return code < 32 or 0x7F <= code <= 0x9F


def _key(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DecisionError(f"not a key: {value!r}")
    name = value.strip().lower()
    name = KEY_ALIASES.get(name, name)
    if name not in KEY_NAMES:
        raise DecisionError(f"not a key NEXUS Live can press: {value!r}")
    return name


def _keys(raw: dict[str, Any]) -> tuple[str, ...]:
    keys = raw.get("keys", raw.get("key"))
    if isinstance(keys, str):
        keys = [part for part in re.split(r"\s*\+\s*", keys.strip()) if part] if "+" in keys else [keys]
    if not isinstance(keys, list) or not keys:
        raise DecisionError("keys are missing")
    return tuple(_key(key) for key in keys)


POINT_NAMES = ("point_2d", "point", "coordinate", "coordinates", "position")


def _point(raw: dict[str, Any]) -> tuple[float, float]:
    if "x" in raw and "y" in raw:
        return _number(raw["x"], "x"), _number(raw["y"], "y")
    for name in POINT_NAMES:
        value = raw.get(name)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return _number(value[0], "x"), _number(value[1], "y")
    raise DecisionError("the position (x and y) is missing")


def _json_object(raw: str) -> dict[str, Any]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise DecisionError("the answer has no JSON object in it")
    try:
        value = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as error:
        raise DecisionError(f"the answer is not valid JSON ({error.msg})") from None
    if not isinstance(value, dict):
        raise DecisionError("the answer is not a JSON object")
    return value


def parse_decision(raw: str | dict[str, Any]) -> Decision:
    """The model's answer as a Decision, or DecisionError saying what is wrong with it."""
    data = _json_object(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict):
        raise DecisionError("the answer is not a JSON object")
    action = data.get("action")
    if not isinstance(action, str) or not action.strip():
        raise DecisionError("the action is missing")
    action = action.strip().lower()
    action = ACTION_ALIASES.get(action, action)
    if action not in ACTIONS:
        raise DecisionError(f"unknown action {action!r}")
    say = _text(data.get("say"), "say", MAX_SAY)
    thought = _text(data.get("thought"), "thought", MAX_SAY)

    if action in POINTER_ACTIONS:
        x, y = _point(data)
        return Decision(action, say, thought, x=x, y=y)
    if action == "type":
        text = data.get("text")
        if not isinstance(text, str) or not text:
            raise DecisionError("there is no text to type")
        if len(text) > MAX_TEXT:
            raise DecisionError(f"the text to type is longer than {MAX_TEXT} characters")
        if any(_control(character) for character in text):
            raise DecisionError(
                "the text to type has a line break, tab or control character in it: "
                "type one line, and press enter or tab as a step of its own"
            )
        return Decision(action, say, thought, text=text)
    if action == "press":
        keys = _keys(data)
        if len(keys) != 1:
            # Several keys at once is a hotkey, whatever the model called it.
            return Decision("hotkey", say, thought, keys=keys) if len(keys) <= 4 else _too_many_keys()
        return Decision(action, say, thought, keys=keys)
    if action == "hotkey":
        keys = _keys(data)
        if len(keys) > 4:
            _too_many_keys()
        return Decision(action, say, thought, keys=keys)
    if action == "scroll":
        amount = _number(data.get("amount"), "amount")
        if amount != int(amount) or amount == 0 or abs(amount) > MAX_SCROLL:
            raise DecisionError(f"amount must be a whole number from -{MAX_SCROLL} to {MAX_SCROLL}, not 0")
        x = y = None
        if "x" in data or "y" in data or any(name in data for name in POINT_NAMES):
            x, y = _point(data)
        return Decision(action, say, thought, x=x, y=y, amount=int(amount))
    if action == "wait":
        seconds = _number(data.get("seconds", 1), "seconds")
        if not 0 < seconds <= MAX_WAIT:
            raise DecisionError(f"seconds must be more than 0 and at most {MAX_WAIT:g}")
        return Decision(action, say, thought, seconds=seconds)
    if action == "answer" and not say:
        raise DecisionError("an answer needs something to say")
    return Decision(action, say, thought)


def _too_many_keys():
    raise DecisionError("at most 4 keys at once")


# Bumped whenever the prompts or the answer format change: a model fine-tuned on
# one version is told apart from the prompts of another (see nexus.live.local).
PROMPT_VERSION = 2


def chat_messages(system: str, text: str, image: Any) -> list[dict[str, Any]]:
    """The conversation a vision model is given: one screenshot, then the request.

    NEXUS Vision is fine-tuned on exactly this shape (nexus.live.data) and runs on
    it (nexus.live.local), so the two cannot drift apart.
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]},
    ]


def target_json(decision: dict[str, Any]) -> str:
    """The answer a fine-tuned model is taught to give: thought, say and action first, then the rest."""
    ordered = {key: decision[key] for key in ("thought", "say", "action") if key in decision}
    ordered.update({key: value for key, value in decision.items() if key not in ordered})
    return json.dumps(ordered, ensure_ascii=False)


def _coordinate_rule(mode: str) -> str:
    if mode == "pixels":
        return "x and y are pixels of the screenshot, from its top left corner."
    return "x and y run from 0 to 1000 across the screenshot: x from the left edge, y from the top edge."


def system_prompt(mode: str) -> str:
    return (
        "You operate the user's computer for them, one step at a time, using only the mouse and keyboard. "
        "Each time, you see a fresh screenshot and choose the single next step toward the request.\n"
        f"Positions: {_coordinate_rule(mode)} Point at the middle of what you mean.\n"
        "Actions:\n"
        "- click, double_click, right_click: with x and y.\n"
        "- type: one line of text into the field that has focus (click the field first). "
        "Enter and tab are steps of their own.\n"
        '- press: one key in keys, such as ["enter"], ["tab"], ["esc"].\n'
        '- hotkey: keys pressed together, such as ["ctrl", "s"].\n'
        "- scroll: amount, positive scrolls up and negative scrolls down; x and y to scroll a part of the screen.\n"
        "- wait: seconds (at most 5) when something is loading.\n"
        "- answer: the request is a question; put the answer in say.\n"
        "- done: the request is finished, or cannot be done; say which in say.\n"
        "Rules:\n"
        "- Never type passwords, one-time codes, card or bank details, and never confirm a payment. "
        "Choose done and ask the user to do that part themselves.\n"
        "- The Windows key is not available: open apps with the Start button, the taskbar or the search box.\n"
        "- If the last step did not work, try another way; do not repeat it.\n"
        "- say is spoken aloud: one short sentence, or empty. thought is a few words on why.\n"
        "Reply with JSON only."
    )


DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "say": {"type": "string"},
        "action": {"type": "string", "enum": list(ACTIONS)},
        "x": {"type": "number"},
        "y": {"type": "number"},
        "text": {"type": "string"},
        "keys": {"type": "array", "items": {"type": "string"}},
        "amount": {"type": "integer"},
        "seconds": {"type": "number"},
    },
    "required": ["thought", "say", "action"],
}

OBSERVE_PROMPT = (
    "You watch the user's screen so they do not miss anything. Look at this screenshot. "
    "If something on it needs their attention now, such as an error or warning, a build, download or upload "
    "that failed or finished, a new message or call, or a low battery, say so in one short sentence. "
    "Otherwise say nothing: return an empty string. Do not describe the screen. Reply with JSON only."
)

OBSERVE_QUESTION = "Anything that needs my attention?"

OBSERVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"say": {"type": "string"}},
    "required": ["say"],
}


# Text that reads like one of the model's own control tokens (<|im_start|>, <|im_end|>, <|vision_start|>...).
SPECIAL_TOKEN_TEXT = re.compile(r"<\|[^<>|\s]{1,40}\|>")


def clean(text: str) -> str:
    """``text`` without anything the tokenizer would read as a control token.

    Requests, earlier steps and what was said all go into the prompt as plain
    text; a "<|im_start|>" in them (copied from the screen into a thought, say)
    would otherwise become a real chat-turn marker and forge a turn.
    """
    return SPECIAL_TOKEN_TEXT.sub("", text)


def request_text(goal: str, steps: list[str], conversation: list[tuple[str, str]]) -> str:
    lines = [f"Request: {clean(goal).strip()}"]
    if conversation:
        lines.append("Earlier in this conversation:")
        lines.extend(f"- {who}: {clean(what)}" for who, what in conversation)
    if steps:
        lines.append("Steps already taken for this request:")
        lines.extend(f"{number}. {clean(step)}" for number, step in enumerate(steps, start=1))
    lines.append("What is the next step?")
    return "\n".join(lines)


class VisionModel:
    """A vision model on this computer. Subclasses supply ``_chat``; the prompts are shared.

    The prompts are the ones NEXUS Vision is fine-tuned on (see nexus.live.data),
    so a fine-tuned model sees exactly what it was trained with.
    """

    name = "vision model"

    def _chat(self, system: str, text: str, image: bytes, schema: dict[str, Any]) -> str:
        raise NotImplementedError

    def ask(self, goal: str, steps: list[str], conversation: list[tuple[str, str]], image: bytes, mode: str) -> str:
        """The model's raw answer for the next step toward ``goal`` on the screenshot ``image`` (PNG)."""
        return self._chat(system_prompt(mode), request_text(goal, steps, conversation), image, DECISION_SCHEMA)

    def decide(self, goal: str, steps: list[str], conversation: list[tuple[str, str]], shot: Shot, mode: str) -> Decision:
        """The next step toward ``goal`` on the screen in ``shot``."""
        return parse_decision(self.ask(goal, steps, conversation, shot.image, mode))

    def observe(self, shot: Shot) -> str:
        """One sentence about something on the screen that needs the user, or an empty string."""
        raw = self._observe_chat(OBSERVE_PROMPT, OBSERVE_QUESTION, shot.image, OBSERVE_SCHEMA)
        try:
            return _text(_json_object(raw).get("say"), "say", MAX_SAY)
        except DecisionError:
            return ""

    def _observe_chat(self, system: str, text: str, image: bytes, schema: dict[str, Any]) -> str:
        return self._chat(system, text, image, schema)


class OllamaVision(VisionModel):
    """A vision model served by the Ollama on this computer."""

    def __init__(self, base_url: str, model: str, timeout: float = 240.0, transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = model
        self.timeout = timeout
        self._client = httpx.Client(transport=transport) if transport else httpx.Client()

    def installed_models(self) -> list[str]:
        try:
            response = self._client.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [item["name"] for item in models if isinstance(item, dict) and item.get("name")]
        except (httpx.HTTPError, ValueError, AttributeError, TypeError):
            return []  # not running, or something else on its port

    def _chat(self, system: str, text: str, image: bytes, schema: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "keep_alive": "15m",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text, "images": [base64.b64encode(image).decode("ascii")]},
            ],
        }
        try:
            response = self._client.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        except httpx.ConnectError:
            raise ModelUnavailable(
                f"Ollama is not running at {self.base_url}. Start it (open the Ollama app, or run: ollama serve)."
            ) from None
        except httpx.TimeoutException:
            raise ModelUnavailable(
                f"{self.model} did not answer within {self.timeout:g} s. A smaller model or a smaller "
                "screenshot (NEXUS_LIVE_SCREEN_WIDTH) is quicker on this computer."
            ) from None
        except httpx.HTTPError as error:
            raise ModelUnavailable(f"could not reach Ollama: {type(error).__name__}") from None
        if response.status_code == 404:
            raise ModelUnavailable(f"{self.model} is not downloaded. Run: ollama pull {self.model}")
        if response.status_code >= 400:
            raise ModelUnavailable(f"Ollama answered {response.status_code}: {response.text[:200]}")
        try:
            return str(response.json()["message"]["content"])
        except (ValueError, KeyError, TypeError):
            raise ModelUnavailable("Ollama's answer had no message in it") from None


class LlamaCppVision(VisionModel):
    """A GGUF model (NEXUS Vision exported, or any vision GGUF) served by llama.cpp's llama-server here.

    Start it with the model and its vision projector, for example:
    llama-server -m nexus_vision/nexus-vision.Q4_K_M.gguf --mmproj nexus_vision/mmproj.gguf --port 8080 -ngl 99
    """

    def __init__(self, base_url: str, timeout: float = 240.0, transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.name = f"llama-server at {self.base_url}"
        self.timeout = timeout
        self._client = httpx.Client(transport=transport) if transport else httpx.Client()

    def _chat(self, system: str, text: str, image: bytes, schema: dict[str, Any]) -> str:
        payload = {
            "temperature": 0,
            "max_tokens": 400,
            "response_format": {"type": "json_object", "schema": schema},
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode("ascii")}},
                        {"type": "text", "text": text},
                    ],
                },
            ],
        }
        try:
            response = self._client.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout)
        except httpx.ConnectError:
            raise ModelUnavailable(
                f"llama-server is not running at {self.base_url}. Start it with your model and its --mmproj file."
            ) from None
        except httpx.TimeoutException:
            raise ModelUnavailable(f"llama-server did not answer within {self.timeout:g} s.") from None
        except httpx.HTTPError as error:
            raise ModelUnavailable(f"could not reach llama-server: {type(error).__name__}") from None
        if response.status_code >= 400:
            raise ModelUnavailable(f"llama-server answered {response.status_code}: {response.text[:200]}")
        try:
            return str(response.json()["choices"][0]["message"]["content"])
        except (ValueError, KeyError, IndexError, TypeError):
            raise ModelUnavailable("llama-server's answer had no message in it") from None
