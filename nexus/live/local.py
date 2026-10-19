"""NEXUS Vision on this computer, loaded in-process with transformers: no server and no network.

The model folder is what nexus.live.finetune writes: either a LoRA adapter
(adapter_config.json, on top of the base model it names) or a merged model
(everything in the folder, the way to run fully offline). Next to the weights
it writes nexus_live.json, the model card that says which prompts it learned.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from nexus.live.model import PROMPT_VERSION, ModelUnavailable, VisionModel, chat_messages

MODEL_CARD = "nexus_live.json"


@dataclass(frozen=True)
class ModelCard:
    prompt_version: int
    coordinates: str
    base_model: str
    screen_width: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "prompt_version": self.prompt_version,
                "coordinates": self.coordinates,
                "base_model": self.base_model,
                "screen_width": self.screen_width,
            },
            indent=2,
        )


def read_card(model_dir: Path) -> ModelCard | None:
    path = Path(model_dir) / MODEL_CARD
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ModelCard(
        prompt_version=int(data.get("prompt_version", 0)),
        coordinates=str(data.get("coordinates", "normalized")),
        base_model=str(data.get("base_model", "")),
        screen_width=int(data.get("screen_width", 1280)),
    )


def card_problem(model_dir: Path, coordinates: str) -> str | None:
    """Why the model in ``model_dir`` would not fit this NEXUS Live, or None."""
    model_dir = Path(model_dir).expanduser()
    if not model_dir.is_dir():
        return (
            f"No NEXUS Vision model in {model_dir}. Download yours with `nexus vision pull YOUR_NAME/nexus-vision`, "
            "train one (docs/NEXUS_VISION.md), or set NEXUS_LIVE_BACKEND=ollama to use a model in Ollama meanwhile."
        )
    if not ((model_dir / "adapter_config.json").exists() or (model_dir / "config.json").exists()):
        return f"{model_dir} has no model in it (no adapter_config.json or config.json)."
    card = read_card(model_dir)
    if card is None:
        return None  # a plain Hugging Face model folder: it runs, with the default prompts
    if card.prompt_version != PROMPT_VERSION:
        return (
            f"The model in {model_dir} was trained on prompt version {card.prompt_version}, and this NEXUS Live "
            f"uses version {PROMPT_VERSION}. Train it again on the current data."
        )
    if card.coordinates != coordinates:
        return f"The model in {model_dir} gives {card.coordinates} positions; set NEXUS_LIVE_COORDINATES={card.coordinates}."
    return None


# Files that make a folder a model folder: an adapter, a whole model, NEXUS Vision's card, or what is left of
# any of them (a download that stopped part-way leaves some files and not others).
MODEL_FILES = (
    "adapter_config.json", "adapter_model.safetensors", "config.json", "model.safetensors",
    "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json",
    "preprocessor_config.json", "chat_template.jinja", MODEL_CARD, ".cache/huggingface",
)
PULL_MARK = ".nexus-pull"  # in a download folder: the repo it is a download of


def is_model_folder(folder: Path) -> bool:
    return any((Path(folder) / name).exists() for name in MODEL_FILES)


def pull_model(
    repo: str,
    folder: Path,
    download: Callable[..., Any],
    retries: int = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download ``repo`` and put it in ``folder``, in place of the model there, all at once.

    It goes into a download folder beside ``folder`` (kept until it is moved in,
    so a download that stops picks up where it was at the next try), and only a
    complete download replaces the old model, whole: downloading over it left
    an adapter's files next to a merged model's, and the old adapter kept
    loading. If moving the new one in fails, the old one is put back. A folder
    that holds other things and no model is refused rather than emptied.
    """
    folder = Path(folder).expanduser()
    if folder.exists() and not folder.is_dir():
        raise ValueError(f"{folder} is a file, not a folder, so it was left alone.")
    if folder.exists() and any(folder.iterdir()) and not is_model_folder(folder):
        raise ValueError(
            f"{folder} has other things in it and no model, so it was left alone. Give an empty folder, "
            "or the folder NEXUS Vision is in."
        )
    folder.parent.mkdir(parents=True, exist_ok=True)
    staging = folder.with_name(f".{folder.name}.download")
    mark = staging / PULL_MARK
    if staging.exists() and not (mark.is_file() and mark.read_text(encoding="utf-8").strip() == repo):
        shutil.rmtree(staging)  # a download of something else: start again
    staging.mkdir(parents=True, exist_ok=True)
    mark.write_text(repo, encoding="utf-8")
    download(repo, local_dir=str(staging), ignore_patterns=["checkpoints/*"])
    previous = None
    if folder.exists():
        previous = folder.with_name(f".{folder.name}.old-{uuid.uuid4().hex[:8]}")
        _rename(folder, previous, retries, sleep)
    try:
        _rename(staging, folder, retries, sleep)
    except OSError:
        if previous is not None:
            _rename(previous, folder, retries, sleep)  # the old model back where it was
        raise
    (folder / PULL_MARK).unlink(missing_ok=True)
    if previous is not None:
        shutil.rmtree(previous, ignore_errors=True)
    return folder


def _rename(source: Path, target: Path, retries: int, sleep: Callable[[float], None]) -> None:
    for attempt in range(retries):
        try:
            source.rename(target)
            return
        except PermissionError:  # on Windows a file just written can still be open (a virus scan, the indexer)
            if attempt == retries - 1:
                raise
            sleep(0.5)


def _want_4bit(setting: str) -> bool:
    if setting == "on":
        return True
    if setting == "off":
        return False
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3
    except Exception:
        return False


def _compute_dtype(torch: Any) -> Any:
    """bfloat16 on graphics cards that do it in hardware (an RTX 30 series and later), else float16.

    Qwen3-VL ships in bfloat16; float16 can overflow, so it is only the fallback for older cards.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def load_options(torch: Any, load_4bit: str) -> dict[str, Any]:
    """How from_pretrained loads the model on this computer: which device, which number format, 4 bits or not."""
    options: dict[str, Any] = {"dtype": "auto"}
    if torch.cuda.is_available():
        options["device_map"] = "auto"
        # Qwen3-VL ships in bfloat16, which cards before the RTX 30 series (a T4) only emulate, slowly.
        options["dtype"] = _compute_dtype(torch)
    if _want_4bit(load_4bit):
        from transformers import BitsAndBytesConfig  # type: ignore

        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=_compute_dtype(torch)
        )
        options["device_map"] = {"": 0}
        # The layers that stay whole (embeddings, norms, the vision tower) use this too. Left out,
        # transformers 4.57 picks float16 for them even on a card with bfloat16, where float16 can overflow.
        options["dtype"] = _compute_dtype(torch)
    return options


class LocalVision(VisionModel):
    """A vision model loaded in this process. ``model_dir`` is a NEXUS Vision folder (checked against its
    model card), or, with ``check_card=False``, any Qwen3-VL folder or Hub name, such as the base model."""

    def __init__(
        self,
        model_dir: str | Path,
        coordinates: str = "normalized",
        load_4bit: str = "auto",
        max_new_tokens: int = 300,
        check_card: bool = True,
    ):
        self.model_dir = Path(model_dir).expanduser() if check_card else model_dir
        self.coordinates = coordinates
        self.load_4bit = load_4bit
        self.max_new_tokens = max_new_tokens
        self.check_card = check_card
        self.name = f"NEXUS Vision ({model_dir})" if check_card else str(model_dir)
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.RLock()
        # The last answer as generated: how many new tokens, and their text with the special tokens left in,
        # so an answer that is only an end-of-turn marker can be told from one that failed.
        self.last_generation: dict[str, Any] = {}

    def load(self) -> None:
        """Load the model now rather than at the first question (so the first answer is not timed with the load)."""
        with self._lock:
            self._load()

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.check_card:
            problem = card_problem(Path(self.model_dir), self.coordinates)
            if problem:
                raise ModelUnavailable(problem)
        try:
            import torch  # type: ignore
            from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore
        except ImportError:
            raise ModelUnavailable("The local backend needs torch and transformers. Run: pip install -e .[vision]") from None

        options = load_options(torch, self.load_4bit)

        folder = Path(self.model_dir)
        adapter_file = folder / "adapter_config.json"
        if folder.is_dir() and adapter_file.exists():
            base = json.loads(adapter_file.read_text(encoding="utf-8")).get("base_model_name_or_path", "")
            if not base:
                raise ModelUnavailable(f"{adapter_file} does not name its base model.")
            try:
                from peft import PeftModel  # type: ignore
            except ImportError:
                raise ModelUnavailable("Loading an adapter needs peft. Run: pip install -e .[vision]") from None

            model = AutoModelForImageTextToText.from_pretrained(base, **options)
            model = PeftModel.from_pretrained(model, str(folder))
            processor_source = str(folder) if _has_processor(folder) else base
        else:
            model = AutoModelForImageTextToText.from_pretrained(str(self.model_dir), **options)
            processor_source = str(self.model_dir)
        model.eval()
        self._processor = AutoProcessor.from_pretrained(processor_source)
        self._model = model

    def _chat(self, system: str, text: str, image: bytes, schema: dict[str, Any]) -> str:
        return self._generate(system, text, image, adapter=True)

    def _observe_chat(self, system: str, text: str, image: bytes, schema: dict[str, Any]) -> str:
        # Watch mode asks the base model: NEXUS Vision was taught to pick steps, not
        # what on a screen is worth mentioning, and the base model does that well.
        return self._generate(system, text, image, adapter=False)

    def _generate(self, system: str, text: str, image: bytes, adapter: bool) -> str:
        with self._lock:  # watch mode and requests share one model: one generation at a time
            self.last_generation = {}
            self._load()
            import torch  # type: ignore
            from PIL import Image  # type: ignore

            picture = Image.open(io.BytesIO(image)).convert("RGB")
            inputs = self._processor.apply_chat_template(
                chat_messages(system, text, picture),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device)
            without_adapter = not adapter and hasattr(self._model, "disable_adapter")
            with torch.inference_mode(), (self._model.disable_adapter() if without_adapter else contextlib.nullcontext()):
                output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            new_tokens = output[:, inputs["input_ids"].shape[1] :]
            self.last_generation = {
                "new_tokens": int(new_tokens.shape[1]),
                "raw": self._processor.batch_decode(new_tokens, skip_special_tokens=False)[0],
            }
            return self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def close(self) -> None:
        """Free the model's memory (and the graphics card's)."""
        with self._lock:
            self._model = None
            self._processor = None
        try:
            import gc

            import torch  # type: ignore

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _has_processor(folder: Path) -> bool:
    return any((folder / name).exists() for name in ("processor_config.json", "preprocessor_config.json"))
