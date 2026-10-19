"""Fine-tune NEXUS Vision: a LoRA adapter on Qwen3-VL-2B-Instruct, taught the steps NEXUS Live takes.

    python -m nexus.live.finetune --data data/vision --data ~/.nexus/live/teach --out nexus_vision

Every example is shown to the model as the exact conversation NEXUS Live has
with it at run time (nexus.live.data.to_messages), and it learns to give the
recorded step as its answer; the loss is on the answer only.

What lands in --out is what NEXUS Live's local backend loads: the adapter, the
processor, nexus_live.json (the prompt version it learned, the position format
and the screenshot width) and training.json (what it was trained on). With
--merge it also writes a standalone model next to it, for running fully offline
or converting to GGUF for llama.cpp.

Needs the vision extra: pip install -e .[vision]. On a graphics card under 12 GB,
or one without bfloat16 (a T4), the base model is loaded in 4 bits (QLoRA).
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from nexus.live.data import Example, read_examples, split, to_messages
from nexus.live.local import MODEL_CARD, ModelCard
from nexus.live.model import PROMPT_VERSION

DEFAULT_BASE = "Qwen/Qwen3-VL-2B-Instruct"
LANGUAGE_LAYERS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
VISION_LAYERS = r".*visual.*\.(qkv|proj|linear_fc1|linear_fc2)"
IGNORE_INDEX = -100
ANSWER_HEADER = "<|im_start|>assistant\n"
ANSWER_END = "<|im_end|>"


@dataclass
class TrainOptions:
    data: list[Path]
    out: Path
    base: str = DEFAULT_BASE
    epochs: float = 1.0
    max_steps: int = -1  # stop after this many optimizer steps (for a quick check), -1 for whole epochs
    learning_rate: float = 1e-4
    batch_size: int = 2
    grad_accum: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    tune_vision: bool = False  # also adapt the vision encoder, not just the language model
    eval_fraction: float = 0.1
    four_bit: str = "auto"  # auto, on, off
    compute: str = "auto"  # auto, bf16, fp16, fp32: auto is bf16 where the card has it, else fp16 falling back to fp32
    max_width: int = 1280
    seed: int = 7
    save_steps: int = 200
    log_steps: int = 10
    workers: int = 2


@dataclass
class TrainResult:
    out: Path
    train_examples: int
    eval_examples: int
    skipped: dict[str, int]
    steps: int
    train_loss: float | None
    eval_loss: float | None
    seconds: float
    precision: str
    sources: dict[str, int] = field(default_factory=dict)


class TrainingDataError(ValueError):
    """The training data cannot be used as it is."""


def load_items(folders: Sequence[Path]) -> tuple[list[tuple[Path, Example]], dict[str, int]]:
    """The usable examples of each dataset folder, and how many were skipped and why."""
    items: list[tuple[Path, Example]] = []
    skipped = {"old prompt version": 0, "invalid": 0, "image missing": 0}
    for folder in folders:
        folder = Path(folder).expanduser()
        for example in read_examples(folder):
            if example.prompt_version != PROMPT_VERSION:
                skipped["old prompt version"] += 1
                continue
            try:
                example.check()
            except ValueError:
                skipped["invalid"] += 1
                continue
            if not (folder / example.image).is_file():
                skipped["image missing"] += 1
                continue
            items.append((folder, example))
    return items, skipped


def load_image(path: Path, max_width: int) -> Any:
    from PIL import Image  # type: ignore

    with Image.open(path) as picture:
        picture = picture.convert("RGB")
        if picture.width > max_width:
            picture = picture.resize((max_width, max(1, round(picture.height * max_width / picture.width))), Image.LANCZOS)
        return picture


def _last_run(values: list[int], run: list[int]) -> int | None:
    """The index just after the last place ``run`` occurs in ``values``, or None."""
    size = len(run)
    for start in range(len(values) - size, -1, -1):
        if values[start : start + size] == run:
            return start + size
    return None


class AnswerCollator:
    """Turns examples into a batch the model learns from: the whole conversation in, loss on the answer only."""

    def __init__(self, processor: Any, max_width: int):
        self.processor = processor
        self.max_width = max_width
        tokenizer = processor.tokenizer
        self.header = tokenizer.encode(ANSWER_HEADER, add_special_tokens=False)
        self.end_id = tokenizer.convert_tokens_to_ids(ANSWER_END)
        if not self.header or self.end_id is None:
            raise TrainingDataError("the processor's tokenizer has no chat tokens; is the base model a Qwen3-VL?")
        # transformers 5 takes the padding option in processor_kwargs; 4.x takes it directly.
        parameters = inspect.signature(processor.apply_chat_template).parameters
        self.padding = {"processor_kwargs": {"padding": True}} if "processor_kwargs" in parameters else {"padding": True}

    def __call__(self, batch: list[tuple[Path, Example]]) -> dict[str, Any]:
        import torch  # type: ignore

        conversations = [
            to_messages(example, load_image(folder / example.image, self.max_width), with_answer=True)
            for folder, example in batch
        ]
        encoded = self.processor.apply_chat_template(
            conversations, tokenize=True, return_dict=True, return_tensors="pt", **self.padding
        )
        input_ids = encoded["input_ids"]
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for row, ids in enumerate(input_ids.tolist()):
            start = _last_run(ids, self.header)
            end = ids.index(self.end_id, start) if start is not None and self.end_id in ids[start:] else None
            if start is None or end is None:
                continue  # no whole answer in it: nothing to learn from this one
            # The answer and the end-of-turn token after it: the model learns to say it and stop.
            labels[row, start : end + 1] = input_ids[row, start : end + 1]
        encoded["labels"] = labels
        return dict(encoded)


@dataclass(frozen=True)
class Precision:
    device: str
    dtype_name: str  # bfloat16, float16, float32
    four_bit: bool

    def describe(self) -> str:
        return f"{self.device}, {self.dtype_name}" + (", 4-bit base (QLoRA)" if self.four_bit else "")


COMPUTE_TYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def native_bf16(torch: Any) -> bool:
    """Whether the first graphics card does bfloat16 in hardware (Ampere, compute capability 8, and later).

    torch.cuda.is_bf16_supported() says yes on older cards too (a T4, Kaggle's and
    Colab's free GPU) because it counts emulation, which is far too slow to train with.
    """
    major, _minor = torch.cuda.get_device_capability(0)
    return major >= 8


def pick_precision(four_bit: str, compute: str = "auto") -> Precision:
    import torch  # type: ignore

    if torch.cuda.is_available():
        bf16 = native_bf16(torch)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        dtype = COMPUTE_TYPES.get(compute) or ("bfloat16" if bf16 else "float16")
        if dtype == "bfloat16" and not bf16:
            raise TrainingDataError("This graphics card has no bfloat16 in hardware: use --compute fp16 or fp32.")
        quantize = four_bit == "on" or (four_bit == "auto" and (memory_gb < 12 or not bf16))
        return Precision("cuda", dtype, quantize)
    if four_bit == "on":
        raise TrainingDataError("4-bit loading needs an NVIDIA graphics card (bitsandbytes).")
    return Precision("cpu", "float32", False)


class NonFiniteLoss(Exception):
    """The training loss became NaN or infinite: the numbers overflowed."""


def free_memory() -> None:
    """Give back the memory of models no longer used, on the graphics card too.

    A trained model is not freed when the last name for it goes: accelerate's
    mixed precision leaves it in a reference cycle (its forward is a method
    bound to itself), which only the garbage collector breaks. Without this,
    NEXUS Vision is evaluated next to the model it was trained from, which a
    4 GB card has no room for.
    """
    import gc

    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def loss_is_bad(logs: dict[str, Any] | None) -> bool:
    loss = (logs or {}).get("loss")
    return isinstance(loss, (int, float)) and not math.isfinite(loss)


def _guard(printer: Callable[[str], None] | None = None) -> Any:
    """A callback that reports the training loss and stops training at the first NaN or infinite one."""
    from transformers import TrainerCallback  # type: ignore

    class LossGuard(TrainerCallback):
        def __init__(self) -> None:
            self.tripped = False

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
            loss = (logs or {}).get("loss")
            if printer is not None and loss is not None and state is not None:
                printer(f"step {state.global_step} of {state.max_steps}: loss {loss:.4f}")
            if loss_is_bad(logs):
                self.tripped = True
                control.should_training_stop = True
            return control

    return LossGuard()


def load_base(base: str, precision: Precision) -> tuple[Any, Any]:
    import torch  # type: ignore
    from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore

    dtype = getattr(torch, precision.dtype_name)
    options: dict[str, Any] = {"dtype": dtype}
    if precision.four_bit:
        from transformers import BitsAndBytesConfig  # type: ignore

        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        options["device_map"] = {"": 0}
    model = AutoModelForImageTextToText.from_pretrained(base, **options)
    if precision.device == "cuda" and not precision.four_bit:
        model = model.to("cuda")
    processor = AutoProcessor.from_pretrained(base)
    processor.tokenizer.padding_side = "right"
    return model, processor


def add_adapter(model: Any, options: TrainOptions, precision: Precision) -> Any:
    from peft import LoraConfig, get_peft_model  # type: ignore

    if precision.four_bit:
        from peft import prepare_model_for_kbit_training  # type: ignore

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    targets = f"({LANGUAGE_LAYERS}|{VISION_LAYERS})" if options.tune_vision else LANGUAGE_LAYERS
    config = LoraConfig(
        r=options.lora_rank,
        lora_alpha=options.lora_alpha,
        lora_dropout=options.lora_dropout,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def write_card(folder: Path, base: str, max_width: int) -> None:
    card = ModelCard(prompt_version=PROMPT_VERSION, coordinates="normalized", base_model=base, screen_width=max_width)
    (Path(folder) / MODEL_CARD).write_text(card.to_json() + "\n", encoding="utf-8")


def train(options: TrainOptions, printer: Callable[[str], None] = print) -> TrainResult:
    """Fine-tune on the examples in ``options.data`` and write NEXUS Vision into ``options.out``."""
    started = time.monotonic()
    items, skipped = load_items(options.data)
    if not items:
        raise TrainingDataError(f"No usable examples in {', '.join(str(folder) for folder in options.data)}.")
    by_example = {id(example): (folder, example) for folder, example in items}
    train_examples, eval_examples = split([example for _, example in items], options.eval_fraction)
    train_items = [by_example[id(example)] for example in train_examples]
    eval_items = [by_example[id(example)] for example in eval_examples]
    if not train_items:
        raise TrainingDataError("Every example landed in the held-out set; add more examples or lower --eval-fraction.")
    sources: dict[str, int] = {}
    for _, example in train_items:
        sources[example.source] = sources.get(example.source, 0) + 1
    printer(f"Training on {len(train_items)} examples, holding out {len(eval_items)}. Sources: {sources}")
    if any(skipped.values()):
        printer(f"Skipped: {skipped}")

    out = Path(options.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    precision = pick_precision(options.four_bit, options.compute)
    try:
        model, processor, output, eval_loss = _fit(options, precision, train_items, eval_items, out, printer)
    except NonFiniteLoss:
        if not (precision.dtype_name == "float16" and options.compute == "auto"):
            raise TrainingDataError(
                "The training loss became NaN. Try a lower --learning-rate, or --compute fp32."
            ) from None
        printer("The loss overflowed in float16 on this graphics card. Starting again in float32: slower, but it holds.")
        shutil.rmtree(out / "checkpoints", ignore_errors=True)
        precision = replace(precision, dtype_name="float32")
        try:
            model, processor, output, eval_loss = _fit(options, precision, train_items, eval_items, out, printer)
        except NonFiniteLoss:
            raise TrainingDataError("The training loss became NaN even in float32. Try a lower --learning-rate.") from None

    model.save_pretrained(out)
    processor.save_pretrained(out)
    write_card(out, options.base, options.max_width)
    del model, processor
    free_memory()
    result = TrainResult(
        out=out,
        train_examples=len(train_items),
        eval_examples=len(eval_items),
        skipped=skipped,
        steps=int(output.global_step),
        train_loss=float(output.training_loss) if output.training_loss is not None else None,
        eval_loss=eval_loss,
        seconds=round(time.monotonic() - started, 1),
        precision=precision.describe(),
        sources=sources,
    )
    summary = {**asdict(result), "out": str(out), "options": {**asdict(options), "data": [str(path) for path in options.data], "out": str(out)}}
    (out / "training.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    printer(f"NEXUS Vision written to {out} in {result.seconds:g} s.")
    return result


def _fit(
    options: TrainOptions,
    precision: Precision,
    train_items: list[tuple[Path, Example]],
    eval_items: list[tuple[Path, Example]],
    out: Path,
    printer: Callable[[str], None],
) -> tuple[Any, Any, Any, float | None]:
    """One training run at ``precision``: the adapter-wrapped model, the processor, the run's output, the held-out loss."""
    import torch  # type: ignore
    from transformers import Trainer, TrainingArguments  # type: ignore

    random.seed(options.seed)
    torch.manual_seed(options.seed)
    printer(f"Loading {options.base} ({precision.describe()})...")
    model, processor = load_base(options.base, precision)
    model = add_adapter(model, options, precision)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    printer(f"Learning {trainable:,} adapter weights.")

    steps_per_epoch = math.ceil(len(train_items) / (options.batch_size * options.grad_accum))
    total_steps = options.max_steps if options.max_steps > 0 else math.ceil(steps_per_epoch * options.epochs)
    arguments = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=options.batch_size,
        per_device_eval_batch_size=options.batch_size,
        gradient_accumulation_steps=options.grad_accum,
        num_train_epochs=options.epochs,
        max_steps=options.max_steps,
        learning_rate=options.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(total_steps * 0.03)),  # a whole number of steps works on every transformers
        weight_decay=0.0,
        logging_steps=options.log_steps,
        logging_first_step=True,  # an overflow shows up at once, not ten steps in
        # Report NaN and infinite losses as they are. By default Trainer logs the running average in their
        # place, so a run that overflowed from the first step logged 0.0 and finished "normally".
        logging_nan_inf_filter=False,
        save_strategy="steps",
        save_steps=options.save_steps,
        save_total_limit=2,
        bf16=precision.dtype_name == "bfloat16",
        fp16=precision.dtype_name == "float16",
        gradient_checkpointing=False,  # already on, set up for the adapter in add_adapter
        remove_unused_columns=False,
        prediction_loss_only=True,  # the held-out loss only: keeping every logit would fill memory
        dataloader_num_workers=options.workers,
        report_to="none",
        seed=options.seed,
        use_cpu=precision.device == "cpu",
    )
    guard = _guard(printer)
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_items,
        eval_dataset=eval_items or None,
        data_collator=AnswerCollator(processor, options.max_width),
        callbacks=[guard],
    )
    printer(f"About {steps_per_epoch} steps per epoch.")
    output = trainer.train()
    overflowed = guard.tripped or (output.training_loss is not None and not math.isfinite(output.training_loss))
    eval_loss = None
    if eval_items and not overflowed:
        eval_loss = float(trainer.evaluate()["eval_loss"])
        printer(f"Held-out loss: {eval_loss:.4f}")
        overflowed = not math.isfinite(eval_loss)
    if overflowed:
        del trainer, model
        free_memory()
        raise NonFiniteLoss()
    return model, processor, output, eval_loss


def merge(adapter_dir: Path, out_dir: Path, printer: Callable[[str], None] = print) -> Path:
    """The adapter folded into its base model: one standalone model folder, for offline use or GGUF."""
    import torch  # type: ignore
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore

    adapter_dir, out_dir = Path(adapter_dir).expanduser(), Path(out_dir).expanduser()
    base = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))["base_model_name_or_path"]
    printer(f"Merging {adapter_dir} into {base}...")
    # On the CPU, in bfloat16: the dtype Qwen3-VL ships in, whatever graphics card this computer has.
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    AutoProcessor.from_pretrained(str(adapter_dir)).save_pretrained(out_dir)
    shutil.copy2(adapter_dir / MODEL_CARD, out_dir / MODEL_CARD)
    printer(f"Standalone model written to {out_dir}.")
    return out_dir


def use_one_gpu() -> None:
    """Train on the first graphics card only. With several (Kaggle's two T4s), Trainer would wrap the
    model to split each batch across them, which a 4-bit model does not survive. Call before torch loads."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def main(argv: Sequence[str] | None = None) -> int:
    use_one_gpu()
    parser = argparse.ArgumentParser(prog="nexus vision train", description="Fine-tune NEXUS Vision on NEXUS Live examples.")
    parser.add_argument("--data", type=Path, action="append", required=True, help="a dataset folder (repeat for more)")
    parser.add_argument("--out", type=Path, default=Path("nexus_vision"))
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--tune-vision", action="store_true")
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--four-bit", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--compute", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--width", type=int, default=1280, help="screenshot width, the same NEXUS Live uses")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2, help="image loading processes")
    parser.add_argument("--merge", type=Path, help="also write a standalone merged model to this folder")
    args = parser.parse_args(argv)
    options = TrainOptions(
        data=args.data,
        out=args.out,
        base=args.base,
        epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        tune_vision=args.tune_vision,
        eval_fraction=args.eval_fraction,
        four_bit=args.four_bit,
        compute=args.compute,
        max_width=args.width,
        seed=args.seed,
        workers=args.workers,
    )
    train(options)
    if args.merge:
        merge(args.out, args.merge)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
