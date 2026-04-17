from __future__ import annotations

import json
import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# COMPATIBILITY SHIM
# ──────────────────────────────────────────────────────────────────────
import torch

_SHIM_DTYPES = (
    [f"int{i}" for i in range(1, 8)]
    + [f"uint{i}" for i in range(1, 8)]
    + ["float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"]
)
for _attr in _SHIM_DTYPES:
    if not hasattr(torch, _attr):
        setattr(torch, _attr, torch.uint8)

import types
if not hasattr(torch.utils, "_pytree"):
    torch.utils._pytree = types.ModuleType("_pytree")
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda x: x
# ──────────────────────────────────────────────────────────────────────

from unsloth import FastLanguageModel
from datasets import load_dataset
from rich.console import Console
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

console = Console()

BASE_MODEL = "unsloth/Phi-3-mini-4k-instruct"
DATASET_PATH = Path("nexus_dataset.jsonl")
TRAIN_OUTPUT_DIR = Path("outputs") / "nexus_sft"
ADAPTER_OUTPUT_DIR = Path("lora_model")

# Tuned conservatively for a 4 GB RTX 3050 laptop GPU.
MAX_SEQ_LENGTH = 512
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 6
LEARNING_RATE = 2e-4


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install the CUDA torch build before training.")


def _load_model_and_tokenizer():
    console.print("[cyan]Loading base model and tokenizer (Unsloth)...[/cyan]")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer


def _prepare_peft_model(model):
    console.print("[cyan]Attaching LoRA adapters using Unsloth...[/cyan]")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    model.print_trainable_parameters()
    return model


def _format_dataset(tokenizer):
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found: {DATASET_PATH}")

    console.print(f"[cyan]Loading curated dataset from {DATASET_PATH}...[/cyan]")
    dataset = load_dataset("json", data_files=str(DATASET_PATH.resolve()), split="train")

    def render_chat(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    formatted = dataset.map(
        render_chat,
        remove_columns=dataset.column_names,
        desc="Formatting training chats",
    )
    console.print(f"[green]Prepared {len(formatted)} training examples[/green]")
    preview = formatted[0]["text"][:300].replace("\n", "\\n")
    console.print(f"[dim]Sample: {preview}...[/dim]")
    return formatted


def run_training() -> None:
    _require_cuda()
    model, tokenizer = _load_model_and_tokenizer()
    model = _prepare_peft_model(model)
    dataset = _format_dataset(tokenizer)

    TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = SFTConfig(
        output_dir=str(TRAIN_OUTPUT_DIR),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        warmup_steps=5,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    console.print("[cyan]Starting fine-tuning run...[/cyan]")
    trainer_stats = trainer.train()
    console.print("[green]Training complete[/green]")

    ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Saving LoRA adapter to {ADAPTER_OUTPUT_DIR}...[/cyan]")
    model.save_pretrained(str(ADAPTER_OUTPUT_DIR))
    tokenizer.save_pretrained(str(ADAPTER_OUTPUT_DIR))

    metrics_path = TRAIN_OUTPUT_DIR / "train_metrics.json"
    metrics = {key: float(value) if isinstance(value, (int, float)) else value for key, value in trainer_stats.metrics.items()}
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    console.print(f"[green]Saved metrics to {metrics_path}[/green]")


if __name__ == "__main__":
    run_training()
