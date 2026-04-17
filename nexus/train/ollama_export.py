from __future__ import annotations
import os, subprocess, types
from pathlib import Path

# ── COMPATIBILITY SHIM ──
import torch
for _a in [f"int{i}" for i in range(1,8)] + [f"uint{i}" for i in range(1,8)] + ["float8_e4m3fn","float8_e5m2","float8_e4m3fnuz","float8_e5m2fnuz"]:
    if not hasattr(torch, _a): setattr(torch, _a, torch.uint8)
if not hasattr(torch.utils, "_pytree"):
    torch.utils._pytree = types.ModuleType("_pytree")
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda x: x
# ── END SHIM ──

from rich.console import Console
from unsloth import FastLanguageModel

console = Console()
MERGED_MODEL_DIR = Path("nexus_model")
GGUF_BASENAME = "nexus_model-unsloth.Q4_K_M.gguf"
UNSLOTH_LLAMA_CPP_PATH = r"D:\unsloth\llama.cpp"

MODELFILE_BODY = """TEMPLATE \"\"\"<|system|>
{{ .System }}<|end|>
<|user|>
{{ .Prompt }}<|end|>
<|assistant|>\"\"\"
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
"""


def export_and_bridge_to_ollama():
    os.environ.setdefault("UNSLOTH_LLAMA_CPP_PATH", UNSLOTH_LLAMA_CPP_PATH)
    Path(UNSLOTH_LLAMA_CPP_PATH).parent.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]Loading trained adapters...[/cyan]")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )

    # Step 1: Try GGUF compilation
    gguf_path = None
    console.print("[cyan]Attempting GGUF compilation (q4_k_m)...[/cyan]")
    try:
        model.save_pretrained_gguf(
            str(MERGED_MODEL_DIR), tokenizer, quantization_method="q4_k_m"
        )
        for candidate in [Path(GGUF_BASENAME), MERGED_MODEL_DIR / GGUF_BASENAME]:
            if candidate.exists():
                gguf_path = candidate
                break
    except Exception as error:
        console.print(f"[yellow]GGUF conversion failed: {error}[/yellow]")
        console.print("[yellow]Falling back to merged safetensors...[/yellow]")

    # Step 2: Fallback to merged safetensors
    if gguf_path is None:
        console.print("[cyan]Saving merged 16-bit model...[/cyan]")
        try:
            model.save_pretrained_merged(
                str(MERGED_MODEL_DIR), tokenizer, save_method="merged_16bit"
            )
            config_file = MERGED_MODEL_DIR / "config.json"
            if config_file.exists():
                import json
                with open(config_file, "r") as f:
                    model_config = json.load(f)
                if "MistralForCausalLM" in model_config.get("architectures", []):
                    console.print("[yellow]Reverting Unsloth Mistral patches to native Phi-3 formatting...[/yellow]")
                    model_config["architectures"] = ["Phi3ForCausalLM"]
                    model_config["model_type"] = "phi3"
                    with open(config_file, "w") as f:
                        json.dump(model_config, f, indent=4)
            console.print(f"[green]Merged model saved to {MERGED_MODEL_DIR.resolve()}[/green]")
        except Exception as e:
            console.print(f"[red]Merge failed: {e}[/red]")
            return

    # Step 3: Write Modelfile and register with Ollama
    if gguf_path and gguf_path.exists():
        from_line = f"FROM ./{gguf_path}"
    else:
        from_line = f"FROM {MERGED_MODEL_DIR.resolve()}"

    modelfile_content = from_line + "\n" + MODELFILE_BODY
    mf_path = Path("Modelfile")
    mf_path.write_text(modelfile_content, encoding="utf-8")
    console.print("[cyan]Modelfile written. Registering with Ollama...[/cyan]")

    try:
        subprocess.run(["ollama", "create", "nexus-trained", "-f", "Modelfile"], check=True)
        console.print("[green]SUCCESS! nexus-trained is now live in Ollama![/green]")
    except Exception as e:
        console.print(f"[red]Ollama registration failed: {e}[/red]")
        console.print("[yellow]The merged model is saved. You can manually run: ollama create nexus-trained -f Modelfile[/yellow]")


if __name__ == "__main__":
    export_and_bridge_to_ollama()
