"""Download the NEXUS LoRA adapter weights from HuggingFace Hub."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ID = "nagu-io/nexus-lora-v1"
LOCAL_DIR = Path(__file__).resolve().parents[1] / "lora_model"
REQUIRED_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]


def main() -> None:
    print(f"Downloading NEXUS LoRA adapter from {REPO_ID}...")
    print(f"Target directory: {LOCAL_DIR}\n")

    # Check if weights already exist
    weights_path = LOCAL_DIR / "adapter_model.safetensors"
    if weights_path.exists():
        size_mb = weights_path.stat().st_size / (1024 * 1024)
        print(f"Model weights already present ({size_mb:.0f} MB). Skipping download.")
        print("Delete lora_model/adapter_model.safetensors to re-download.")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("Install it with: pip install huggingface-hub")
        sys.exit(1)

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    for filename in REQUIRED_FILES:
        print(f"  Downloading {filename}...")
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=str(LOCAL_DIR),
                local_dir_use_symlinks=False,
            )
        except Exception as error:
            print(f"  WARNING: Could not download {filename}: {error}")

    print(f"\nDone! Model files saved to {LOCAL_DIR}")
    print("Set NEXUS_LOCAL_BACKEND=adapter in your .env to use the model.")


if __name__ == "__main__":
    main()
