"""Upload the NEXUS LoRA adapter to HuggingFace Hub.

Usage:
    1. Get a HuggingFace token from https://huggingface.co/settings/tokens
    2. Run: python tools/upload_model.py --token YOUR_HF_TOKEN
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "nagu-io/nexus-lora-v1"
LOCAL_DIR = Path(__file__).resolve().parents[1] / "lora_model"

FILES_TO_UPLOAD = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "README.md",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload NEXUS LoRA adapter to HuggingFace Hub")
    parser.add_argument("--token", required=True, help="HuggingFace API token (write access)")
    parser.add_argument("--repo-id", default=REPO_ID, help=f"Target repo (default: {REPO_ID})")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("Install it with: pip install huggingface-hub")
        sys.exit(1)

    api = HfApi(token=args.token)

    # Create the repo if it doesn't exist
    print(f"Creating repo {args.repo_id} (if it doesn't exist)...")
    try:
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
        print(f"  Repo ready: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"  Note: {e}")

    # Upload each file
    for filename in FILES_TO_UPLOAD:
        filepath = LOCAL_DIR / filename
        if not filepath.exists():
            print(f"  SKIP: {filename} (not found)")
            continue

        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  Uploading {filename} ({size_mb:.1f} MB)...")
        try:
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=args.repo_id,
                repo_type="model",
            )
            print(f"  ✓ {filename} uploaded")
        except Exception as e:
            print(f"  ERROR uploading {filename}: {e}")

    print(f"\nDone! Model available at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
