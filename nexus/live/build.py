"""Build NEXUS Vision in one go: fetch the data, train, test it against the base model, and share it with yourself.

    nexus vision build --systems windows --limit 4000 --push YOUR_NAME/nexus-vision
    (or: python -m nexus.live.build ...)

1. Fetches OS-Atlas desktop screenshots for each --systems (only the ones it
   needs, see nexus.live.atlas), and adds your own --teach recordings: a folder,
   or a private dataset repo on your Hugging Face account.
2. Fine-tunes Qwen3-VL-2B-Instruct on them (nexus.live.finetune).
3. Puts the held-out examples to the base model and to NEXUS Vision, and writes
   the comparison into eval.json and the model's README.md.
4. With --push, uploads the model to a private repo on your account, so your
   laptop can download it (nexus vision pull) and run it offline.

This is what the Hugging Face Jobs script (scripts/nexus_vision_job.py) and the
Colab notebook run. It also runs on your own graphics card.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from nexus.live.data import EXAMPLES_FILE, write_examples

HUB_REPO = os.getenv("NEXUS_VISION_REPO", "")
LICENSE_NOTE = (
    "Base model: Qwen/Qwen3-VL-2B-Instruct (Apache-2.0). Grounding data: OS-Copilot/OS-Atlas-data (Apache-2.0)."
)


def fetch_teach(source: str, work: Path, printer: Callable[[str], None]) -> Path:
    """A teach dataset as a local folder: the folder itself, or a download of a Hub dataset repo."""
    local = Path(source).expanduser()
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download  # type: ignore

    printer(f"Downloading your recordings from {source}...")
    target = work / "teach" / source.replace("/", "__")
    snapshot_download(source, repo_type="dataset", local_dir=str(target))
    if not (target / EXAMPLES_FILE).exists():
        raise FileNotFoundError(f"{source} has no {EXAMPLES_FILE}; upload the whole teach folder.")
    return target


def model_readme(repo: str, training: dict[str, Any], results: dict[str, dict[str, Any]], table: str) -> str:
    sources = ", ".join(f"{name}: {count}" for name, count in sorted(training.get("sources", {}).items()))
    return f"""---
base_model: Qwen/Qwen3-VL-2B-Instruct
library_name: peft
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- nexus-live
- gui-agent
- qwen3-vl
- lora
datasets:
- OS-Copilot/OS-Atlas-data
---

# NEXUS Vision

The vision model of NEXUS Live, the local screen, voice and action assistant in
[nexus](https://github.com/nagu-io/nexus). It looks at a screenshot and a request
and answers with the next mouse or keyboard step, as JSON.

It is a LoRA adapter on Qwen3-VL-2B-Instruct, fine-tuned on NEXUS Live's own
prompts: {training.get("train_examples", 0)} examples ({sources}), {training.get("steps", 0)} steps.
Positions are 0 to 1000 across the screenshot. It runs on your own computer:

```
nexus vision pull {repo or "YOUR_NAME/nexus-vision"}
nexus live --voice
```

## How it does on held-out examples

{table}

A step is right when it is the same action and, for a click, lands inside the
element's box (or within 3% of the screen of the recorded click). Invalid means
an answer NEXUS Live could not act on.

{LICENSE_NOTE}
"""


def run(args: argparse.Namespace, printer: Callable[[str], None] = print) -> dict[str, Any]:
    from nexus.live.atlas import fetch_os_atlas
    from nexus.live.evaluate import evaluate, held_out, report_table
    from nexus.live.finetune import TrainOptions, load_items, train
    from nexus.live.local import LocalVision

    work = Path(args.work).expanduser()
    atlas_folder = work / "atlas"
    if args.limit > 0 and not (args.reuse_data and (atlas_folder / EXAMPLES_FILE).exists()):
        if (atlas_folder / EXAMPLES_FILE).exists():
            (atlas_folder / EXAMPLES_FILE).unlink()
        systems = [system.strip() for system in args.systems.split(",") if system.strip()]
        share = max(1, args.limit // max(1, len(systems)))
        for system in systems:
            examples = fetch_os_atlas(system, atlas_folder, limit=share, max_width=args.width, seed=args.seed, printer=printer)
            write_examples(atlas_folder, examples, append=True)
            printer(f"{system}: {len(examples)} examples.")
    data = [atlas_folder] if (atlas_folder / EXAMPLES_FILE).exists() else []
    data += [fetch_teach(source, work, printer) for source in args.teach or []]
    if not data:
        raise SystemExit("No data: give --limit above 0 for OS-Atlas examples, or --teach with your recordings.")

    out = Path(args.out).expanduser()
    options = TrainOptions(
        data=data,
        out=out,
        base=args.base,
        epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        four_bit=args.four_bit,
        max_width=args.width,
        seed=args.seed,
        workers=args.workers,
    )
    result = train(options, printer)
    training = json.loads((out / "training.json").read_text(encoding="utf-8"))

    results: dict[str, dict[str, Any]] = {}
    if args.eval_limit > 0:
        items, _ = load_items(data)
        held = held_out(items, options.eval_fraction)
        printer(f"Testing on {min(len(held), args.eval_limit)} held-out examples.")
        for name, model in (("NEXUS Vision", LocalVision(out)), ("Base model", LocalVision(args.base, check_card=False))):
            printer(f"{name}:")
            summary, _ = evaluate(model, held, args.eval_limit, printer)
            results[name] = summary
            model.close()
    table = report_table(results) if results else "(not evaluated)"
    (out / "eval.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(model_readme(args.push or "", training, results, table), encoding="utf-8")
    printer(table)

    if args.push:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi()
        api.create_repo(args.push, private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(out),
            repo_id=args.push,
            ignore_patterns=["checkpoints/*", "*.part"],
            commit_message=f"NEXUS Vision: {result.train_examples} examples, {result.steps} steps",
        )
        printer(f"Uploaded to https://huggingface.co/{args.push} (private).")
    return {"training": asdict(result) | {"out": str(out)}, "results": results}


def parser() -> argparse.ArgumentParser:
    from nexus.live.finetune import DEFAULT_BASE

    parser = argparse.ArgumentParser(prog="nexus vision build", description="Fetch data, train, evaluate and (optionally) upload NEXUS Vision.")
    parser.add_argument("--systems", default="windows", help="OS-Atlas systems, comma separated: windows, macos, linux")
    parser.add_argument("--limit", type=int, default=4000, help="OS-Atlas elements in all (0 for none)")
    parser.add_argument("--teach", action="append", help="your recordings: a folder, or a dataset repo such as you/nexus-teach")
    parser.add_argument("--work", default="nexus_vision_work", help="where the data is kept")
    parser.add_argument("--reuse-data", action="store_true", help="keep OS-Atlas examples fetched before")
    parser.add_argument("--out", default="nexus_vision")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--four-bit", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2)
    parser.add_argument("--eval-limit", type=int, default=200, help="held-out examples to test each model on (0 to skip)")
    parser.add_argument("--push", default=HUB_REPO, help="upload to this model repo on your account, kept private")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
