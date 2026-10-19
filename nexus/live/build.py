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

This is what the Kaggle and Colab notebook (notebooks/train_nexus_vision.ipynb)
and the Hugging Face Jobs script (tools/nexus_vision_job.py) run. It also runs on
your own graphics card.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
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


class PublicRepo(ValueError):
    """A repo to upload into exists, and is public."""


def ensure_private(api: Any, repo: str, repo_type: str = "model") -> None:
    """Make ``repo`` on the Hub, private; or check that the one already there is private.

    create_repo leaves an existing repo as it is, public or not: uploads into
    a public one (a model's raw answers, teach screenshots) would be public.
    """
    api.create_repo(repo, repo_type=repo_type, private=True, exist_ok=True)
    if not getattr(api.repo_info(repo, repo_type=repo_type), "private", False):
        raise PublicRepo(
            f"{repo} already exists and is public, so nothing was uploaded to it. Make it private in its settings "
            "on huggingface.co, or choose another name."
        )


class RunLog:
    """Prints each line, and keeps the run's log in the model repo as run.log.

    A run on Kaggle or Colab can then be followed (and a failure read) from
    anywhere with access to the repo, without the notebook open. The log is
    uploaded every ``every`` seconds and at each stage; an upload that fails is
    reported and the run goes on.
    """

    def __init__(self, repo: str, printer: Callable[[str], None] = print, every: float = 600.0,
                 api: Any = None, clock: Callable[[], float] = time.monotonic):
        if api is None:
            from huggingface_hub import HfApi  # type: ignore

            api = HfApi()
        self.repo = repo
        self.printer = printer
        self.every = every
        self.api = api
        self.clock = clock
        self.lines: list[str] = []
        self.last_upload: float | None = None
        ensure_private(api, repo)  # also proves the token can write, before any work

    def print(self, line: str) -> None:
        self.printer(line)
        self.lines.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}")
        if self.last_upload is None or self.clock() - self.last_upload >= self.every:
            self.upload()

    def upload(self) -> None:
        self.last_upload = self.clock()
        try:
            self.api.upload_file(
                path_or_fileobj=("\n".join(self.lines) + "\n").encode("utf-8"),
                path_in_repo="run.log",
                repo_id=self.repo,
                commit_message="Training log",
            )
        except Exception as error:  # the log is a convenience: never let it stop the run
            self.printer(f"(could not update run.log: {type(error).__name__})")

    def failed(self, error: BaseException) -> None:
        self.lines.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} FAILED: {type(error).__name__}: {error}")
        self.lines.extend(traceback.format_exception(error)[-12:])
        self.upload()


def run(args: argparse.Namespace, printer: Callable[[str], None] = print) -> dict[str, Any]:
    log = RunLog(args.push, printer) if args.push else None
    try:
        return _run(args, log.print if log else printer, log)
    except BaseException as error:
        if log:
            log.failed(error)
        raise


def environment() -> str:
    """The versions and graphics card a run used, for its log."""
    import importlib.metadata as metadata

    parts = []
    for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            parts.append(f"{package} {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            parts.append(f"{package} missing")
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            parts.append(f"{torch.cuda.get_device_name(0)} (compute {major}.{minor})")
        else:
            parts.append("no graphics card")
    except ImportError:
        pass
    return ", ".join(parts)


def _run(args: argparse.Namespace, printer: Callable[[str], None], log: RunLog | None) -> dict[str, Any]:
    from nexus.live.atlas import fetch_os_atlas
    from nexus.live.evaluate import evaluate, held_out, report_table
    from nexus.live.finetune import TrainOptions, free_memory, load_items, train
    from nexus.live.local import LocalVision

    printer(environment())
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
    if log:
        log.upload()

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
        compute=args.compute,
        max_width=args.width,
        seed=args.seed,
        workers=args.workers,
    )
    result = train(options, printer)
    training = json.loads((out / "training.json").read_text(encoding="utf-8"))
    if log:
        log.upload()

    results: dict[str, dict[str, Any]] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    if args.eval_limit > 0:
        items, _ = load_items(data)
        held = held_out(items, options.eval_fraction)
        printer(f"Testing on {min(len(held), args.eval_limit)} held-out examples.")
        for name, model in (("NEXUS Vision", LocalVision(out)), ("Base model", LocalVision(args.base, check_card=False))):
            free_memory()  # nothing from training or the model before may stay on the graphics card
            printer(f"{name}:")
            summary, grades = evaluate(model, held, args.eval_limit, printer)
            results[name] = summary
            details[name] = [asdict(item) for item in grades]
            model.close()
    table = report_table(results) if results else "(not evaluated)"
    (out / "eval.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (out / "eval_details.json").write_text(json.dumps(details, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "README.md").write_text(model_readme(args.push or "", training, results, table), encoding="utf-8")
    printer(table)

    if log:
        api = log.api
        api.upload_folder(
            folder_path=str(out),
            repo_id=args.push,
            ignore_patterns=["checkpoints/*", "*.part"],
            commit_message=f"NEXUS Vision: {result.train_examples} examples, {result.steps} steps",
        )
        printer(f"Uploaded to https://huggingface.co/{args.push} (private).")
        log.upload()
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
    parser.add_argument("--compute", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2)
    parser.add_argument("--eval-limit", type=int, default=200, help="held-out examples to test each model on (0 to skip)")
    parser.add_argument("--push", default=HUB_REPO, help="upload to this model repo on your account, kept private")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from nexus.live.finetune import use_one_gpu

    use_one_gpu()
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
