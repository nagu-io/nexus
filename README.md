# NEXUS

> Local-first autonomous AI developer workspace with experimental Hive intelligence, a bundled fine-tuned adapter path, hallucination gating, and CanaryRAG trust checks.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Desktop](https://img.shields.io/badge/desktop-electron-black)

NEXUS is not a plain chat window. It takes a goal, compiles it into a workflow, executes against a local workspace, scores output risk before serving it, and exposes the full system through a dashboard, API, CLI, and packaged desktop app.

The repo already contains the major pieces of that stack:

- autonomous repo-first coding runtime
- packaged Electron desktop app
- experimental Hive distributed-search runtime
- ReflectScore hallucination gate
- CanaryRAG / CanaryVaults trust layer
- bundled LoRA adapter path for desktop builds
- SafeBench benchmark suite

## What Ships Today

- Goal compiler: `IntentParser -> PlannerEngine -> BlueprintGenerator -> Orchestrator`
- Repo-aware execution with file editing, terminal execution, retries, traces, and critics
- React dashboard for chat, files, runtime telemetry, model control, and Hive inspection
- Electron desktop shell with bundled backend sidecar
- ReflectScore risk scoring on model output before answers are served
- Canary-based leak detection and trust workflows
- `CompressX` compression and model artifact management
- Experimental Hive route for trust-scored distributed answer search
- SafeBench for honesty, safety, recovery, and task-success benchmarking

## Hive - Distributed Intelligence

NEXUS Hive does not try to split one forward pass across the internet. It distributes the search for the best answer.

- trusted nodes race the same or related subtasks
- canary tasks check whether nodes are still trustworthy
- ReflectScore ranks returned candidates
- top candidates can be assembled into one final answer

What is implemented now:

- Hive runtime code in [`nexus/hive/`](nexus/hive/)
- routing integration in [`nexus/router/mind_router.py`](nexus/router/mind_router.py)
- API endpoints at `/hive/status` and `/hive/demo`
- CLI entry point: `nexus hive "build me a full authentication system"`
- desktop Hive panel in the dashboard

Read the architecture note in [`docs/NEXUS_HIVE.md`](docs/NEXUS_HIVE.md).

Honest status: Hive is real in the repo, but the internet-wide peer transport and hardened remote sandbox are still experimental work, not a finished public mesh.

## Embedded Model

NEXUS now has a real bundled local model path instead of only depending on a separate manual model install.

- the fine-tuned adapter artifacts live in [`lora_model/`](lora_model/)
- the desktop packager bundles that adapter as `model-packs/default-adapter`
- the packaged Electron shell auto-detects the bundled adapter and can boot in adapter mode
- `CompressX` lives in [`nexus/compress/`](nexus/compress/) and manages compression artifacts and launch-pack accounting

This matters because the packaged desktop build can ship a local-first runtime without forcing users to manually wire up a separate fine-tuned model first.

Honest status: the local stack is strong for offline and local-first workflows, but it is not claiming frontier Claude/Codex-level quality from a tiny bundled model.

## Trust Stack

NEXUS has a full trust story instead of only a generation story.

- [`nexus/reflect/`](nexus/reflect/): ReflectScore hallucination scoring and answer gating
- [`nexus/canary/`](nexus/canary/): CanaryRAG and CanaryVaults integration
- [`nexus/critics/`](nexus/critics/): correctness, safety, and efficiency evaluation
- [`safebench/`](safebench/): benchmark scaffold for honesty, safety, recovery, and task success

The important idea is simple: NEXUS tries to decide whether an answer should be shown at all, not just how to generate one.

## Desktop App

The active desktop product is:

- backend/runtime in [`nexus/`](nexus/)
- dashboard UI in [`dashboard/`](dashboard/)
- Electron shell in [`desktop/`](desktop/)

The packaging flow bundles:

- dashboard build output
- FastAPI backend sidecar
- default local adapter pack

Read the packaging note in [`docs/DESKTOP_PACKAGING.md`](docs/DESKTOP_PACKAGING.md).

### Run The Web Stack

Install Python and frontend dependencies:

```bash
pip install -e .
npm --prefix dashboard install
```

Optional local-model setup if you want the Ollama path:

```bash
ollama pull phi3:mini
ollama serve
```

Run the backend:

```bash
python -m uvicorn nexus.api:app --host 127.0.0.1 --port 8000
```

Run the dashboard:

```bash
npm --prefix dashboard run dev
```

### Run The Desktop App

Install desktop dependencies:

```bash
npm --prefix desktop install
```

Run Electron in development:

```bash
npm --prefix desktop run dev
```

Build the desktop installer:

```bash
python -m pip install pyinstaller
npm --prefix desktop run dist
```

Installer outputs are written under [`desktop/release/`](desktop/release/).

## Workflow

Typical repo-first flow:

1. Launch the dashboard or desktop app.
2. Open a repository in the `Files` tab.
3. Switch to `Chat`.
4. Turn on repo/workspace mode when needed.
5. Send a task like `fix the login flow` or `build an auth page`.
6. Watch runtime decisions, traces, and trust scores update live.

For existing repositories, NEXUS writes its own runtime artifacts under `.nexus/` so it does not overwrite the repo's real top-level docs during analysis loops.

## Architecture

High-level execution flow:

```text
goal
  -> IntentParser
  -> PlannerEngine
  -> BlueprintGenerator
  -> Orchestrator
       -> WiringEngine
       -> PolicyEngine
       -> StrategyEngine
       -> FileTool / TerminalTool / CodeExecutor / ProjectExecutor / GitTool
       -> MultiCritic
       -> ReflectScore
  -> RuntimeInsights / API / Dashboard / Desktop
```

Main code areas:

- [`nexus/`](nexus/): compiler, runtime, agents, API, memory, router, trust systems
- [`dashboard/`](dashboard/): React dashboard
- [`desktop/`](desktop/): Electron desktop shell and installer config
- [`tests/`](tests/): regression suite
- [`tools/`](tools/): backend bundling and dev helpers
- [`docs/`](docs/): architecture, packaging, and launch material
- [`safebench/`](safebench/): benchmark scaffold

## Useful Commands

```bash
nexus doctor
nexus status
nexus chat
nexus reflect
nexus protect --status
nexus hive "build me a full authentication system"

python main.py --build --explain "create react login page"
python main.py --analyze --explain "analyze this repository"

python -m unittest discover -s tests -v
npm --prefix dashboard run build
npm --prefix desktop run dist
```

## Important Repo Notes

- The active product lives in `nexus/`, `dashboard/`, `desktop/`, `tests/`, and `docs/`.
- [`backend/`](backend/) and [`frontend/`](frontend/) are older sample/prototype app folders, not the main NEXUS product.
- [`src-tauri/`](src-tauri/) is an older Tauri desktop prototype; Electron is the active desktop path.
- `lora_model/` contains the adapter artifacts used by the bundled desktop model path.
- `nexus_model/` contains much larger full model checkpoint artifacts.

## License

MIT. See [`LICENSE`](LICENSE).
