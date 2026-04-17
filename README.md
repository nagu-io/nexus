# NEXUS

> Local-first autonomous AI developer workspace with a fine-tuned LoRA adapter, repo-aware execution, hallucination gating, and a cinematic desktop shell.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Desktop](https://img.shields.io/badge/desktop-tauri-black)

NEXUS is not a plain chat window. It takes a goal, compiles it into a workflow, executes against a local workspace, scores output risk before serving it, and exposes the full system through a dashboard, API, CLI, and packaged desktop app.

## What Ships Today

- **Goal compiler:** `IntentParser → PlannerEngine → BlueprintGenerator → Orchestrator`
- **Repo-aware execution** with file editing, terminal execution, retries, traces, and critics
- **React dashboard** for chat, files, runtime telemetry, model control, and Hive inspection
- **Tauri desktop shell** with bundled Python backend sidecar
- **ReflectScore** risk scoring on model output before answers are served
- **CanaryRAG / CanaryVaults** leak detection and trust workflows
- **Bundled LoRA adapter** path for fully local, offline inference
- **Experimental Hive** route for trust-scored distributed answer search
- **SafeBench** for honesty, safety, recovery, and task-success benchmarking

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Rust toolchain (for Tauri desktop builds only)

### Install Dependencies

```bash
# Python backend
pip install -e .

# Dashboard frontend
npm --prefix dashboard install
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env to set your preferences
```

**Adapter mode** (bundled LoRA model — no Ollama needed):
```env
NEXUS_LOCAL_BACKEND=adapter
NEXUS_LOCAL_MODEL_DIR=lora_model
```

**Ollama mode** (if you prefer Ollama):
```env
OLLAMA_BASE_URL=http://localhost:11434
NEXUS_MODEL=phi3:mini
```

**Cloud fallback** (optional — for higher quality answers):
```env
OPENROUTER_API_KEY=your-key-here
OPENROUTER_MODEL=openrouter/auto
```

### Download the Model (Adapter Mode)

The LoRA adapter weights are hosted separately to keep the repo lightweight:

```bash
python tools/download_model.py
```

Or manually download from [HuggingFace Hub](https://huggingface.co/) and place files in `lora_model/`.

### Run the Web Stack

```bash
# Start the backend API
python -m uvicorn nexus.api:app --host 127.0.0.1 --port 8000

# In another terminal, start the dashboard
npm --prefix dashboard run dev
```

### Run the Desktop App

```bash
# Install Tauri CLI (one-time)
cargo install tauri-cli

# Development mode
cargo tauri dev

# Production build
cargo tauri build
```

The desktop app bundles the dashboard and Python backend into a single native window.

## Architecture

```text
goal
  → IntentParser
  → PlannerEngine
  → BlueprintGenerator
  → Orchestrator
       → WiringEngine
       → PolicyEngine
       → StrategyEngine
       → FileTool / TerminalTool / CodeExecutor / ProjectExecutor / GitTool
       → MultiCritic
       → ReflectScore
  → RuntimeInsights / API / Dashboard / Desktop
```

### Code Layout

| Directory | Purpose |
|---|---|
| [`nexus/`](nexus/) | Compiler, runtime, agents, API, memory, router, trust systems |
| [`dashboard/`](dashboard/) | React dashboard (Vite + Radix UI) |
| [`src-tauri/`](src-tauri/) | Tauri desktop shell and native config |
| [`tests/`](tests/) | Backend regression suite |
| [`tools/`](tools/) | Backend bundling and dev helpers |
| [`docs/`](docs/) | Architecture, packaging, and design docs |
| [`safebench/`](safebench/) | Benchmark scaffold for safety and honesty evaluation |
| [`lora_model/`](lora_model/) | LoRA adapter config (weights downloaded separately) |

## Embedded Model

NEXUS ships with a real bundled local model path:

- Fine-tuned LoRA adapter trained on the NEXUS codebase
- 4-bit quantized via BitsAndBytes for low memory usage
- Loads through standard HuggingFace `transformers` + `peft`
- Fallback scaffolding when model output isn't parseable

> **Honest status:** The local adapter is strong for offline, repo-first workflows but does not claim frontier-model quality from a small bundled model. Cloud fallback via OpenRouter is available for higher quality.

## Trust Stack

NEXUS has a full trust story instead of only a generation story:

- [`nexus/reflect/`](nexus/reflect/) — ReflectScore hallucination scoring and answer gating
- [`nexus/canary/`](nexus/canary/) — CanaryRAG and CanaryVaults integration
- [`nexus/critics/`](nexus/critics/) — Correctness, safety, and efficiency evaluation
- [`safebench/`](safebench/) — Benchmark scaffold for honesty, safety, recovery, and task success

The important idea: NEXUS tries to decide whether an answer should be shown at all, not just how to generate one.

## Hive — Distributed Intelligence

NEXUS Hive distributes the search for the best answer, not the forward pass:

- Trusted nodes race the same or related subtasks
- Canary tasks check whether nodes are still trustworthy
- ReflectScore ranks returned candidates
- Top candidates can be assembled into one final answer

Read the architecture note in [`docs/NEXUS_HIVE.md`](docs/NEXUS_HIVE.md).

> **Honest status:** Hive is real in the repo, but the internet-wide peer transport is still experimental, not a finished public mesh.

## Workflow

1. Launch the dashboard or desktop app
2. Open a repository in the **Files** tab
3. Switch to **Chat** and enable **repo execution** mode
4. Send a task like `build an auth page` or `fix the login flow`
5. Watch the agent create files, run commands, and log traces in real time

## Useful Commands

```bash
# CLI
nexus doctor
nexus status
nexus chat
nexus reflect
nexus protect --status
nexus hive "build me a full authentication system"

# Direct execution
python main.py --build --explain "create react login page"
python main.py --analyze --explain "analyze this repository"

# Testing
python -m pytest tests/ -v
npm --prefix dashboard run build
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR guidelines.

## License

MIT. See [LICENSE](LICENSE).
