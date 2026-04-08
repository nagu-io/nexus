# NEXUS

> Autonomous local AI execution system that plans, codes, runs, self-corrects, and learns. Runs on phi3:mini. Free. No cloud required.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Stars](https://img.shields.io/github/stars/nagu-io/nexus)

NEXUS is not a chat wrapper or an agent demo. It is a full autonomous execution system.

Give it a goal. It compiles a plan, selects agents by capability, executes through a policy-governed orchestrator, evaluates every output with multiple critics, self-corrects when confidence is low, caches decisions that work, decays decisions that stale, and exposes the entire run through structured traces. Everything runs offline on your own hardware.

## What Landed

This repo now includes the runtime pieces that make long autonomous runs practical:

- Streaming execution with idle watchdogs so builds, scripts, and servers no longer fail silently.
- Multi-file project materialization, dependency install, run, and targeted fix support.
- Git checkpoints and rollback support inside the runtime loop.
- Local-first conversation history with SQLite-backed session persistence and search.
- Live dashboard updates over WebSocket, including terminal-style runtime events.
- Context reduction for oversized chat history, workspace prompts, logs, and file-heavy tasks before they hit the active model.
- Plugin discovery for agents, critics, and tools.
- Automatic project docs generation for `README.md` and `ARCHITECTURE.md` after builds complete.

## What Makes This Real

| Capability | How |
|---|---|
| **Autonomous execution loop** | `CodeExecutor` writes code → runs it → captures errors → feeds errors to the coding agent → gets a fix → retries. Up to 3 self-correction cycles per task. |
| **Policy-governed orchestration** | 47KB orchestrator with runtime policy engine, strategy adaptation, parallel scheduling, and a 24-step tool budget per agent cycle. |
| **Multi-critic evaluation** | Correctness, efficiency, and safety critics score every output. Confidence thresholds gate acceptance. Low scores trigger retries or agent switching. |
| **Skill memory** | Successful workflows are cached with ranked reuse. Stale patterns decay automatically. The system gets better the more you use it. |
| **Decision cache** | Agent/strategy/confidence combos that worked are remembered and reused, skipping expensive re-evaluation when the same pattern recurs. |
| **Context reduction** | Oversized chat history, workspace-grounded prompts, logs, and file dumps are reduced to fit the model budget while raw history stays stored and traceable. |
| **Full explainability** | CLI, API, and dashboard read from the same `RuntimeInsights` layer. Every decision, retry, and fallback is traced. |
| **Offline-first** | Optimized for Ollama + `phi3:mini`. Optional cloud fallback to Anthropic or Groq. |

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Start the local model
ollama pull phi3:mini
ollama serve

# 3. Install the dashboard
npm run dashboard:install
npm run dashboard:build

# 4. Verify
nexus doctor
nexus init
```

## Run It

```bash
# Backend + dashboard together
npm run dev

# Or with pnpm
pnpm install && pnpm dev
```

## First Runs

```bash
# Build with full explain trace
python main.py --build --explain "create react login page"

# Analyze a codebase
python main.py --analyze --explain "analyze this repository"

# Build, scaffold to disk, install deps, and launch locally
python main.py --build --run --output-dir generated/login-system \
  "build a full stack login system with Express backend, API routes, and basic frontend form"
```

Each run shows the compiled plan, runtime decisions, retry/fallback strategy, and final confidence score.

**`--write`** materializes generated files to disk. Default: `./generated/<scaffold-name>`. Add `--force` to overwrite.

**`--run`** builds on `--write`: materializes, installs npm dependencies, picks a free port starting at 3010, launches the app, and prints the URL.

## The Execution Loop

```
goal
  → intent parser (classify intent + complexity)
  → planner engine (generate task graph with dependencies)
  → blueprint generator (create executable blueprint)
  → orchestrator
      → policy engine (govern execution behavior)
      → agent selection (capability-based wiring)
      → agent execution (think → act → observe → reflect)
      → tool dispatch (file_tool, terminal_tool, code_executor)
      → multi-critic evaluation (correctness + efficiency + safety)
      → strategy engine (retry, fallback, agent switch)
      → decision cache (reuse what works, decay what stales)
  → skill memory (persist successful workflows)
  → runtime insights (expose everything to CLI/API/dashboard)
```

## Architecture

```
CLI / API / Dashboard
  → RuntimeInsights (single source of truth)
  → Compiler
       → IntentParser → PlannerEngine → BlueprintGenerator
  → Orchestrator (47KB — policy, strategy, tool dispatch, parallel scheduling)
       → PolicyEngine, StrategyEngine, SharedMemory, WiringEngine
  → Agents
       → CodingAgent (27KB — autonomous mode, stack validation, fix loops)
       → ResearchAgent, MemoryAgent, FileAgent, CanaryAgent
  → Runtime Tools
       → CodeExecutor (write → run → error → fix loop)
       → ContextReducer (prompt budget management for long history/log/file inputs)
       → FileTool, TerminalTool
  → Evaluation
       → CorrectnessCritic, EfficiencyCritic, SafetyCritic → MultiCritic
  → Memory
       → SkillMemory (ranked workflow reuse with decay)
       → DecisionCache (agent/strategy confidence reuse)
       → EnvironmentMemory (per-project session persistence)
       → ExecutionTrace, DecisionLog
  → Extensions
       → MindRouter (Ollama/Anthropic/Groq provider routing)
       → ReflectScore (self-evaluation + benchmarking)
       → ModelCompressor (GPTQ quantization)
       → Voice Interface (Whisper + TTS)
```

## Product Surfaces

| Surface | Purpose |
|---------|---------|
| **CLI** | Direct execution, `--explain` mode, `nexus doctor/chat/reflect/protect` |
| **FastAPI** | Programmatic access, dashboard data endpoints |
| **React Dashboard** | Run history, success/retry trends, cache health, skill patterns, live agent status |

All three read from the same `RuntimeInsights` layer.

## Core Commands

```bash
nexus doctor                           # Environment checks
nexus status                           # System status
nexus init                             # Initialize workspace
nexus code "build a FastAPI auth service"  # Direct code generation
nexus chat                             # Interactive chat
nexus chat --dashboard                 # Chat + dashboard
nexus reflect                          # Self-evaluation
nexus reflect --compare original       # Compare against baseline
nexus reflect --live                   # Live reflect loop
nexus protect --check                  # Security scan
nexus protect --seed https://example.com/docs  # Seed threat intelligence
```

## Dashboard

```bash
# One command from repo root
npm run dev

# Two-terminal fallback
python -m uvicorn nexus.api:app --port 8000
npm run dashboard:dev
```

The dashboard exposes: run history, success rate and retry trends, cache health, top skill-memory patterns, and live system/agent status.

## Context Reduction

NEXUS keeps raw history, traces, and stored conversations intact, but reduces oversized prompts before they hit the active model. This applies to long chat sessions, workspace-grounded repo prompts, runtime logs, and large file-heavy task instructions.

Configuration:

```bash
NEXUS_CONTEXT_REDUCTION_ENABLED=true
NEXUS_CONTEXT_REDUCTION_BACKEND=heuristic
NEXUS_CONTEXT_REDUCTION_THRESHOLD_CHARS=12000
NEXUS_CONTEXT_REDUCTION_TARGET_CHARS=6000
NEXUS_CONTEXT_REDUCTION_MODEL=
```

The reducer is wired through the orchestrator, API chat path, direct `MindRouter` routes, the CLI, and Jarvis. The dashboard surfaces the active reducer backend and shows a per-response reduction badge when a prompt was compacted.

## Repository Layout

```
nexus/                  Python core — compiler, runtime, agents, API
  agents/               CodingAgent, ResearchAgent, FileAgent, MemoryAgent
  compiler/             PlannerEngine
  critics/              Correctness, Efficiency, Safety → MultiCritic
  memory/               SkillMemory, EnvironmentMemory, SupabaseMemory
  runtime/              PolicyEngine, StrategyEngine, DecisionCache,
                        CodeExecutor, FileTool, TerminalTool, Trace, Insights
  canary/               CanaryAgent, RiskEngine
  reflect/              ReflectScore, Benchmarks, Evaluator
  router/               MindRouter (provider routing)
  compress/             Model compression (GPTQ)
  jarvis/               Voice interface (Whisper + TTS)
dashboard/              React + Tailwind frontend
tests/                  Regression suite (72KB+)
docs/launch/            Demo script, HN draft, launch checklist
tools/                  Dev orchestrator
main.py                 Compiler → orchestrator runtime entrypoint
```

## Testing

```bash
# Full regression suite
python -m unittest discover -s tests -v

# Executor tests only
python -m unittest tests/test_executor.py -v

# Dashboard build check
npm run dashboard:build
```

`npm audit --prefix dashboard` is currently clean with 0 known vulnerabilities.

## Launch Assets

Launch materials live in `docs/launch/`: `DEMO_SCRIPT.md`, `HACKERNEWS_DRAFT.md`, `LAUNCH_CHECKLIST.md`, `COLD_INSTALL_REPORT.md`.

## License

MIT. See `LICENSE`.
