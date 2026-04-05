# NEXUS

NEXUS is a local-first AI developer OS that runs on your machine, routes tasks across local and cloud models, scores answers before serving them, and adds canary-based RAG protection.

## What It Includes

- `CompressX` for model compression
- `AEON` for routing and agent execution
- `ReflectScore` for hallucination detection and response gating
- `CanaryRAG` and `CanaryVaults` for canary seeding and leak monitoring
- a Python CLI, FastAPI backend, and React dashboard

## Launch Path

NEXUS is currently optimized for one first-class setup:

- Ollama
- `phi3:mini`
- Python 3.11+

That is the path the docs, checks, and launch UX are built around.

## Quick Start

```bash
pip install -e .
ollama pull phi3:mini
ollama serve
nexus doctor
nexus init
nexus chat
```

If `phi3:mini` is missing, `nexus doctor` and `nexus init` will tell you to run:

```bash
ollama pull phi3:mini
```

## What You Can Do Right Now

```bash
nexus init
nexus chat
nexus code "build a FastAPI auth service"
nexus reflect
nexus protect --seed https://example.com/docs
```

Dashboard mode:

```bash
nexus chat --dashboard
```

## Free Core, Optional Upgrades

NEXUS core is free forever for local use.

Free local path:

- `nexus init` prepares compression artifacts and runs a serving-model benchmark
- `nexus chat` runs on the local model
- `nexus code` uses the local route by default
- `nexus reflect` works with heuristic scoring
- `nexus protect --seed` has a local fallback plan

Optional keys:

- `GROQ_API_KEY` for live ReflectScore scoring
- `ANTHROPIC_API_KEY` for stronger cloud fallback
- `SUPABASE_URL` and `SUPABASE_KEY` for persistent memory

Hosted products:

- `CanaryVaults` for real remote leak monitoring and alerting
- future `NEXUS Cloud` for synced memory and multi-device state

## Honest Runtime Notes

- Ollama is required for the local launch path.
- `nexus init` defaults to `phi3:mini`.
- `nexus init` does not claim to benchmark the saved compressed artifact directly. It reports a serving-model proxy benchmark unless direct artifact execution is implemented.
- `nexus protect --seed` works in local fallback mode.
- `nexus protect --check` is guidance/fallback mode unless CanaryVaults is configured.

Run this any time:

```bash
nexus doctor
```

It checks Ollama, the default model, and optional dependencies, then tells you exactly what is missing.

## Core Commands

```bash
nexus doctor
nexus status
nexus init
nexus code "build a FastAPI auth service"
nexus chat
nexus chat --dashboard
nexus reflect
nexus reflect --compare original
nexus reflect --live
nexus protect --check
nexus protect --seed https://example.com/docs
```

## ReflectScore Trust Layer

ReflectScore sits between model output and the user:

- `< 0.3` serve
- `0.3 to < 0.6` warn
- `>= 0.6` block and reroute to a stronger model

If both the first answer and the stronger fallback are high risk, NEXUS withholds the response instead of showing a likely hallucination.

## Architecture

```text
NEXUS CLI / API / Dashboard
  |- CompressX adapter
  |- AEON-style router
  |    |- local model
  |    |- cloud model
  |    `- specialist agents
  |         `- ReflectScore trust gate
  |- Jarvis interface
  `- CanaryRAG / CanaryVaults layer
```

## Testing And CI

The repo now includes a small regression suite for the launch-critical paths we fixed:

- file path safety
- `doctor` dependency checks
- ReflectScore heuristic scoring
- benchmark path validation
- `phi3:mini` alias handling

Run locally:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions also runs:

- `python -m compileall nexus`
- `python -m unittest discover -s tests -v`

## Repository Layout

This repo intentionally keeps only the NEXUS product structure:

- `nexus/` for the Python app, CLI, API, agents, and runtime modules
- `dashboard/` for the React frontend
- `docs/` for launch assets and project docs
- `tests/` for the regression suite

## Dashboard

Start everything together:

```bash
nexus chat --dashboard
```

Or run it manually:

```bash
python -m uvicorn nexus.api:app --port 8000
cd dashboard
npm install
npm run dev
```

## Launch Assets

Ready-to-use launch materials are in `docs/launch/`:

- `DEMO_SCRIPT.md`
- `HACKERNEWS_DRAFT.md`
- `LAUNCH_CHECKLIST.md`
- `COLD_INSTALL_REPORT.md`

## License

MIT. See `LICENSE`.
