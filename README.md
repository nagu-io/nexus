# NEXUS

Private AI developer OS for local coding, routing, trust scoring, and RAG protection.

NEXUS packages the ideas behind CompressX, AEON, ReflectScore, CanaryRAG, and CanaryVaults into one clean product repo.

## What Is It?

NEXUS is a local-first AI developer OS that runs on your machine, routes tasks across local and cloud models, checks answers before serving them, and protects RAG systems with canary-based security.

Core pieces:

- `CompressX` for model compression
- `AEON` for routing and agent execution
- `ReflectScore` for hallucination detection and response gating
- `CanaryRAG` and `CanaryVaults` for canary seeding and leak monitoring

## Does It Work On My Machine?

Yes, if you can run Ollama locally.

Minimum path:

- Python 3.11+
- Ollama
- one local model: `phi3:mini`

Setup:

```bash
pip install -e .
ollama pull phi3:mini
ollama serve
nexus doctor
```

`nexus doctor` tells you exactly what is missing and how to fix it.

## What Can I Do Right Now?

Copy-paste these after install:

```bash
nexus chat
nexus code "build a FastAPI auth service"
nexus reflect
```

You can also start the dashboard:

```bash
nexus chat --dashboard
```

## Free Forever

NEXUS core is free forever for local use.

If you run Ollama on your machine, NEXUS can already do useful work without any paid account:

- `nexus init` to prepare and benchmark a local model
- `nexus chat` for local-first chat
- `nexus code "task"` for local coding help
- `nexus reflect` with heuristic trust scoring
- `nexus protect --seed <url>` with local canary fallback when the hosted API is unavailable

That means the default experience is:

- no per-query billing from NEXUS
- no required API key to get started
- your hardware, your model, your cost

## Bring Your Own Keys

You only need external keys when you want stronger behavior than the local stack can provide:

- `GROQ_API_KEY` improves ReflectScore from heuristic checks to live model-based scoring
- `ANTHROPIC_API_KEY` enables stronger cloud fallback for hard tasks
- `SUPABASE_URL` and `SUPABASE_KEY` enable persistent memory across sessions

These are optional upgrades. NEXUS still runs without them.

## Paid Products

Hosted products are separate from the free local core:

- `CanaryVaults` for real remote leak monitoring and alerting
- future `NEXUS Cloud` for synced memory and multi-device state

This keeps the product honest:

- free local usage does not create platform cost for NEXUS
- paid revenue starts when users opt into hosted services
- people can trust the product before they ever pay for anything

## What Is Built

- Python package and CLI in `nexus/`
- FastAPI backend in `nexus/api.py`
- React dashboard in `dashboard/`
- Specialist agents for coding, research, memory, files, and canary workflows
- Internal compression, trust-scoring, and canary modules inside `nexus/`

## Quickstart

### 1. Install

Core install:

```bash
pip install -e .
```

Full install with optional cloud, compression, memory, and voice features:

```bash
pip install -e ".[full]"
```

Feature-specific installs:

```bash
pip install -e ".[cloud]"
pip install -e ".[compression]"
pip install -e ".[memory]"
pip install -e ".[voice]"
```

### 2. Configure Environment

On PowerShell:

```powershell
Copy-Item .env.example .env
```

On macOS/Linux:

```bash
cp .env.example .env
```

Then fill in the keys you want to use.

### 3. Run Setup Checks

```bash
nexus doctor
```

### 4. Start NEXUS

```bash
ollama serve
nexus init
nexus chat
```

## Core Commands

```bash
nexus doctor
nexus status
nexus init --model mistralai/Mistral-7B-Instruct-v0.2 --bits 4
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

ReflectScore sits between model output and the user.

- `score < 0.3`: serve the answer
- `0.3 <= score < 0.6`: serve with warning
- `score >= 0.6`: block and reroute to a stronger model

That makes ReflectScore both a benchmark layer and a live response gate.

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

## Runtime Dependencies

NEXUS is designed to degrade gracefully when optional services are missing.

- Ollama: required for local model execution
- Groq API key: optional, enables live ReflectScore scoring instead of heuristic fallback
- Anthropic API key: optional, enables stronger cloud reroute fallback
- Supabase: optional, enables persistent memory
- CanaryVaults API key: optional, enables real remote leak monitoring

Run `nexus doctor` any time to see what is configured and exactly what to fix.

## Repository Layout

This repo intentionally keeps only the NEXUS product structure:

- `nexus/` for the Python app, CLI, API, and runtime modules
- `dashboard/` for the React frontend
- `docs/` for launch and project docs

## Dashboard

Start the API and frontend together with:

```bash
nexus chat --dashboard
```

Or run them manually:

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

## License

MIT. See `LICENSE`.
