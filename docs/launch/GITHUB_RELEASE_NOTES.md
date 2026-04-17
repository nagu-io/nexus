# GitHub Release Notes

## Title

NEXUS v0.3.0 - local-first AI desktop with Hive, bundled adapter, and trust gating

## Release Body

NEXUS is a local-first AI developer workspace that can plan, edit, execute, retry, and explain work on your machine.

This release is the first version where the repo story matches the actual stack:

- packaged Electron desktop app
- bundled backend sidecar
- bundled LoRA adapter path for local desktop builds
- experimental Hive distributed-search runtime
- ReflectScore hallucination scoring and answer gating
- CanaryRAG / CanaryVaults trust and leak protection
- SafeBench benchmark scaffold

### What is included

- Python runtime and orchestrator
- FastAPI backend
- React dashboard
- Electron desktop shell
- model control center
- Hive panel and Hive API
- packaged Windows installer flow

### Core modules

- `nexus/compress` - CompressX compression and artifact management
- `nexus/router` - AEON-style routing and provider runtime
- `nexus/reflect` - ReflectScore trust gate
- `nexus/canary` - CanaryRAG and leak-monitoring workflows
- `nexus/hive` - experimental distributed Hive runtime
- `safebench` - benchmark scaffold for honesty, safety, recovery, and task success

### Quick start

```bash
pip install -e .
nexus doctor

python -m uvicorn nexus.api:app --host 127.0.0.1 --port 8000
npm --prefix dashboard install
npm --prefix dashboard run dev
```

Desktop development:

```bash
npm --prefix desktop install
npm --prefix desktop run dev
```

Desktop packaging:

```bash
python -m pip install pyinstaller
npm --prefix desktop run dist
```

### What works today

- local chat and repo-aware coding workflows
- ReflectScore-based trust gating
- Canary protection with local fallback flows
- Hive runtime demo surface in CLI, API, chat, and desktop UI
- bundled desktop installer path
- regression tests and CI

### Honest status notes

- Hive is implemented but still experimental; the public internet peer mesh is not the finished state yet
- the bundled local model path is real, but it is not making a frontier-model quality claim
- some optional security and cloud features still require external service configuration

### Repo

https://github.com/nagu-io/nexus
