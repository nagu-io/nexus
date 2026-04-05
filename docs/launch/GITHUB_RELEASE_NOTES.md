# GitHub Release Notes

## Title

NEXUS v0.1.0 - local-first AI developer OS launch

## Release Body

NEXUS is a local-first AI developer OS that runs on your machine, routes tasks across local and cloud paths, scores answers before serving them, and adds canary-based RAG protection.

### What is included

- `CompressX` for compression
- `AEON` for routing
- `ReflectScore` for hallucination scoring and response gating
- `CanaryRAG` and `CanaryVaults` for canary seeding and leak monitoring
- Python CLI
- FastAPI backend
- React dashboard

### Launch path

NEXUS is currently optimized for:

- Ollama
- `phi3:mini`
- Python 3.11+

### Quick start

```bash
pip install -e .
ollama pull phi3:mini
ollama serve
nexus doctor
nexus init
nexus chat
```

### What works today

- local chat and routed coding tasks
- ReflectScore benchmark and trust gating
- environment checks through `nexus doctor`
- canary seeding with local fallback
- dashboard and API
- regression tests and CI

### Honest runtime notes

- `nexus init` currently reports a serving-model proxy benchmark, not direct compressed-artifact execution
- `nexus protect --seed` works in local fallback mode
- `nexus protect --check` is a hosted-service path unless CanaryVaults is configured

### Repo

https://github.com/nagu-io/nexus
