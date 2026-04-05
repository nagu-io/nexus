# Hacker News Draft

Show HN: NEXUS - local-first AI developer OS with trust gating and RAG protection

I built NEXUS as a CS student from India. It is a private AI developer OS that runs locally, routes tasks across local and cloud paths, scores answers before serving them, and adds canary-based RAG protection.

The main idea is that most local AI tools can generate answers, but very few try to decide whether an answer should be shown at all.

NEXUS has a trust layer called ReflectScore that sits between the model output and the user:

- low-risk answers are served
- medium-risk answers are shown with a warning
- high-risk answers are blocked and can be rerouted to a stronger model

Current shipped pieces:

- `CompressX` for compression
- `AEON` for routing
- `ReflectScore` for hallucination scoring and response gating
- `CanaryRAG` and `CanaryVaults` for canary seeding and leak monitoring
- CLI, FastAPI backend, and React dashboard

The launch path is intentionally narrow right now:

- Ollama
- `phi3:mini`
- `nexus doctor` for setup checks

What already works:

- install works
- CLI works
- API works
- dashboard builds
- tests and CI are in place
- local fallback modes exist when optional services are missing

What is still honest about the current state:

- `nexus init` currently reports a serving-model proxy benchmark, not direct compressed-artifact execution
- `nexus protect --seed` has a local fallback
- `nexus protect --check` is still mostly a hosted-service path unless CanaryVaults is configured

Repo:

https://github.com/nagu-io/nexus

Would love feedback on:

- whether the trust-gating model is useful
- whether the local-first + hosted-upgrade path is compelling
- whether the canary/RAG protection story feels valuable in practice
