# Hacker News Draft

Show HN: NEXUS - open-source private AI developer OS with compression, routing, hallucination checks, and RAG protection

I built NEXUS as a local-first AI developer OS that combines several smaller projects I had already been working on into one product.

What it does:

- compresses local models through a CompressX adapter
- routes work across local models, cloud models, and specialist agents
- scores every response with ReflectScore before serving it
- blocks or warns on suspicious answers
- seeds canaries into RAG sources and checks for leaks
- ships with a CLI, API, and React dashboard

The project is open source and runs on your machine. The app is built in Python with a React dashboard, and the repo is kept focused on the shipped NEXUS product rather than a pile of separate experiments.

What I think is interesting is the trust layer. A lot of local AI tooling can generate answers, but much less of it tries to decide whether an answer should be shown at all. ReflectScore is the layer in NEXUS that serves, warns, or blocks before the response reaches the user.

Current state:

- install works
- CLI works
- FastAPI endpoints work
- dashboard builds
- runtime setup is checked with `nexus doctor`
- local fallback modes are in place when optional services are missing

I would love feedback on:

- the trust-gating approach
- how useful the compression plus benchmark story is
- whether the canary/RAG protection layer feels valuable in practice

Repo structure:

- `nexus/` is the shipped product
- `dashboard/` is the frontend
- `docs/` contains launch and project docs
