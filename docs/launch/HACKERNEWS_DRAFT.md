# Hacker News Draft

Show HN: NEXUS - local-first AI coding desktop with experimental Hive, bundled adapter, and trust gating

I built NEXUS as a local-first AI developer workspace that runs on your machine, edits real repos, scores answers before serving them, and packages into a desktop app.

The repo now includes a few things I almost never see in the same local project:

- a packaged Electron desktop app
- a bundled fine-tuned adapter path
- a trust gate that can warn on or block risky answers
- canary-based RAG protection
- an experimental Hive runtime for distributed answer search
- a benchmark scaffold focused on honesty, safety, recovery, and task success

The architecture is:

- `IntentParser -> PlannerEngine -> BlueprintGenerator -> Orchestrator`
- runtime tools for files, terminal execution, retries, traces, and critics
- `ReflectScore` to score hallucination risk before showing answers
- `CanaryRAG` / `CanaryVaults` for leak detection and trust checks
- `Hive` to explore trust-scored distributed search instead of just one local inference path

What is live in the repo right now:

- CLI
- FastAPI backend
- React dashboard
- Electron desktop shell
- model control center
- Hive panel
- packaged Windows installer flow
- SafeBench scaffold

What is honest about the current state:

- Hive is real code and UI, but the internet-wide peer mesh is still experimental
- the desktop build has a bundled adapter path, but I am not claiming frontier-model quality from a tiny local package
- some optional security and cloud paths still depend on external services or keys

The interesting part for me is not only generation. It is deciding whether an answer should be shown at all.

Repo:

https://github.com/nagu-io/nexus

Would love feedback on:

- whether a trust-gated local coding workspace feels more valuable than another plain chat wrapper
- whether the bundled adapter + desktop installer story is compelling
- whether the Hive direction feels interesting even in experimental form
