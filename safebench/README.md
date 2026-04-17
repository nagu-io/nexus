# NEXUS SafeBench

NEXUS SafeBench is a standalone benchmark scaffold for evaluating autonomous coding agents on more than raw task completion.

The long-term goal is to measure four things together:

1. Can the agent solve real coding tasks?
2. Does it stay honest about uncertainty, commands, files, and results?
3. Does it resist prompt injection and secret exfiltration attempts?
4. Can it recover safely from failing runs and partial progress?

This folder is intentionally self-contained so it can be moved into its own Git repository later with minimal cleanup.

## Initial Scope

The first scaffold includes:

- a manifest-driven benchmark definition
- a small CLI for overview, validation, run planning, dataset inspection, and dry-run execution
- starter benchmark tracks for task success, honesty, security, and recovery
- a first scenario dataset for the coding honesty track
- unit tests for manifest loading, dataset loading, and run generation

This is not the full benchmark harness yet. It is the project skeleton we can now grow into a publishable benchmark.

## Folder Layout

```text
safebench/
  datasets/
    coding_honesty.json
  manifests/
    core.json
  src/nexus_safebench/
    cli.py
    dataset.py
    manifest.py
    models.py
    runner.py
    scoring.py
  tests/
    test_dataset.py
    test_manifest.py
    test_runner.py
```

## Getting Started

From this folder:

```bash
pip install -e .
safebench overview
safebench validate
safebench scenarios --suite coding_honesty
safebench plan --suite coding_honesty
safebench run --suite coding_honesty
python -m unittest discover -s tests -v
```

`safebench run` currently executes a dry-run adapter using scripted scenario traces from the dataset and emits benchmark artifacts under `runs/`.

## Suggested Next Steps

1. Add real execution adapters for NEXUS sessions and external agent traces.
2. Add a sandbox adapter layer for Docker-backed benchmark execution.
3. Add scenario datasets for secret leak resilience, issue resolution, and recovery.
4. Add richer scoring for calibration, refusal quality, and policy compliance.
5. Publish methodology before publishing leaderboard claims.
