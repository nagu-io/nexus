# NEXUS Demo Script

## Goal

Show that NEXUS installs, checks its environment, routes tasks, scores trust, and protects RAG sources.

## Terminal Demo Flow

### 1. Show the repo

```bash
dir
```

Point out:

- `nexus/` is the product runtime
- `dashboard/` is the UI
- the repo is intentionally kept to just the shipped NEXUS app and docs

### 2. Verify setup

```bash
nexus doctor
```

Say:

"NEXUS tells you exactly what is missing before you hit runtime errors."

### 3. Show system status

```bash
nexus status
```

Call out the active model and ReflectScore thresholds.

### 4. Run a routed task

```bash
nexus code "build a Python CLI that validates JWT expiration timestamps"
```

Say:

"This goes through AEON routing first, then ReflectScore scores the answer before it is served."

### 5. Run the benchmark layer

```bash
nexus reflect
```

Say:

"ReflectScore is both the benchmark system and the live trust gate."

### 6. Show canary protection

```bash
nexus protect --seed https://example.com/internal-docs
```

Say:

"If the remote service is unavailable, NEXUS still falls back to a local canary plan."

### 7. Launch the dashboard

```bash
nexus chat --dashboard
```

Open:

- `http://localhost:8000`
- `http://localhost:3000`

### 8. Close with the pitch

"NEXUS is a private AI developer OS that compresses models, routes tasks intelligently, detects hallucinations before they reach you, and protects your RAG from leaks."
