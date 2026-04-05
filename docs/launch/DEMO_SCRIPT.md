# NEXUS Demo Script

## Goal

Record a 60-second demo that shows:

- NEXUS runs locally
- `nexus doctor` explains setup clearly
- routing + ReflectScore are visible
- canary protection has a useful fallback story

## Recording Plan

### 0-5 seconds: face intro

Say:

"I am a CS student from India, and I built NEXUS, a private AI developer OS that runs locally on your machine."

### 5-10 seconds: show the repo

```powershell
Get-ChildItem
```

Say:

"This repo is just the shipped product: runtime, dashboard, docs, and tests."

### 10-20 seconds: show setup check

```powershell
nexus doctor
```

Say:

"NEXUS checks Ollama, the default launch model, and optional services before runtime."

### 20-30 seconds: show the launch path

```powershell
nexus init
```

If Ollama is not ready, keep the output in frame.

Say:

"The launch path is intentionally opinionated: Ollama plus phi3:mini."

### 30-42 seconds: show a routed coding task

```powershell
nexus code "build a Python CLI that validates JWT expiration timestamps"
```

Say:

"AEON routes the task, then ReflectScore decides whether to serve, warn, or block the answer."

### 42-52 seconds: show canary protection

```powershell
nexus protect --seed https://example.com/internal-docs
```

Say:

"If CanaryVaults is not configured, NEXUS still builds a local canary plan instead of failing."

### 52-60 seconds: close

Say:

"NEXUS is open source, runs locally, and is free forever at the core. If you want real-time RAG leak monitoring, that upgrade is CanaryVaults."

## Backup Commands

Use these if you want a longer version after the 60-second cut:

```powershell
nexus status
nexus reflect
nexus chat --dashboard
```
