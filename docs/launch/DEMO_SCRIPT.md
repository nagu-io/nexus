# NEXUS Demo Script

## Goal

Record a 60-90 second demo that shows:

- NEXUS is a real desktop app, not just a repo
- the app runs local-first
- Hive is visible in the product
- trust gating is visible in the product
- the bundled model / model control story is visible

## Recording Plan

### 0-5 seconds: face intro

Say:

"I built NEXUS, a local-first AI developer workspace with experimental Hive intelligence, a bundled adapter path, and trust-gated outputs."

### 5-12 seconds: show the desktop app

Open the desktop shell and keep the title bar, sidebar, and mission cards in frame.

Say:

"This is the real desktop app. It is not just a chat window. It has files, runtime telemetry, model control, and Hive built in."

### 12-22 seconds: show model control

Open the `Models` tab and show the local runtime plus bundled adapter state.

Say:

"The desktop build ships with a bundled local adapter path, so the app can manage its own local runtime instead of sending users off to wire everything manually."

### 22-35 seconds: show a routed repo task

Open `Files`, load a repo, then go to `Chat` and send a short coding request.

Suggested prompt:

```text
fix the login flow and explain what changed
```

Say:

"NEXUS compiles a goal into a workflow, runs against the repo, and streams runtime decisions while it works."

### 35-47 seconds: show trust gating

Keep the runtime cards and Reflect meter visible while the response comes back.

Say:

"ReflectScore sits between the model output and the user. Low-risk answers are served, medium-risk answers are warned on, and risky answers can be blocked."

### 47-60 seconds: show Hive

Open the `Hive` tab and run a Hive demo prompt.

Suggested prompt:

```text
build me a full authentication system
```

Say:

"Hive is the experimental distributed layer. Instead of splitting one inference across the internet, it distributes the search for the best answer, scores returned candidates, and assembles the strongest result."

### 60-75 seconds: show benchmark / trust proof

Flash the repo tree or SafeBench folder.

Say:

"I also built SafeBench to benchmark honesty, safety, recovery, and task success instead of only raw completion."

### Close

Say:

"NEXUS is local-first, packaged as a desktop app, and already includes the trust and distributed pieces most projects only talk about."

## Backup Commands

Use these if you want a longer cut or terminal-based version:

```powershell
nexus doctor
nexus status
nexus reflect
nexus hive "build me a full authentication system"
python -m uvicorn nexus.api:app --host 127.0.0.1 --port 8000
npm --prefix desktop run dev
```
