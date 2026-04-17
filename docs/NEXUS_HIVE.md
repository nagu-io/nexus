# NEXUS Hive

NEXUS Hive is the distributed layer that can sit on top of the current local-first runtime without replacing it.

The key idea is not "split one forward pass across the internet." That loses on latency and coordination. The practical version is "distribute the search for the best answer":

- AEON decomposes or replicates the task.
- Trusted nodes run the same or related subtasks in parallel while idle.
- ReflectScore scores returned candidates.
- The best answer wins, and the top few can be assembled into one stronger response.
- Canary-backed trust scoring keeps poisoned or low-quality nodes from staying in the network.

## How It Fits This Repo

- `nexus.router.mind_router.MindRouter`
  Already decides where work should go. Hive extends that decision from local or cloud routes into trusted peer routes.

- `nexus.orchestrator.Orchestrator`
  Already knows how to split work into tasks, evaluate outputs, and retry. Hive can reuse that shape for distributed subtasks instead of inventing a second execution model.

- `nexus.reflect.reflect_score.ReflectScore`
  Already scores answer risk. Hive uses it to rank returned candidates from multiple nodes and block low-trust outputs.

- `nexus.canary`
  Already provides the right trust language. Hive turns canaries into node health checks rather than only RAG leak checks.

## What Is Implemented Now

The first foundation slice lives in `nexus/hive/`:

- `models.py`
  Node, task, candidate, dispatch-plan, and consensus data models.

- `trust.py`
  `NodeTrustAssessor`, which weights canary pass rate, runtime reliability, idle capacity, and security-risk penalties into a single node trust score.

- `coordinator.py`
  `HiveCoordinator`, which:
  - filters nodes by capability and trust floor
  - creates a replicated dispatch plan
  - enforces canary pass/fail results during ranking
  - runs ReflectScore against returned candidates
  - ranks the final winner with quality, trust, and latency all considered

- `privacy.py`
  Builds sealed local task envelopes so nodes only receive masked task metadata plus an opaque payload.

- `canary_runtime.py`
  Injects mixed canary challenges into the selected node set and blocks nodes that fail them.

- `assembly.py`
  Synthesizes the top clean candidates into one higher-signal final answer.

- `runtime.py`
  Ties envelopes, canaries, candidate racing, and response assembly into one end-to-end Hive run.

This means the core local-first Hive runtime is now grounded in the repo, even though transport is still simulated.

## Product Surface

Hive now has an end-to-end experimental surface in the app:

- API: `GET /hive/status` and `POST /hive/demo`
- CLI: `nexus hive "build me a full authentication system"`
- Chat shortcut: start a prompt with `/hive`
- AEON router: Hive can now be selected as a real route for explicit Hive/swarm/distributed prompts
- Desktop app: open the `Hive` tab in the left sidebar

Because the Electron shell already hosts the dashboard, the Hive panel is available in the desktop app automatically.

## What Still Needs To Be Built

1. Peer transport
   Secure node discovery, encrypted task envelopes, and request-response messaging.

2. Remote execution sandbox
   Donated idle compute must run constrained workloads, not arbitrary user code.

3. Real envelope cryptography
   The current sealed payloads are local simulation primitives. Production transport will need authenticated encryption and key exchange.

## Suggested Rollout

Phase 0: local simulation

- Feed synthetic node pools into `HiveCoordinator`
- Benchmark trust and selection logic
- Tune trust thresholds before any network traffic exists

Phase 1: trusted LAN cluster

- Same-user devices only
- Plain capability ads plus signed node identity
- No internet-wide routing yet

Phase 2: internet volunteer mesh

- Encrypted envelopes
- Canary-backed reputation
- ReflectScore-based winner selection
- Standby nodes and trust decay

Phase 3: full distributed intelligence

- AEON splits tasks across the mesh
- ReflectScore and canary trust jointly gate answers
- Top candidates are assembled, not merely picked

## Important Constraint

NEXUS Hive should preserve the local-first promise:

- Local machine keeps the full prompt and user context.
- Remote nodes receive only the minimum executable work unit.
- Trust never depends on a central company deciding who is allowed to participate.

That keeps Hive aligned with the rest of NEXUS instead of turning it into a normal hosted AI backend.
