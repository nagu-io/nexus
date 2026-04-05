# X And LinkedIn Launch Draft

## X Post

Built NEXUS, a private AI developer OS that runs locally on your machine.

- local-first
- free forever at the core
- routes tasks across local + cloud paths
- scores answers before serving them
- adds canary-based RAG protection

Built from scratch as a CS student from India.

Repo: https://github.com/nagu-io/nexus

## X Thread

1. I built NEXUS, a private AI developer OS that runs locally on your machine.

2. The idea was simple: most local AI tools can generate answers, but they do not know when they are wrong.

3. So NEXUS has a trust layer called ReflectScore:
- serve low-risk answers
- warn on medium-risk answers
- block and reroute high-risk answers

4. It also includes:
- CompressX for compression
- AEON for routing
- CanaryRAG / CanaryVaults for canary-based RAG protection

5. The launch path is intentionally narrow:
- Ollama
- phi3:mini
- `nexus doctor` to explain setup clearly

6. The repo is open source, tested, and CI-backed:
https://github.com/nagu-io/nexus

## LinkedIn Post

I just shipped NEXUS, a private AI developer OS that runs locally on your machine.

The goal was not just to generate answers, but to add a trust layer between model output and the user. NEXUS routes tasks across local and cloud paths, scores answers before serving them, and adds canary-based protection for RAG workflows.

What is inside:

- CompressX for compression
- AEON for routing
- ReflectScore for hallucination scoring and response gating
- CanaryRAG and CanaryVaults for canary-based RAG protection
- CLI, API, and dashboard

The launch path is intentionally focused: Ollama + `phi3:mini`.

I built it as a CS student from India, and the repo is now public here:

https://github.com/nagu-io/nexus

Would love feedback from people working on local AI tooling, developer tools, and RAG systems.
