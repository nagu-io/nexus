# Contributing to NEXUS

Thanks for your interest in contributing to NEXUS! This guide will help you get set up.

## Development Setup

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** (with npm)
- **Rust toolchain** (only needed for Tauri desktop builds)
- **Git**

### Clone and Install

```bash
git clone https://github.com/nagu-io/nexus.git
cd nexus

# Python backend
pip install -e .

# Dashboard frontend
npm --prefix dashboard install

# Copy environment config
cp .env.example .env
```

### Running Locally

```bash
# Terminal 1: Backend API
python -m uvicorn nexus.api:app --host 127.0.0.1 --port 8000

# Terminal 2: Dashboard dev server
npm --prefix dashboard run dev
```

### Running Tests

```bash
# Backend tests
python -m pytest tests/ -v

# Dashboard build check
npm --prefix dashboard run build
```

## Project Structure

```
nexus/              # Core Python backend — agents, runtime, API, router
dashboard/          # React dashboard (Vite)
src-tauri/          # Tauri desktop shell
tests/              # Backend test suite
tools/              # Dev helpers and build scripts
docs/               # Architecture and design docs
safebench/          # Safety and honesty benchmarks
lora_model/         # LoRA adapter config (weights hosted externally)
```

## Code Style

- **Python:** Follow PEP 8. Use type hints. Keep functions focused.
- **JavaScript/JSX:** Use functional React components. Prefer `const` over `let`.
- **Commits:** Use clear, descriptive commit messages. One logical change per commit.

## Pull Request Guidelines

1. **Fork** the repo and create a feature branch from `main`
2. **Test** your changes: run `pytest` and ensure the dashboard builds
3. **Document** any new features or config options
4. **Keep PRs focused** — one feature or fix per PR
5. **Describe** what your PR does and why in the PR description

## Areas Where Help Is Welcome

- Cross-platform testing (macOS, Linux)
- Model quality improvements and training data
- Dashboard UI/UX polish
- Additional agent capabilities
- Documentation and tutorials
- SafeBench scenario expansion

## Reporting Issues

- Use GitHub Issues
- Include steps to reproduce
- Mention your OS, Python version, and Node version
- Attach relevant logs from the terminal or dashboard

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
