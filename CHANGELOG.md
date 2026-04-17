# Changelog

All notable changes to NEXUS will be documented in this file.

## [0.2.8] — 2026-04-18

### Added
- **Tauri desktop shell** replacing the previous Electron prototype
- **Aura Obsidian design system** — cinematic dark UI with micro-animations
- **Bundled LoRA adapter** path for fully offline local inference
- **Fallback scaffold system** — guarantees file creation when local model output isn't parseable
- **Session isolation** — chat history and memory are now scoped per workspace
- **Live Terminal** panel for real-time agent activity tracing
- **Intent routing fix** — workspace action tasks always route to coding agent
- **SafeBench** benchmark scaffold for honesty, safety, and task-success evaluation
- **NEXUS Hive** experimental distributed intelligence runtime
- **CanaryRAG / CanaryVaults** trust layer integration
- **CompressX** model compression and artifact management
- **ReflectScore** hallucination gating with serve/warn/block decisions

### Fixed
- Agent routing: action verbs (`fix`, `build`, `inspect`) now correctly route to coding agent in workspace mode
- Chat session leakage across different repository workspaces
- `BuildArtifactMaterializer` extraction failures on empty model output
- `max_new_tokens` increased from 220 to 3000 for complete code generation
- Removed redundant `quantization_config` that caused HuggingFace warnings
- Windows `torchao` compatibility patches for PyTorch 2.5+

### Known Issues
- `Failed to find CUDA` warning appears on systems without CUDA toolkit (harmless — CPU fallback works)
- `torch_dtype is deprecated` warning from HuggingFace (cosmetic)
- Local LoRA model generates short text; fallback scaffold compensates
