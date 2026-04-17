"""Desktop-facing model control surfaces for local runtime and packaging flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nexus.compress.compressor import CompressXEngine
from nexus.router.provider_runtime import configured_local_backend, preferred_cloud_provider

MB = 1024 * 1024
SINGLE_APP_BUDGET_MB = 1024
DESKTOP_RUNTIME_RESERVE_MB = 180


class ModelControlCenter:
    """Summarize local model artifacts and single-app readiness for the desktop UI."""

    def __init__(
        self,
        *,
        config: Any,
        compress_engine: CompressXEngine | None = None,
        repo_root: str | Path | None = None,
    ):
        self.config = config
        self.compress_engine = compress_engine or CompressXEngine()
        self.repo_root = Path(repo_root or Path.cwd()).expanduser().resolve()

    def overview(self) -> dict[str, Any]:
        adapter_dir = self._resolve_local_model_dir()
        checkpoint_dir = (self.repo_root / "nexus_model").resolve()
        compress_root = self.compress_engine.output_dir.resolve()

        adapter_artifact = self._artifact_payload(
            artifact_id="adapter",
            path=adapter_dir,
            label="Adapter Pack",
            description="The local LoRA adapter pack used by the embedded adapter runtime.",
        )
        checkpoint_artifact = self._artifact_payload(
            artifact_id="checkpoint",
            path=checkpoint_dir,
            label="Source Checkpoint",
            description="The full training checkpoint used for export, merge, and heavy packaging flows.",
        )
        compress_cache = self._artifact_payload(
            artifact_id="compress_cache",
            path=compress_root,
            label="CompressX Cache",
            description="Generated compression artifacts and manifests for ship-ready local packs.",
        )

        compressed_models = [self._compressed_model_payload(item) for item in self.compress_engine.list_models()]
        packaging = self._packaging_payload(adapter_artifact, compressed_models)

        return {
            "runtime": {
                "backend": configured_local_backend(self.config),
                "launch_model": str(getattr(self.config, "nexus_model", "phi3:mini") or "phi3:mini"),
                "resolved_launch_model": self.compress_engine._resolve_model_name(
                    str(getattr(self.config, "nexus_model", "phi3:mini") or "phi3:mini")
                ),
                "local_model_dir": str(adapter_dir),
                "adapter_ready": adapter_artifact["exists"],
                "cloud_fallback": preferred_cloud_provider(self.config) or "none",
                "single_app_mode": configured_local_backend(self.config) == "adapter",
            },
            "artifacts": [adapter_artifact, checkpoint_artifact, compress_cache],
            "compressed_models": compressed_models,
            "packaging": packaging,
            "compressx": {
                "manifest_path": str(self.compress_engine.manifest_path),
                "available_outputs": len(compressed_models),
                "launch_alias_supported": True,
            },
        }

    def update_runtime(
        self,
        *,
        backend: str | None = None,
        local_model_dir: str | None = None,
        launch_model: str | None = None,
    ) -> dict[str, Any]:
        if backend is not None:
            normalized = str(backend).strip().lower()
            if normalized not in {"ollama", "adapter"}:
                raise ValueError("Local backend must be 'ollama' or 'adapter'.")
            self.config.local_model_backend = normalized

        if local_model_dir is not None and str(local_model_dir).strip():
            self.config.local_model_dir = str(local_model_dir).strip()

        if launch_model is not None and str(launch_model).strip():
            self.config.nexus_model = str(launch_model).strip()

        return self.overview()

    def compress_launch_model(self, *, bits: int = 4) -> dict[str, Any]:
        quant_bits = max(2, min(int(bits), 8))
        artifact_path = self.compress_engine.compress(str(self.config.nexus_model), bits=quant_bits)
        overview = self.overview()
        artifact = next(
            (
                item
                for item in overview["compressed_models"]
                if Path(item["path"]).resolve() == artifact_path.resolve()
            ),
            None,
        )
        return {
            "ok": True,
            "bits": quant_bits,
            "artifact": artifact,
            "overview": overview,
        }

    def _resolve_local_model_dir(self) -> Path:
        raw = str(getattr(self.config, "local_model_dir", "lora_model") or "lora_model")
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.repo_root / candidate).resolve()

    def _artifact_payload(
        self,
        *,
        artifact_id: str,
        path: Path,
        label: str,
        description: str,
    ) -> dict[str, Any]:
        exists = path.exists()
        size_bytes = self._path_size(path) if exists else 0
        return {
            "id": artifact_id,
            "label": label,
            "description": description,
            "path": str(path),
            "exists": exists,
            "kind": "directory" if path.is_dir() else "file" if exists else "missing",
            "entry_count": self._entry_count(path) if exists and path.is_dir() else 0,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / MB, 1),
        }

    def _compressed_model_payload(self, item: dict[str, Any]) -> dict[str, Any]:
        output_path = Path(str(item.get("path", "") or "")).expanduser().resolve()
        exists = output_path.exists()
        size_bytes = self._path_size(output_path) if exists else 0
        return {
            "name": item.get("name"),
            "path": str(output_path),
            "exists": exists,
            "bits": item.get("bits"),
            "ratio": item.get("ratio"),
            "source": item.get("source", "unknown"),
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / MB, 1),
            "real_measurement": item.get("source") == "native_gptq",
        }

    def _packaging_payload(
        self,
        adapter_artifact: dict[str, Any],
        compressed_models: list[dict[str, Any]],
    ) -> dict[str, Any]:
        real_outputs = [item for item in compressed_models if item["real_measurement"] and item["exists"]]
        launch_model = str(getattr(self.config, "nexus_model", "phi3:mini") or "phi3:mini")
        selected_output = next((item for item in real_outputs if item["name"] == launch_model), None)
        if selected_output is None and real_outputs:
            selected_output = min(real_outputs, key=lambda item: item["size_bytes"])

        estimated_total_mb = None
        sub_gb_possible = None
        readiness = "prototype"
        message = (
            "Package one small launch model plus adapter packs inside the desktop app. "
            "That is the credible path to a single-install local runtime."
        )

        if selected_output is not None:
            estimated_total_mb = round(
                DESKTOP_RUNTIME_RESERVE_MB + float(selected_output["size_mb"]) + float(adapter_artifact["size_mb"]),
                1,
            )
            sub_gb_possible = estimated_total_mb <= SINGLE_APP_BUDGET_MB
            readiness = "ready" if sub_gb_possible else "over_budget"
            message = (
                "The current measured launch pack can fit inside a single installer."
                if sub_gb_possible
                else "The current measured launch pack is still too large for a sub-1GB single installer."
            )
        elif any(item["source"] == "mock" for item in compressed_models):
            readiness = "mock"
            message = (
                "CompressX has mock outputs only right now. Install the real compression backend before trusting any "
                "sub-1GB packaging estimate."
            )
        elif not compressed_models:
            message = (
                "No compressed launch pack exists yet. Run CompressX from the desktop app to generate a ship-ready "
                "local model artifact."
            )

        return {
            "budget_mb": SINGLE_APP_BUDGET_MB,
            "runtime_reserve_mb": DESKTOP_RUNTIME_RESERVE_MB,
            "adapter_pack_mb": adapter_artifact["size_mb"],
            "selected_launch_pack_mb": selected_output["size_mb"] if selected_output is not None else None,
            "selected_launch_pack_name": selected_output["name"] if selected_output is not None else None,
            "estimated_total_mb": estimated_total_mb,
            "sub_gb_possible": sub_gb_possible,
            "readiness": readiness,
            "message": message,
        }

    def _path_size(self, path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        return 0

    def _entry_count(self, path: Path) -> int:
        if not path.is_dir():
            return 0
        return sum(1 for item in path.rglob("*") if item.is_file())
