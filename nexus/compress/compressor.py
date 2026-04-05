"""
CompressX Engine - self-contained compression layer for NEXUS.
Uses direct GPTQ when available and falls back to a mock compression flow.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rich.console import Console

console = Console()

MODEL_ALIASES = {
    "phi3:mini": "microsoft/Phi-3-mini-4k-instruct",
}


class CompressXEngine:
    """Main NEXUS compression engine."""

    def __init__(self):
        from nexus.config import config

        self.config = config
        self.output_dir = config.data_dir / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = config.data_dir / "compress_manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        """Load the compression manifest from disk."""
        if self.manifest_path.exists():
            with open(self.manifest_path, encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def _resolve_model_name(self, model_name: str) -> str:
        """Map friendly runtime aliases to benchmark/compression source models."""
        return MODEL_ALIASES.get(model_name, model_name)

    def _slugify_model_name(self, model_name: str) -> str:
        """Create a filesystem-safe folder name for model artifacts."""
        return re.sub(r'[<>:"/\\|?*\s]+', "_", model_name).strip("_")

    def _save_manifest(self):
        """Save the compression manifest to disk."""
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, indent=2)

    def compress(self, model_name: str, bits: int = 4) -> Path:
        """Compress a HuggingFace model with GPTQ when available."""
        requested_name = model_name
        source_model_name = self._resolve_model_name(model_name)
        safe_name = self._slugify_model_name(requested_name)
        output_path = self.output_dir / f"{safe_name}_gptq_{bits}bit"

        if output_path.exists():
            console.print(f"[yellow]Model already compressed at {output_path}[/yellow]")
            return output_path

        console.print(f"[bold cyan]CompressX: Compressing {source_model_name} at {bits}-bit...[/bold cyan]")
        if source_model_name != requested_name:
            console.print(
                f"[dim]Resolved model alias '{requested_name}' to '{source_model_name}' for compression.[/dim]"
            )
        return self._compress_with_direct_gptq(
            source_model_name,
            bits,
            output_path,
            requested_name=requested_name,
        )

    def _compress_with_direct_gptq(
        self,
        model_name: str,
        bits: int,
        output_path: Path,
        requested_name: str | None = None,
    ) -> Path:
        """Direct GPTQ compression path."""
        manifest_key = requested_name or model_name
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            from transformers import AutoTokenizer

            console.print("[cyan]Step 1/5: loading tokenizer...[/cyan]")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

            console.print("[cyan]Step 2/5: setting up quantization config...[/cyan]")
            quantize_config = BaseQuantizeConfig(bits=bits, group_size=128, desc_act=False)

            console.print("[cyan]Step 3/5: loading model weights...[/cyan]")
            model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config=quantize_config)

            console.print("[cyan]Step 4/5: running GPTQ calibration...[/cyan]")
            calibration_data = self._get_calibration_data(tokenizer)
            model.quantize(calibration_data)

            console.print("[cyan]Step 5/5: saving compressed model...[/cyan]")
            model.save_quantized(str(output_path))
            tokenizer.save_pretrained(str(output_path))

            original_size = self._estimate_original_size(model_name)
            compressed_size = self._get_dir_size(output_path)
            ratio = original_size / max(compressed_size, 1)

            self.manifest[manifest_key] = {
                "compressed_path": str(output_path),
                "bits": bits,
                "compression_ratio": ratio,
                "original_size_mb": original_size / (1024 * 1024),
                "compressed_size_mb": compressed_size / (1024 * 1024),
                "source": "native_gptq",
                "source_model": model_name,
            }
            self._save_manifest()

            console.print("[bold green]Compression complete![/bold green]")
            console.print(f"  Ratio: {ratio:.1f}x | Output: {output_path}")
            return output_path

        except ImportError:
            console.print("[yellow]auto-gptq not available. Using mock compression for demo.[/yellow]")
            return self._mock_compress(model_name, output_path, bits, requested_name=requested_name)

    def _mock_compress(
        self,
        model_name: str,
        output_path: Path,
        bits: int,
        requested_name: str | None = None,
    ) -> Path:
        """Mock compression for demo/testing when no real compression backend is available."""
        manifest_key = requested_name or model_name
        output_path.mkdir(parents=True, exist_ok=True)
        mock_meta = {
            "model_name": model_name,
            "bits": bits,
            "compression_ratio": 3.6,
            "status": "mock_compressed",
            "note": "Install the optional compression dependencies for real GPTQ compression.",
        }
        with open(output_path / "compress_meta.json", "w", encoding="utf-8") as handle:
            json.dump(mock_meta, handle, indent=2)

        self.manifest[manifest_key] = {
            "compressed_path": str(output_path),
            "bits": bits,
            "compression_ratio": 3.6,
            "source": "mock",
            "source_model": model_name,
        }
        self._save_manifest()
        console.print(f"[yellow]Mock compression complete at {output_path}[/yellow]")
        return output_path

    def _get_calibration_data(self, tokenizer, n_samples: int = 512) -> list:
        """Get calibration dataset for GPTQ."""
        try:
            from datasets import load_dataset

            data = load_dataset("c4", "en", split="train", streaming=True)
            samples = []
            for index, item in enumerate(data):
                if index >= n_samples:
                    break
                encoded = tokenizer(item["text"], return_tensors="pt", max_length=512, truncation=True)
                samples.append(encoded)
            return samples
        except Exception as exc:
            console.print(f"[yellow]Calibration data load failed: {exc}. Using synthetic data.[/yellow]")
            return [{"input_ids": [[1] * 128]} for _ in range(32)]

    def _estimate_original_size(self, model_name: str) -> int:
        """Estimate original model size in bytes."""
        size_map = {
            "phi": 2_000_000_000,
            "mistral": 7_000_000_000,
            "llama": 8_000_000_000,
            "opt-125m": 125_000_000,
        }
        for key, size in size_map.items():
            if key in model_name.lower():
                return size * 2
        return 4_000_000_000

    def _get_dir_size(self, path: Path) -> int:
        """Get total size of a directory in bytes."""
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    def list_models(self) -> list:
        """List all compressed models."""
        return [
            {
                "name": name,
                "ratio": info.get("compression_ratio", 0),
                "path": info.get("compressed_path"),
                "bits": info.get("bits"),
                "source": info.get("source", "unknown"),
            }
            for name, info in self.manifest.items()
        ]
