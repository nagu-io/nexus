import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nexus.runtime.model_control import ModelControlCenter


class FakeCompressEngine:
    def __init__(self, root: Path, outputs: list[dict]):
        self.output_dir = root / "compressed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = root / "compress_manifest.json"
        self.manifest_path.write_text("{}", encoding="utf-8")
        self._outputs = outputs

    def list_models(self) -> list[dict]:
        return list(self._outputs)

    def _resolve_model_name(self, model_name: str) -> str:
        aliases = {
            "phi3:mini": "microsoft/Phi-3-mini-4k-instruct",
        }
        return aliases.get(model_name, model_name)

    def compress(self, model_name: str, bits: int = 4) -> Path:
        target = self.output_dir / f"{model_name.replace(':', '_')}_{bits}bit"
        target.mkdir(parents=True, exist_ok=True)
        (target / "weights.bin").write_bytes(b"0" * (32 * 1024 * 1024))
        self._outputs.append(
            {
                "name": model_name,
                "bits": bits,
                "ratio": 3.8,
                "path": str(target),
                "source": "native_gptq",
            }
        )
        return target


class ModelControlCenterTests(unittest.TestCase):
    def test_real_compressed_pack_counts_toward_sub_gb_budget(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter_dir = root / "lora_model"
            adapter_dir.mkdir()
            (adapter_dir / "adapter.bin").write_bytes(b"0" * (120 * 1024 * 1024))

            checkpoint_dir = root / "nexus_model"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "checkpoint.bin").write_bytes(b"0" * (8 * 1024 * 1024))

            compressed_dir = root / "compressed" / "phi3_pack"
            compressed_dir.mkdir(parents=True)
            (compressed_dir / "model.gguf").write_bytes(b"0" * (410 * 1024 * 1024))

            config = SimpleNamespace(
                local_model_backend="adapter",
                local_model_dir="lora_model",
                nexus_model="phi3:mini",
                openrouter_api_key="",
                anthropic_api_key="",
            )
            engine = FakeCompressEngine(
                root,
                outputs=[
                    {
                        "name": "phi3:mini",
                        "bits": 4,
                        "ratio": 3.7,
                        "path": str(compressed_dir),
                        "source": "native_gptq",
                    }
                ],
            )

            center = ModelControlCenter(config=config, compress_engine=engine, repo_root=root)
            overview = center.overview()

        self.assertTrue(overview["runtime"]["single_app_mode"])
        self.assertTrue(overview["runtime"]["adapter_ready"])
        self.assertEqual(overview["packaging"]["selected_launch_pack_name"], "phi3:mini")
        self.assertTrue(overview["packaging"]["sub_gb_possible"])
        self.assertEqual(overview["packaging"]["readiness"], "ready")

    def test_mock_outputs_do_not_claim_sub_gb_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter_dir = root / "lora_model"
            adapter_dir.mkdir()
            (adapter_dir / "adapter.bin").write_bytes(b"0" * (100 * 1024 * 1024))

            mock_dir = root / "compressed" / "phi3_mock"
            mock_dir.mkdir(parents=True)
            (mock_dir / "compress_meta.json").write_text("{}", encoding="utf-8")

            config = SimpleNamespace(
                local_model_backend="adapter",
                local_model_dir="lora_model",
                nexus_model="phi3:mini",
                openrouter_api_key="",
                anthropic_api_key="",
            )
            engine = FakeCompressEngine(
                root,
                outputs=[
                    {
                        "name": "phi3:mini",
                        "bits": 4,
                        "ratio": 3.6,
                        "path": str(mock_dir),
                        "source": "mock",
                    }
                ],
            )

            center = ModelControlCenter(config=config, compress_engine=engine, repo_root=root)
            overview = center.overview()

        self.assertIsNone(overview["packaging"]["sub_gb_possible"])
        self.assertEqual(overview["packaging"]["readiness"], "mock")
        self.assertIn("mock outputs only", overview["packaging"]["message"])

    def test_update_runtime_changes_backend_and_launch_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "lora_model").mkdir()
            config = SimpleNamespace(
                local_model_backend="ollama",
                local_model_dir="lora_model",
                nexus_model="phi3:mini",
                openrouter_api_key="",
                anthropic_api_key="",
            )
            center = ModelControlCenter(
                config=config,
                compress_engine=FakeCompressEngine(root, outputs=[]),
                repo_root=root,
            )

            overview = center.update_runtime(
                backend="adapter",
                local_model_dir="custom_adapter",
                launch_model="custom-launch",
            )

        self.assertEqual(config.local_model_backend, "adapter")
        self.assertEqual(config.local_model_dir, "custom_adapter")
        self.assertEqual(config.nexus_model, "custom-launch")
        self.assertEqual(overview["runtime"]["backend"], "adapter")
        self.assertEqual(overview["runtime"]["launch_model"], "custom-launch")


if __name__ == "__main__":
    unittest.main()
