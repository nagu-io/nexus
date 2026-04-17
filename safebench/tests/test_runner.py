import json
import tempfile
import unittest
from pathlib import Path

from nexus_safebench.dataset import load_dataset
from nexus_safebench.manifest import load_manifest
from nexus_safebench.runner import run_dataset


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.core_manifest = self.project_root / "manifests" / "core.json"
        self.coding_honesty_dataset = self.project_root / "datasets" / "coding_honesty.json"

    def test_run_dataset_emits_summary_and_results(self):
        manifest = load_manifest(self.core_manifest)
        dataset = load_dataset(self.coding_honesty_dataset)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_dataset(manifest, dataset, output_root=temp_dir)
            output_dir = Path(result.output_dir)

            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "scenario_results.jsonl").exists())
            self.assertTrue((output_dir / "dataset_snapshot.json").exists())

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["suite_id"], "coding_honesty")
            self.assertEqual(summary["total_scenarios"], 4)
            self.assertIn("unsupported_claim_rate", summary["metric_averages"])
            self.assertEqual(len(result.scenario_results), 4)


if __name__ == "__main__":
    unittest.main()
