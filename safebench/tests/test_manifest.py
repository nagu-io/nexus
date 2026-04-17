import json
import tempfile
import unittest
from pathlib import Path

from nexus_safebench.manifest import ManifestError, load_manifest
from nexus_safebench.runner import build_run_plan


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.core_manifest = self.project_root / "manifests" / "core.json"

    def test_core_manifest_loads(self):
        manifest = load_manifest(self.core_manifest)
        self.assertEqual(manifest.name, "NEXUS SafeBench Core")
        self.assertIn("coding_honesty", manifest.suite_ids())
        self.assertIn("secret_exfiltration_rate", manifest.metric_ids())

    def test_run_plan_can_target_single_suite(self):
        manifest = load_manifest(self.core_manifest)
        plan = build_run_plan(manifest, suite_id="coding_honesty")
        self.assertEqual(len(plan.selected_suites), 1)
        self.assertEqual(plan.selected_suites[0].id, "coding_honesty")
        self.assertIn("unsupported_claim_rate", {metric.id for metric in plan.selected_metrics})

    def test_manifest_rejects_unknown_metric_reference(self):
        payload = {
            "name": "Broken",
            "version": "0.1.0",
            "description": "Broken manifest",
            "metrics": [
                {"id": "known_metric", "label": "Known", "description": "Known metric"}
            ],
            "suites": [
                {
                    "id": "broken_suite",
                    "label": "Broken Suite",
                    "objective": "Break validation",
                    "description": "References a missing metric",
                    "scenario_types": ["placeholder"],
                    "key_metrics": ["missing_metric"],
                    "acceptance_criteria": ["none"]
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "broken.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ManifestError):
                load_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
