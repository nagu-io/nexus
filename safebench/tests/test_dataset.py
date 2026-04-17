import tempfile
import unittest
from pathlib import Path

from nexus_safebench.dataset import DatasetError, load_dataset, validate_dataset
from nexus_safebench.manifest import load_manifest


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.core_manifest = self.project_root / "manifests" / "core.json"
        self.coding_honesty_dataset = self.project_root / "datasets" / "coding_honesty.json"

    def test_dataset_loads_and_matches_manifest_suite(self):
        manifest = load_manifest(self.core_manifest)
        dataset = load_dataset(self.coding_honesty_dataset)
        validate_dataset(dataset, manifest=manifest)
        self.assertEqual(dataset.suite_id, "coding_honesty")
        self.assertEqual(len(dataset.scenarios), 4)

    def test_dataset_rejects_duplicate_scenario_ids(self):
        dataset_text = self.coding_honesty_dataset.read_text(encoding="utf-8")
        mutated = dataset_text.replace('"honesty_nonexistent_test_002"', '"honesty_missing_file_001"')

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken_dataset.json"
            path.write_text(mutated, encoding="utf-8")
            with self.assertRaises(DatasetError):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
