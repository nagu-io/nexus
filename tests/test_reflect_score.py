import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from nexus.reflect.reflect_score import ReflectScore


class ReflectScoreTests(unittest.TestCase):
    def test_heuristic_scores_dates_urls_and_percentages(self):
        scorer = ReflectScore()
        score = scorer._heuristic_score(
            "This happened in 2024. See https://example.com and 12.5% of users were affected."
        )
        self.assertGreaterEqual(score, 0.1)

    def test_benchmark_model_requires_existing_artifact(self):
        scorer = ReflectScore()
        with self.assertRaises(FileNotFoundError):
            asyncio.run(scorer.benchmark_model("D:/definitely-missing-model", n_samples=0))

    def test_benchmark_model_reports_proxy_mode(self):
        scorer = ReflectScore()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = temp_path / "phi3_mini_gptq_4bit"
            model_dir.mkdir()
            with open(model_dir / "compress_meta.json", "w", encoding="utf-8") as handle:
                json.dump({"compression_ratio": 3.6}, handle)

            scorer.results_dir = temp_path / "reflect_results"
            scorer.results_dir.mkdir()

            result = asyncio.run(scorer.benchmark_model(model_dir, n_samples=0))

        self.assertEqual(result["benchmark_mode"], "active_serving_backend_proxy")
        self.assertIn("active NEXUS serving model", result["benchmark_warning"])


if __name__ == "__main__":
    unittest.main()
