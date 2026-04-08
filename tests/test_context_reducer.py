"""Unit tests for prompt-sized context reduction."""

import unittest

from nexus.runtime.context_reducer import HeuristicContextReducer


class HeuristicContextReducerTests(unittest.TestCase):
    def test_short_text_passes_through_unchanged(self):
        reducer = HeuristicContextReducer(threshold_chars=400, target_chars=220)
        text = "Workflow goal: hello\n\nTask: keep this prompt unchanged"

        result = reducer.reduce(text)

        self.assertFalse(result.reduced)
        self.assertEqual(result.text, text)
        self.assertEqual(result.strategy, "pass_through")

    def test_long_prompt_is_reduced_and_keeps_core_headers(self):
        reducer = HeuristicContextReducer(threshold_chars=500, target_chars=320)
        long_text = (
            "Workflow goal: Build a resilient app\n\n"
            "Primary intent: coding\n\n"
            "Task type: solution\n\n"
            "Task: Use the gathered context to produce code\n\n"
            "Shared memory context:\n"
            + ("dependency output line\n" * 120)
            + "\n\nWorkspace files:\n"
            + "\n".join(f"- src/file_{i}.py" for i in range(50))
        )

        result = reducer.reduce(long_text)

        self.assertTrue(result.reduced)
        self.assertLessEqual(len(result.text), 320)
        self.assertIn("Workflow goal:", result.text)
        self.assertIn("Task:", result.text)
        self.assertIn("Context reducer note:", result.text)


if __name__ == "__main__":
    unittest.main()
