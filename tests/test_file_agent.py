import unittest
from pathlib import Path

from nexus.agents.file_agent import FileAgent


class FileAgentSafetyTests(unittest.TestCase):
    def setUp(self):
        self.agent = FileAgent()

    def test_disallows_prefix_sibling_path(self):
        repo_root = Path.cwd().resolve()
        sibling_path = repo_root.parent / f"{repo_root.name}-malicious" / "secret.txt"
        self.assertFalse(self.agent._is_safe_path(sibling_path))

    def test_allows_repo_path(self):
        self.assertTrue(self.agent._is_safe_path(Path.cwd() / "README.md"))


if __name__ == "__main__":
    unittest.main()
