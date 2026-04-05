import unittest

from nexus.compress.compressor import CompressXEngine


class CompressXEngineTests(unittest.TestCase):
    def test_phi3_launch_alias_maps_to_safe_source_and_slug(self):
        engine = CompressXEngine()
        self.assertEqual(engine._resolve_model_name("phi3:mini"), "microsoft/Phi-3-mini-4k-instruct")
        self.assertEqual(engine._slugify_model_name("phi3:mini"), "phi3_mini")


if __name__ == "__main__":
    unittest.main()
