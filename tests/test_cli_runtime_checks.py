import unittest
from contextlib import contextmanager
from unittest.mock import patch

import nexus.cli as cli_module
from nexus.config import config


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


@contextmanager
def patched_config(**updates):
    originals = {key: getattr(config, key) for key in updates}
    try:
        for key, value in updates.items():
            setattr(config, key, value)
        yield
    finally:
        for key, value in originals.items():
            setattr(config, key, value)


def fake_find_spec_missing(*missing_names: str):
    def _find_spec(name: str):
        return None if name in missing_names else object()

    return _find_spec


class RuntimeChecksTests(unittest.TestCase):
    def test_warns_when_launch_model_is_missing(self):
        with patched_config(
            nexus_model="phi3:mini",
            groq_api_key="",
            anthropic_api_key="",
            supabase_url="",
            supabase_key="",
            canaryvaults_api_key="",
        ):
            with patch("httpx.get", return_value=FakeResponse({"models": [{"name": "llama3:8b"}]})):
                checks = cli_module._runtime_checks()

        ollama_check = next(check for check in checks if check["name"] == "Ollama")
        self.assertEqual(ollama_check["level"], "warn")
        self.assertIn("ollama pull phi3:mini", ollama_check["message"])

    def test_warns_when_anthropic_sdk_is_missing(self):
        with patched_config(
            nexus_model="phi3:mini",
            groq_api_key="",
            anthropic_api_key="test-key",
            supabase_url="",
            supabase_key="",
            canaryvaults_api_key="",
        ):
            with patch("httpx.get", side_effect=RuntimeError("offline")):
                with patch("importlib.util.find_spec", side_effect=fake_find_spec_missing("anthropic")):
                    checks = cli_module._runtime_checks()

        anthropic_check = next(check for check in checks if check["name"] == "Anthropic API")
        self.assertEqual(anthropic_check["level"], "warn")
        self.assertIn("pip install anthropic", anthropic_check["message"])


if __name__ == "__main__":
    unittest.main()
