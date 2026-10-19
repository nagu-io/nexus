"""Tests for the `nexus vision pull` and `nexus vision share-teach` commands, with the Hub faked."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

import nexus.cli as cli_module
from nexus.live.model import PROMPT_VERSION

CARD = json.dumps({"prompt_version": PROMPT_VERSION, "coordinates": "normalized", "base_model": "Qwen/Qwen3-VL-2B-Instruct",
                   "screen_width": 1280})
REPOS = {
    "me/adapter": {"adapter_config.json": json.dumps({"base_model_name_or_path": "Qwen/Qwen3-VL-2B-Instruct"}),
                   "adapter_model.safetensors": "weights", "nexus_live.json": CARD},
    "me/merged": {"config.json": "{}", "model.safetensors": "weights", "nexus_live.json": CARD},
}


def fake_hub(downloads, public=()):
    hub = ModuleType("huggingface_hub")

    def snapshot_download(repo, local_dir=None, ignore_patterns=None, **kwargs):
        downloads.append((repo, local_dir))
        if local_dir is None:
            return "/cache/" + repo
        for name, content in REPOS[repo].items():
            (Path(local_dir) / name).parent.mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / name).write_text(content)
        return local_dir

    class HfApi:
        uploaded = []

        def create_repo(self, repo, repo_type=None, private=False, exist_ok=False):
            pass

        def repo_info(self, repo, repo_type=None):
            return SimpleNamespace(private=repo not in public)

        def upload_folder(self, **kwargs):
            HfApi.uploaded.append(kwargs)

    hub.snapshot_download = snapshot_download
    hub.HfApi = HfApi
    return hub


class PullTests(unittest.TestCase):
    def pull(self, repo, folder, downloads):
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub(downloads)}):
            return CliRunner().invoke(cli_module.cli, ["vision", "pull", repo, "--to", str(folder)])

    def test_a_merged_model_replaces_an_adapter_whole(self):
        # Downloading over the old folder left adapter_config.json behind, so the old adapter kept loading.
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "nexus_vision"
            downloads = []
            self.assertEqual(self.pull("me/adapter", folder, downloads).exit_code, 0)
            self.assertTrue((folder / "adapter_config.json").exists())
            result = self.pull("me/merged", folder, downloads)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertFalse((folder / "adapter_config.json").exists())
            self.assertTrue((folder / "config.json").exists())
            self.assertEqual(sorted(path.name for path in Path(root).iterdir()), ["nexus_vision"])
            self.assertIn("Ready.", result.output)

    def test_an_adapter_brings_its_base_model_too(self):
        # Otherwise the 4.3 GB base model downloads at the first request, not "once, then offline".
        with tempfile.TemporaryDirectory() as root:
            downloads = []
            result = self.pull("me/adapter", Path(root) / "nexus_vision", downloads)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual([repo for repo, _ in downloads], ["me/adapter", "Qwen/Qwen3-VL-2B-Instruct"])
            self.assertIsNone(downloads[1][1])  # into the Hugging Face cache, where loading looks
            self.assertIn("downloading that too", result.output)

    def test_a_folder_with_other_things_in_it_is_refused_with_its_reason(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "notes.txt").write_text("mine")
            downloads = []
            result = self.pull("me/merged", Path(root), downloads)
            self.assertIn("has other things in it and no model", " ".join(result.output.split()))
            self.assertEqual(((Path(root) / "notes.txt").read_text(), downloads), ("mine", []))

    def test_a_merged_model_needs_nothing_more(self):
        with tempfile.TemporaryDirectory() as root:
            downloads = []
            self.pull("me/merged", Path(root) / "nexus_vision", downloads)
            self.assertEqual([repo for repo, _ in downloads], ["me/merged"])


class ShareTeachTests(unittest.TestCase):
    def share(self, repo, public):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "examples.jsonl").write_text("")
            hub = fake_hub([], public=public)
            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                result = CliRunner().invoke(cli_module.cli, ["vision", "share-teach", repo, "--from", root], input="y\n")
            return result, hub.HfApi.uploaded

    def test_a_public_repo_is_refused(self):
        result, uploaded = self.share("me/nexus-teach", public={"me/nexus-teach"})
        self.assertEqual(uploaded, [])
        self.assertIn("public", result.output)

    def test_a_private_repo_gets_the_upload(self):
        result, uploaded = self.share("me/nexus-teach", public=())
        self.assertEqual(len(uploaded), 1, result.output)
        self.assertEqual(uploaded[0]["repo_type"], "dataset")


if __name__ == "__main__":
    unittest.main()
