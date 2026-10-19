"""Tests for nexus.live.local (the model card) and nexus.live.app's backend wiring."""

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from nexus.live import app
from nexus.live.local import MODEL_CARD, PULL_MARK, LocalVision, ModelCard, card_problem, pull_model, read_card
from nexus.live.model import PROMPT_VERSION, LlamaCppVision, ModelUnavailable, OllamaVision
from nexus.live.settings import LiveSettings


def model_folder(root: Path, card: dict | None = None, kind: str = "adapter") -> Path:
    folder = root / "nexus_vision"
    folder.mkdir()
    (folder / ("adapter_config.json" if kind == "adapter" else "config.json")).write_text("{}")
    if card is not None:
        (folder / MODEL_CARD).write_text(json.dumps(card))
    return folder


class CardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def card(self, **changes):
        values = {"prompt_version": PROMPT_VERSION, "coordinates": "normalized", "base_model": "Qwen/Qwen3-VL-2B-Instruct", "screen_width": 1280}
        values.update(changes)
        return values

    def test_a_matching_model_has_no_problem(self):
        self.assertIsNone(card_problem(model_folder(self.root, self.card()), "normalized"))

    def test_a_merged_model_folder_counts_too(self):
        self.assertIsNone(card_problem(model_folder(self.root, self.card(), kind="merged"), "normalized"))

    def test_no_folder_says_how_to_get_one(self):
        problem = card_problem(self.root / "missing", "normalized")
        self.assertIn("No NEXUS Vision model", problem)
        self.assertIn("nexus vision pull", problem)
        self.assertIn("NEXUS_LIVE_BACKEND=ollama", problem)

    def test_a_folder_without_a_model(self):
        folder = self.root / "empty"
        folder.mkdir()
        self.assertIn("has no model in it", card_problem(folder, "normalized"))

    def test_a_model_trained_on_other_prompts_is_refused(self):
        for version in (PROMPT_VERSION - 1, PROMPT_VERSION + 1):
            with self.subTest(version=version):
                with tempfile.TemporaryDirectory() as other:
                    problem = card_problem(model_folder(Path(other), self.card(prompt_version=version)), "normalized")
                    self.assertIn(f"prompt version {version}", problem)

    def test_a_model_giving_other_positions_is_refused(self):
        problem = card_problem(model_folder(self.root, self.card(coordinates="pixels")), "normalized")
        self.assertIn("NEXUS_LIVE_COORDINATES=pixels", problem)

    def test_no_card_runs_with_the_default_prompts(self):
        self.assertIsNone(card_problem(model_folder(self.root), "normalized"))

    def test_read_card_round_trip_and_defaults(self):
        folder = model_folder(self.root)
        self.assertIsNone(read_card(folder))
        card = ModelCard(PROMPT_VERSION, "normalized", "base", 960)
        (folder / MODEL_CARD).write_text(card.to_json())
        self.assertEqual(read_card(folder), card)
        (folder / MODEL_CARD).write_text("{}")
        self.assertEqual(read_card(folder), ModelCard(0, "normalized", "", 1280))

    def test_the_home_folder_is_expanded(self):
        with mock.patch.dict(os.environ, {"HOME": str(self.root), "USERPROFILE": str(self.root)}):
            model_folder(self.root, self.card())
            self.assertIsNone(card_problem(Path("~/nexus_vision"), "normalized"))
            self.assertEqual(LocalVision("~/nexus_vision").model_dir, self.root / "nexus_vision")

    def test_local_vision_refuses_before_loading_anything(self):
        vision = LocalVision(self.root / "missing")
        with self.assertRaisesRegex(ModelUnavailable, "No NEXUS Vision model"):
            vision.ask("open notepad", [], [], b"png", "normalized")

    def test_a_failed_question_leaves_no_generation_from_before(self):
        vision = LocalVision(self.root / "missing")
        vision.last_generation = {"new_tokens": 7, "raw": "an earlier answer"}
        with self.assertRaises(ModelUnavailable):
            vision.ask("open notepad", [], [], b"png", "normalized")
        self.assertEqual(vision.last_generation, {})


class BackendWiringTests(unittest.TestCase):
    def test_each_backend_gets_its_model(self):
        base = LiveSettings()
        self.assertIsInstance(app.make_vision(replace(base, backend="local"), "x"), LocalVision)
        self.assertIsInstance(app.make_vision(replace(base, backend="llamacpp"), "x"), LlamaCppVision)
        ollama = app.make_vision(replace(base, backend="ollama"), "qwen3-vl:4b")
        self.assertIsInstance(ollama, OllamaVision)
        self.assertEqual(ollama.model, "qwen3-vl:4b")

    def test_the_default_backend_is_your_own_model(self):
        self.assertEqual(LiveSettings().backend, "local")
        name, why = app.resolve_model(LiveSettings())
        self.assertIn("NEXUS Vision", name)

    def test_naming_an_ollama_model_switches_to_ollama(self):
        settings = app.session_settings(LiveSettings(), dry_run=False, model="qwen3-vl:2b", confirm=None)
        self.assertEqual((settings.backend, settings.model), ("ollama", "qwen3-vl:2b"))

    def test_an_explicit_backend_wins(self):
        settings = app.session_settings(LiveSettings(), dry_run=True, model="m", confirm="risky", backend="llamacpp")
        self.assertEqual((settings.backend, settings.dry_run, settings.confirm), ("llamacpp", True, "risky"))

    def test_no_changes_keeps_the_settings(self):
        base = replace(LiveSettings(), backend="ollama")
        self.assertIs(app.session_settings(base, dry_run=False, model=None, confirm=None), base)


class ModelCardWidthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name) / "nv"
        self.folder.mkdir()
        (self.folder / "adapter_config.json").write_text("{}")
        self.environment = mock.patch.dict(os.environ)
        self.environment.start()
        os.environ.pop("NEXUS_LIVE_SCREEN_WIDTH", None)

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def settings(self, width_on_card):
        (self.folder / MODEL_CARD).write_text(ModelCard(PROMPT_VERSION, "normalized", "b", width_on_card).to_json())
        return replace(LiveSettings(), model_dir=str(self.folder))

    def test_uses_the_width_the_model_was_trained_at(self):
        self.assertEqual(app.apply_model_card(self.settings(960)).screen_max_width, 960)

    def test_a_width_you_set_wins(self):
        with mock.patch.dict(os.environ, {"NEXUS_LIVE_SCREEN_WIDTH": "1280"}):
            self.assertEqual(app.apply_model_card(self.settings(960)).screen_max_width, 1280)

    def test_nonsense_widths_are_ignored(self):
        self.assertEqual(app.apply_model_card(self.settings(100)).screen_max_width, 1280)
        self.assertEqual(app.apply_model_card(self.settings(479)).screen_max_width, 1280)
        self.assertEqual(app.apply_model_card(self.settings(480)).screen_max_width, 480)
        self.assertEqual(app.apply_model_card(self.settings(3840)).screen_max_width, 3840)
        self.assertEqual(app.apply_model_card(self.settings(3841)).screen_max_width, 1280)

    def test_only_for_the_local_backend(self):
        settings = replace(self.settings(960), backend="ollama")
        self.assertEqual(app.apply_model_card(settings).screen_max_width, 1280)

    def test_a_broken_card_is_ignored(self):
        (self.folder / MODEL_CARD).write_text("{not json")
        settings = replace(LiveSettings(), model_dir=str(self.folder))
        self.assertEqual(app.apply_model_card(settings).screen_max_width, 1280)


class ChecksTests(unittest.TestCase):
    def test_local_backend_reports_a_missing_model_as_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            settings = replace(LiveSettings(), model_dir=str(Path(root) / "missing"))
            with mock.patch.object(app, "capture", side_effect=RuntimeError("no screen")):
                lines = app.checks(settings)
        vision = [line for line in lines if line[1] == "NEXUS Vision"]
        self.assertEqual(vision[0][0], "error")
        self.assertIn("nexus vision pull", vision[0][2])

    def test_local_backend_finds_a_trained_model(self):
        with tempfile.TemporaryDirectory() as root:
            folder = model_folder(Path(root), {"prompt_version": PROMPT_VERSION, "coordinates": "normalized"})
            (folder / "training.json").write_text(json.dumps({"train_examples": 4321}))
            settings = replace(LiveSettings(), model_dir=str(folder))
            with mock.patch.object(app, "capture", side_effect=RuntimeError("no screen")):
                lines = app.checks(settings)
        vision = [line for line in lines if line[1] == "NEXUS Vision"]
        self.assertEqual(vision[0][0], "ok")
        self.assertIn("4321 examples", vision[0][2])

    def test_the_desktop_readers_are_tried(self):
        settings = replace(LiveSettings(), backend="llamacpp", llamacpp_url="http://127.0.0.1:9")
        with mock.patch.object(app, "capture", side_effect=RuntimeError("no screen")), \
             mock.patch.object(app, "desktop_lines", return_value=[("ok", "Window in front", "reads as 'x'")]):
            lines = app.checks(settings)
        self.assertIn(("ok", "Window in front", "reads as 'x'"), lines)

    def test_llamacpp_backend_checks_the_server(self):
        settings = replace(LiveSettings(), backend="llamacpp", llamacpp_url="http://127.0.0.1:9")
        with mock.patch.object(app, "capture", side_effect=RuntimeError("no screen")):
            lines = app.checks(settings)
        server = [line for line in lines if line[1] == "llama-server"]
        self.assertEqual(server[0][0], "error")


class FakeDownload:
    """snapshot_download into a folder: writes the repo's files, or fails part-way when told to."""

    def __init__(self, files, fail=False):
        self.files = files
        self.fail = fail
        self.calls = []
        self.found = []  # what was already in the download folder at each call

    def __call__(self, repo, local_dir, ignore_patterns):
        folder = Path(local_dir)
        self.calls.append(repo)
        self.found.append(sorted(path.name for path in folder.iterdir()))
        for index, (name, content) in enumerate(self.files.items()):
            (folder / name).write_text(content)
            if self.fail and index == 0:
                raise ConnectionError("the connection dropped")


MERGED = {"config.json": "{}", "model.safetensors": "new weights"}


def names(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir()) if folder.exists() else []


class PullTests(unittest.TestCase):
    """`nexus vision pull`: a complete download replaces the model whole, and nothing else is ever lost."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, ignore_errors=True))
        self.folder = self.root / "nexus_vision"

    def old_adapter(self):
        self.folder.mkdir()
        (self.folder / "adapter_config.json").write_text("{}")
        (self.folder / "adapter_model.safetensors").write_text("old weights")

    def test_a_new_model_replaces_the_old_one_whole(self):
        self.old_adapter()
        pull_model("me/merged", self.folder, FakeDownload(MERGED))
        self.assertEqual(names(self.folder), ["config.json", "model.safetensors"])
        self.assertEqual(names(self.root), ["nexus_vision"])  # no download or backup folder left behind

    def test_into_a_new_or_empty_folder(self):
        pull_model("me/merged", self.folder, FakeDownload(MERGED))
        self.assertEqual(names(self.folder), ["config.json", "model.safetensors"])
        other = self.root / "empty"
        other.mkdir()
        pull_model("me/merged", other, FakeDownload(MERGED))
        self.assertEqual(names(other), ["config.json", "model.safetensors"])

    def test_a_folder_with_other_things_in_it_is_left_alone(self):
        models = self.root / "models"
        (models / "llama").mkdir(parents=True)
        (models / "notes.txt").write_text("mine")
        download = FakeDownload(MERGED)
        with self.assertRaisesRegex(ValueError, "has other things in it and no model"):
            pull_model("me/merged", models, download)
        self.assertEqual((names(models), download.calls), (["llama", "notes.txt"], []))

    def test_what_is_left_of_a_model_is_still_a_model_folder(self):
        # An older pull that stopped part-way, or a download folder of the Hub's.
        for leftover in ("adapter_model.safetensors", "tokenizer.json", ".cache/huggingface/download/x.lock"):
            with self.subTest(leftover=leftover):
                folder = self.root / leftover.replace("/", "_")
                (folder / leftover).parent.mkdir(parents=True, exist_ok=True)
                (folder / leftover).write_text("partial")
                pull_model("me/merged", folder, FakeDownload(MERGED))
                self.assertEqual(names(folder), ["config.json", "model.safetensors"])

    def test_a_file_is_left_alone(self):
        target = self.root / "nexus_vision"
        target.write_text("not a folder")
        with self.assertRaisesRegex(ValueError, "is a file"):
            pull_model("me/merged", target, FakeDownload(MERGED))
        self.assertEqual(target.read_text(), "not a folder")

    def test_a_backup_of_the_user_s_own_is_never_touched(self):
        self.old_adapter()
        mine = self.root / "nexus_vision.old"
        mine.mkdir()
        (mine / "adapter_config.json").write_text("my backup")
        pull_model("me/merged", self.folder, FakeDownload(MERGED))
        self.assertEqual((mine / "adapter_config.json").read_text(), "my backup")

    def test_a_download_that_stops_leaves_the_old_model_and_resumes_next_time(self):
        self.old_adapter()
        with self.assertRaises(ConnectionError):
            pull_model("me/merged", self.folder, FakeDownload(MERGED, fail=True))
        self.assertEqual(names(self.folder), ["adapter_config.json", "adapter_model.safetensors"])
        again = FakeDownload(MERGED)
        pull_model("me/merged", self.folder, again)
        self.assertEqual(again.found, [[PULL_MARK, "config.json"]])  # what was downloaded before is still there
        self.assertEqual(names(self.folder), ["config.json", "model.safetensors"])

    def test_a_download_of_another_repo_starts_again(self):
        with self.assertRaises(ConnectionError):
            pull_model("me/other", self.folder, FakeDownload({"stale.bin": "x", "more.bin": "y"}, fail=True))
        download = FakeDownload(MERGED)
        pull_model("me/merged", self.folder, download)
        self.assertEqual(download.found, [[PULL_MARK]])
        self.assertEqual(names(self.folder), ["config.json", "model.safetensors"])

    def test_if_the_new_model_cannot_be_moved_in_the_old_one_is_put_back(self):
        self.old_adapter()
        real_rename = Path.rename

        def rename(source, target):
            if source.name.endswith(".download"):
                raise PermissionError("a virus scan has a file open")
            return real_rename(source, target)

        with mock.patch.object(Path, "rename", rename):
            with self.assertRaises(PermissionError):
                pull_model("me/merged", self.folder, FakeDownload(MERGED), sleep=lambda seconds: None)
        self.assertEqual(names(self.folder), ["adapter_config.json", "adapter_model.safetensors"])
        self.assertEqual(names(self.root), [".nexus_vision.download", "nexus_vision"])  # kept, to move in next time
        again = FakeDownload(MERGED)
        pull_model("me/merged", self.folder, again)
        self.assertEqual(again.found, [[PULL_MARK, "config.json", "model.safetensors"]])
        self.assertEqual(names(self.folder), ["config.json", "model.safetensors"])

    def test_a_file_held_open_for_a_moment_is_waited_out(self):
        self.old_adapter()
        real_rename = Path.rename
        failures = [PermissionError("in use"), PermissionError("in use")]
        slept = []

        def rename(source, target):
            if source.name.endswith(".download") and failures:
                raise failures.pop(0)
            return real_rename(source, target)

        with mock.patch.object(Path, "rename", rename):
            pull_model("me/merged", self.folder, FakeDownload(MERGED), sleep=slept.append)
        self.assertEqual(slept, [0.5, 0.5])
        self.assertEqual(names(self.folder), ["config.json", "model.safetensors"])

    def test_other_rename_errors_are_not_retried(self):
        self.old_adapter()
        calls = []
        real_rename = Path.rename

        def rename(source, target):
            if source.name.endswith(".download"):
                calls.append(source)
                raise OSError("the disk is full")
            return real_rename(source, target)

        with mock.patch.object(Path, "rename", rename):
            with self.assertRaisesRegex(OSError, "the disk is full"):
                pull_model("me/merged", self.folder, FakeDownload(MERGED), sleep=lambda seconds: None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(names(self.folder), ["adapter_config.json", "adapter_model.safetensors"])


if __name__ == "__main__":
    unittest.main()
