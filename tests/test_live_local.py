"""Tests for nexus.live.local (the model card) and nexus.live.app's backend wiring."""

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from nexus.live import app
from nexus.live.local import MODEL_CARD, LocalVision, ModelCard, card_problem, read_card
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

    def test_llamacpp_backend_checks_the_server(self):
        settings = replace(LiveSettings(), backend="llamacpp", llamacpp_url="http://127.0.0.1:9")
        with mock.patch.object(app, "capture", side_effect=RuntimeError("no screen")):
            lines = app.checks(settings)
        server = [line for line in lines if line[1] == "llama-server"]
        self.assertEqual(server[0][0], "error")


if __name__ == "__main__":
    unittest.main()
