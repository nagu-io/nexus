"""Tests for nexus.live.build's run log: the progress of a training run, kept in the model repo."""

import argparse
import unittest
from types import SimpleNamespace
from unittest import mock

from nexus.live import build
from nexus.live.build import PublicRepo, RunLog, ensure_private


class FakeApi:
    """The Hub: ``public`` names repos that already exist and are public."""

    def __init__(self, fail_uploads=False, public=()):
        self.created = []
        self.uploads = []
        self.fail_uploads = fail_uploads
        self.public = set(public)

    def create_repo(self, repo, repo_type=None, private=False, exist_ok=False):
        self.created.append((repo, private, exist_ok))

    def repo_info(self, repo, repo_type=None):
        return SimpleNamespace(private=repo not in self.public)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, commit_message):
        if self.fail_uploads:
            raise ConnectionError("offline")
        self.uploads.append((path_in_repo, repo_id, path_or_fileobj.decode("utf-8")))


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class RunLogTests(unittest.TestCase):
    def make(self, **api_options):
        self.api = FakeApi(**api_options)
        self.clock = Clock()
        self.printed = []
        return RunLog("me/nexus-vision", self.printed.append, every=600, api=self.api, clock=self.clock)

    def test_the_repo_is_made_private_before_anything_else(self):
        self.make()
        self.assertEqual(self.api.created, [("me/nexus-vision", True, True)])
        self.assertEqual(self.api.uploads, [])

    def test_lines_are_printed_and_uploaded_at_most_every_interval(self):
        log = self.make()
        log.print("Reading the windows annotations...")
        self.assertEqual(len(self.api.uploads), 1)
        self.clock.now = 599
        log.print("step 1 of 300: loss 2.1")
        self.assertEqual(len(self.api.uploads), 1)
        self.clock.now = 600
        log.print("step 2 of 300: loss 1.9")
        self.assertEqual(len(self.api.uploads), 2)
        path, repo, text = self.api.uploads[-1]
        self.assertEqual((path, repo), ("run.log", "me/nexus-vision"))
        self.assertEqual(text.count("\n"), 3)
        self.assertIn("step 2 of 300: loss 1.9", text)
        self.assertEqual(self.printed, ["Reading the windows annotations...", "step 1 of 300: loss 2.1", "step 2 of 300: loss 1.9"])

    def test_a_failed_upload_never_stops_the_run(self):
        log = self.make(fail_uploads=True)
        log.print("hello")
        self.assertEqual(self.printed, ["hello", "(could not update run.log: ConnectionError)"])

    def test_a_failure_is_uploaded_with_its_traceback(self):
        log = self.make()
        try:
            raise RuntimeError("CUDA out of memory")
        except RuntimeError as error:
            log.failed(error)
        text = self.api.uploads[-1][2]
        self.assertIn("FAILED: RuntimeError: CUDA out of memory", text)
        self.assertIn("Traceback", text)


class PrivateRepoTests(unittest.TestCase):
    def test_a_new_or_private_repo_is_fine(self):
        api = FakeApi()
        ensure_private(api, "me/nexus-teach", "dataset")
        self.assertEqual(api.created, [("me/nexus-teach", True, True)])

    def test_an_existing_public_repo_is_refused(self):
        # create_repo(exist_ok=True) leaves a public repo public: the run log, raw answers or teach
        # screenshots would have gone public while the console said "private".
        api = FakeApi(public={"me/nexus-vision"})
        with self.assertRaisesRegex(PublicRepo, "already exists and is public"):
            ensure_private(api, "me/nexus-vision")

    def test_the_run_log_refuses_a_public_repo_before_any_work(self):
        api = FakeApi(public={"me/nexus-vision"})
        with self.assertRaises(PublicRepo):
            RunLog("me/nexus-vision", lambda line: None, api=api)
        self.assertEqual(api.uploads, [])


class RunTests(unittest.TestCase):
    def test_a_failing_run_records_the_failure_and_still_fails(self):
        api = FakeApi()
        args = argparse.Namespace(push="me/nexus-vision")
        with mock.patch.object(build, "RunLog", lambda repo, printer: RunLog(repo, printer, api=api)), \
             mock.patch.object(build, "_run", side_effect=MemoryError("too big")):
            with self.assertRaises(MemoryError):
                build.run(args, printer=lambda line: None)
        self.assertIn("FAILED: MemoryError: too big", api.uploads[-1][2])

    def test_no_repo_no_log(self):
        args = argparse.Namespace(push="")
        with mock.patch.object(build, "RunLog", side_effect=AssertionError("no log without a repo")), \
             mock.patch.object(build, "_run", return_value={"ok": True}) as inner:
            self.assertEqual(build.run(args, printer=print), {"ok": True})
        self.assertIsNone(inner.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
