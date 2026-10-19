"""Tests for nexus.live.build's run log: the progress of a training run, kept in the model repo."""

import argparse
import unittest
from unittest import mock

from nexus.live import build
from nexus.live.build import RunLog


class FakeApi:
    def __init__(self, fail_uploads=False):
        self.created = []
        self.uploads = []
        self.fail_uploads = fail_uploads

    def create_repo(self, repo, private, exist_ok):
        self.created.append((repo, private, exist_ok))

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
