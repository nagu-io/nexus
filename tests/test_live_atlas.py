"""Tests for nexus.live.atlas: reading a few files out of a big split zip without downloading it."""

import http.server
import importlib.util
import io
import json
import os
import random
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

from nexus.live.atlas import SYSTEMS, JoinedFile, RangeFile, RemoteArchive, UnreadableMember, fetch_os_atlas, read_member


def make_archive(files: dict[str, bytes], force_zip64: bool = True) -> bytes:
    """A zip of ``files`` (half stored, half deflated), with ZIP64 records like the 71 GB OS-Atlas one has."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for index, (name, data) in enumerate(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2024, 8, 22, 0, 18, 44))
            info.compress_type = zipfile.ZIP_DEFLATED if index % 2 else zipfile.ZIP_STORED
            with archive.open(info, "w", force_zip64=force_zip64) as handle:
                handle.write(data)
    return buffer.getvalue()


def split_parts(blob: bytes, sizes: list[int]) -> list[bytes]:
    parts, start = [], 0
    for size in sizes:
        parts.append(blob[start : start + size])
        start += size
    parts.append(blob[start:])
    return [part for part in parts if part]


def sample_files(count=12, seed=1) -> dict[str, bytes]:
    rng = random.Random(seed)
    files = {}
    for index in range(count):
        body = bytes(rng.getrandbits(8) for _ in range(rng.randint(50, 3000))) + b"pattern" * rng.randint(1, 400)
        files[f"windows_images/2024/shot_{index:03d}_sub{index % 3}.png"] = body
    return files


class Opened:
    """An opener over in-memory parts that counts how often each part is opened and read."""

    def __init__(self, parts: dict[str, bytes]):
        self.parts = parts
        self.opened: list[str] = []
        self.reads = 0

    def __call__(self, name: str):
        self.opened.append(name)
        outer = self

        class Counting(io.BytesIO):
            def read(self, size=-1):
                outer.reads += 1
                return super().read(size)

        return Counting(self.parts[name])


class JoinedFileTests(unittest.TestCase):
    def setUp(self):
        self.blob = bytes(range(256)) * 40  # 10240 bytes
        pieces = split_parts(self.blob, [1000, 4000, 3])
        self.names = [(f"part{index}", len(piece)) for index, piece in enumerate(pieces)]
        self.opener = Opened({name: piece for (name, _), piece in zip(self.names, pieces)})
        self.joined = JoinedFile(self.names, self.opener)

    def test_reads_across_part_boundaries_exactly(self):
        for start, length in ((0, 10), (995, 10), (999, 2), (1000, 4003), (4990, 30), (5002, 2), (10230, 50), (0, 20000)):
            with self.subTest(start=start, length=length):
                self.joined.seek(start)
                data = self.joined.read(length)
                self.assertEqual(data, self.blob[start : start + length])
                self.assertEqual(self.joined.tell(), min(len(self.blob), start + length))

    def test_seek_from_the_end_and_current(self):
        self.assertEqual(self.joined.seek(-22, io.SEEK_END), len(self.blob) - 22)
        self.assertEqual(self.joined.read(22), self.blob[-22:])
        self.joined.seek(100)
        self.joined.seek(5, io.SEEK_CUR)
        self.assertEqual(self.joined.tell(), 105)
        with self.assertRaises(ValueError):
            self.joined.seek(-1)

    def test_reading_the_end_only_opens_the_parts_it_touches(self):
        self.joined.seek(-50, io.SEEK_END)
        self.joined.read(50)
        self.assertEqual(self.opener.opened, ["part3"])
        self.joined.seek(4999)
        self.joined.read(5)  # the last byte of part1, all 3 of part2, the first of part3
        self.assertEqual(self.opener.opened, ["part3", "part1", "part2"])

    def test_one_read_fills_the_whole_request_across_parts(self):
        self.joined.seek(990)
        self.assertEqual(len(self.joined.read(4100)), 4100)

    def test_a_short_part_is_an_error_not_silent_garbage(self):
        pieces = {"a": b"x" * 10, "b": b"y" * 5}
        joined = JoinedFile([("a", 10), ("b", 9)], Opened(pieces))
        joined.seek(12)
        self.assertEqual(joined.read(3), b"yyy")
        with self.assertRaises(EOFError):
            joined.read(10)

    def test_past_the_end_reads_nothing(self):
        self.joined.seek(len(self.blob) + 10)
        self.assertEqual(self.joined.read(5), b"")


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.files = sample_files()
        self.blob = make_archive(self.files)
        pieces = split_parts(self.blob, [7000, 9000, 11])
        self.parts = [(f"windows_image_a{chr(97 + index)}", len(piece)) for index, piece in enumerate(pieces)]
        self.opener = Opened({name: piece for (name, _), piece in zip(self.parts, pieces)})

    def test_table_of_contents_by_file_name(self):
        archive = RemoteArchive.open(self.parts, self.opener)
        self.assertEqual(sorted(archive.by_name), sorted(Path(name).name for name in self.files))

    def test_fetches_just_the_files_asked_for_across_parts(self):
        archive = RemoteArchive.open(self.parts, self.opener)
        wanted = ["shot_001_sub1.png", "shot_007_sub1.png", "windows_images/2024/shot_010_sub1.png", "missing.png"]
        found, missing = archive.fetch(wanted, workers=3)
        self.assertEqual(missing, ["missing.png"])
        self.assertEqual(found["shot_001_sub1.png"], self.files["windows_images/2024/shot_001_sub1.png"])
        self.assertEqual(found["shot_007_sub1.png"], self.files["windows_images/2024/shot_007_sub1.png"])
        self.assertEqual(found["shot_010_sub1.png"], self.files["windows_images/2024/shot_010_sub1.png"])

    def test_every_file_reads_back_exactly_stored_or_deflated(self):
        archive = RemoteArchive.open(self.parts, self.opener)
        found, missing = archive.fetch(list(self.files), workers=4)
        self.assertEqual(missing, [])
        self.assertEqual({Path(name).name: data for name, data in self.files.items()}, found)

    def test_one_read_per_file_when_the_header_fits_the_slack(self):
        archive = RemoteArchive.open(self.parts, self.opener)
        source = JoinedFile(self.parts, self.opener)
        info = archive.by_name["shot_004_sub1.png"]
        before = self.opener.reads
        read_member(source, info)
        self.assertLessEqual(self.opener.reads - before, 2)  # 2 when the file straddles two parts

    def test_progress_is_reported(self):
        archive = RemoteArchive.open(self.parts, self.opener)
        calls = []
        archive.fetch(["shot_000_sub0.png", "shot_002_sub2.png"], workers=1, progress=lambda done, total: calls.append((done, total)))
        self.assertEqual(calls, [(1, 2), (2, 2)])

    def test_a_corrupted_file_is_reported_missing_not_returned(self):
        corrupt = bytearray(self.blob)
        archive_info = zipfile.ZipFile(io.BytesIO(self.blob)).getinfo("windows_images/2024/shot_002_sub2.png")
        corrupt[archive_info.header_offset + 30 + len(archive_info.filename) + 40] ^= 0xFF
        pieces = {"whole": bytes(corrupt)}
        archive = RemoteArchive.open([("whole", len(corrupt))], Opened(pieces))
        found, missing = archive.fetch(["shot_002_sub2.png", "shot_000_sub0.png"], workers=1)
        self.assertEqual(missing, ["shot_002_sub2.png"])
        self.assertIn("shot_000_sub0.png", found)

    def test_unsupported_compression_is_unreadable(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_BZIP2) as archive:
            archive.writestr("a.png", b"hello" * 100)
        blob = buffer.getvalue()
        info = zipfile.ZipFile(io.BytesIO(blob)).getinfo("a.png")
        with self.assertRaisesRegex(UnreadableMember, "compression method"):
            read_member(io.BytesIO(blob), info)

    def test_a_header_longer_than_the_slack_still_reads(self):
        info = zipfile.ZipFile(io.BytesIO(self.blob)).getinfo("windows_images/2024/shot_005_sub2.png")
        self.assertEqual(read_member(io.BytesIO(self.blob), info, slack=0), self.files["windows_images/2024/shot_005_sub2.png"])

    def test_not_a_file_header_where_the_contents_point(self):
        info = zipfile.ZipFile(io.BytesIO(self.blob)).getinfo("windows_images/2024/shot_003_sub0.png")
        info.header_offset += 1
        with self.assertRaisesRegex(UnreadableMember, "no file header"):
            read_member(io.BytesIO(self.blob), info)


class RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves files with HTTP range requests, and a redirect, like the Hub's resolve URLs."""

    files: dict[str, bytes] = {}
    requests: list[str] = []
    honour_ranges = True

    def log_message(self, *args):
        pass

    def do_GET(self):
        type(self).requests.append(self.path + " " + self.headers.get("Range", ""))
        if self.path.startswith("/resolve/"):
            self.send_response(302)
            self.send_header("Location", "/cdn/" + self.path[len("/resolve/") :])
            self.end_headers()
            return
        name = self.path[len("/cdn/") :]
        data = self.files.get(name)
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        header = self.headers.get("Range")
        if not header or not self.honour_ranges:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        start_text, end_text = header.replace("bytes=", "").split("-")
        start, end = int(start_text), min(int(end_text), len(data) - 1)
        if start >= len(data):
            self.send_response(416)
            self.end_headers()
            return
        chunk = data[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)


class RangeFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
        os.environ["no_proxy"] = os.environ["NO_PROXY"]
        cls.files = sample_files(8, seed=5)
        blob = make_archive(cls.files)
        pieces = split_parts(blob, [5000])
        RangeHandler.files = {f"part{index}": piece for index, piece in enumerate(pieces)}
        cls.parts = [(f"part{index}", len(piece)) for index, piece in enumerate(pieces)]
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        RangeHandler.requests = []
        RangeHandler.honour_ranges = True

    def opener(self, name):
        return RangeFile(f"{self.base}/resolve/{name}")

    def test_fetches_files_out_of_a_split_archive_over_http(self):
        archive = RemoteArchive.open(self.parts, self.opener)
        found, missing = archive.fetch(["shot_001_sub1.png", "shot_006_sub0.png"], workers=2)
        self.assertEqual(missing, [])
        self.assertEqual(found["shot_006_sub0.png"], self.files["windows_images/2024/shot_006_sub0.png"])
        self.assertTrue(all("bytes=" in request for request in RangeHandler.requests if request.startswith("/cdn/")))

    def test_each_read_asks_for_just_its_bytes(self):
        handle = RangeFile(f"{self.base}/resolve/part0")
        handle.seek(10)
        self.assertEqual(handle.read(5), RangeHandler.files["part0"][10:15])
        self.assertIn("/cdn/part0 bytes=10-14", RangeHandler.requests)
        handle.close()

    def test_past_the_end_is_empty(self):
        handle = RangeFile(f"{self.base}/resolve/part1")
        handle.seek(10_000_000)
        self.assertEqual(handle.read(10), b"")
        handle.close()

    def test_a_server_that_ignores_ranges_is_an_error_not_a_full_download(self):
        RangeHandler.honour_ranges = False
        handle = RangeFile(f"{self.base}/resolve/part0")
        with self.assertRaisesRegex(OSError, "does not support ranged reads"):
            handle.read(10)
        handle.close()

    def test_missing_file(self):
        handle = RangeFile(f"{self.base}/resolve/nope", retries=0)
        with self.assertRaisesRegex(OSError, "answered 404"):
            handle.read(10)
        handle.close()


def png(seed):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (640, 360), (seed * 20 % 255, 80, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


@unittest.skipUnless(importlib.util.find_spec("PIL"), "needs Pillow")
class FetchTests(unittest.TestCase):
    """The whole fetch, with the Hub replaced by files on disk laid out as OS-Atlas lays them out."""

    def test_windows_annotations_and_split_archive_become_examples(self):
        names = [f"20240822_00184{index}_before_screenshot_sub{index % 3}.png" for index in range(6)]
        annotations = [
            {"img_filename": name, "elements": [
                {"instruction": "Help and feedback", "bbox": [0.742, 0.797, 0.966, 0.842]},
                {"instruction": "Copilot", "bbox": [0.182, 0.0, 0.382, 0.049]},
                {"instruction": "Off the edge", "bbox": [0.9, 0.1, 1.5, 0.2]},
            ]}
            for name in names[:5]  # the sixth screenshot has no annotations
        ]
        files = {f"windows_images/{name}": png(index) for index, name in enumerate(names)}
        blob = make_archive(files)
        part_paths = SYSTEMS["windows"][1]
        third = len(blob) // 3
        pieces = split_parts(blob, [third, third, 7])
        self.assertEqual(len(pieces), len(part_paths))
        stored = dict(zip(part_paths, pieces))
        with tempfile.TemporaryDirectory() as folder:
            annotations_file = Path(folder) / "windows_splited.json"
            annotations_file.write_text(json.dumps(annotations))
            out = Path(folder) / "out"
            lines = []
            examples = fetch_os_atlas(
                "windows", out, limit=6, per_image=2, done_share=0.0, workers=2, printer=lines.append,
                download=lambda path: annotations_file,
                sizes=lambda paths: [(path, len(stored[path])) for path in paths],
                opener=Opened(stored),
            )
            self.assertEqual(len(examples), 6)
            self.assertEqual({example.source for example in examples}, {"os-atlas-windows"})
            self.assertEqual({example.request for example in examples} <= {'Click "Help and feedback"', 'Click "Copilot"',
                              'Click on "Help and feedback"', 'Click on "Copilot"', 'Select "Help and feedback"', 'Select "Copilot"',
                              'Find "Help and feedback" and click it', 'Find "Copilot" and click it'}, True)
            self.assertEqual(len({example.image for example in examples}), 3)
            for example in examples:
                self.assertTrue((out / example.image).is_file())
            self.assertIn("Chose 6 elements on 3 screenshots.", lines)

    def test_an_unknown_system_is_refused(self):
        with self.assertRaisesRegex(ValueError, "system must be one of"):
            fetch_os_atlas("android", Path("."), download=lambda path: path)


if __name__ == "__main__":
    unittest.main()
