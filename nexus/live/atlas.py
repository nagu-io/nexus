"""OS-Atlas screenshots from the Hugging Face Hub, without downloading whole archives.

OS-Atlas keeps its desktop screenshots in big zip files: 3 GB for Linux, 5.5 GB
for macOS, and 71 GB for Windows, split into four parts. Training NEXUS Vision
needs a few thousand of them, so this reads the zip's table of contents from
the end of the last part, then fetches just the chosen screenshots with ranged
reads. A few thousand Windows screenshots come to a few GB instead of 71.

    python -m nexus.live.atlas --system windows --limit 4000 --out data/vision
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import threading
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Sequence

from nexus.live.data import Example, atlas_elements, atlas_examples, choose, write_examples

REPO = "OS-Copilot/OS-Atlas-data"
# The annotations and the archive parts of each desktop system, in the dataset repo.
SYSTEMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "windows": (
        "desktop_domain/windows_splited.json",
        tuple(f"desktop_domain/windows_image_a{letter}" for letter in "abcd"),
    ),
    "macos": ("desktop_domain/macos_splited.json", ("desktop_domain/macos_images.zip",)),
    "linux": ("desktop_domain/linux_splited.json", ("desktop_domain/linux_images.zip",)),
}
LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
LOCAL_HEADER_SIGNATURE = 0x04034B50


class JoinedFile(io.RawIOBase):
    """Several files read as one, end to end: the parts of a split archive.

    ``parts`` are (name, size) pairs in order; ``opener`` opens one by name for
    reading. Parts are opened when first read, so a reader that only needs the
    end of the archive never touches its start.
    """

    def __init__(self, parts: Sequence[tuple[str, int]], opener: Callable[[str], BinaryIO]):
        super().__init__()
        self._parts = list(parts)
        self._starts: list[int] = []
        total = 0
        for _, size in self._parts:
            self._starts.append(total)
            total += size
        self._size = total
        self._opener = opener
        self._handles: dict[int, BinaryIO] = {}
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def _part_at(self, position: int) -> int:
        low, high = 0, len(self._starts) - 1
        while low < high:
            middle = (low + high + 1) // 2
            if self._starts[middle] <= position:
                low = middle
            else:
                high = middle - 1
        return low

    def readinto(self, buffer: Any) -> int:
        """Fills ``buffer`` as far as the end of the archive, across as many parts as it takes."""
        view = memoryview(buffer).cast("B")
        filled = 0
        while filled < len(view) and self._position < self._size:
            index = self._part_at(self._position)
            name, size = self._parts[index]
            inside = self._position - self._starts[index]
            wanted = min(len(view) - filled, size - inside)
            handle = self._handles.get(index)
            if handle is None:
                handle = self._handles[index] = self._opener(name)
            handle.seek(inside)
            data = handle.read(wanted)
            if not data:
                raise EOFError(f"{name} ended at {inside} bytes, before the {size} it should have")
            view[filled : filled + len(data)] = data
            filled += len(data)
            self._position += len(data)
        return filled

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()
        super().close()


class UnreadableMember(Exception):
    """A file in the archive that this reader cannot unpack."""


def read_member(source: BinaryIO, info: zipfile.ZipInfo, slack: int = 1024) -> bytes:
    """The contents of one file in a zip, with as few reads as possible (one, usually).

    Uses the sizes from the zip's table of contents, which are right even when
    the file's own header leaves them out. Checks the contents against the CRC.
    """
    name_length = len(info.orig_filename.encode("utf-8" if info.flag_bits & 0x800 else "cp437"))
    source.seek(info.header_offset)
    blob = source.read(LOCAL_HEADER.size + name_length + slack + info.compress_size)
    if len(blob) < LOCAL_HEADER.size:
        raise UnreadableMember(f"{info.filename}: the archive ends early")
    fields = LOCAL_HEADER.unpack_from(blob)
    if fields[0] != LOCAL_HEADER_SIGNATURE:
        raise UnreadableMember(f"{info.filename}: no file header where the table of contents points")
    start = LOCAL_HEADER.size + fields[9] + fields[10]
    data = blob[start : start + info.compress_size]
    if len(data) < info.compress_size:  # a longer header than the slack allowed for
        source.seek(info.header_offset + start + len(data))
        data += source.read(info.compress_size - len(data))
    if info.compress_type == zipfile.ZIP_STORED:
        contents = data
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        try:
            contents = zlib.decompress(data, -15)
        except zlib.error as error:
            raise UnreadableMember(f"{info.filename}: {error}") from None
    else:
        raise UnreadableMember(f"{info.filename}: compression method {info.compress_type} is not supported")
    if zlib.crc32(contents) != info.CRC:
        raise UnreadableMember(f"{info.filename}: the contents do not match their checksum")
    return contents


@dataclass
class RemoteArchive:
    """A zip (in one or more parts) read in place, with its table of contents by file name."""

    parts: list[tuple[str, int]]
    opener: Callable[[str], BinaryIO]
    by_name: dict[str, zipfile.ZipInfo]

    @classmethod
    def open(cls, parts: Sequence[tuple[str, int]], opener: Callable[[str], BinaryIO]) -> "RemoteArchive":
        joined = io.BufferedReader(JoinedFile(parts, opener), buffer_size=1 << 20)
        try:
            with zipfile.ZipFile(joined) as archive:
                infos = archive.infolist()
        finally:
            joined.close()
        by_name: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            if not info.is_dir():
                by_name.setdefault(Path(info.filename).name, info)
        return cls(list(parts), opener, by_name)

    def fetch(
        self, names: Sequence[str], workers: int = 8, progress: Callable[[int, int], None] | None = None
    ) -> tuple[dict[str, bytes], list[str]]:
        """The files named (by file name), fetched in parallel; and the names that could not be read."""
        wanted = [name for name in dict.fromkeys(Path(name).name for name in names)]
        found: dict[str, bytes] = {}
        missing: list[str] = [name for name in wanted if name not in self.by_name]
        lock = threading.Lock()
        local = threading.local()
        sources: list[JoinedFile] = []
        done = [0]

        def one(name: str) -> None:
            source = getattr(local, "source", None)
            if source is None:
                source = local.source = JoinedFile(self.parts, self.opener)
                with lock:
                    sources.append(source)
            try:
                data = read_member(source, self.by_name[name])
            except (UnreadableMember, EOFError, OSError):
                with lock:
                    missing.append(name)
                return
            with lock:
                found[name] = data
                done[0] += 1
                if progress:
                    progress(done[0], len(wanted))

        present = [name for name in wanted if name in self.by_name]
        try:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                list(pool.map(one, present))
        finally:
            for source in sources:
                source.close()
        return found, missing


class RangeFile(io.RawIOBase):
    """A file on a web server, read with HTTP range requests: each read fetches just the bytes asked for."""

    def __init__(self, url: str, headers: dict[str, str] | None = None, retries: int = 3, timeout: float = 120.0):
        super().__init__()
        import httpx

        self.url = url
        self._headers = dict(headers or {})
        self._retries = retries
        self._client = httpx.Client(follow_redirects=True, timeout=timeout)
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._position = offset
        elif whence == io.SEEK_CUR:
            self._position += offset
        else:
            raise ValueError("a ranged file can only seek from its start or the current position")
        if self._position < 0:
            raise ValueError("negative seek position")
        return self._position

    def readinto(self, buffer: Any) -> int:
        import time

        import httpx

        view = memoryview(buffer).cast("B")
        if not len(view):
            return 0
        end = self._position + len(view) - 1
        headers = {**self._headers, "Range": f"bytes={self._position}-{end}"}
        for attempt in range(self._retries + 1):
            try:
                response = self._client.get(self.url, headers=headers)
            except httpx.HTTPError as error:
                if attempt == self._retries:
                    raise OSError(f"could not read {self.url}: {type(error).__name__}") from None
                time.sleep(2**attempt)
                continue
            if response.status_code == 416:  # asked for bytes past the end
                return 0
            if response.status_code == 206:
                break
            if response.status_code in (429, 500, 502, 503, 504) and attempt < self._retries:
                time.sleep(2**attempt)
                continue
            if response.status_code == 200:
                raise OSError(f"{self.url} does not support ranged reads")
            raise OSError(f"{self.url} answered {response.status_code}")
        data = response.content[: len(view)]
        view[: len(data)] = data
        self._position += len(data)
        return len(data)

    def close(self) -> None:
        self._client.close()
        super().close()


def hub_url(path: str, repo: str = REPO, revision: str = "main") -> str:
    import os
    from urllib.parse import quote

    endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/datasets/{repo}/resolve/{quote(revision, safe='')}/{quote(path)}"


def hub_opener(repo: str = REPO, revision: str = "main") -> Callable[[str], BinaryIO]:
    """Opens files of a Hub dataset for ranged reading. Sends your Hugging Face token if you are logged in."""
    headers: dict[str, str] = {}
    try:
        from huggingface_hub import get_token  # type: ignore

        token = get_token()
    except Exception:
        token = None
    if token:
        headers["Authorization"] = f"Bearer {token}"  # httpx drops it on a redirect to another host

    def opener(path: str) -> BinaryIO:
        return RangeFile(hub_url(path, repo, revision), headers)  # type: ignore[return-value]

    return opener


def hub_sizes(paths: Sequence[str], repo: str = REPO, revision: str = "main") -> list[tuple[str, int]]:
    """The size of each file, in order: a split archive is read by where each part starts."""
    from huggingface_hub import HfApi  # type: ignore

    infos = {info.path: info for info in HfApi().get_paths_info(repo, list(paths), repo_type="dataset", revision=revision)}
    missing = [path for path in paths if path not in infos]
    if missing:
        raise FileNotFoundError(f"not in {repo}: {', '.join(missing)}")
    return [(path, int(infos[path].size)) for path in paths]


def fetch_os_atlas(
    system: str,
    out_folder: Path,
    limit: int = 4000,
    per_image: int = 3,
    max_width: int = 1280,
    seed: int = 7,
    done_share: float = 0.15,
    workers: int = 8,
    printer: Callable[[str], None] = print,
    download: Callable[[str], str | Path] | None = None,
    sizes: Callable[[Sequence[str]], list[tuple[str, int]]] | None = None,
    opener: Callable[[str], BinaryIO] | None = None,
) -> list[Example]:
    """Fetch ``limit`` OS-Atlas elements of ``system`` from the Hub into the dataset ``out_folder``.

    ``download``, ``sizes`` and ``opener`` reach the Hub; they are parameters so
    the whole flow can be run against files on disk.
    """
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {', '.join(SYSTEMS)}, not {system!r}")
    if download is None:
        from huggingface_hub import hf_hub_download  # type: ignore

        def download(path: str) -> str:
            return hf_hub_download(REPO, path, repo_type="dataset")

    annotations_path, part_paths = SYSTEMS[system]
    printer(f"Reading the {system} annotations...")
    entries = json.loads(Path(download(annotations_path)).read_text(encoding="utf-8"))
    chosen = choose(atlas_elements(entries), limit, per_image, seed)
    printer(f"Chose {sum(len(elements) for _, elements in chosen)} elements on {len(chosen)} screenshots.")

    printer("Reading the archive's table of contents...")
    archive = RemoteArchive.open((sizes or hub_sizes)(part_paths), opener or hub_opener())

    def report(done: int, total: int) -> None:
        if done % 200 == 0 or done == total:
            printer(f"  {done} of {total} screenshots")

    found, missing = archive.fetch([name for name, _ in chosen], workers=workers, progress=report)
    if missing:
        printer(f"{len(missing)} screenshots could not be read and are left out.")
    examples = atlas_examples(
        chosen,
        lambda name: found.get(Path(name).name),
        out_folder,
        max_width=max_width,
        seed=seed,
        done_share=done_share,
    )
    for example in examples:  # tell the systems apart, so one can be held out or weighed
        example.source = f"os-atlas-{system}"
    return examples


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nexus vision fetch", description="Fetch OS-Atlas desktop grounding examples for NEXUS Vision.")
    parser.add_argument("--system", choices=sorted(SYSTEMS), default="windows")
    parser.add_argument("--limit", type=int, default=4000, help="elements to take")
    parser.add_argument("--per-image", type=int, default=3)
    parser.add_argument("--width", type=int, default=1280, help="screenshots are shrunk to this width")
    parser.add_argument("--done-share", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True, help="the dataset folder to add to")
    args = parser.parse_args(argv)
    examples = fetch_os_atlas(
        args.system,
        args.out,
        limit=args.limit,
        per_image=args.per_image,
        max_width=args.width,
        seed=args.seed,
        done_share=args.done_share,
        workers=args.workers,
    )
    count = write_examples(args.out, examples, append=True)
    print(f"Added {count} examples to {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
