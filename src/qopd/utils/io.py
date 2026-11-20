"""Small, dependable file helpers: JSON / JSONL (orjson), atomic writes, hashing."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import orjson

_OPTS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Write via a temp file + rename so readers never see a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(path: str | os.PathLike[str], obj: Any) -> None:
    atomic_write_bytes(path, orjson.dumps(obj, option=_OPTS | orjson.OPT_INDENT_2) + b"\n")


def read_json(path: str | os.PathLike[str]) -> Any:
    return orjson.loads(Path(path).read_bytes())


def append_jsonl(path: str | os.PathLike[str], obj: Any) -> None:
    with Path(path).open("ab") as f:
        f.write(orjson.dumps(obj, option=_OPTS) + b"\n")


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Any]) -> int:
    data = b"".join(orjson.dumps(r, option=_OPTS) + b"\n" for r in rows)
    atomic_write_bytes(path, data)
    return data.count(b"\n")


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[Any]:
    with Path(path).open("rb") as f:
        for line in f:
            if line.strip():
                yield orjson.loads(line)


def read_jsonl(path: str | os.PathLike[str]) -> list[Any]:
    return list(iter_jsonl(path))


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
