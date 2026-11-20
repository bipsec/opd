"""Memory snapshots for the unified-memory GB10.

CLAUDE.md: every function that loads a model logs memory before and after.
Use ``with track_memory("load T16"):`` around any model load.
nvidia-smi reports no GPU memory on GB10, so system numbers come from /proc/meminfo
and the process RSS; torch allocator numbers are added if torch is already imported.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from qopd.utils.env import read_meminfo

_GB_KB = 1024**2


@dataclass(frozen=True)
class MemSnapshot:
    tag: str
    mem_total_gb: float
    mem_available_gb: float
    mem_used_gb: float
    proc_rss_gb: float
    torch_allocated_gb: float | None
    torch_reserved_gb: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def process_rss_gb(pid: int | None = None) -> float:
    """Resident set size of ``pid`` (default: this process) in GB; NaN if unreadable."""
    try:
        with open(f"/proc/{pid or os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / _GB_KB
    except OSError:
        pass
    return float("nan")


def snapshot(tag: str = "") -> MemSnapshot:
    mi = read_meminfo()
    alloc = reserved = None
    torch = sys.modules.get("torch")  # never import torch just to measure
    if torch is not None and torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
    return MemSnapshot(
        tag=tag,
        mem_total_gb=mi["MemTotal"] / _GB_KB,
        mem_available_gb=mi["MemAvailable"] / _GB_KB,
        mem_used_gb=(mi["MemTotal"] - mi["MemAvailable"]) / _GB_KB,
        proc_rss_gb=process_rss_gb(),
        torch_allocated_gb=alloc,
        torch_reserved_gb=reserved,
    )


def log_memory(tag: str, logger: logging.Logger | None = None) -> MemSnapshot:
    from qopd.utils.logging import get_logger  # noqa: PLC0415

    s = snapshot(tag)
    torch_part = (
        f" torch_alloc={s.torch_allocated_gb:.2f}GB" if s.torch_allocated_gb is not None else ""
    )
    (logger or get_logger("memory")).info(
        "[mem] %s: used=%.1fGB avail=%.1fGB rss=%.2fGB%s",
        tag,
        s.mem_used_gb,
        s.mem_available_gb,
        s.proc_rss_gb,
        torch_part,
    )
    return s


@contextmanager
def track_memory(tag: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Log memory before and after the block, plus elapsed time and the change in use."""
    before = log_memory(f"{tag} (before)", logger)
    t0 = time.perf_counter()
    yield
    after = log_memory(f"{tag} (after, {time.perf_counter() - t0:.1f}s)", logger)
    from qopd.utils.logging import get_logger  # noqa: PLC0415

    (logger or get_logger("memory")).info(
        "[mem] %s: delta used=%+.1fGB", tag, after.mem_used_gb - before.mem_used_gb
    )
