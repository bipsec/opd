"""1 Hz power + memory sampler writing ``monitor.csv``.

Power: NVML (nvidia-ml-py) first, falling back to ``nvidia-smi --query-gpu=power.draw``.
Memory: /proc/meminfo (GB10 has unified memory; nvidia-smi reports none) and process RSS.
"""

from __future__ import annotations

import csv
import math
import os
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType

from qopd.utils.env import read_meminfo
from qopd.utils.memory import process_rss_gb

COLUMNS = (
    "t_unix",
    "elapsed_s",
    "power_w",
    "mem_total_gb",
    "mem_available_gb",
    "mem_used_gb",
    "proc_rss_gb",
)
PowerFn = Callable[[], float]


def write_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(COLUMNS)


def nvml_power_fn() -> PowerFn | None:
    try:
        import pynvml  # noqa: PLC0415  (provided by nvidia-ml-py)

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        pynvml.nvmlDeviceGetPowerUsage(handle)  # probe once; raises if unsupported
    except Exception:
        return None
    return lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0


def nvidia_smi_power() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return math.nan


def default_power_fn() -> PowerFn:
    return nvml_power_fn() or nvidia_smi_power


class Monitor:
    """Background sampler; use as a context manager around a job.

    ``pids`` lists extra processes whose RSS is summed into ``proc_rss_gb``
    (e.g. a vLLM EngineCore child); this process is always included.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        interval_s: float = 1.0,
        power_fn: PowerFn | None = None,
        pids: Sequence[int] = (),
    ) -> None:
        self.path = Path(path)
        self.interval_s = interval_s
        self.power_fn = power_fn
        self.pids = list(pids)
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def sample(self) -> list[float]:
        mi = read_meminfo()
        gb = 1024**2
        rss = process_rss_gb() + sum(process_rss_gb(p) for p in self.pids)
        now = time.time()
        power = self.power_fn() if self.power_fn else math.nan
        return [
            round(now, 3),
            round(now - self._t0, 3),
            power,
            round(mi["MemTotal"] / gb, 3),
            round(mi["MemAvailable"] / gb, 3),
            round((mi["MemTotal"] - mi["MemAvailable"]) / gb, 3),
            round(rss, 3),
        ]

    def _loop(self) -> None:
        with self.path.open("a", newline="") as f:
            w = csv.writer(f)
            while True:
                w.writerow(self.sample())
                f.flush()
                self.samples += 1
                if self._stop.wait(self.interval_s):
                    break

    def start(self) -> Monitor:
        if not self.path.exists() or self.path.stat().st_size == 0:
            write_header(self.path)
        if self.power_fn is None:
            self.power_fn = default_power_fn()
        self._t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="qopd-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def __enter__(self) -> Monitor:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
