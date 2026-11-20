"""Environment probe (PLAN step 0.1).

Standard library only, so the same code runs on the bare host (no torch) and
inside the NGC container. Every item is either a value or an explicit
``error`` string saying why it could not be read; nothing is guessed.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKAGES = ("torch", "vllm", "transformers", "flash_attn", "triton", "pydantic", "numpy")


@dataclass
class Item:
    """One probed value, or the reason it could not be read."""

    value: Any = None
    error: str | None = None

    def render(self) -> str:
        if self.error is not None:
            return f"could not read: {self.error}"
        return str(self.value)


@dataclass
class EnvReport:
    label: str
    timestamp: str
    items: dict[str, Item] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "timestamp": self.timestamp,
            "items": {k: asdict(v) for k, v in self.items.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnvReport:
        return cls(
            label=d["label"],
            timestamp=d["timestamp"],
            items={k: Item(**v) for k, v in d["items"].items()},
        )


def _run(cmd: list[str], timeout: float = 20.0) -> str:
    """Run a command and return stdout; raise RuntimeError with a short reason."""
    if shutil.which(cmd[0]) is None:
        raise RuntimeError(f"`{cmd[0]}` not found on PATH")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"`{' '.join(cmd)}` timed out") from e
    if out.returncode != 0:
        msg = (out.stderr or out.stdout).strip().splitlines()
        raise RuntimeError(f"`{' '.join(cmd)}` exit {out.returncode}: {msg[-1] if msg else ''}")
    return out.stdout


def _safe(fn: Any, *args: Any) -> Item:
    try:
        return Item(value=fn(*args))
    except Exception as e:
        return Item(error=f"{type(e).__name__}: {e}")


# --- individual probes -------------------------------------------------------


def in_container() -> bool:
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def read_meminfo() -> dict[str, int]:
    """Return /proc/meminfo values in kB."""
    vals: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        vals[key] = int(rest.strip().split()[0])
    return vals


def memory_gb() -> dict[str, float]:
    mi = read_meminfo()
    gb = 1024**2
    return {
        "MemTotal_GB": round(mi["MemTotal"] / gb, 1),
        "MemAvailable_GB": round(mi["MemAvailable"] / gb, 1),
        "SwapTotal_GB": round(mi.get("SwapTotal", 0) / gb, 1),
        "SwapFree_GB": round(mi.get("SwapFree", 0) / gb, 1),
    }


def gpu_nvidia_smi() -> dict[str, str]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.total,power.draw",
            "--format=csv,noheader",
        ]
    )
    name, cap, drv, mem, power = (s.strip() for s in out.strip().splitlines()[0].split(","))
    return {
        "name": name,
        "compute_cap": cap,
        "driver": drv,
        "memory.total": mem,
        "power.draw": power,
    }


def cuda_driver_version() -> str:
    for line in _run(["nvidia-smi"]).splitlines():
        if "CUDA Version" in line:
            return line.split("CUDA Version:")[1].strip(" |")
    raise RuntimeError("no 'CUDA Version' line in nvidia-smi output")


def gpu_torch() -> dict[str, Any]:
    import torch  # noqa: PLC0415 - optional dependency

    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__} sees no CUDA device")
    return {
        "name": torch.cuda.get_device_name(0),
        "capability": "sm_{}{}".format(*torch.cuda.get_device_capability(0)),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def _rss_gb(pid: int) -> str:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return f"{int(line.split()[1]) / 1024**2:.1f} GB"
        return "VmRSS missing"
    except OSError as e:
        return f"unreadable ({type(e).__name__}; pid namespace?)"


def gpu_processes() -> list[dict[str, str]]:
    out = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )
    procs = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, name, used = (s.strip() for s in line.split(","))
        procs.append(
            {
                "pid": pid,
                "name": name,
                "nvidia_smi_used": used,
                "rss": _rss_gb(int(pid)) if pid.isdigit() else "n/a",
            }
        )
    if not procs and in_container():
        raise RuntimeError(
            "nvidia-smi lists no processes inside the container "
            "(host PIDs are hidden by the pid namespace); see the host column"
        )
    return procs


def package_versions() -> dict[str, str]:
    vers: dict[str, str] = {}
    for p in PACKAGES:
        try:
            vers[p] = md.version(p.replace("_", "-"))
        except md.PackageNotFoundError:
            try:
                vers[p] = md.version(p)
            except md.PackageNotFoundError:
                vers[p] = "not installed"
    return vers


def hf_cache_dir() -> Path:
    if "HF_HUB_CACHE" in os.environ:
        return Path(os.environ["HF_HUB_CACHE"])
    if "HF_HOME" in os.environ:
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def disk_free(path: Path) -> dict[str, Any]:
    p = path
    while not p.exists():
        p = p.parent
    du = shutil.disk_usage(p)
    return {
        "path": str(path),
        "measured_at": str(p),
        "free_GB": round(du.free / 1e9, 1),
        "total_GB": round(du.total / 1e9, 1),
    }


def ngc_images() -> list[str]:
    out = _run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}} ({{.Size}})"])
    imgs = [line for line in out.splitlines() if line.startswith("nvcr.io/nvidia/")]
    return imgs or ["none found"]


def running_containers() -> list[str]:
    out = _run(["docker", "ps", "--format", "{{.Names}} | {{.Image}} | {{.Status}}"])
    return [line for line in out.splitlines() if line.strip()] or ["none running"]


def cpu_info() -> dict[str, Any]:
    return {
        "arch": platform.machine(),
        "cpus": os.cpu_count(),
        "kernel": platform.release(),
        "os": platform.platform(),
    }


# --- top level ---------------------------------------------------------------


def probe(label: str | None = None) -> EnvReport:
    """Collect every item PLAN step 0.1 asks for."""
    ctr = in_container()
    rep = EnvReport(
        label=label or ("container" if ctr else "host"),
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    it = rep.items
    it["in_container"] = Item(value=ctr)
    it["python"] = Item(value=f"{sys.version.split()[0]} ({sys.executable})")
    it["cpu"] = _safe(cpu_info)
    it["gpu (nvidia-smi)"] = _safe(gpu_nvidia_smi)
    it["gpu (torch)"] = _safe(gpu_torch)
    it["cuda driver version"] = _safe(cuda_driver_version)
    it["unified memory (/proc/meminfo)"] = _safe(memory_gb)
    it["gpu processes"] = _safe(gpu_processes)
    it["package versions"] = _safe(package_versions)
    it["hf cache disk"] = _safe(lambda: disk_free(hf_cache_dir()))
    it["project disk"] = _safe(lambda: disk_free(Path.cwd()))
    it["ngc docker images"] = _safe(ngc_images)
    it["running docker containers"] = _safe(running_containers)
    return rep


def render_markdown(reports: list[EnvReport]) -> str:
    """Render one or more probe reports side by side."""
    keys: list[str] = []
    for r in reports:
        keys += [k for k in r.items if k not in keys]
    lines = [
        "# Environment report",
        "",
        "Generated by `scripts/env_probe.py` (`qopd.utils.env`). Values are read, not assumed;",
        "an item that could not be read says why.",
        "",
    ]
    for r in reports:
        lines.append(f"- **{r.label}** probed at {r.timestamp}")
    lines.append("")
    for k in keys:
        lines.append(f"## {k}")
        lines.append("")
        for r in reports:
            item = r.items.get(k, Item(error="not probed"))
            lines.append(f"**{r.label}:**")
            lines.append("")
            lines.extend(_render_value(item))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_value(item: Item) -> list[str]:
    if item.error is not None:
        return [f"- could not read: {item.error}"]
    v = item.value
    if isinstance(v, dict):
        return [f"- {k}: {val}" for k, val in v.items()]
    if isinstance(v, list):
        if not v:
            return ["- (none)"]
        if isinstance(v[0], dict):
            return ["- " + ", ".join(f"{k}={val}" for k, val in d.items()) for d in v]
        return [f"- {x}" for x in v]
    return [f"- {v}"]


def snapshot() -> dict[str, Any]:
    """Fast subset of :func:`probe` for each run's ``env.json`` (no docker, no torch import)."""
    rep = EnvReport(label="run", timestamp=datetime.now(UTC).isoformat(timespec="seconds"))
    rep.items["hostname"] = Item(value=platform.node())
    rep.items["in_container"] = Item(value=in_container())
    rep.items["python"] = Item(value=sys.version.split()[0])
    rep.items["cpu"] = _safe(cpu_info)
    rep.items["gpu (nvidia-smi)"] = _safe(gpu_nvidia_smi)
    rep.items["unified memory (/proc/meminfo)"] = _safe(memory_gb)
    rep.items["package versions"] = _safe(package_versions)
    return rep.to_dict()
