"""Two-Spark helpers: run read-only commands on the peer over SSH and check if it is free.

Authentication uses the password in ``.env`` (``PASSWD=...``) through ``SSH_ASKPASS``, so
the password never appears on a command line, in the environment, or in logs. Key-based
SSH, if configured later, is used automatically (ssh tries keys first).

Policy (CLAUDE.md): never kill or disturb a job on the peer; use it only once it is free.
"""

from __future__ import annotations

import os
import shlex
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLUSTER_FILE = PROJECT_ROOT / "configs" / "cluster.yaml"


@dataclass(frozen=True)
class Node:
    name: str
    host: str
    user: str
    hostname: str = ""
    role: str = "secondary"

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


def load_nodes(path: Path = DEFAULT_CLUSTER_FILE) -> tuple[dict[str, Node], Path]:
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    nodes = {n["name"]: Node(**n) for n in raw["nodes"]}
    env_file = Path(raw.get("env_file", ".env"))
    if not env_file.is_absolute():
        env_file = PROJECT_ROOT / env_file
    return nodes, env_file


def _askpass_script(env_file: Path, workdir: Path) -> Path:
    script = workdir / "askpass.sh"
    # The script reads the password at call time; it does not contain it.
    script.write_text(
        "#!/bin/sh\n"
        f"sed -n 's/^PASSWD=//p' {shlex.quote(str(env_file))} | head -n1 "
        "| sed -e 's/^\"//' -e 's/\"$//' -e \"s/^'//\" -e \"s/'$//\"\n"
    )
    script.chmod(stat.S_IRWXU)
    return script


def ssh_run(
    node: Node,
    command: str,
    env_file: Path,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    """Run ``command`` on ``node``; returns the completed process (never raises on exit code)."""
    if not env_file.exists():
        raise FileNotFoundError(f"{env_file} not found (needs PASSWD=...)")
    with tempfile.TemporaryDirectory(prefix="qopd-ssh-") as d:
        env = dict(os.environ)
        env.update(
            SSH_ASKPASS=str(_askpass_script(env_file, Path(d))),
            SSH_ASKPASS_REQUIRE="force",
            DISPLAY=env.get("DISPLAY", "none"),
        )
        cmd = [
            "ssh",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "NumberOfPasswordPrompts=1",
            node.target,
            command,
        ]
        return subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from any tty so SSH_ASKPASS is used
            check=False,
        )


@dataclass(frozen=True)
class NodeStatus:
    node: str
    reachable: bool
    gpu: str
    mem_available_gb: float | None
    gpu_processes: list[str]
    error: str | None = None

    @property
    def free(self) -> bool:
        """True when reachable and no GPU compute process is running."""
        return self.reachable and not self.gpu_processes


_STATUS_CMD = (
    "nvidia-smi --query-gpu=name --format=csv,noheader; echo ---; "
    "grep MemAvailable /proc/meminfo; echo ---; "
    "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"
)


def parse_status(node: str, stdout: str) -> NodeStatus:
    gpu, mem, procs = ([*stdout.split("---"), "", ""])[:3]
    mem_kb = mem.split()[1] if len(mem.split()) > 1 else None
    return NodeStatus(
        node=node,
        reachable=True,
        gpu=gpu.strip(),
        mem_available_gb=round(int(mem_kb) / 1024**2, 1) if mem_kb else None,
        gpu_processes=[p.strip() for p in procs.strip().splitlines() if p.strip()],
    )


def node_status(node: Node, env_file: Path) -> NodeStatus:
    """Read-only status of a node (GPU name, free memory, GPU processes)."""
    try:
        out = ssh_run(node, _STATUS_CMD, env_file, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return NodeStatus(node.name, False, "", None, [], error=f"{type(e).__name__}: {e}")
    if out.returncode != 0:
        msg = (out.stderr.strip().splitlines() or ["?"])[-1]
        return NodeStatus(node.name, False, "", None, [], error=msg)
    return parse_status(node.name, out.stdout)
