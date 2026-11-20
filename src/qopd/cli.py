"""`qopd` command-line entry point (a single typer app).

Commands are thin: they parse arguments and call library functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from qopd import __version__

app = typer.Typer(
    name="qopd",
    help="Quantized-teacher on-policy distillation (qopd).",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
config_app = typer.Typer(help="Inspect and validate configs (configs/).", no_args_is_help=True)
app.add_typer(config_app, name="config")
models_app = typer.Typer(
    help="Model registry, downloads and tokenizer checks.", no_args_is_help=True
)
app.add_typer(models_app, name="models")


def _print_version(value: bool) -> None:
    if value:
        console.print(f"qopd {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", help="Print the version and exit.", is_eager=True, callback=_print_version
        ),
    ] = False,
) -> None:
    """Quantized-teacher on-policy distillation (qopd)."""


@app.command()
def env(
    label: Annotated[
        str | None, typer.Option(help="Report label (default: host/container).")
    ] = None,
    write: Annotated[
        bool, typer.Option("--write", help="Also write reports/env.<label>.json and env.md.")
    ] = False,
    out_dir: Annotated[Path, typer.Option(help="Report directory for --write.")] = Path("reports"),
) -> None:
    """Environment report: GPU, memory, versions, disk, containers."""
    from qopd.utils.env import EnvReport, probe, render_markdown  # noqa: PLC0415

    rep = probe(label)
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"env.{rep.label}.json").write_text(json.dumps(rep.to_dict(), indent=2))
        reports = [
            EnvReport.from_dict(json.loads(p.read_text()))
            for p in sorted(out_dir.glob("env.*.json"))
        ]
        (out_dir / "env.md").write_text(render_markdown(reports))
        console.print(f"wrote {out_dir / 'env.md'}")
    console.print(render_markdown([rep]), markup=False, highlight=False)


@app.command()
def status() -> None:
    """Queue and run status (the job queue arrives in step 8.1)."""
    from qopd.utils.run import default_runs_root  # noqa: PLC0415

    root = default_runs_root()
    runs = sorted(p.parent for p in root.glob("*/*/state.json")) if root.exists() else []
    if not runs:
        console.print(f"no runs under {root}; no queue yet (step 8.1)")
        return
    table = Table("experiment", "run", "status", "done")
    for rd in runs:
        state = json.loads((rd / "state.json").read_text())
        table.add_row(
            rd.parent.name, rd.name, str(state.get("status", "?")), str((rd / "DONE").exists())
        )
    console.print(table)


@app.command()
def peer(
    cluster_file: Annotated[Path | None, typer.Option(help="Cluster YAML.")] = None,
) -> None:
    """Read-only status of every non-primary Spark (is its GPU free?)."""
    from qopd.utils.cluster import DEFAULT_CLUSTER_FILE, load_nodes, node_status  # noqa: PLC0415

    nodes, env_file = load_nodes(cluster_file or DEFAULT_CLUSTER_FILE)
    table = Table("node", "host", "reachable", "gpu", "mem avail GB", "gpu processes", "free")
    for node in nodes.values():
        if node.role == "primary":
            continue
        st = node_status(node, env_file)
        table.add_row(
            node.name,
            node.host,
            str(st.reachable) if st.error is None else f"no ({st.error})",
            st.gpu,
            "" if st.mem_available_gb is None else str(st.mem_available_gb),
            "\n".join(st.gpu_processes) or "-",
            "yes" if st.free else "no",
        )
    console.print(table)


@config_app.command("show")
def config_show(
    path: Annotated[Path, typer.Argument(help="YAML file under configs/.")],
    overrides: Annotated[list[str] | None, typer.Argument(help="dotted.key=value ...")] = None,
) -> None:
    """Print the fully resolved config (defaults composed, overrides applied, validated)."""
    import yaml  # noqa: PLC0415

    from qopd.config import load, to_dict  # noqa: PLC0415

    cfg = load(path, overrides or [])
    console.print(f"# {type(cfg).__name__} from {path}", markup=False)
    console.print(yaml.safe_dump(to_dict(cfg), sort_keys=False), markup=False, highlight=False)


@config_app.command("check")
def config_check() -> None:
    """Validate every YAML file under configs/."""
    from qopd.config import CONFIG_ROOT, all_config_files, load  # noqa: PLC0415

    failed = 0
    for path in all_config_files():
        rel = path.relative_to(CONFIG_ROOT)
        try:
            cfg = load(path)
            console.print(f"ok    {rel}  ({type(cfg).__name__})", markup=False)
        except Exception as e:
            failed += 1
            console.print(f"FAIL  {rel}: {e}", markup=False)
    if failed:
        raise typer.Exit(1)


@models_app.command("list")
def models_list() -> None:
    """Show every registry id with its source, backend, bits and pinned revision."""
    from qopd.models.registry import load_registry  # noqa: PLC0415

    table = Table("id", "role", "bits", "backend", "source", "revision")
    for cfg in load_registry().values():
        table.add_row(
            cfg.id, cfg.role, f"{cfg.bits:g}", cfg.backend, cfg.source, cfg.revision or "-"
        )
    console.print(table)


@models_app.command("verify")
def models_verify(
    ids: Annotated[list[str], typer.Argument(help="Model ids (e.g. S1 S2 T16 D16).")],
) -> None:
    """Check each HF repo id (at its pinned revision) or local rung directory exists."""
    from qopd.models.registry import ModelNotFoundError, resolve, verify_exists  # noqa: PLC0415

    failed = False
    for mid in ids:
        cfg = resolve(mid)
        try:
            console.print(f"ok    {mid:7s} {cfg.source} @ {verify_exists(cfg)}", markup=False)
        except ModelNotFoundError as e:
            failed = True
            console.print(f"FAIL  {e}", markup=False)
    if failed:
        raise typer.Exit(1)


@models_app.command("download")
def models_download(
    ids: Annotated[list[str], typer.Argument(help="Model ids to download.")],
    tokenizer_only: Annotated[bool, typer.Option(help="Only tokenizer/config files.")] = False,
    pin: Annotated[bool, typer.Option(help="Write the commit sha into configs/models.")] = True,
) -> None:
    """Download HF models into the cache and pin their revision (approval needed > 10 GB)."""
    from qopd.models.loading import download  # noqa: PLC0415
    from qopd.models.registry import pin_revision, resolve  # noqa: PLC0415

    for mid in ids:
        cfg = resolve(mid)
        path, sha = download(cfg, tokenizer_only=tokenizer_only)
        if pin and cfg.revision != sha:
            pin_revision(mid, sha)
        console.print(f"{mid}: {path} (revision {sha})", markup=False)


@models_app.command("tokcheck")
def models_tokcheck(
    teacher: Annotated[str, typer.Argument(help="Teacher (or first) model id.")],
    students: Annotated[list[str], typer.Argument(help="Student (or other) model ids.")],
) -> None:
    """Hard tokenizer identity check; exits 1 on any mismatch."""
    from qopd.models.loading import load_tokenizer  # noqa: PLC0415
    from qopd.models.registry import resolve  # noqa: PLC0415
    from qopd.models.tokenizer_check import compare_tokenizers  # noqa: PLC0415

    ref = load_tokenizer(resolve(teacher))
    failed = False
    for sid in students:
        diff = compare_tokenizers(ref, load_tokenizer(resolve(sid)))
        status = "ok  " if diff.identical else "FAIL"
        failed |= not diff.identical
        console.print(f"{status} {teacher} vs {sid}: {diff.summary()}", markup=False)
    if failed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
