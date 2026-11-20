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
teacher_app = typer.Typer(help="Teacher scoring backends.", no_args_is_help=True)
app.add_typer(teacher_app, name="teacher")
rollout_app = typer.Typer(help="Student rollout engine.", no_args_is_help=True)
app.add_typer(rollout_app, name="rollout")
data_app = typer.Typer(
    help="Data pipeline (prompt pool, eval suites, splits).", no_args_is_help=True
)
app.add_typer(data_app, name="data")


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


@teacher_app.command("agree")
def teacher_agree(
    model: Annotated[str, typer.Option(help="Stand-in teacher id.")] = "S2",
    n: Annotated[int, typer.Option(help="Number of sequences.")] = 20,
    max_tokens: Annotated[int, typer.Option(help="Max response tokens.")] = 512,
    top_k: Annotated[int, typer.Option(help="Top-k log-probs.")] = 20,
    seed: Annotated[int, typer.Option()] = 0,
    kv_cache_gb: Annotated[float, typer.Option(help="vLLM KV cache size (GB).")] = 4.0,
) -> None:
    """vLLM vs HF backend agreement (PLAN 1.3); writes runs/agreement/<run_id>/."""
    from qopd.teacher.agreement import run_agreement  # noqa: PLC0415

    summary = run_agreement(model, n, max_tokens, top_k, seed, kv_cache_gb)
    for k, v in summary.items():
        console.print(f"{k:26s} {v}", markup=False)
    if not summary["pass"]:
        raise typer.Exit(1)


@rollout_app.command("bench")
def rollout_bench(
    model: Annotated[str, typer.Option(help="Student id.")] = "S2",
    prompts: Annotated[int, typer.Option(help="Number of prompts.")] = 8,
    n: Annotated[int, typer.Option(help="Samples per prompt.")] = 2,
    max_tokens: Annotated[int | None, typer.Option(help="Default: configs/train/opd.yaml.")] = None,
    seed: Annotated[int, typer.Option()] = 0,
    kv_cache_gb: Annotated[float, typer.Option(help="vLLM KV cache size (GB).")] = 6.0,
) -> None:
    """Rollout throughput and memory at training settings (PLAN 1.4)."""
    from qopd.rollout.bench import run_bench  # noqa: PLC0415

    summary = run_bench(model, prompts, n, max_tokens, seed, kv_cache_gb)
    for k, v in summary.items():
        console.print(f"{k:30s} {v}", markup=False)
    if not summary["valid"]:
        raise typer.Exit(1)


@data_app.command("pool")
def data_pool(
    config: Annotated[Path, typer.Option(help="DataCfg YAML.")] = Path("configs/data/default.yaml"),
    overrides: Annotated[list[str] | None, typer.Argument(help="dotted.key=value ...")] = None,
) -> None:
    """Build the training prompt pool (PLAN 2.1) and refresh reports/data.md."""
    from qopd.config import DataCfg, load  # noqa: PLC0415
    from qopd.data.pool import build_pool  # noqa: PLC0415

    stats = build_pool(load(config, overrides or [], cls=DataCfg))
    for st in stats["stages"]:
        console.print(f"{st['stage']:12s} in={st['in']:6d} kept={st['kept']:6d}", markup=False)
    console.print(f"pool: {stats['pool']['n']} problems -> {stats['pool']['file']}", markup=False)


@data_app.command("evals")
def data_evals(
    suites: Annotated[list[str] | None, typer.Argument(help="Suites (default: all).")] = None,
    config: Annotated[Path, typer.Option(help="DataCfg YAML.")] = Path("configs/data/default.yaml"),
) -> None:
    """Download and convert the evaluation suites (PLAN 2.2); refresh reports/data.md."""
    from qopd.config import DataCfg, load  # noqa: PLC0415
    from qopd.data.evalsets import build_eval_suites  # noqa: PLC0415
    from qopd.data.pool import data_dir  # noqa: PLC0415
    from qopd.data.report import write_report  # noqa: PLC0415

    cfg = load(config, cls=DataCfg)
    stats = build_eval_suites(cfg, suites)
    write_report(data_dir(cfg))
    for name, r in stats["suites"].items():
        extra = r.get("reason", f"n={r.get('n')} sha256={str(r.get('sha256'))[:12]}")
        console.print(f"{name:13s} {r['status']:8s} {extra}", markup=False)


@data_app.command("report")
def data_report() -> None:
    """Re-render reports/data.md from the stats files."""
    from qopd.config import CONFIG_ROOT, DataCfg, load  # noqa: PLC0415
    from qopd.data.pool import data_dir  # noqa: PLC0415
    from qopd.data.report import write_report  # noqa: PLC0415

    cfg = load(CONFIG_ROOT / "data" / "default.yaml", cls=DataCfg)
    console.print(f"wrote {write_report(data_dir(cfg))}", markup=False)


if __name__ == "__main__":
    app()
