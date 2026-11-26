"""Paper figures 1-4 (PLAN 10.2), rendered only from result files (CLAUDE.md rule 6).

Static vector PDFs for the LaTeX paper (white page). Colour has two jobs only:
- identity (students; teacher ladders): categorical slots 1-2, blue ``#2a78d6`` and orange
  ``#eb6834`` (validated all-pairs on white: CVD ΔE 9.2+, normal-vision 24+);
- magnitude (the quantized rungs, ordered by bits): one blue ramp light->dark
  ``#86b6ef … #104281`` (validated as an ordinal ramp on white), BF16 drawn in primary ink.
Reference lines (cold start, SFT, RLVR) are neutral grey, dashed, and labelled directly; every
series is also direct-labelled or in a legend, so identity never relies on colour alone.
One y-axis per panel (small multiples instead of dual axes).

Inputs (each figure is skipped when its inputs do not exist yet):
- E1 fidelity: ``runs/e1/<student>/fidelity/<rung>.json`` (``FidelityReport.to_dict``)
- student scores: ``reports/results/student_scores.json`` -
  ``[{"student", "teacher", "kind", "seed", "math500"}, ...]`` (kind: opd | coldstart | sft | rlvr)
- rung memory: ``reports/results/rungs.json`` - ``{"<rung>": {"memory_gb": ...}, ...}``
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from qopd.config import PROJECT_ROOT
from qopd.utils import io

FIG_DIR = PROJECT_ROOT / "paper" / "figures"
RESULTS = PROJECT_ROOT / "reports" / "results"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
STUDENT_COLOR = {"S1": "#2a78d6", "S2": "#eb6834"}
LADDER_COLOR = {"30B-A3B": "#2a78d6", "8B": "#eb6834"}
RUNG_COLOR = {
    "T16": INK,
    "T8": "#104281",
    "T4-fp": "#1c5cab",
    "T4-int": "#2a78d6",
    "T3": "#5598e7",
    "T2": "#86b6ef",
}
RUNG_ORDER = ("T16", "T8", "T4-fp", "T4-int", "T3", "T2")
REF_STYLE = {
    "coldstart": ("no OPD", (0, (1, 2))),
    "sft": ("off-policy SFT", (0, (4, 2))),
    "rlvr": ("RLVR (GRPO)", (0, (6, 2, 1, 2))),
}
FIDELITY_PANELS = (
    ("pearson", "Pearson $r$ of $r_t$"),
    ("mae", "mean $|\\Delta r_t|$"),
    ("topk_kl", "top-20 KL"),
    ("top1_agreement", "top-1 agreement"),
    ("sign_flip_rate", "sign flips"),
)
WIDTH = 6.5  # inches: \textwidth of the paper


def _plt() -> Any:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "figure.dpi": 150,
        }
    )
    return plt


def _save(fig: Any, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)  # for quick viewing
    return out


def spread(ys: Sequence[float], min_gap: float) -> list[float]:
    """Nudge label positions apart so neighbours are at least ``min_gap`` apart (order kept)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in pairwise(order):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    shift = (sum(out) - sum(ys)) / len(ys) if ys else 0.0  # re-centre the group
    return [y - shift for y in out]


def _label_ends(ax: Any, items: Sequence[tuple[float, float, str]], frac: float = 0.09) -> None:
    """Direct labels at line ends, spread vertically so they never overlap. Text wears ink."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    ys = spread([y for _, y, _ in items], frac * (hi - lo))
    for (x, _, text), y in zip(items, ys, strict=True):
        ax.annotate(
            text,
            (x, y),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=INK2,
            annotation_clip=False,
        )


# --- Figure 1: fidelity vs teacher bits ----------------------------------------------------------


def fig_fidelity(reports: dict[str, dict[str, dict[str, Any]]], out: Path) -> Path:
    """``reports[student][rung]`` = FidelityReport dict. One panel per metric."""
    plt = _plt()
    rungs = [r for r in RUNG_ORDER[1:] if any(r in v for v in reports.values())]
    fig, axes = plt.subplots(1, len(FIDELITY_PANELS), figsize=(WIDTH, 1.7), sharex=True)
    for ax, (key, title) in zip(axes, FIDELITY_PANELS, strict=True):
        ends = []
        for student in sorted(reports):
            xs = [i for i, r in enumerate(rungs) if r in reports[student]]
            ys = [reports[student][rungs[i]][key] for i in xs]
            ax.plot(xs, ys, marker="o", color=STUDENT_COLOR.get(student, INK), label=student)
            if xs:
                ends.append((xs[-1], ys[-1], student))
        ax.set_title(title, color=INK)
        ax.set_xticks(range(len(rungs)), rungs, rotation=45)
        if ax is axes[-1]:
            _label_ends(ax, ends)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.tight_layout(w_pad=1.0)
    return _save(fig, out)


# --- Figure 2: student accuracy vs teacher rung -------------------------------------------------


def fig_bits(scores: Sequence[dict[str, Any]], out: Path) -> Path:
    """One panel per student: OPD seeds (dots) + mean (line), reference lines for baselines."""
    plt = _plt()
    students = sorted({s["student"] for s in scores})
    fig, axes = plt.subplots(1, len(students), figsize=(WIDTH, 2.2), squeeze=False)
    for ax, student in zip(axes[0], students, strict=True):
        mine = [s for s in scores if s["student"] == student]
        opd = defaultdict(list)
        for s in mine:
            if s["kind"] == "opd":
                opd[s["teacher"]].append(100 * s["math500"])
        rungs = [r for r in RUNG_ORDER if r in opd]
        color = STUDENT_COLOR.get(student, INK)
        for i, r in enumerate(rungs):
            ax.scatter([i] * len(opd[r]), opd[r], s=12, color=color, alpha=0.5, linewidths=0)
        means = [statistics.fmean(opd[r]) for r in rungs]
        ax.plot(range(len(rungs)), means, marker="o", color=color, zorder=3)
        refs = []
        for kind, (name, dash) in REF_STYLE.items():
            vals = [100 * s["math500"] for s in mine if s["kind"] == kind]
            if vals:
                y = statistics.fmean(vals)
                # line stops before the label column so text is never struck through
                ax.hlines(
                    y, -0.4, len(rungs) - 0.8, color=MUTED, linestyle=dash, linewidth=1.0, zorder=1
                )
                refs.append((len(rungs) - 0.8, y, name))
        ax.set_xticks(range(len(rungs)), rungs)
        ax.set_title(f"student {student}", color=INK)
        ax.set_xlabel("teacher rung (bits decrease →)")
        ax.set_xlim(-0.4, len(rungs) - 0.2 + 1.2)
        _label_ends(ax, refs, frac=0.13)  # after limits are final, spread so labels never touch
    axes[0][0].set_ylabel("MATH-500 pass@1 (%)")
    fig.tight_layout()
    return _save(fig, out)


# --- Figure 3: reward error by entropy bucket ---------------------------------------------------


def fig_entropy(reports: dict[str, dict[str, Any]], out: Path, metric: str = "mae") -> Path:
    """``reports[rung]`` = FidelityReport dict for one student; lines per rung (ordinal ramp)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(WIDTH / 2, 2.2))
    rungs = [r for r in RUNG_ORDER[1:] if r in reports]
    buckets = list(next(iter(reports.values()))["by_entropy_bucket"]) if reports else []
    ends: list[tuple[float, float, str]] = []
    for r in rungs:
        ys = [reports[r]["by_entropy_bucket"][b].get(metric, float("nan")) for b in buckets]
        ax.plot(range(len(buckets)), ys, marker="o", color=RUNG_COLOR[r], label=r)
        ends.append((len(buckets) - 1, ys[-1], r))
    _label_ends(ax, ends)
    ax.set_xticks(range(len(buckets)), buckets, rotation=30)
    ax.set_xlabel("student entropy bucket (nats)")
    ax.set_ylabel("mean $|\\Delta r_t|$ vs BF16")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return _save(fig, out)


# --- Figure 4: accuracy vs teacher memory ---------------------------------------------

LADDERS = {"30B-A3B": ("T16", "T8", "T4-fp", "T4-int", "T3", "T2"), "8B": ("D16", "D8", "D4")}


def fig_memory(
    scores: Sequence[dict[str, Any]],
    rungs: dict[str, dict[str, Any]],
    out: Path,
    student: str = "S1",
) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(WIDTH / 2, 2.4))
    for ladder, ids in LADDERS.items():
        pts = []
        for t in ids:
            vals = [
                100 * s["math500"]
                for s in scores
                if s["student"] == student and s["kind"] == "opd" and s["teacher"] == t
            ]
            if vals and t in rungs and rungs[t].get("memory_gb") is not None:
                pts.append((rungs[t]["memory_gb"], statistics.fmean(vals), t))
        if not pts:
            continue
        pts.sort()
        color = LADDER_COLOR[ladder]
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker="o",
            color=color,
            label=f"Qwen3-{ladder}",
        )
        for x, y, t in pts:
            ax.annotate(
                t, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6, color=INK2
            )
    ax.set_xlabel("teacher weight memory (GB)")
    ax.set_ylabel(f"{student} MATH-500 pass@1 (%)")
    ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    return _save(fig, out)


# --- loading + driver ----------------------------------------------------------------------------


def load_e1(runs: Path = PROJECT_ROOT / "runs" / "e1") -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for f in sorted(runs.glob("*/fidelity/*.json")):
        out.setdefault(f.parent.parent.name, {})[f.stem] = io.read_json(f)
    return out


def make_all(
    fig_dir: Path = FIG_DIR, results: Path = RESULTS, e1_runs: Path = PROJECT_ROOT / "runs" / "e1"
) -> dict[str, Path | None]:
    """Render every figure whose inputs exist; others stay placeholders in the paper."""
    made: dict[str, Path | None] = {"fidelity": None, "bits": None, "entropy": None, "memory": None}
    e1 = load_e1(e1_runs)
    if e1:
        made["fidelity"] = fig_fidelity(e1, fig_dir / "fidelity.pdf")
        student = "S1" if "S1" in e1 else sorted(e1)[0]
        made["entropy"] = fig_entropy(e1[student], fig_dir / "entropy.pdf")
    scores_f, rungs_f = results / "student_scores.json", results / "rungs.json"
    if scores_f.exists():
        scores = io.read_json(scores_f)
        made["bits"] = fig_bits(scores, fig_dir / "bits.pdf")
        if rungs_f.exists():
            made["memory"] = fig_memory(scores, io.read_json(rungs_f), fig_dir / "memory.pdf")
    return made
