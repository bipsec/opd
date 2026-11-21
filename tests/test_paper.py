"""paper/: tables regenerate, and the LaTeX is structurally consistent (no TeX on this machine)."""

import re
from pathlib import Path

from qopd.paper.tables import TABLES_ALL, num, sci, tex, write_all

PAPER = Path(__file__).resolve().parents[1] / "paper"


def _all_tex() -> dict[Path, str]:
    return {p: p.read_text() for p in [PAPER / "main.tex", *sorted(PAPER.glob("sections/*.tex"))]}


def test_helpers():
    assert tex("a_b & 50%") == r"a\_b \& 50\%"
    assert num(16000) == "16{,}000" and num(None) == r"\TBD{}"
    assert sci(1e-8) == "$10^{-8}$" and sci(2e-5) == r"$2\times10^{-5}$"


def test_tables_regenerate(tmp_path):
    paths = write_all(tmp_path)
    assert {p.stem for p in paths} == set(TABLES_ALL) | {"numbers"}
    for p in paths:
        t = p.read_text()
        assert t.count("{") == t.count("}"), p.name
        if p.stem != "numbers":
            assert t.count(r"\begin{table}") == t.count(r"\end{table}") == 1, p.name
            label = re.search(r"\\label\{(tab:[^}]+)\}", t)
            assert label, p.name


def test_inputs_exist():
    for path, text in _all_tex().items():
        for target in re.findall(r"\\input\{([^}]+)\}", text):
            f = PAPER / (target if target.endswith(".tex") else f"{target}.tex")
            assert f.exists(), f"{path.name}: \\input{{{target}}} missing"


def test_refs_resolve():
    texts = list(_all_tex().values()) + [p.read_text() for p in PAPER.glob("tables/*.tex")]
    joined = "\n".join(texts)
    labels = set(re.findall(r"\\label\{([^}]+)\}", joined))
    labels |= {f"fig:{n}" for n in re.findall(r"\\placeholderfig\{([^}]+)\}", joined)}
    refs = set(re.findall(r"\\ref\{([^}]+)\}", joined))
    assert refs <= labels, sorted(refs - labels)


def test_number_macros_defined():
    defined = set(
        re.findall(r"\\newcommand\{\\(\w+)\}", (PAPER / "tables" / "numbers.tex").read_text())
    )
    used = set()
    for text in _all_tex().values():
        used |= {
            m
            for m in re.findall(
                r"\\(n[A-Z]\w*|agree\w+|rolloutMaxTok|teacherTopK|entropyBuckets|san\w+)", text
            )
        }
    assert used <= defined, sorted(used - defined)


def test_braces_balance_in_sections():
    for path, text in _all_tex().items():
        stripped = re.sub(r"\\[{}]", "", re.sub(r"(?<!\\)%.*", "", text))
        assert stripped.count("{") == stripped.count("}"), path.name


def test_no_hand_typed_results_in_results_sections():
    """E1-E5 sections must not contain decimal numbers (results come from tables/macros)."""
    for name in ("04_fidelity", "05_distillation", "06_mechanism", "07_memory_cost"):
        text = re.sub(r"(?<!\\)%.*", "", (PAPER / "sections" / f"{name}.tex").read_text())
        assert not re.findall(r"(?<![\w{])\d+\.\d+", text), name
