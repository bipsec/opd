# Paper draft

Working title: *How Many Bits Does a Teacher Need? On-Policy Distillation with Quantized Teachers*
(arXiv, cs.LG with cs.CL). The structure follows the paper outline in `EXPERIMENT_PLAN.md`.

## Layout

| Path | What | Edited by |
| --- | --- | --- |
| `main.tex` | Preamble, macros (`\TBD`, `\todo`, `\placeholderfig`), section order | hand |
| `sections/00_abstract.tex` … `09_appendix.tex` | One file per section of the outline | hand |
| `tables/*.tex`, `tables/numbers.tex` | Every table and every number quoted in the prose | **generated** |
| `refs.bib` | References; doc-cited works have author lists marked TBD | hand (verify all) |
| `figures/` | Figures 1–4 (produced in Phase 10 by `scripts/make_figures.py`) | generated |

## Rules

- **No number is typed by hand.** Results come from `tables/` and the macros in
  `tables/numbers.tex` (`\nPoolClean`, `\agreeMeanAbs`, …). Both are regenerated from `runs/`,
  `reports/`, the `data/**` stats files behind `reports/data.md`, and `configs/`:

  ```bash
  python scripts/make_paper_tables.py        # from the repo root (inside the dev container)
  ```

  Design constants that are part of the method (13-gram, group size 128, the ±1.0 TOST margin, …)
  appear inline because they come from the experiment plan, not from a run.
- Anything not available yet is `\TBD` (red in the PDF). `\todo{…}` (blue) marks author notes.
  Count what is left: `grep -c '\\TBD' paper/sections/*.tex paper/tables/*.tex`.
- Results sections (E1–E5) have their structure, captions and tests in place; their prose is
  written once the numbers exist.

## Build

No TeX is installed on the Spark. Either upload `paper/` to Overleaf, or on a machine with TeX
Live run `make` here (`latexmk -pdf main.tex`).
