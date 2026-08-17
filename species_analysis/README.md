# Species differences in GFR parameters

This directory reproduces the mouse-vs-human comparison of fitted GFR parameters
reported in the manuscript section **"Species Differences in GFR Parameters"**
(Figure `fig:human_mouse`).

## Provenance

As stated in the manuscript, this analysis was produced with **AI assistance and
then reviewed, validated, and interpreted by the authors**. The agents operated
only on the fitted GFR parameter tables derived from this repository; all reported
findings were author-curated. The scripts here are the cleaned, self-contained
versions of that workflow, adapted to run directly against the data already in
this repository (no external downloads).

## Contents

| File | Produces | Description |
|------|----------|-------------|
| `extract_gfr_params.py` | `data/human_vs_mouse_gfr_params.csv` | Extracts per-cell GFR parameters (current/firing-rate kernel weights, activation-gain terms) for the `(20, 20)` fit config, labels species (from `donor__species`) and E/I class (from dendrite morphology), and keeps well-fit cells (EVR2 > 0.5). Result: 1130 mouse + 89 human neurons. |
| `statistical_tests.py` | `data/statistical_tests_summary.csv` | Mann–Whitney U tests (mouse vs. human) for each GFR parameter, split by All / Excitatory / Inhibitory, reporting medians, IQRs, and p-values. |
| `make_figure_human_mouse.py` | `figures/fig_human_mouse_v2.pdf` (and `.png`) | The manuscript figure: current- and firing-rate-kernel overlays (panels a, b) and violin comparisons of key parameters α₈, β₅, γ, c₁ (panels c–f) with significance annotations. |

Each script is independent and reads the fitted parameters directly from the
repository; they do not depend on one another.

## Data source

Both inputs already live in this repository:

- `../model/best_params.pickle` — fitted GFR parameters per cell and config.
- `../data/metadata.csv` — Allen Cell Types metadata (species, dendrite type,
  Cre line, layer).

## Running

Use a Python environment with `numpy`, `pandas`, `scipy`, `matplotlib`, and
`torch` installed (see the repository `requirements.txt`). From this directory:

```bash
python extract_gfr_params.py       # -> data/human_vs_mouse_gfr_params.csv
python statistical_tests.py        # -> data/statistical_tests_summary.csv
python make_figure_human_mouse.py  # -> figures/fig_human_mouse_v2.pdf, .png
```

The regenerated `figures/fig_human_mouse_v2.pdf` is the figure used in the
manuscript.

## Scope note

The scripts here reproduce the figure and the parameter-level statistics (EVR and
the α/β/γ/c₁ comparisons) cited in the manuscript. A small number of additional
species statistics mentioned in the text — the inhibitory **subclass** breakdown
(Pvalb-like / Sst-like maximum firing rate γ) and the **classifier** results
(logistic-regression species ROC-AUC, and the excitatory-vs-inhibitory
mouse→human transfer ROC-AUC) — come from the broader author-reviewed,
AI-assisted analysis.
