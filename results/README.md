# `results/` — frozen artifacts for the revision's quantitative tables

This folder holds the exact, frozen outputs that back the quantitative L-MNIST
tables in the revised manuscript, together with the trained checkpoints needed to
reproduce them. It is a **supplement** to the paper's core resource (the fitted
GFR dataset in [`../model/`](../model/) and the cell metadata in
[`../data/`](../data/)); it is *not* the resource itself. Nothing here is
required to load or use the GFR dataset.

Each subfolder pairs a rendered LaTeX table (the exact object in the manuscript)
with the machine-readable results it was rendered from and the checkpoints that
produced those results.

## Contents and mapping to the manuscript

### `per_step_accuracy/` → Table `tab:per-step-accuracy`

Per-step readout accuracy on L-MNIST for four **from-scratch** models trained at
a matched parameter budget (~7.3k params): GFR-RNN (hidden 65), vanilla RNN
(hidden 68), SNN-LIF (hidden 68), and SNN-Synaptic (hidden 68). Mean ± std over
5 runs.

| File | What it is |
| --- | --- |
| `per_step_accuracy_table.tex` | The rendered manuscript table (`tab:per-step-accuracy`). |
| `per_step_accuracy_table.csv` | Same numbers, machine-readable. |
| `eval_single_step_accuracy.json` | Per-run, per-step accuracies (keys `test`/`train` → model → `runs[]` with `step1`…`step5`, `vote_avg`). |
| `gfr_rnn_snn_log_train_loss.pdf` | Training-loss curves for the four models. |
| `checkpoints/gfr_lmnist_stage1_run{1..5}.pt` | GFR-RNN param-matched runs. |
| `checkpoints/rnn_run{1..5}.pt` | Vanilla RNN runs. |
| `checkpoints/snn_lif_run{1..5}.pt` | SNN-LIF runs. |
| `checkpoints/snn_synaptic_run{1..5}.pt` | SNN-Synaptic runs. |

Regenerate:

```bash
# Train (each writes 5 seeded checkpoints):
python run_stage1_rnn_param_match.py     # GFR-RNN (hidden 65), param-matched
python run_rnn_checkpoint_rerun.py       # vanilla RNN (hidden 68)
python run_snn_phase1_param_match.py     # SNN-LIF and SNN-Synaptic (hidden 68)
# Evaluate + render the table from the checkpoints above:
python eval_single_step_accuracy.py      # -> eval_single_step_accuracy.json + per_step_accuracy_table.{tex,csv}
```

### `bio_init/` → Table `tab:bio-init-per-step-percent`

Best-checkpoint per-step accuracy for 128-unit GFR-RNNs initialized with random
vs. biological GFR parameters, under trainable vs. frozen recurrent dynamics.
Mean ± std over 5 seeds; best checkpoint selected by test-set Step 1. Biological
rows use Δt = Δt' = 20, Noise-2 EVR > 0.7.

| File | What it is |
| --- | --- |
| `bio_init_per_step_table_percent.tex` | The rendered manuscript table (`tab:bio-init-per-step-percent`). |
| `results.json` | Full per-seed, per-step results for all three conditions. |
| `checkpoints/random_no_freeze_seed{1..5}_best.pt` | Random init, trainable. |
| `checkpoints/bio_no_freeze_seed{1..5}_best.pt` | Biological init, trainable. |
| `checkpoints/bio_freeze_seed{1..5}_best.pt` | Biological init, frozen recurrent dynamics. |

Regenerate:

```bash
python reproduce_bio_init_table.py       # trains the 3×5 runs -> results.json (+ *_best.pt checkpoints)
python generate_bio_init_latex_table.py  # renders bio_init_per_step_table_percent.tex from results.json
```

## Related results produced elsewhere

- **Activation-function comparison** (`tab:acti-compare`) is produced inline by
  [`../activation_benchmark.ipynb`](../activation_benchmark.ipynb).
- **GFR-RNN vs. SNN appendix** figures/analysis are in
  [`../appendix_gfr_rnn_vs_snn.ipynb`](../appendix_gfr_rnn_vs_snn.ipynb).
- **Human vs. mouse species analysis** (figure + statistics) is in
  [`../species_analysis/`](../species_analysis/).

## Notes

- Checkpoints are small (`state_dict`s for ~7k–24k-parameter models) and are kept
  so the reported numbers can be re-evaluated without re-training.
- Random seeds are fixed inside each training script; re-running reproduces the
  tabulated means to within the stated standard deviations.
