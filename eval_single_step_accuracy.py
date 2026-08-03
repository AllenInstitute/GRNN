"""
Evaluate all L-MNIST model checkpoints at each individual readout step (steps 1–5)
and with 5-step softmax voting.

Reports test accuracy for each run and mean/std per model type.
Outputs a paper-ready LaTeX table.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from l_mnist_training_pipeline import (
    ExperimentConfig,
    RNNClassifier,
    make_gfr_model,
)
from network import Network
from run_snn_phase1_param_match import config_from_dict, make_snn_model
from snn_network import SNNNetwork, SNNNetworkSynaptic
from utils import reshape_image


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_test_loader(batch_size: int = 512) -> DataLoader:
    transform = torchvision.transforms.ToTensor()
    test_set = torchvision.datasets.MNIST(
        "data/mnist/test", download=True, train=False, transform=transform
    )
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)


def get_train_loader(batch_size: int = 512) -> DataLoader:
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(
        "data/mnist/train", download=True, train=True, transform=transform
    )
    return DataLoader(train_set, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_gfr_checkpoint(ckpt_path: Path, device: torch.device) -> Network:
    payload = torch.load(ckpt_path, map_location=device)
    cfg = ExperimentConfig(**payload["config"])
    model = make_gfr_model(
        hidden_dim=cfg.gfr_hidden_dim,
        freeze_neurons=False,
        freeze_activation=True,
        device=device,
        bio_units=cfg.bio_units,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def load_rnn_checkpoint(ckpt_path: Path, device: torch.device) -> RNNClassifier:
    payload = torch.load(ckpt_path, map_location=device)
    hidden_size = int(payload.get("hidden_size", 68))
    model = RNNClassifier(input_size=28, hidden_size=hidden_size, num_classes=10).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def load_snn_checkpoint(
    ckpt_path: Path, model_key: str, device: torch.device
) -> torch.nn.Module:
    payload = torch.load(ckpt_path, map_location=device)
    cfg = config_from_dict(payload["config"])
    model = make_snn_model(model_key, cfg, device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation: per-step accuracy (steps 1–5) and vote
# ---------------------------------------------------------------------------
@torch.inference_mode()
def eval_per_step_and_vote(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_readout: int = 5,
) -> Dict[str, float]:
    """Return accuracy at each readout step and the 5-step vote accuracy."""
    step_correct = [0] * n_readout
    vote_correct = 0
    total = 0

    for images, labels in loader:
        sequence = reshape_image(images, variant="l").to(device)
        labels = labels.to(device)
        batch = sequence.shape[0]

        model.reset(batch)
        for t in range(sequence.shape[1]):
            model(sequence[:, t, :])

        vote_scores = None
        for step_idx in range(n_readout):
            logits = model(model.zero_input(batch))
            pred = torch.argmax(logits, dim=1)
            step_correct[step_idx] += (pred == labels).sum().item()

            softmax_scores = torch.softmax(logits, dim=1)
            if vote_scores is None:
                vote_scores = softmax_scores
            else:
                vote_scores += softmax_scores

        vote_pred = torch.argmax(vote_scores, dim=1)
        vote_correct += (vote_pred == labels).sum().item()
        total += labels.numel()

    result = {}
    for step_idx in range(n_readout):
        result[f"step{step_idx+1}"] = step_correct[step_idx] / max(total, 1)
    result["vote_avg"] = vote_correct / max(total, 1)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_latex_table(results: Dict[str, Dict[str, Any]], metric_keys: List[str], n_readout: int, split: str) -> str:
    """Generate a LaTeX table string for the given results."""
    caption = f"Per-step readout accuracy on the L-MNIST {split} set (mean $\\pm$ std over 5 runs)."
    label = f"tab:per-step-accuracy-{split}"

    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(f"    \\caption{{{caption}}}")
    latex_lines.append(f"    \\label{{{label}}}")
    latex_lines.append(r"    \centering")
    latex_lines.append(r"    \small")
    latex_lines.append(r"    \begin{tabular}{l" + "c" * (n_readout + 1) + "}")
    latex_lines.append(r"    \toprule")
    col_headers = " & ".join([r"\textbf{Model}"] + [f"\\textbf{{Step {i+1}}}" for i in range(n_readout)] + [r"\textbf{Vote (1--5)}"])
    latex_lines.append(f"    {col_headers} \\\\")
    latex_lines.append(r"    \midrule")

    for group_name, r in results.items():
        short_name = group_name.replace(" (hidden=65)", "").replace(" (hidden=86)", " (large)").replace(" (hidden=68)", "")
        cells = [short_name]
        for key in metric_keys:
            mean_val = r[f"{key}_mean"]
            std_val = r[f"{key}_std"]
            cells.append(f"${mean_val:.3f} \\pm {std_val:.3f}$")
        latex_lines.append(f"    {' & '.join(cells)} \\\\")

    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"    \end{tabular}")
    latex_lines.append(r"\end{table}")
    return "\n".join(latex_lines)


def generate_latex_table_combined_transposed(
    test_results: Dict[str, Dict[str, Any]],
    train_results: Dict[str, Dict[str, Any]],
    metric_keys: List[str],
    n_readout: int,
) -> str:
    """Generate a transposed LaTeX table with train/test in one table.

    Rows = readout steps + vote; Columns = models (one sub-column per model).
    Two row groups: Train and Test.
    """
    model_names = list(test_results.keys())
    short_names = [
        name.replace(" (hidden=65)", "").replace(" (hidden=86)", " (large)").replace(" (hidden=68)", "")
        for name in model_names
    ]
    n_models = len(model_names)

    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"    \caption{Per-step readout accuracy (\%) on the L-MNIST task (mean $\pm$ std over 5 runs).}")
    latex_lines.append(r"    \label{tab:per-step-accuracy}")
    latex_lines.append(r"    \centering")
    latex_lines.append(r"    \small")
    # columns: readout step | model1 | model2 | ...
    latex_lines.append(r"    \begin{tabular}{l" + "c" * n_models + "}")
    latex_lines.append(r"    \toprule")
    col_headers = " & ".join([r"\textbf{Readout Step}"] + [f"\\textbf{{{s}}}" for s in short_names])
    latex_lines.append(f"    {col_headers} \\\\")

    # --- Test section ---
    latex_lines.append(r"    \midrule")
    latex_lines.append(r"    \multicolumn{" + str(n_models + 1) + r"}{l}{\textit{Test set}} \\")
    latex_lines.append(r"    \midrule")

    for key in metric_keys:
        if key == "vote_avg":
            row_label = "Vote (1--5)"
        else:
            step_num = key.replace("step", "")
            row_label = f"Step {step_num}"
        cells = [row_label]
        for model_name in model_names:
            r = test_results[model_name]
            mean_val = r[f"{key}_mean"] * 100
            std_val = r[f"{key}_std"] * 100
            cells.append(f"${mean_val:.1f} \\pm {std_val:.1f}$")
        latex_lines.append(f"    {' & '.join(cells)} \\\\")

    # --- Train section ---
    latex_lines.append(r"    \midrule")
    latex_lines.append(r"    \multicolumn{" + str(n_models + 1) + r"}{l}{\textit{Train set}} \\")
    latex_lines.append(r"    \midrule")

    for key in metric_keys:
        if key == "vote_avg":
            row_label = "Vote (1--5)"
        else:
            step_num = key.replace("step", "")
            row_label = f"Step {step_num}"
        cells = [row_label]
        for model_name in model_names:
            r = train_results[model_name]
            mean_val = r[f"{key}_mean"] * 100
            std_val = r[f"{key}_std"] * 100
            cells.append(f"${mean_val:.1f} \\pm {std_val:.1f}$")
        latex_lines.append(f"    {' & '.join(cells)} \\\\")

    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"    \end{tabular}")
    latex_lines.append(r"\end{table}")
    return "\n".join(latex_lines)


def generate_csv(results: Dict[str, Dict[str, Any]], metric_keys: List[str], n_readout: int) -> str:
    """Generate CSV content for the given results."""
    header = "Model," + ",".join(f"Step {i+1} Mean,Step {i+1} Std" for i in range(n_readout)) + ",Vote Mean,Vote Std\n"
    rows = [header]
    for group_name, r in results.items():
        row_vals = [group_name]
        for key in metric_keys:
            row_vals.append(f"{r[f'{key}_mean']:.4f}")
            row_vals.append(f"{r[f'{key}_std']:.4f}")
        rows.append(",".join(row_vals) + "\n")
    return "".join(rows)


def evaluate_all(
    checkpoint_groups: Dict[str, List[Tuple[str, Path, str]]],
    loader: DataLoader,
    device: torch.device,
    n_readout: int,
    split_name: str,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all checkpoint groups on a given loader."""
    metric_keys = [f"step{i}" for i in range(1, n_readout + 1)] + ["vote_avg"]
    results: Dict[str, Dict[str, Any]] = {}

    for group_name, checkpoints in checkpoint_groups.items():
        print(f"\n{'='*60}")
        print(f"  {group_name} [{split_name}]")
        print(f"{'='*60}")

        run_results: List[Dict[str, float]] = []

        for run_label, ckpt_path, loader_type in checkpoints:
            if not ckpt_path.exists():
                print(f"  [{run_label}] SKIP — {ckpt_path} not found")
                continue

            if loader_type == "gfr":
                model = load_gfr_checkpoint(ckpt_path, device)
            elif loader_type == "rnn":
                model = load_rnn_checkpoint(ckpt_path, device)
            elif loader_type in ("snn_lif", "snn_synaptic"):
                model = load_snn_checkpoint(ckpt_path, loader_type, device)
            else:
                raise ValueError(f"Unknown loader type: {loader_type}")

            metrics = eval_per_step_and_vote(model, loader, device, n_readout=n_readout)
            run_results.append(metrics)

            step_strs = " | ".join(f"s{i+1}:{metrics[f'step{i+1}']:.4f}" for i in range(n_readout))
            print(f"  [{run_label}] {step_strs} | vote:{metrics['vote_avg']:.4f}")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if run_results:
            summary: Dict[str, Any] = {"runs": run_results}
            for key in metric_keys:
                vals = [r[key] for r in run_results]
                summary[f"{key}_mean"] = float(np.mean(vals))
                summary[f"{key}_std"] = float(np.std(vals))

            print(f"  -------")
            for key in metric_keys:
                print(f"  {key}: {summary[f'{key}_mean']:.4f} ± {summary[f'{key}_std']:.4f}")

            results[group_name] = summary

    return results


def print_summary(results: Dict[str, Dict[str, Any]], metric_keys: List[str], n_readout: int, split_name: str):
    """Print a console summary table."""
    print(f"\n\n{'='*110}")
    print(f"  SUMMARY ({split_name} Accuracy)")
    print(f"{'='*110}")
    header = f"{'Model':<30}" + "".join(f"{'Step '+str(i+1):>14}" for i in range(n_readout)) + f"{'Vote (1-5)':>14}"
    print(header)
    print(f"{'-'*110}")
    for group_name, r in results.items():
        row = f"{group_name:<30}"
        for key in metric_keys:
            row += f"{r[f'{key}_mean']:.4f}±{r[f'{key}_std']:.4f}".rjust(14)
        print(row)
    print(f"{'='*110}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_loader = get_test_loader(batch_size=512)
    train_loader = get_train_loader(batch_size=512)

    # ---- Define checkpoint groups ----
    checkpoint_groups: Dict[str, List[Tuple[str, Path, str]]] = {
        "GFR-RNN (hidden=65)": [
            (f"run{i}", Path(f"runs/l_mnist_stage1_rnn_param_match_20260518_115200/gfr_lmnist_stage1_run{i}.pt"), "gfr")
            for i in range(1, 6)
        ],
        "RNN (hidden=68)": [
            (f"run{i}", Path(f"runs/l_mnist_rnn_checkpoint_rerun_20260521_a100_ckpt/rnn_run{i}.pt"), "rnn")
            for i in range(1, 6)
        ],
        "SNN-LIF (hidden=68)": [
            (f"run{i}", Path(f"runs_revision_snn/l_mnist_snn_phase1_param_match_20260521_a100_ckpt/snn_lif_run{i}.pt"), "snn_lif")
            for i in range(1, 6)
        ],
        "SNN-Synaptic (hidden=68)": [
            (f"run{i}", Path(f"runs_revision_snn/l_mnist_snn_phase1_param_match_20260521_a100_ckpt/snn_synaptic_run{i}.pt"), "snn_synaptic")
            for i in range(1, 6)
        ],
    }

    n_readout = 5
    metric_keys = [f"step{i}" for i in range(1, n_readout + 1)] + ["vote_avg"]

    # ---- Evaluate on test set ----
    print("\n" + "#" * 70)
    print("#  EVALUATING ON TEST SET")
    print("#" * 70)
    test_results = evaluate_all(checkpoint_groups, test_loader, device, n_readout, "Test")
    print_summary(test_results, metric_keys, n_readout, "Test")

    # ---- Evaluate on train set ----
    print("\n" + "#" * 70)
    print("#  EVALUATING ON TRAIN SET")
    print("#" * 70)
    train_results = evaluate_all(checkpoint_groups, train_loader, device, n_readout, "Train")
    print_summary(train_results, metric_keys, n_readout, "Train")

    # ---- Generate LaTeX tables ----
    combined_latex = generate_latex_table_combined_transposed(
        test_results, train_results, metric_keys, n_readout
    )
    print(f"\n\nLaTeX Table (Combined Transposed):\n{combined_latex}")

    # ---- Save results ----
    output_dir = Path("runs_revision_snn")
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    all_results = {"test": test_results, "train": train_results}
    json_path = output_dir / "eval_single_step_accuracy.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # LaTeX (combined transposed)
    tex_path = output_dir / "per_step_accuracy_table.tex"
    with open(tex_path, "w") as f:
        f.write(combined_latex + "\n")
    print(f"LaTeX table saved to: {tex_path}")

    # CSV
    csv_path = output_dir / "per_step_accuracy_table_test.csv"
    with open(csv_path, "w") as f:
        f.write(generate_csv(test_results, metric_keys, n_readout))
    print(f"CSV table (test) saved to: {csv_path}")

    csv_path_train = output_dir / "per_step_accuracy_table_train.csv"
    with open(csv_path_train, "w") as f:
        f.write(generate_csv(train_results, metric_keys, n_readout))
    print(f"CSV table (train) saved to: {csv_path_train}")


if __name__ == "__main__":
    main()
