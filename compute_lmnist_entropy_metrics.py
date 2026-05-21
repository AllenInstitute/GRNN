import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from l_mnist_training_pipeline import ExperimentConfig, RNNClassifier, get_lmnist_data_loaders, make_gfr_model
from run_snn_phase1_param_match import config_from_dict, make_snn_model
from utils import reshape_image


def resolve_path(root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_paths(
    root: Path,
    run_dir: Path,
    summary: Dict[str, Any],
    summary_key: str,
    history_pattern: str,
    checkpoint_patterns: Iterable[str],
) -> List[Path]:
    paths: List[Path] = []
    payload = summary.get(summary_key, {})
    for raw_path in payload.get("checkpoints", []):
        paths.append(resolve_path(root, raw_path))

    histories_dir = run_dir / "histories"
    for history_path in sorted(histories_dir.glob(history_pattern)):
        history_payload = load_json(history_path)
        if "checkpoint" in history_payload:
            paths.append(resolve_path(root, history_payload["checkpoint"]))

    for pattern in checkpoint_patterns:
        paths.extend(path.resolve() for path in sorted(run_dir.glob(pattern)))
        paths.extend(path.resolve() for path in sorted(histories_dir.glob(pattern)))

    unique_paths: List[Path] = []
    seen = set()
    for path in paths:
        path = path.resolve()
        key = str(path)
        if key not in seen:
            unique_paths.append(path)
            seen.add(key)
    return [path for path in unique_paths if path.exists()]


def load_model(model_key: str, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location=device)

    if model_key == "gfr_rnn":
        cfg = ExperimentConfig(**payload["config"])
        model = make_gfr_model(
            hidden_dim=cfg.gfr_hidden_dim,
            freeze_neurons=False,
            freeze_activation=True,
            device=device,
            bio_units=cfg.bio_units,
        )
    elif model_key == "rnn":
        cfg = ExperimentConfig(**payload["config"])
        hidden_size = int(payload.get("hidden_size", cfg.rnn_hidden_dim))
        model = RNNClassifier(input_size=28, hidden_size=hidden_size, num_classes=cfg.num_classes).to(device)
    elif model_key in {"snn_lif", "snn_synaptic"}:
        cfg = config_from_dict(payload["config"])
        model = make_snn_model(model_key, cfg, device)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=1)


def first_and_decision_probs(
    model: torch.nn.Module,
    sequence: torch.Tensor,
    readout_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "forward_sequence"):
        logits = model.forward_sequence(sequence)
    else:
        model.reset(sequence.shape[0])
        for step in range(sequence.shape[1]):
            model(sequence[:, step, :])
        logits = model(model.zero_input(sequence.shape[0]))

    first_probs = torch.softmax(logits, dim=1)
    vote_scores = first_probs.clone()
    for _ in range(readout_steps - 1):
        if hasattr(model, "forward_zero_input"):
            logits = model.forward_zero_input()
        else:
            logits = model(model.zero_input(sequence.shape[0]))
        vote_scores += torch.softmax(logits, dim=1)

    decision_probs = vote_scores / float(readout_steps)
    return first_probs, decision_probs


def evaluate_checkpoint(
    model_key: str,
    checkpoint_path: Path,
    test_loader,
    device: torch.device,
    readout_steps: int,
) -> Dict[str, Any]:
    model = load_model(model_key, checkpoint_path, device)
    first_entropy_correct: List[float] = []
    first_entropy_incorrect: List[float] = []
    decision_entropy_correct: List[float] = []
    decision_entropy_incorrect: List[float] = []
    correct_count = 0
    total_count = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            sequence = reshape_image(images, variant="l").to(device)
            labels = labels.to(device)

            first_probs, decision_probs = first_and_decision_probs(model, sequence, readout_steps)
            predictions = torch.argmax(decision_probs, dim=1)
            correct = predictions == labels

            first_entropy = entropy_from_probs(first_probs)
            decision_entropy = entropy_from_probs(decision_probs)

            first_entropy_correct.extend(first_entropy[correct].detach().cpu().tolist())
            first_entropy_incorrect.extend(first_entropy[~correct].detach().cpu().tolist())
            decision_entropy_correct.extend(decision_entropy[correct].detach().cpu().tolist())
            decision_entropy_incorrect.extend(decision_entropy[~correct].detach().cpu().tolist())

            correct_count += int(correct.sum().item())
            total_count += int(labels.numel())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "checkpoint": str(checkpoint_path),
        "accuracy": correct_count / max(total_count, 1),
        "n_correct": correct_count,
        "n_incorrect": total_count - correct_count,
        "first_entropy_correct": float(np.mean(first_entropy_correct)) if first_entropy_correct else float("nan"),
        "first_entropy_incorrect": float(np.mean(first_entropy_incorrect)) if first_entropy_incorrect else float("nan"),
        "decision_entropy_correct": float(np.mean(decision_entropy_correct)) if decision_entropy_correct else float("nan"),
        "decision_entropy_incorrect": float(np.mean(decision_entropy_incorrect)) if decision_entropy_incorrect else float("nan"),
    }


def summarize_rows(label: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"model": label, "n_runs": len(rows)}
    for key in [
        "accuracy",
        "first_entropy_correct",
        "first_entropy_incorrect",
        "decision_entropy_correct",
        "decision_entropy_incorrect",
    ]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values))
    summary["n_correct_mean"] = float(np.mean([row["n_correct"] for row in rows]))
    summary["n_incorrect_mean"] = float(np.mean([row["n_incorrect"] for row in rows]))
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pm(row: Dict[str, Any], key: str, digits: int = 3) -> str:
    return f"${row[f'{key}_mean']:.{digits}f} \\pm {row[f'{key}_std']:.{digits}f}$"


def write_latex(path: Path, rows: List[Dict[str, Any]], readout_steps: int) -> None:
    lines = [
        "\\begin{table}",
        "    \\caption{Predictive entropy on correctly and incorrectly classified L-MNIST test samples.}",
        "    \\label{tab:lmnist-entropy}",
        "    \\centering",
        "    \\begin{tabular}{llllll}",
        "    \\toprule",
        (
            "    \\textbf{Model} & \\textbf{Test Acc.} & \\textbf{First Correct} & \\textbf{First Incorrect} & "
            f"\\textbf{{{readout_steps}-Readout Correct}} & \\textbf{{{readout_steps}-Readout Incorrect}} "
            "\\\\"
        ),
        "    \\midrule",
    ]
    for row in rows:
        lines.append(
            f"    {row['model']} & {pm(row, 'accuracy')} & {pm(row, 'first_entropy_correct')} & "
            f"{pm(row, 'first_entropy_incorrect')} & {pm(row, 'decision_entropy_correct')} & "
            f"{pm(row, 'decision_entropy_incorrect')} \\\\"
        )
    lines.extend([
        "    \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute L-MNIST entropy diagnostics from saved model checkpoints.")
    parser.add_argument("--gfr-rnn-dir", type=Path, default=Path("runs/l_mnist_stage1_rnn_param_match_20260518_115200"))
    parser.add_argument("--rnn-dir", type=Path, default=None, help="Optional separate RNN checkpoint rerun directory.")
    parser.add_argument("--snn-dir", type=Path, default=Path("runs_revision_snn/l_mnist_snn_phase1_param_match_20260520_171817"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs_revision_snn/l_mnist_gfr_rnn_snn_combined_20260521"))
    parser.add_argument("--readout-steps", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = Path.cwd()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    gfr_rnn_summary = load_json(args.gfr_rnn_dir / "results.json")
    rnn_dir = args.rnn_dir or args.gfr_rnn_dir
    rnn_summary = load_json(rnn_dir / "results.json")
    snn_summary = load_json(args.snn_dir / "results.json")
    batch_size = int(gfr_rnn_summary["config"]["batch_size"])
    _, test_loader = get_lmnist_data_loaders(batch_size)

    model_specs = [
        (
            "GFR-RNN",
            "gfr_rnn",
            args.gfr_rnn_dir,
            gfr_rnn_summary,
            "stage1",
            "stage1_run*.json",
            ["gfr_lmnist_stage1_run*.pt"],
        ),
        (
            "RNN",
            "rnn",
            rnn_dir,
            rnn_summary,
            "rnn",
            "rnn_run*.json",
            ["rnn_run*.pt"],
        ),
        (
            "SNN-LIF",
            "snn_lif",
            args.snn_dir,
            snn_summary,
            "snn_lif",
            "snn_lif_run*.json",
            ["snn_lif_run*.pt"],
        ),
        (
            "SNN-Synaptic",
            "snn_synaptic",
            args.snn_dir,
            snn_summary,
            "snn_synaptic",
            "snn_synaptic_run*.json",
            ["snn_synaptic_run*.pt"],
        ),
    ]

    per_run_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    missing: Dict[str, str] = {}
    for label, model_key, run_dir, summary, summary_key, history_pattern, patterns in model_specs:
        paths = checkpoint_paths(root, run_dir, summary, summary_key, history_pattern, patterns)
        if not paths:
            missing[label] = f"No checkpoint files found in {run_dir}"
            continue

        model_rows: List[Dict[str, Any]] = []
        for run_idx, checkpoint_path in enumerate(paths, start=1):
            row = evaluate_checkpoint(model_key, checkpoint_path, test_loader, device, args.readout_steps)
            row = {"model": label, "run_idx": run_idx, **row}
            model_rows.append(row)
            per_run_rows.append(row)
        summary_rows.append(summarize_rows(label, model_rows))

    write_csv(output_dir / "entropy_metrics_per_run.csv", per_run_rows)
    write_csv(output_dir / "entropy_metrics_summary.csv", summary_rows)
    write_latex(output_dir / "entropy_metrics_table.tex", summary_rows, args.readout_steps)
    (output_dir / "entropy_metrics_missing.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")

    print(f"Wrote entropy diagnostics to {output_dir}")
    if missing:
        print("Missing checkpoints:")
        for label, reason in missing.items():
            print(f"  {label}: {reason}")


if __name__ == "__main__":
    main()