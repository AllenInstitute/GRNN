import argparse
import concurrent.futures
import json
import logging
import math
import multiprocessing as mp
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from l_mnist_training_pipeline import (
    ExperimentConfig,
    RNNClassifier,
    count_parameters,
    make_gfr_model,
    parse_gpu_ids,
    plot_mean_std_curve,
    save_json,
    setup_logging,
    summarize_run_group,
)
from rerun_gfr_per_seed_pipeline import _copy_baseline_artifacts, _train_stage1


def choose_stage1_hidden_size(target_params: int, bio_units: bool, max_hidden: int = 256) -> Tuple[int, int]:
    best_hidden = 1
    best_params = 0
    best_gap = math.inf
    device = torch.device("cpu")
    for hidden_dim in range(1, max_hidden + 1):
        model = make_gfr_model(
            hidden_dim=hidden_dim,
            freeze_neurons=False,
            freeze_activation=True,
            device=device,
            bio_units=bio_units,
        )
        params = count_parameters(model, trainable_only=True)
        gap = abs(params - target_params)
        if gap < best_gap:
            best_gap = gap
            best_hidden = hidden_dim
            best_params = params
    return best_hidden, best_params


def run_parallel(tasks: List[Dict[str, Any]], max_workers: int, logger: logging.Logger) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        future_to_task = {executor.submit(_train_stage1, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            results.append(result)
            logger.info("Completed stage1_run%d on GPU %s", task["run_idx"], task["gpu_id"])
    results.sort(key=lambda item: int(item["run_idx"]))
    return results


def plot_stage1_vs_rnn(stage1_stats: Dict[str, Any], rnn_stats: Dict[str, Any], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, payload, color in [
        ("GFR-RNN", stage1_stats, "tab:blue"),
        ("RNN", rnn_stats, "tab:orange"),
    ]:
        mean = np.asarray(payload["train_loss_mean"], dtype=np.float64)
        std = np.asarray(payload["train_loss_std"], dtype=np.float64)
        epochs = np.arange(1, len(mean) + 1)
        ax.plot(epochs, mean, label=label, color=color, linewidth=2.2)
        ax.fill_between(
            epochs,
            np.maximum(mean - std, 1e-12),
            np.maximum(mean + std, 1e-12),
            color=color,
            alpha=0.18,
            linewidth=0,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Cross-Entropy Loss")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_rnn_log_train_loss.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage-1 GFR-RNN with trainable params matched to the hidden-size-68 RNN."
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=Path("runs/l_mnist_gfr_per_seed_20260515_100137"),
        help="Run directory containing the RNN baseline histories/results.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gpu-ids", type=str, default="")
    parser.add_argument("--rnn-hidden-dim", type=int, default=68)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("runs") / f"l_mnist_stage1_rnn_param_match_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "histories").mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    source_results_path = args.source_run_dir / "results.json"
    with open(source_results_path, "r", encoding="utf-8") as f:
        source_results = json.load(f)

    cfg = ExperimentConfig(**source_results["config"])
    rnn_target = count_parameters(
        RNNClassifier(input_size=28, hidden_size=args.rnn_hidden_dim, num_classes=cfg.num_classes),
        trainable_only=True,
    )
    hidden_dim, stage1_params = choose_stage1_hidden_size(rnn_target, bio_units=cfg.bio_units)
    cfg.gfr_hidden_dim = hidden_dim
    cfg.gfr_stage2_target_params = rnn_target
    cfg.parallel_runs = True
    cfg.gpu_ids = args.gpu_ids

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        raise RuntimeError("No GPU IDs available. Pass --gpu-ids explicitly.")

    run_seeds = list(source_results.get("run_seeds", [cfg.base_seed + i for i in range(cfg.n_runs)]))
    logger.info("Source run: %s", args.source_run_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("GPU pool: %s", gpu_ids)
    logger.info("Run seeds: %s", run_seeds)
    logger.info(
        "RNN hidden_dim=%d trainable_params=%d; matched Stage-1 GFR hidden_dim=%d trainable_params=%d",
        args.rnn_hidden_dim,
        rnn_target,
        hidden_dim,
        stage1_params,
    )

    with open(output_dir / "command.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(["python", "run_stage1_rnn_param_match.py"] + __import__("sys").argv[1:]) + "\n")

    _copy_baseline_artifacts(args.source_run_dir, output_dir)

    tasks = [
        {
            "config": asdict(cfg),
            "run_idx": i + 1,
            "seed": int(seed),
            "gpu_id": gpu_ids[i % len(gpu_ids)],
            "output_dir": str(output_dir),
        }
        for i, seed in enumerate(run_seeds)
    ]
    stage1_results = run_parallel(tasks, max_workers=len(gpu_ids), logger=logger)
    stage1_histories = [item["history"] for item in stage1_results]
    stage1_stats = summarize_run_group(stage1_histories)

    rnn_stats = dict(source_results["rnn"])
    if "histories" not in rnn_stats:
        rnn_histories = []
        for path in sorted((args.source_run_dir / "histories").glob("rnn_run*.json")):
            with open(path, "r", encoding="utf-8") as f:
                rnn_histories.append(json.load(f)["history"])
        rnn_stats = summarize_run_group(rnn_histories)

    plot_mean_std_curve(
        stage1_stats["train_loss_mean"],
        stage1_stats["train_loss_std"],
        title="GFR-RNN Stage-1 Train Loss (mean +/- std)",
        ylabel="Train Loss",
        save_path=output_dir / "stage1_train_loss.png",
    )
    plot_mean_std_curve(
        stage1_stats["test_loss_mean"],
        stage1_stats["test_loss_std"],
        title="GFR-RNN Stage-1 Test Loss (mean +/- std)",
        ylabel="Test Loss",
        save_path=output_dir / "stage1_test_loss.png",
    )
    plot_stage1_vs_rnn(stage1_stats, rnn_stats, output_dir)

    summary = {
        "config": asdict(cfg),
        "source_run_dir": str(args.source_run_dir),
        "run_seeds": run_seeds,
        "rnn_hidden_dim": args.rnn_hidden_dim,
        "rnn_trainable_params": int(rnn_target),
        "stage1_hidden_dim": int(hidden_dim),
        "stage1_trainable_params": int(stage1_params),
        "stage1": {
            "checkpoints": [item["checkpoint"] for item in stage1_results],
            "trainable_params": int(stage1_params),
            **stage1_stats,
        },
        "rnn": rnn_stats,
    }
    save_json(output_dir / "results.json", summary)

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Model,Trainable Params,Train Acc Mean,Train Acc Std,Test Acc Mean,Test Acc Std\n")
        f.write(
            f"GFR-RNN,{stage1_params},"
            f"{summary['stage1']['train_acc_mean']:.6f},{summary['stage1']['train_acc_std']:.6f},"
            f"{summary['stage1']['acc_mean']:.6f},{summary['stage1']['acc_std']:.6f}\n"
        )
        f.write(
            f"RNN,{rnn_target},"
            f"{summary['rnn']['train_acc_mean']:.6f},{summary['rnn']['train_acc_std']:.6f},"
            f"{summary['rnn']['acc_mean']:.6f},{summary['rnn']['acc_std']:.6f}\n"
        )

    logger.info(
        "Final | GFR-RNN test acc mean/std: %.4f / %.4f | RNN: %.4f / %.4f",
        summary["stage1"]["acc_mean"],
        summary["stage1"]["acc_std"],
        summary["rnn"]["acc_mean"],
        summary["rnn"]["acc_std"],
    )
    logger.info("Stage-1 parameter-matched run complete")


if __name__ == "__main__":
    main()
