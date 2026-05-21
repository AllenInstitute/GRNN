import argparse
import concurrent.futures
import json
import multiprocessing as mp
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from l_mnist_training_pipeline import (
    ExperimentConfig,
    RNNClassifier,
    count_parameters,
    parse_gpu_ids,
    plot_mean_std_curve,
    save_json,
    setup_logging,
    summarize_run_group,
    train_parallel_run,
)


def run_parallel(tasks: List[Dict[str, Any]], max_workers: int, logger) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        future_to_task = {executor.submit(train_parallel_run, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            results.append(result)
            logger.info("Completed rnn_run%d on GPU %s", task["run_idx"], task["gpu_id"])
    results.sort(key=lambda item: int(item["run_idx"]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun the L-MNIST RNN baseline and save per-seed checkpoints.")
    parser.add_argument("--source-run-dir", type=Path, default=Path("runs/l_mnist_stage1_rnn_param_match_20260518_115200"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gpu-ids", type=str, default="")
    parser.add_argument("--runs-per-gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with (args.source_run_dir / "results.json").open("r", encoding="utf-8") as handle:
        source_results = json.load(handle)

    cfg = ExperimentConfig(**source_results["config"])
    if args.epochs is not None:
        cfg.epochs_rnn = args.epochs
    cfg.gpu_ids = args.gpu_ids
    cfg.parallel_runs = True

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        raise RuntimeError("No GPU IDs available. Pass --gpu-ids explicitly.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("runs") / f"l_mnist_rnn_checkpoint_rerun_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "histories").mkdir(parents=True, exist_ok=True)
    (output_dir / "progress").mkdir(parents=True, exist_ok=True)
    (output_dir / "run_logs").mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    run_seeds = list(source_results.get("run_seeds", [cfg.base_seed + i for i in range(cfg.n_runs)]))
    cfg.n_runs = len(run_seeds)
    rnn_params = count_parameters(
        RNNClassifier(input_size=28, hidden_size=cfg.rnn_hidden_dim, num_classes=cfg.num_classes),
        trainable_only=True,
    )

    logger.info("Starting RNN checkpoint rerun")
    logger.info("Source run: %s", args.source_run_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("GPU pool: %s", gpu_ids)
    logger.info("Run seeds: %s", run_seeds)
    logger.info("RNN hidden_dim=%d trainable_params=%d epochs=%d", cfg.rnn_hidden_dim, rnn_params, cfg.epochs_rnn)

    with (output_dir / "command.txt").open("w", encoding="utf-8") as handle:
        handle.write(" ".join(["python", "run_rnn_checkpoint_rerun.py"] + __import__("sys").argv[1:]) + "\n")

    cfg_payload = asdict(cfg)
    tasks = [
        {
            "config": cfg_payload,
            "model_name": "rnn",
            "run_idx": idx,
            "seed": int(seed),
            "gpu_id": gpu_ids[(idx - 1) % len(gpu_ids)],
            "output_dir": str(output_dir),
        }
        for idx, seed in enumerate(run_seeds, start=1)
    ]

    results = run_parallel(tasks, max_workers=len(gpu_ids) * max(1, args.runs_per_gpu), logger=logger)
    histories = [item["history"] for item in results]
    stats = summarize_run_group(histories)
    summary = {
        "config": asdict(cfg),
        "source_run_dir": str(args.source_run_dir),
        "run_seeds": run_seeds,
        "rnn": {
            "hidden_size": int(cfg.rnn_hidden_dim),
            "trainable_params": int(rnn_params),
            "checkpoints": [item["checkpoint"] for item in results if "checkpoint" in item],
            **stats,
        },
    }
    save_json(output_dir / "results.json", summary)

    plot_mean_std_curve(
        stats["train_loss_mean"],
        stats["train_loss_std"],
        title="RNN Train Loss (mean +/- std)",
        ylabel="Train Loss",
        save_path=output_dir / "rnn_train_loss.png",
    )
    plot_mean_std_curve(
        stats["test_loss_mean"],
        stats["test_loss_std"],
        title="RNN Test Loss (mean +/- std)",
        ylabel="Test Loss",
        save_path=output_dir / "rnn_test_loss.png",
    )

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Model,Trainable Params,Train Acc Mean,Train Acc Std,Test Acc Mean,Test Acc Std\n")
        handle.write(
            f"RNN,{rnn_params},{summary['rnn']['train_acc_mean']:.6f},{summary['rnn']['train_acc_std']:.6f},"
            f"{summary['rnn']['acc_mean']:.6f},{summary['rnn']['acc_std']:.6f}\n"
        )

    logger.info(
        "Final | RNN test acc mean/std: %.4f / %.4f",
        summary["rnn"]["acc_mean"],
        summary["rnn"]["acc_std"],
    )
    logger.info("RNN checkpoint rerun complete")


if __name__ == "__main__":
    main()