import argparse
import concurrent.futures
import csv
import json
import logging
import multiprocessing as mp
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from l_mnist_training_pipeline import (
    ExperimentConfig,
    count_parameters,
    effective_trainable_parameters,
    get_lmnist_data_loaders,
    load_frozen_gfr_neurons_from_checkpoint,
    make_gfr_model,
    parse_gpu_ids,
    plot_log_train_loss_comparison,
    plot_mean_std_curve,
    save_json,
    set_seed,
    set_trainable_recurrent_only,
    setup_logging,
    setup_run_logging,
    summarize_run_group,
    train_one_run,
)


def _copy_baseline_artifacts(source_dir: Path, output_dir: Path) -> None:
    histories_dir = output_dir / "histories"
    histories_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ("rnn_run*.json", "lstm_run*.json"):
        for path in sorted((source_dir / "histories").glob(pattern)):
            shutil.copy2(path, histories_dir / path.name)

    for name in (
        "rnn_train_loss.png",
        "rnn_test_loss.png",
        "lstm_train_loss.png",
        "lstm_test_loss.png",
    ):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def _train_stage1(task: Dict[str, Any]) -> Dict[str, Any]:
    cfg = ExperimentConfig(**task["config"])
    run_idx = int(task["run_idx"])
    seed = int(task["seed"])
    gpu_id = int(task["gpu_id"])
    output_dir = Path(task["output_dir"])

    set_seed(seed)
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    log_dir = output_dir / "run_logs"
    progress_dir = output_dir / "progress"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"stage1_run{run_idx}"
    logger = setup_run_logging(log_dir / f"{run_name}.log", f"gfr_per_seed.{run_name}")
    logger.info("%s | assigned_gpu=%d | seed=%d", run_name, gpu_id, seed)

    train_loader, test_loader = get_lmnist_data_loaders(cfg.batch_size)
    model = make_gfr_model(
        hidden_dim=cfg.gfr_hidden_dim,
        freeze_neurons=False,
        freeze_activation=True,
        device=device,
        bio_units=cfg.bio_units,
    )
    trainable_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)
    logger.info("%s | trainable_params=%d | epochs=%d", run_name, trainable_params, cfg.epochs_stage1)

    history = train_one_run(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=cfg.epochs_stage1,
        lr=cfg.lr,
        is_gfr=True,
        device=device,
        logger=logger,
        run_name=run_name,
        progress_path=progress_dir / f"{run_name}.json",
    )

    checkpoint_path = output_dir / f"gfr_lmnist_stage1_run{run_idx}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "history": history,
            "seed": seed,
            "run_idx": run_idx,
        },
        checkpoint_path,
    )

    result = {
        "model_name": "stage1",
        "run_idx": run_idx,
        "seed": seed,
        "gpu_id": gpu_id,
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    save_json(output_dir / "histories" / f"{run_name}.json", result)
    logger.info("%s | checkpoint=%s | complete", run_name, checkpoint_path)
    return result


def _train_stage2(task: Dict[str, Any]) -> Dict[str, Any]:
    cfg = ExperimentConfig(**task["config"])
    run_idx = int(task["run_idx"])
    seed = int(task["seed"])
    gpu_id = int(task["gpu_id"])
    output_dir = Path(task["output_dir"])
    stage1_ckpt = Path(task["stage1_ckpt"])

    set_seed(seed)
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    log_dir = output_dir / "run_logs"
    progress_dir = output_dir / "progress"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"stage2_run{run_idx}"
    logger = setup_run_logging(log_dir / f"{run_name}.log", f"gfr_per_seed.{run_name}")
    logger.info("%s | assigned_gpu=%d | seed=%d | stage1_ckpt=%s", run_name, gpu_id, seed, stage1_ckpt)

    train_loader, test_loader = get_lmnist_data_loaders(cfg.batch_size)
    model = make_gfr_model(
        hidden_dim=cfg.gfr_hidden_dim,
        freeze_neurons=False,
        freeze_activation=True,
        device=device,
        bio_units=cfg.bio_units,
    )
    load_frozen_gfr_neurons_from_checkpoint(model, stage1_ckpt, device)
    set_trainable_recurrent_only(model, target_params=cfg.gfr_stage2_target_params)
    trainable_params = effective_trainable_parameters(model)
    total_params = count_parameters(model, trainable_only=False)
    logger.info("%s | trainable_params=%d | epochs=%d", run_name, trainable_params, cfg.epochs_stage2)

    history = train_one_run(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=cfg.epochs_stage2,
        lr=cfg.lr,
        is_gfr=True,
        device=device,
        logger=logger,
        run_name=run_name,
        progress_path=progress_dir / f"{run_name}.json",
    )

    result = {
        "model_name": "stage2",
        "run_idx": run_idx,
        "seed": seed,
        "gpu_id": gpu_id,
        "stage1_checkpoint": str(stage1_ckpt),
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "history": history,
    }
    save_json(output_dir / "histories" / f"{run_name}.json", result)
    logger.info("%s | complete", run_name)
    return result


def _run_parallel(tasks: List[Dict[str, Any]], worker, max_workers: int, logger: logging.Logger, label: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        future_to_task = {executor.submit(worker, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            results.append(result)
            logger.info("Completed %s_run%d on GPU %s", label, task["run_idx"], task["gpu_id"])
    results.sort(key=lambda item: int(item["run_idx"]))
    return results


def _write_loss_artifacts(output_dir: Path, summary: Dict[str, Any]) -> None:
    curves_dir = output_dir / "loss_artifacts"
    curves_dir.mkdir(parents=True, exist_ok=True)
    model_keys = ["stage1", "stage2", "rnn", "lstm"]

    npz_payload: Dict[str, np.ndarray] = {}
    for key in model_keys:
        for curve_key in ["train_loss_mean", "train_loss_std", "test_loss_mean", "test_loss_std"]:
            npz_payload[f"{key}_{curve_key}"] = np.asarray(summary[key][curve_key], dtype=np.float64)
    np.savez(curves_dir / "loss_curves.npz", **npz_payload)

    with open(curves_dir / "loss_curves.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epoch", "train_loss_mean", "train_loss_std", "test_loss_mean", "test_loss_std"])
        for key in model_keys:
            n_epochs = len(summary[key]["train_loss_mean"])
            for epoch in range(n_epochs):
                writer.writerow(
                    [
                        key,
                        epoch + 1,
                        summary[key]["train_loss_mean"][epoch],
                        summary[key]["train_loss_std"][epoch],
                        summary[key]["test_loss_mean"][epoch],
                        summary[key]["test_loss_std"][epoch],
                    ]
                )

    with open(curves_dir / "run_histories.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "run_idx", "epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        for key in model_keys:
            for run_idx, history in enumerate(summary[key]["histories"], start=1):
                n_epochs = len(history["train_loss"])
                for epoch in range(n_epochs):
                    writer.writerow(
                        [
                            key,
                            run_idx,
                            epoch + 1,
                            history["train_loss"][epoch],
                            history["train_acc"][epoch],
                            history["test_loss"][epoch],
                            history["test_acc"][epoch],
                        ]
                    )


def _baseline_from_source(source_results: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    payload = dict(source_results[model_name])
    payload["histories"] = list(source_results[model_name]["histories"])
    return payload


def _build_summary(
    cfg: ExperimentConfig,
    source_results: Dict[str, Any],
    output_dir: Path,
    gpu_ids: List[int],
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stage1_histories = [item["history"] for item in stage1_results]
    stage2_histories = [item["history"] for item in stage2_results]
    stage1_stats = summarize_run_group(stage1_histories)
    stage2_stats = summarize_run_group(stage2_histories)
    rnn_payload = _baseline_from_source(source_results, "rnn")
    lstm_payload = _baseline_from_source(source_results, "lstm")

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
    plot_mean_std_curve(
        stage2_stats["train_loss_mean"],
        stage2_stats["train_loss_std"],
        title="GFR-RNN Stage-2 Train Loss (mean +/- std)",
        ylabel="Train Loss",
        save_path=output_dir / "stage2_train_loss.png",
    )
    plot_mean_std_curve(
        stage2_stats["test_loss_mean"],
        stage2_stats["test_loss_std"],
        title="GFR-RNN Stage-2 Test Loss (mean +/- std)",
        ylabel="Test Loss",
        save_path=output_dir / "stage2_test_loss.png",
    )
    plot_log_train_loss_comparison(
        {
            "RNN": (
                np.asarray(rnn_payload["train_loss_mean"], dtype=np.float64),
                np.asarray(rnn_payload["train_loss_std"], dtype=np.float64),
            ),
            "GFR-RNN Stage-1": (stage1_stats["train_loss_mean"], stage1_stats["train_loss_std"]),
            "GFR-RNN Stage-2": (stage2_stats["train_loss_mean"], stage2_stats["train_loss_std"]),
            "LSTM": (
                np.asarray(lstm_payload["train_loss_mean"], dtype=np.float64),
                np.asarray(lstm_payload["train_loss_std"], dtype=np.float64),
            ),
        },
        title="L-MNIST Log Train Loss: RNN, GFR-RNN Stage-1/2, LSTM",
        save_path=output_dir / "all_models_log_train_loss.png",
    )
    shutil.copy2(output_dir / "all_models_log_train_loss.png", output_dir / "all_models_log_train_loss_with_stage1.png")
    plot_log_train_loss_comparison(
        {
            "RNN": (
                np.asarray(rnn_payload["train_loss_mean"], dtype=np.float64),
                np.asarray(rnn_payload["train_loss_std"], dtype=np.float64),
            ),
            "GFR-RNN Stage-1": (stage1_stats["train_loss_mean"], stage1_stats["train_loss_std"]),
            "GFR-RNN Stage-2": (stage2_stats["train_loss_mean"], stage2_stats["train_loss_std"]),
        },
        title="L-MNIST Log Train Loss: RNN vs GFR-RNN Stage-1/2",
        save_path=output_dir / "rnn_gfr_log_train_loss.png",
    )

    summary = {
        "config": asdict(cfg),
        "source_baseline_run": str(source_results.get("_source_path", "")),
        "run_seeds": [cfg.base_seed + i for i in range(cfg.n_runs)],
        "gpu_ids": gpu_ids,
        "stage1": {
            "checkpoints": [item["checkpoint"] for item in stage1_results],
            "trainable_params": int(stage1_results[0]["trainable_params"]),
            "total_params": int(stage1_results[0]["total_params"]),
            **stage1_stats,
        },
        "stage2": {
            "trainable_params": int(stage2_results[0]["trainable_params"]),
            "total_params": int(stage2_results[0]["total_params"]),
            **stage2_stats,
        },
        "rnn": rnn_payload,
        "lstm": lstm_payload,
    }
    _write_loss_artifacts(output_dir, summary)
    save_json(output_dir / "results.json", summary)

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Model,Trainable Params,Train Acc Mean,Train Acc Std,Test Acc Mean,Test Acc Std\n")
        f.write(
            f"GFR-RNN (Stage-1),{summary['stage1']['trainable_params']},"
            f"{summary['stage1']['train_acc_mean']:.6f},{summary['stage1']['train_acc_std']:.6f},"
            f"{summary['stage1']['acc_mean']:.6f},{summary['stage1']['acc_std']:.6f}\n"
        )
        f.write(
            f"GFR-RNN (Stage-2),{summary['stage2']['trainable_params']},"
            f"{summary['stage2']['train_acc_mean']:.6f},{summary['stage2']['train_acc_std']:.6f},"
            f"{summary['stage2']['acc_mean']:.6f},{summary['stage2']['acc_std']:.6f}\n"
        )
        f.write(
            f"RNN,{summary['rnn']['trainable_params']},"
            f"{summary['rnn']['train_acc_mean']:.6f},{summary['rnn']['train_acc_std']:.6f},"
            f"{summary['rnn']['acc_mean']:.6f},{summary['rnn']['acc_std']:.6f}\n"
        )
        lstm_params = summary["lstm"].get("matched_trainable_params", summary["lstm"].get("trainable_params", 0))
        f.write(
            f"LSTM (matched),{lstm_params},"
            f"{summary['lstm']['train_acc_mean']:.6f},{summary['lstm']['train_acc_std']:.6f},"
            f"{summary['lstm']['acc_mean']:.6f},{summary['lstm']['acc_std']:.6f}\n"
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun GFR Stage-1/Stage-2 per seed and reuse saved RNN/LSTM baselines.")
    parser.add_argument("--source-run-dir", default="runs/l_mnist_pipeline_20260514_102656")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gpu-ids", default="4,5")
    args = parser.parse_args()

    source_dir = Path(args.source_run_dir)
    with (source_dir / "results.json").open("r", encoding="utf-8") as f:
        source_results = json.load(f)
    source_results["_source_path"] = str(source_dir / "results.json")

    cfg = ExperimentConfig(**source_results["config"])
    cfg.gpu_ids = args.gpu_ids
    gpu_ids = parse_gpu_ids(cfg.gpu_ids)
    if not gpu_ids:
        raise RuntimeError("No GPU ids available for rerun.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_root) / f"l_mnist_gfr_per_seed_{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "histories").mkdir(parents=True, exist_ok=True)
    (output_dir / "progress").mkdir(parents=True, exist_ok=True)
    (output_dir / "run_logs").mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("Source baseline run: %s", source_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("GPU pool: %s", gpu_ids)
    logger.info("Run seeds: %s", [cfg.base_seed + i for i in range(cfg.n_runs)])
    _copy_baseline_artifacts(source_dir, output_dir)

    cfg_payload = asdict(cfg)
    run_seeds = [cfg.base_seed + i for i in range(cfg.n_runs)]
    stage1_tasks = [
        {
            "config": cfg_payload,
            "run_idx": idx,
            "seed": seed,
            "gpu_id": gpu_ids[(idx - 1) % len(gpu_ids)],
            "output_dir": str(output_dir),
        }
        for idx, seed in enumerate(run_seeds, start=1)
    ]
    logger.info("Launching %d Stage-1 reruns with %d workers", len(stage1_tasks), len(gpu_ids))
    stage1_results = _run_parallel(stage1_tasks, _train_stage1, len(gpu_ids), logger, "stage1")

    stage2_tasks = [
        {
            "config": cfg_payload,
            "run_idx": int(stage1_result["run_idx"]),
            "seed": int(stage1_result["seed"]),
            "gpu_id": gpu_ids[(int(stage1_result["run_idx"]) - 1) % len(gpu_ids)],
            "output_dir": str(output_dir),
            "stage1_ckpt": stage1_result["checkpoint"],
        }
        for stage1_result in stage1_results
    ]
    logger.info("Launching %d Stage-2 reruns with %d workers", len(stage2_tasks), len(gpu_ids))
    stage2_results = _run_parallel(stage2_tasks, _train_stage2, len(gpu_ids), logger, "stage2")

    summary = _build_summary(cfg, source_results, output_dir, gpu_ids, stage1_results, stage2_results)
    logger.info(
        "Final | Stage-1 test acc mean/std: %.4f / %.4f | Stage-2: %.4f / %.4f | RNN copied: %.4f / %.4f | LSTM copied: %.4f / %.4f",
        summary["stage1"]["acc_mean"],
        summary["stage1"]["acc_std"],
        summary["stage2"]["acc_mean"],
        summary["stage2"]["acc_std"],
        summary["rnn"]["acc_mean"],
        summary["rnn"]["acc_std"],
        summary["lstm"]["acc_mean"],
        summary["lstm"]["acc_std"],
    )
    logger.info("GFR per-seed rerun complete")


if __name__ == "__main__":
    main()
