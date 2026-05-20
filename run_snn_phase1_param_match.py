import argparse
import concurrent.futures
import csv
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from l_mnist_training_pipeline import (
    ExperimentConfig,
    count_parameters,
    parse_gpu_ids,
    plot_log_train_loss_comparison,
    plot_mean_std_curve,
    save_json,
    set_seed,
    setup_logging,
    setup_run_logging,
    summarize_run_group,
    train_one_run,
)
from snn_network import SNNNetwork, SNNNetworkSynaptic


def config_to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    return {
        "batch_size": int(cfg.batch_size),
        "lr": float(cfg.lr),
        "epochs_snn": int(cfg.epochs_snn),
        "n_runs": int(cfg.n_runs),
        "base_seed": int(cfg.base_seed),
        "num_classes": int(cfg.num_classes),
        "parallel_runs": bool(cfg.parallel_runs),
        "gpu_ids": str(cfg.gpu_ids),
        "snn_hidden_dim": int(cfg.snn_hidden_dim),
        "snn_beta": float(cfg.snn_beta),
        "snn_alpha": float(cfg.snn_alpha),
    }


def config_from_dict(payload: Dict[str, Any]) -> ExperimentConfig:
    cfg = ExperimentConfig()
    for key, value in payload.items():
        setattr(cfg, key, value)
    return cfg


def get_seeded_lmnist_data_loaders(batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader]:
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
        ]
    )
    train_set = torchvision.datasets.MNIST("data/mnist/train", download=True, train=True, transform=transform)
    test_set = torchvision.datasets.MNIST("data/mnist/test", download=True, train=False, transform=transform)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def make_snn_model(model_name: str, cfg: ExperimentConfig, device: torch.device) -> torch.nn.Module:
    if model_name == "snn_lif":
        return SNNNetwork(
            in_dim=28,
            hidden_dim=cfg.snn_hidden_dim,
            out_dim=cfg.num_classes,
            beta=cfg.snn_beta,
            device=device,
        ).to(device)
    if model_name == "snn_synaptic":
        return SNNNetworkSynaptic(
            in_dim=28,
            hidden_dim=cfg.snn_hidden_dim,
            out_dim=cfg.num_classes,
            alpha=cfg.snn_alpha,
            beta=cfg.snn_beta,
            device=device,
        ).to(device)
    raise ValueError(f"Unknown SNN model: {model_name}")


def train_snn_run(task: Dict[str, Any]) -> Dict[str, Any]:
    cfg = config_from_dict(task["config"])
    model_name = str(task["model_name"])
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

    run_name = f"{model_name}_run{run_idx}"
    logger = setup_run_logging(log_dir / f"{run_name}.log", f"snn_phase1.{run_name}")
    logger.info("%s | assigned_gpu=%d | seed=%d", run_name, gpu_id, seed)

    train_loader, test_loader = get_seeded_lmnist_data_loaders(cfg.batch_size, seed=seed)
    model = make_snn_model(model_name, cfg, device)
    trainable_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)
    logger.info("%s | trainable_params=%d | total_params=%d | epochs=%d", run_name, trainable_params, total_params, cfg.epochs_snn)

    history = train_one_run(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=cfg.epochs_snn,
        lr=cfg.lr,
        is_gfr=False,
        device=device,
        logger=logger,
        run_name=run_name,
        progress_path=progress_dir / f"{run_name}.json",
    )

    result = {
        "model_name": model_name,
        "run_idx": run_idx,
        "seed": seed,
        "gpu_id": gpu_id,
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "history": history,
    }
    save_json(output_dir / "histories" / f"{run_name}.json", result)
    logger.info("%s | complete", run_name)
    return result


def run_parallel(tasks: List[Dict[str, Any]], max_workers: int, logger, parallel: bool) -> List[Dict[str, Any]]:
    if not parallel or max_workers <= 1:
        results = []
        for task in tasks:
            result = train_snn_run(task)
            results.append(result)
            logger.info("Completed %s_run%d", task["model_name"], task["run_idx"])
        return results

    results = []
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        future_to_task = {executor.submit(train_snn_run, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            results.append(result)
            logger.info("Completed %s_run%d on GPU %s", task["model_name"], task["run_idx"], task["gpu_id"])
    return results


def write_loss_artifacts(output_dir: Path, summary: Dict[str, Any], model_keys: List[str]) -> None:
    curves_dir = output_dir / "loss_artifacts"
    curves_dir.mkdir(parents=True, exist_ok=True)

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


def build_summary(cfg: ExperimentConfig, output_dir: Path, run_seeds: List[int], gpu_ids: List[int], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    model_keys = ["snn_lif", "snn_synaptic"]
    grouped: Dict[str, List[Dict[str, Any]]] = {key: [] for key in model_keys}
    for result in results:
        grouped[result["model_name"]].append(result)
    for values in grouped.values():
        values.sort(key=lambda item: int(item["run_idx"]))

    summary: Dict[str, Any] = {
        "config": config_to_dict(cfg),
        "run_seeds": run_seeds,
        "gpu_ids": gpu_ids,
        "phase": "phase1_parameter_matched_snn_baselines",
        "notes": "SNN neuronal decay hyperparameters are fixed; trainable parameters are fc1/fc2/fc3 weights and biases only.",
    }

    for key in model_keys:
        histories = [item["history"] for item in grouped[key]]
        stats = summarize_run_group(histories)
        summary[key] = {
            "hidden_size": int(cfg.snn_hidden_dim),
            "trainable_params": int(grouped[key][0]["trainable_params"]),
            "total_params": int(grouped[key][0]["total_params"]),
            "beta": float(cfg.snn_beta),
            **stats,
        }
        if key == "snn_synaptic":
            summary[key]["alpha"] = float(cfg.snn_alpha)

    write_loss_artifacts(output_dir, summary, model_keys)
    save_json(output_dir / "results.json", summary)

    plot_mean_std_curve(
        summary["snn_lif"]["train_loss_mean"],
        summary["snn_lif"]["train_loss_std"],
        title="SNN-LIF Train Loss (mean +/- std)",
        ylabel="Train Loss",
        save_path=output_dir / "snn_lif_train_loss.png",
    )
    plot_mean_std_curve(
        summary["snn_lif"]["test_loss_mean"],
        summary["snn_lif"]["test_loss_std"],
        title="SNN-LIF Test Loss (mean +/- std)",
        ylabel="Test Loss",
        save_path=output_dir / "snn_lif_test_loss.png",
    )
    plot_mean_std_curve(
        summary["snn_synaptic"]["train_loss_mean"],
        summary["snn_synaptic"]["train_loss_std"],
        title="SNN-Synaptic Train Loss (mean +/- std)",
        ylabel="Train Loss",
        save_path=output_dir / "snn_synaptic_train_loss.png",
    )
    plot_mean_std_curve(
        summary["snn_synaptic"]["test_loss_mean"],
        summary["snn_synaptic"]["test_loss_std"],
        title="SNN-Synaptic Test Loss (mean +/- std)",
        ylabel="Test Loss",
        save_path=output_dir / "snn_synaptic_test_loss.png",
    )
    plot_log_train_loss_comparison(
        {
            "SNN-LIF": (summary["snn_lif"]["train_loss_mean"], summary["snn_lif"]["train_loss_std"]),
            "SNN-Synaptic": (summary["snn_synaptic"]["train_loss_mean"], summary["snn_synaptic"]["train_loss_std"]),
        },
        title="L-MNIST Log Train Loss: SNN Baselines",
        save_path=output_dir / "snn_log_train_loss.png",
    )

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Model,Trainable Params,Train Acc Mean,Train Acc Std,Test Acc Mean,Test Acc Std\n")
        f.write(
            f"SNN-LIF,{summary['snn_lif']['trainable_params']},"
            f"{summary['snn_lif']['train_acc_mean']:.6f},{summary['snn_lif']['train_acc_std']:.6f},"
            f"{summary['snn_lif']['acc_mean']:.6f},{summary['snn_lif']['acc_std']:.6f}\n"
        )
        f.write(
            f"SNN-Synaptic,{summary['snn_synaptic']['trainable_params']},"
            f"{summary['snn_synaptic']['train_acc_mean']:.6f},{summary['snn_synaptic']['train_acc_std']:.6f},"
            f"{summary['snn_synaptic']['acc_mean']:.6f},{summary['snn_synaptic']['acc_std']:.6f}\n"
        )

    return summary


def main() -> None:
    default_cfg = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Run Phase-1 parameter-matched SNN baselines on L-MNIST.")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gpu-ids", type=str, default="")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=68, help="SNN hidden size; 68 gives 7354 trainable params.")
    parser.add_argument("--batch-size", type=int, default=default_cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=default_cfg.epochs_stage1)
    parser.add_argument("--lr", type=float, default=default_cfg.lr)
    parser.add_argument("--base-seed", type=int, default=default_cfg.base_seed)
    parser.add_argument("--n-runs", type=int, default=default_cfg.n_runs)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.9)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. Please run on a CUDA-enabled environment.")

    cfg = ExperimentConfig()
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.epochs_snn = args.epochs
    cfg.n_runs = args.n_runs
    cfg.base_seed = args.base_seed
    cfg.snn_hidden_dim = args.hidden_dim
    cfg.snn_beta = args.beta
    cfg.snn_alpha = args.alpha
    cfg.gpu_ids = args.gpu_ids
    cfg.parallel_runs = not args.no_parallel

    gpu_ids = parse_gpu_ids(cfg.gpu_ids)
    if not gpu_ids:
        raise RuntimeError("No GPU IDs available. Pass --gpu-ids explicitly or set CUDA_VISIBLE_DEVICES.")

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_root / f"l_mnist_snn_phase1_param_match_{timestamp}"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "histories").mkdir(parents=True, exist_ok=True)
    (output_dir / "progress").mkdir(parents=True, exist_ok=True)
    (output_dir / "run_logs").mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    run_seeds = [cfg.base_seed + i for i in range(cfg.n_runs)]
    cfg_payload = config_to_dict(cfg)
    model_keys = ["snn_lif", "snn_synaptic"]

    logger.info("Starting Phase-1 parameter-matched SNN baseline run")
    logger.info("Output directory: %s", output_dir)
    logger.info("Config: %s", config_to_dict(cfg))
    logger.info("Run seeds: %s", run_seeds)
    logger.info("GPU pool: %s", gpu_ids)

    for model_name in model_keys:
        model = make_snn_model(model_name, cfg, torch.device("cpu"))
        logger.info("%s | hidden_dim=%d | trainable_params=%d", model_name, cfg.snn_hidden_dim, count_parameters(model, trainable_only=True))

    tasks = []
    for model_name in model_keys:
        for idx, seed in enumerate(run_seeds, start=1):
            tasks.append(
                {
                    "config": cfg_payload,
                    "model_name": model_name,
                    "run_idx": idx,
                    "seed": seed,
                    "gpu_id": gpu_ids[len(tasks) % len(gpu_ids)],
                    "output_dir": str(output_dir),
                }
            )

    results = run_parallel(tasks, max_workers=len(gpu_ids), logger=logger, parallel=cfg.parallel_runs)
    summary = build_summary(cfg, output_dir, run_seeds, gpu_ids, results)
    logger.info(
        "Final | SNN-LIF test acc mean/std: %.4f / %.4f | SNN-Synaptic: %.4f / %.4f",
        summary["snn_lif"]["acc_mean"],
        summary["snn_lif"]["acc_std"],
        summary["snn_synaptic"]["acc_mean"],
        summary["snn_synaptic"]["acc_std"],
    )
    logger.info("Phase-1 parameter-matched SNN baseline run complete")


if __name__ == "__main__":
    main()