import argparse
import concurrent.futures
import csv
import multiprocessing as mp
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from l_mnist_training_pipeline import (
    ExperimentConfig,
    RNNClassifier,
    count_parameters,
    make_gfr_model,
    parse_gpu_ids,
    plot_log_train_loss_comparison,
    plot_mean_std_curve,
    save_json,
    save_progress,
    set_seed,
    setup_logging,
    setup_run_logging,
    summarize_run_group,
)
from snn_network import SNNNetwork, SNNNetworkSynaptic
from utils import reshape_image


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
        "snn_param_target": str(getattr(cfg, "snn_param_target", "gfr-stage1")),
        "snn_target_gfr_hidden_dim": int(getattr(cfg, "snn_target_gfr_hidden_dim", cfg.gfr_hidden_dim)),
        "snn_target_rnn_hidden_dim": int(getattr(cfg, "snn_target_rnn_hidden_dim", cfg.rnn_hidden_dim)),
        "snn_target_trainable_params": int(getattr(cfg, "snn_target_trainable_params", 0)),
        "snn_beta": float(cfg.snn_beta),
        "snn_alpha": float(cfg.snn_alpha),
        "snn_num_workers": int(cfg.snn_num_workers),
        "snn_pin_memory": bool(cfg.snn_pin_memory),
        "snn_persistent_workers": bool(cfg.snn_persistent_workers),
        "snn_prefetch_factor": int(cfg.snn_prefetch_factor),
        "snn_fast_math": bool(cfg.snn_fast_math),
        "snn_amp": bool(cfg.snn_amp),
        "snn_runs_per_gpu": int(cfg.snn_runs_per_gpu),
    }


def config_from_dict(payload: Dict[str, Any]) -> ExperimentConfig:
    cfg = ExperimentConfig()
    for key, value in payload.items():
        setattr(cfg, key, value)
    return cfg


def configure_torch_runtime(fast_math: bool) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = fast_math
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = fast_math
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if fast_math else "highest")


def get_seeded_lmnist_data_loaders(
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> Tuple[DataLoader, DataLoader]:
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST("data/mnist/train", download=True, train=True, transform=transform)
    test_set = torchvision.datasets.MNIST("data/mnist/test", download=True, train=False, transform=transform)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def forward_snn_sequence(model: torch.nn.Module, sequence: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_sequence"):
        return model.forward_sequence(sequence)

    model.reset(sequence.shape[0])
    for step in range(sequence.shape[1]):
        model(sequence[:, step, :])
    return model(model.zero_input(sequence.shape[0]))


def forward_snn_zero_input(model: torch.nn.Module) -> torch.Tensor:
    if hasattr(model, "forward_zero_input"):
        return model.forward_zero_input()
    return model(model.zero_input(model.spk.shape[0]))


def evaluate_snn_sequence_classifier(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    non_blocking: bool,
) -> Tuple[float, float]:
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            sequence = reshape_image(images, variant="l").to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            with autocast_context(device, use_amp):
                logits = forward_snn_sequence(model, sequence)
                loss = criterion(logits, labels)
                vote_scores = torch.softmax(logits, dim=1)
                for _ in range(4):
                    vote_scores += torch.softmax(forward_snn_zero_input(model), dim=1)

            total_loss += loss.item()
            prediction = torch.argmax(vote_scores, dim=1)
            correct += (prediction == labels).sum().item()
            total += labels.numel()

    return total_loss, correct / max(total, 1)


def train_snn_one_run(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    logger,
    run_name: str,
    progress_path: Optional[Path] = None,
    use_amp: bool = False,
    non_blocking: bool = False,
) -> Dict[str, object]:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            sequence = reshape_image(images, variant="l").to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                logits = forward_snn_sequence(model, sequence)
                loss = criterion(logits, labels)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=False)
                optimizer.step()

            total_train_loss += loss.item()
            prediction = torch.argmax(logits.detach(), dim=1)
            train_correct += (prediction == labels).sum().item()
            train_total += labels.numel()

        test_loss, test_acc = evaluate_snn_sequence_classifier(
            model=model,
            data_loader=test_loader,
            device=device,
            use_amp=amp_enabled,
            non_blocking=non_blocking,
        )

        history["train_loss"].append(total_train_loss)
        history["train_acc"].append(train_correct / max(train_total, 1))
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        logger.info(
            "%s | Epoch %03d | train_loss=%.4f | train_acc=%.4f | test_loss=%.4f | test_acc=%.4f",
            run_name,
            epoch + 1,
            history["train_loss"][-1],
            history["train_acc"][-1],
            history["test_loss"][-1],
            history["test_acc"][-1],
        )
        save_progress(
            progress_path,
            {
                "run_name": run_name,
                "status": "running",
                "epoch": epoch + 1,
                "epochs": epochs,
                "history": history,
            },
        )

    eval_train_loss, eval_train_acc = evaluate_snn_sequence_classifier(
        model=model,
        data_loader=train_loader,
        device=device,
        use_amp=amp_enabled,
        non_blocking=non_blocking,
    )
    history["eval_train_loss"] = eval_train_loss
    history["eval_train_acc"] = eval_train_acc
    logger.info(
        "%s | Final train_eval_loss=%.4f | train_eval_acc=%.4f",
        run_name,
        eval_train_loss,
        eval_train_acc,
    )
    save_progress(
        progress_path,
        {
            "run_name": run_name,
            "status": "complete",
            "epoch": epochs,
            "epochs": epochs,
            "history": history,
        },
    )

    return history


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


def choose_gfr_stage1_hidden_size(target_params: int, bio_units: bool, max_hidden: int = 256) -> Tuple[int, int]:
    best_hidden_dim = 1
    best_params = 0
    best_gap = float("inf")
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
            best_hidden_dim = hidden_dim
            best_params = params
            best_gap = gap
    return best_hidden_dim, best_params


def choose_snn_hidden_size(target_params: int, max_hidden: int = 256) -> Tuple[int, int]:
    best_hidden_dim = 1
    best_params = 0
    best_gap = float("inf")
    device = torch.device("cpu")
    for hidden_dim in range(1, max_hidden + 1):
        model = SNNNetwork(in_dim=28, hidden_dim=hidden_dim, out_dim=10, beta=0.95, device=device)
        params = count_parameters(model, trainable_only=True)
        gap = abs(params - target_params)
        if gap < best_gap:
            best_hidden_dim = hidden_dim
            best_params = params
            best_gap = gap
    return best_hidden_dim, best_params


def resolve_parameter_target(cfg: ExperimentConfig) -> Dict[str, Any]:
    target = str(getattr(cfg, "snn_param_target", "gfr-stage1"))
    device = torch.device("cpu")

    if target == "gfr-stage1":
        hidden_dim = int(getattr(cfg, "snn_target_gfr_hidden_dim", cfg.gfr_hidden_dim))
        model = make_gfr_model(
            hidden_dim=hidden_dim,
            freeze_neurons=False,
            freeze_activation=True,
            device=device,
            bio_units=cfg.bio_units,
        )
        return {
            "target": target,
            "label": f"GFR-RNN Phase-1 hidden={hidden_dim}",
            "trainable_params": int(count_parameters(model, trainable_only=True)),
            "hidden_dim": hidden_dim,
            "notes": "GFR fc1/fc2/fc3 and neuron a/b parameters are trainable; activation g is frozen as in the existing Phase-1 scripts.",
        }

    if target == "rnn":
        hidden_dim = int(getattr(cfg, "snn_target_rnn_hidden_dim", cfg.rnn_hidden_dim))
        model = RNNClassifier(input_size=28, hidden_size=hidden_dim, num_classes=cfg.num_classes)
        return {
            "target": target,
            "label": f"RNN hidden={hidden_dim}",
            "trainable_params": int(count_parameters(model, trainable_only=True)),
            "hidden_dim": hidden_dim,
            "notes": "Plain RNN baseline target.",
        }

    if target == "gfr-stage2":
        return {
            "target": target,
            "label": "GFR-RNN Stage-2 effective target",
            "trainable_params": int(cfg.gfr_stage2_target_params),
            "hidden_dim": int(cfg.gfr_hidden_dim),
            "notes": "Stage-2 recurrent-only effective trainable parameter target.",
        }

    if target == "manual":
        params = int(getattr(cfg, "snn_manual_target_params", 0))
        if params <= 0:
            raise ValueError("--param-target manual requires --target-trainable-params > 0")
        return {
            "target": target,
            "label": f"manual target={params}",
            "trainable_params": params,
            "hidden_dim": None,
            "notes": "Manual trainable parameter target.",
        }

    raise ValueError(f"Unknown parameter target: {target}")


def build_parameter_match_report(cfg: ExperimentConfig) -> Dict[str, Any]:
    device = torch.device("cpu")
    target_info = resolve_parameter_target(cfg)
    target_params = int(target_info["trainable_params"])
    nearest_snn_hidden_dim, nearest_snn_params = choose_snn_hidden_size(target_params)

    rnn_hidden_dim = int(getattr(cfg, "snn_target_rnn_hidden_dim", cfg.rnn_hidden_dim))
    rnn_model = RNNClassifier(input_size=28, hidden_size=rnn_hidden_dim, num_classes=cfg.num_classes)
    rnn_params = count_parameters(rnn_model, trainable_only=True)

    snn_lif_params = count_parameters(make_snn_model("snn_lif", cfg, device), trainable_only=True)
    snn_synaptic_params = count_parameters(make_snn_model("snn_synaptic", cfg, device), trainable_only=True)

    target_gfr_hidden_dim = int(getattr(cfg, "snn_target_gfr_hidden_dim", cfg.gfr_hidden_dim))
    gfr_target = make_gfr_model(
        hidden_dim=target_gfr_hidden_dim,
        freeze_neurons=False,
        freeze_activation=True,
        device=device,
        bio_units=cfg.bio_units,
    )
    gfr_target_params = count_parameters(gfr_target, trainable_only=True)

    gfr_same_hidden = make_gfr_model(
        hidden_dim=cfg.snn_hidden_dim,
        freeze_neurons=False,
        freeze_activation=True,
        device=device,
        bio_units=cfg.bio_units,
    )
    gfr_same_hidden_params = count_parameters(gfr_same_hidden, trainable_only=True)
    gfr_hidden_dim, gfr_params = choose_gfr_stage1_hidden_size(rnn_params, bio_units=cfg.bio_units)

    return {
        "target": target_info["target"],
        "target_label": target_info["label"],
        "target_trainable_params": target_params,
        "target_notes": target_info["notes"],
        "rnn_hidden_dim": rnn_hidden_dim,
        "rnn_trainable_params": int(rnn_params),
        "snn_hidden_dim": int(cfg.snn_hidden_dim),
        "snn_lif_trainable_params": int(snn_lif_params),
        "snn_synaptic_trainable_params": int(snn_synaptic_params),
        "snn_lif_gap_vs_target": int(snn_lif_params - target_params),
        "snn_synaptic_gap_vs_target": int(snn_synaptic_params - target_params),
        "snn_matches_target": bool(snn_lif_params == target_params and snn_synaptic_params == target_params),
        "nearest_snn_hidden_dim": int(nearest_snn_hidden_dim),
        "nearest_snn_trainable_params": int(nearest_snn_params),
        "nearest_snn_gap_vs_target": int(nearest_snn_params - target_params),
        "snn_lif_gap_vs_rnn": int(snn_lif_params - rnn_params),
        "snn_synaptic_gap_vs_rnn": int(snn_synaptic_params - rnn_params),
        "snn_matches_rnn": bool(snn_lif_params == rnn_params and snn_synaptic_params == rnn_params),
        "gfr_stage1_target_hidden_dim": int(target_gfr_hidden_dim),
        "gfr_stage1_target_trainable_params": int(gfr_target_params),
        "snn_lif_gap_vs_gfr_stage1_target": int(snn_lif_params - gfr_target_params),
        "snn_synaptic_gap_vs_gfr_stage1_target": int(snn_synaptic_params - gfr_target_params),
        "gfr_stage1_same_hidden_dim": int(cfg.snn_hidden_dim),
        "gfr_stage1_same_hidden_trainable_params": int(gfr_same_hidden_params),
        "gfr_stage1_same_hidden_gap_vs_rnn": int(gfr_same_hidden_params - rnn_params),
        "gfr_stage1_closest_hidden_dim": int(gfr_hidden_dim),
        "gfr_stage1_closest_trainable_params": int(gfr_params),
        "gfr_stage1_closest_gap_vs_rnn": int(gfr_params - rnn_params),
        "gfr_stage2_effective_target_params": int(cfg.gfr_stage2_target_params),
        "snn_lif_gap_vs_gfr_stage2_target": int(snn_lif_params - cfg.gfr_stage2_target_params),
        "snn_synaptic_gap_vs_gfr_stage2_target": int(snn_synaptic_params - cfg.gfr_stage2_target_params),
    }


def log_parameter_match_report(logger, report: Dict[str, Any]) -> None:
    logger.info(
        "Parameter target | %s | trainable=%d | nearest SNN hidden=%d trainable=%d gap=%+d",
        report["target_label"],
        report["target_trainable_params"],
        report["nearest_snn_hidden_dim"],
        report["nearest_snn_trainable_params"],
        report["nearest_snn_gap_vs_target"],
    )
    logger.info(
        "Parameter match | SNN-LIF hidden=%d trainable=%d gap_vs_target=%+d | "
        "SNN-Synaptic trainable=%d gap_vs_target=%+d",
        report["snn_hidden_dim"],
        report["snn_lif_trainable_params"],
        report["snn_lif_gap_vs_target"],
        report["snn_synaptic_trainable_params"],
        report["snn_synaptic_gap_vs_target"],
    )
    logger.info(
        "Parameter reference | RNN hidden=%d trainable=%d | SNN gaps vs RNN: LIF=%+d Synaptic=%+d",
        report["rnn_hidden_dim"],
        report["rnn_trainable_params"],
        report["snn_lif_gap_vs_rnn"],
        report["snn_synaptic_gap_vs_rnn"],
    )
    logger.info(
        "Parameter reference | Stage-1 GFR target hidden=%d trainable=%d | same hidden as SNN=%d trainable=%d | closest GFR to RNN hidden=%d trainable=%d gap_vs_rnn=%+d",
        report["gfr_stage1_target_hidden_dim"],
        report["gfr_stage1_target_trainable_params"],
        report["gfr_stage1_same_hidden_dim"],
        report["gfr_stage1_same_hidden_trainable_params"],
        report["gfr_stage1_closest_hidden_dim"],
        report["gfr_stage1_closest_trainable_params"],
        report["gfr_stage1_closest_gap_vs_rnn"],
    )
    logger.info(
        "Parameter match | default Stage-2 GFR effective target=%d | SNN gaps: LIF=%+d Synaptic=%+d",
        report["gfr_stage2_effective_target_params"],
        report["snn_lif_gap_vs_gfr_stage2_target"],
        report["snn_synaptic_gap_vs_gfr_stage2_target"],
    )


def print_parameter_match_report(report: Dict[str, Any]) -> None:
    print("Parameter check")
    print(f"Target: {report['target_label']}: {report['target_trainable_params']} trainable params")
    print(
        f"Nearest SNN to target: hidden={report['nearest_snn_hidden_dim']}, "
        f"{report['nearest_snn_trainable_params']} trainable params "
        f"(gap vs target {report['nearest_snn_gap_vs_target']:+d})"
    )
    print(
        f"SNN-LIF hidden={report['snn_hidden_dim']}: {report['snn_lif_trainable_params']} "
        f"trainable params (gap vs target {report['snn_lif_gap_vs_target']:+d})"
    )
    print(
        f"SNN-Synaptic hidden={report['snn_hidden_dim']}: {report['snn_synaptic_trainable_params']} "
        f"trainable params (gap vs target {report['snn_synaptic_gap_vs_target']:+d})"
    )
    print(f"RNN hidden={report['rnn_hidden_dim']}: {report['rnn_trainable_params']} trainable params")
    print(
        f"Stage-1 GFR target hidden={report['gfr_stage1_target_hidden_dim']}: "
        f"{report['gfr_stage1_target_trainable_params']} trainable params"
    )
    print(
        f"Stage-1 GFR same hidden={report['gfr_stage1_same_hidden_dim']}: "
        f"{report['gfr_stage1_same_hidden_trainable_params']} trainable params "
        f"(gap vs RNN {report['gfr_stage1_same_hidden_gap_vs_rnn']:+d})"
    )
    print(
        f"Closest Stage-1 GFR to RNN: hidden={report['gfr_stage1_closest_hidden_dim']}, "
        f"{report['gfr_stage1_closest_trainable_params']} trainable params "
        f"(gap vs RNN {report['gfr_stage1_closest_gap_vs_rnn']:+d})"
    )
    print(
        f"Default Stage-2 GFR effective target: {report['gfr_stage2_effective_target_params']} "
        f"(SNN gap {report['snn_lif_gap_vs_gfr_stage2_target']:+d})"
    )


def train_snn_run(task: Dict[str, Any]) -> Dict[str, Any]:
    cfg = config_from_dict(task["config"])
    model_name = str(task["model_name"])
    run_idx = int(task["run_idx"])
    seed = int(task["seed"])
    gpu_id = int(task["gpu_id"])
    output_dir = Path(task["output_dir"])

    set_seed(seed)
    torch.cuda.set_device(gpu_id)
    configure_torch_runtime(bool(cfg.snn_fast_math))
    device = torch.device(f"cuda:{gpu_id}")

    log_dir = output_dir / "run_logs"
    progress_dir = output_dir / "progress"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{model_name}_run{run_idx}"
    logger = setup_run_logging(log_dir / f"{run_name}.log", f"snn_phase1.{run_name}")
    logger.info("%s | assigned_gpu=%d | seed=%d", run_name, gpu_id, seed)

    train_loader, test_loader = get_seeded_lmnist_data_loaders(
        batch_size=cfg.batch_size,
        seed=seed,
        num_workers=cfg.snn_num_workers,
        pin_memory=cfg.snn_pin_memory,
        persistent_workers=cfg.snn_persistent_workers,
        prefetch_factor=cfg.snn_prefetch_factor,
    )
    model = make_snn_model(model_name, cfg, device)
    trainable_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)
    logger.info(
        "%s | trainable_params=%d | total_params=%d | epochs=%d | amp=%s | fast_math=%s | num_workers=%d",
        run_name,
        trainable_params,
        total_params,
        cfg.epochs_snn,
        cfg.snn_amp,
        cfg.snn_fast_math,
        cfg.snn_num_workers,
    )

    history = train_snn_one_run(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=cfg.epochs_snn,
        lr=cfg.lr,
        device=device,
        logger=logger,
        run_name=run_name,
        progress_path=progress_dir / f"{run_name}.json",
        use_amp=cfg.snn_amp,
        non_blocking=cfg.snn_pin_memory,
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


def build_summary(
    cfg: ExperimentConfig,
    output_dir: Path,
    run_seeds: List[int],
    gpu_ids: List[int],
    results: List[Dict[str, Any]],
    parameter_match: Dict[str, Any],
) -> Dict[str, Any]:
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
        "parameter_match": parameter_match,
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
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="SNN hidden size. If omitted, use the nearest SNN size for --param-target.",
    )
    parser.add_argument(
        "--param-target",
        choices=["gfr-stage1", "rnn", "gfr-stage2", "manual"],
        default="gfr-stage1",
        help="Trainable-parameter target used when auto-selecting SNN hidden size.",
    )
    parser.add_argument(
        "--target-gfr-hidden-dim",
        type=int,
        default=65,
        help="Phase-1 GFR hidden size to target when --param-target=gfr-stage1.",
    )
    parser.add_argument(
        "--target-rnn-hidden-dim",
        type=int,
        default=default_cfg.rnn_hidden_dim,
        help="RNN hidden size to target when --param-target=rnn.",
    )
    parser.add_argument(
        "--target-trainable-params",
        type=int,
        default=7355,
        help="Manual trainable-parameter target when --param-target=manual.",
    )
    parser.add_argument("--batch-size", type=int, default=default_cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=default_cfg.epochs_stage1)
    parser.add_argument("--lr", type=float, default=default_cfg.lr)
    parser.add_argument("--base-seed", type=int, default=default_cfg.base_seed)
    parser.add_argument("--n-runs", type=int, default=default_cfg.n_runs)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per training process.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when workers are enabled.")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pinned host memory for CUDA transfers.")
    parser.add_argument("--no-persistent-workers", action="store_true", help="Restart DataLoader workers each epoch.")
    parser.add_argument("--no-fast-math", action="store_true", help="Disable TF32/cudnn benchmark runtime settings.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision for SNN training.")
    parser.add_argument("--runs-per-gpu", type=int, default=1, help="Concurrent SNN runs to launch per GPU.")
    parser.add_argument("--check-params-only", action="store_true", help="Print the parameter-match report and exit without training.")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.epochs_snn = args.epochs
    cfg.n_runs = args.n_runs
    cfg.base_seed = args.base_seed
    cfg.snn_param_target = args.param_target
    cfg.snn_target_gfr_hidden_dim = args.target_gfr_hidden_dim
    cfg.snn_target_rnn_hidden_dim = args.target_rnn_hidden_dim
    cfg.snn_manual_target_params = args.target_trainable_params
    target_info = resolve_parameter_target(cfg)
    cfg.snn_target_trainable_params = int(target_info["trainable_params"])
    if args.hidden_dim is None:
        cfg.snn_hidden_dim, _ = choose_snn_hidden_size(cfg.snn_target_trainable_params)
    else:
        cfg.snn_hidden_dim = args.hidden_dim
    cfg.snn_beta = args.beta
    cfg.snn_alpha = args.alpha
    cfg.gpu_ids = args.gpu_ids
    cfg.parallel_runs = not args.no_parallel
    cfg.snn_num_workers = max(0, args.num_workers)
    cfg.snn_pin_memory = not args.no_pin_memory
    cfg.snn_persistent_workers = not args.no_persistent_workers
    cfg.snn_prefetch_factor = max(1, args.prefetch_factor)
    cfg.snn_fast_math = not args.no_fast_math
    cfg.snn_amp = args.amp
    cfg.snn_runs_per_gpu = max(1, args.runs_per_gpu)

    parameter_match = build_parameter_match_report(cfg)
    if args.check_params_only:
        print_parameter_match_report(parameter_match)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. Please run on a CUDA-enabled environment.")
    configure_torch_runtime(bool(cfg.snn_fast_math))

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
    logger.info("Concurrent workers: %d (%d run(s) per GPU)", len(gpu_ids) * cfg.snn_runs_per_gpu, cfg.snn_runs_per_gpu)
    log_parameter_match_report(logger, parameter_match)

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

    results = run_parallel(tasks, max_workers=len(gpu_ids) * cfg.snn_runs_per_gpu, logger=logger, parallel=cfg.parallel_runs)
    summary = build_summary(cfg, output_dir, run_seeds, gpu_ids, results, parameter_match)
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